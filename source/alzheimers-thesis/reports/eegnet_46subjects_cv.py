#!/usr/bin/env python3
"""EEGNet-style 5-fold subject-level AD vs control baseline for OpenNeuro ds004504.

Run from the alzheimers-thesis repository root:
    python eegnet_46subjects_cv.py

Inputs:
    reports/openneuro_5min_manifest_v2.csv
    derivative EEG .set files referenced by that manifest

Outputs:
    reports/eegnet_cv_predictions.csv
    reports/eegnet_cv_summary.csv
    models/eegnet_cv/fold_{1..5}.pt

Preliminary exploratory evaluation; not an external clinical validation.
"""
from pathlib import Path
import csv
import os
import random
from collections import Counter

import mne
import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

SEED = 42
FS_ORIGINAL = 500
FS_MODEL = 125
N_CHANNELS = 19
WINDOW_SECONDS = 300
EPOCH_SECONDS = 2
CLIP_SECONDS = 10
N_FOLDS = 5
MAX_EPOCHS = 25
PATIENCE = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subjects(root):
    manifest = root / 'reports/openneuro_5min_manifest_v2.csv'
    with manifest.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['status'] == 'CANDIDATE' and r['group'] in ('A', 'C')]
    labels = Counter(r['group'] for r in rows)
    if len(rows) != 46 or labels != {'A': 23, 'C': 23}:
        raise RuntimeError(f'Expected 23 AD + 23 controls, found {dict(labels)}')
    if len({r['subject'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate subjects found.')
    return rows


def prepare_windows(rows):
    windows = []
    channel_order = None
    total = 0
    for i, row in enumerate(rows, 1):
        file = Path(row['file'])
        raw = mne.io.read_raw_eeglab(str(file), preload=False, verbose='ERROR')
        sf = float(raw.info['sfreq'])
        if abs(sf - FS_ORIGINAL) > 0.01:
            raise RuntimeError(f"Unexpected sampling rate: {row['subject']} {sf}")
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        names = tuple(raw.ch_names[p] for p in picks)
        if len(picks) != N_CHANNELS:
            raise RuntimeError(f"Wrong channel count for {row['subject']}")
        if channel_order is None:
            channel_order = names
        if names != channel_order:
            raise RuntimeError(f"Channel order mismatch for {row['subject']}")

        start = int(round(float(row['start_sec']) * sf))
        stop = start + FS_ORIGINAL * WINDOW_SECONDS
        if start < 0 or stop > raw.n_times:
            raise RuntimeError(f"Invalid 300s window for {row['subject']}")
        data = raw.get_data(picks=picks, start=start, stop=stop)
        data = data.astype(np.float32) * 1e6  # V to microvolts
        if not np.isfinite(data).all():
            raise RuntimeError(f"Nonfinite data for {row['subject']}")

        samples_2s = FS_ORIGINAL * EPOCH_SECONDS
        two_second = data.reshape(N_CHANNELS, 150, samples_2s)
        ptp = np.ptp(two_second, axis=2)
        sd = np.std(two_second, axis=2)
        bad_2s = ((ptp > 200).sum(axis=0) >= 3) | (
            (ptp > 300).any(axis=0)
        ) | ((sd < 0.5).any(axis=0))

        clips = []
        for k in range(WINDOW_SECONDS // CLIP_SECONDS):
            # 10s clips comprise five consecutive 2s intervals.
            left, right = k * 5, (k + 1) * 5
            if bad_2s[left:right].any():
                continue
            clip = data[:, left*samples_2s:right*samples_2s]
            clip = resample_poly(clip, 1, 4, axis=1).astype(np.float32)
            # Predefined per-clip centering and fixed scaling; no data-learned statistics.
            clip -= clip.mean(axis=1, keepdims=True)
            clip /= 100.0
            if clip.shape != (N_CHANNELS, FS_MODEL * CLIP_SECONDS):
                raise RuntimeError('Unexpected resampled shape')
            clips.append(clip)
        if len(clips) < 3:
            raise RuntimeError(f"Too few usable clips: {row['subject']} ({len(clips)})")
        windows.append(np.stack(clips))
        total += len(clips)
        if i % 10 == 0 or i == len(rows):
            print(f'Prepared {i}/{len(rows)} participants')
    return windows, channel_order


class CompactEEGNet(nn.Module):
    """EEGNet-8,2-style architecture; independently implemented, not a Braindecode clone."""
    def __init__(self, channels=N_CHANNELS, samples=FS_MODEL * CLIP_SECONDS):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConstantPad2d((31, 32, 0, 0), 0.0),
            nn.Conv2d(1, 8, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(8, eps=1e-3, momentum=0.01),
            nn.Conv2d(8, 16, kernel_size=(channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5),
            nn.ConstantPad2d((7, 8, 0, 0), 0.0),
            nn.Conv2d(16, 16, kernel_size=(1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5),
        )
        # Temporal convolutions preserve the input length, pooling reduces by 4*8.
        self.classifier = nn.Linear(16 * (samples // 4 // 8), 1)

    def forward(self, x):  # x: batch x channels x samples
        x = self.features(x.unsqueeze(1)).flatten(1)
        return self.classifier(x).squeeze(-1)


def predict_subjects(model, indices, windows, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for index in indices:
            clips = torch.from_numpy(windows[int(index)]).to(device)
            # Average the probabilities of the *same* subject; not majority vote by clip.
            probabilities.append(float(torch.sigmoid(model(clips)).mean().item()))
    return np.asarray(probabilities, dtype=float)


def main():
    seed_everything(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path.cwd()
    report_dir = root / 'reports'
    model_dir = root / 'models/eegnet_cv'
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = load_subjects(root)
    y = np.asarray([int(r['group'] == 'A') for r in rows], dtype=int)

    print('===== EEGNET-STYLE EXPERIMENT =====')
    print('Device:', device)
    print('Subjects:', len(rows), 'AD:', int(y.sum()), 'Healthy:', int((1-y).sum()))
    windows, channel_order = prepare_windows(rows)
    print('Total nonoverlapping 10s clips:', sum(map(len, windows)))
    print('Fewest clips per participant:', min(map(len, windows)))

    probe = CompactEEGNet().to(device)
    with torch.no_grad():
        shape = tuple(probe(torch.zeros(2, N_CHANNELS, 1250, device=device)).shape)
    assert shape == (2,), shape
    print('Model parameters:', sum(p.numel() for p in probe.parameters()))
    del probe

    oof_prob = np.full(len(rows), np.nan)
    folds = np.zeros(len(rows), dtype=int)
    best_epochs = {}
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (outer_train, outer_test) in enumerate(outer.split(np.zeros(len(y)), y), 1):
        seed_everything(SEED + fold)
        train_ids, val_ids = train_test_split(
            outer_train, test_size=0.25, stratify=y[outer_train],
            random_state=100 + fold
        )
        assert not (set(train_ids) & set(val_ids))
        assert not (set(outer_test) & (set(train_ids) | set(val_ids)))

        x_train = np.concatenate([windows[int(i)] for i in train_ids])
        y_train = np.concatenate([
            np.full(len(windows[int(i)]), y[int(i)], dtype=np.float32)
            for i in train_ids
        ])
        weights = np.concatenate([
            np.full(len(windows[int(i)]), 1.0/len(windows[int(i)]))
            for i in train_ids
        ])
        data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True,
            generator=torch.Generator().manual_seed(SEED + fold)
        )
        loader = DataLoader(data, batch_size=32, sampler=sampler, num_workers=0)
        model = CompactEEGNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_loss, best_state, best_epoch, stale = float('inf'), None, 0, 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            val_probs = np.clip(
                predict_subjects(model, val_ids, windows, device), 1e-6, 1-1e-6
            )
            val_labels = y[val_ids]
            val_loss = float(np.mean(-val_labels*np.log(val_probs)
                                     -(1-val_labels)*np.log(1-val_probs)))
            if val_loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= PATIENCE:
                break

        model.load_state_dict(best_state)
        prob = predict_subjects(model, outer_test, windows, device)
        oof_prob[outer_test] = prob
        folds[outer_test] = fold
        best_epochs[fold] = best_epoch
        pred = (prob >= 0.5).astype(int)
        print(f'Fold {fold}: best epoch={best_epoch}, '
              f'balanced accuracy={balanced_accuracy_score(y[outer_test], pred):.3f}, '
              f'AUC={roc_auc_score(y[outer_test], prob):.3f}')
        torch.save({
            'state_dict': best_state, 'channels': channel_order,
            'sampling_rate_hz': FS_MODEL, 'fold': fold,
            'best_epoch': best_epoch, 'architecture': 'EEGNet-8,2-style'
        }, model_dir / f'fold_{fold}.pt')

    if not np.isfinite(oof_prob).all() or not (folds > 0).all():
        raise RuntimeError('Some participant predictions are missing')
    pred = (oof_prob >= 0.5).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    metrics = {
        'subjects': len(y), 'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'roc_auc': roc_auc_score(y, oof_prob),
        'sensitivity': tp/(tp+fn), 'specificity': tn/(tn+fp),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    print('\n===== EEGNET OUT-OF-FOLD RESULTS =====')
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')
    print('Confusion matrix [Healthy, AD]:', [[tn, fp], [fn, tp]])

    with (report_dir / 'eegnet_cv_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject','group','fold','segments','true_label',
                         'predicted_label','ad_probability'])
        for i, r in enumerate(rows):
            writer.writerow([r['subject'], r['group'], int(folds[i]),
                             len(windows[i]), int(y[i]), int(pred[i]),
                             round(float(oof_prob[i]), 6)])
    with (report_dir / 'eegnet_cv_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print('\nPredictions: reports/eegnet_cv_predictions.csv')
    print('Summary: reports/eegnet_cv_summary.csv')
    print('Models: models/eegnet_cv/fold_*.pt')
    print('Original EEG unchanged. Subject-level cross-validation only.')
    print('===== EXPERIMENT COMPLETE =====')


if __name__ == '__main__':
    main()
