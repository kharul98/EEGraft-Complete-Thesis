#!/usr/bin/env python3
"""Audit EEGNet predictions and diagnose fold-1 training (no test-set retraining).

Run from ~/alzheimers-thesis, with eegnet_46subjects_cv.py alongside this file:
    python eegnet_diagnostic_fold1.py

Inputs: existing predictions, manifest and derivative EEG .set files.
Writes only reports/eegnet_fold1_learning_diagnostic.csv.
No modification to source data or model checkpoints.
"""
from pathlib import Path
import csv
import os
from collections import Counter

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from eegnet_46subjects_cv import (
    CompactEEGNet, prepare_windows, load_subjects,
    predict_subjects, seed_everything,
)

ROOT = Path.cwd()
REPORTS = ROOT / 'reports'
SEED = 42
MAX_DIAG_EPOCHS = 35


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def by_subject(rows, path):
    grouped = {r['subject']: r for r in rows}
    if len(grouped) != len(rows):
        raise RuntimeError(f'Duplicate subjects in {path}')
    return grouped


def inspect_saved_predictions():
    print('===== EXISTING PREDICTIONS AUDIT =====')
    specs = [
        ('SVM', REPORTS / 'baseline_predictions_v2.csv',
         'SVM_prediction', 'SVM_ad_probability'),
        ('LogisticRegression', REPORTS / 'baseline_predictions_v2.csv',
         'LogisticRegression_prediction', 'LogisticRegression_ad_probability'),
        ('BasicCNN', REPORTS / 'eeg_cnn_cv_predictions.csv',
         'predicted_label', 'ad_probability'),
        ('EEGNet', REPORTS / 'eegnet_cv_predictions.csv',
         'predicted_label', 'ad_probability'),
    ]
    imported = {}
    for name, path, predkey, probkey in specs:
        if not path.exists():
            raise FileNotFoundError(f'Missing predictions file: {path}')
        records = by_subject(read_rows(path), path)
        imported[name] = (records, predkey, probkey)

    universe = set(imported['EEGNet'][0])
    if len(universe) != 46:
        raise RuntimeError(f'Expected 46 EEGNet subjects; got {len(universe)}')
    ref = imported['EEGNet'][0]
    for name, (records, _, _) in imported.items():
        if set(records) != universe:
            raise RuntimeError(f'{name}: mismatch in subject IDs')
        different = [sid for sid in universe if (
            records[sid]['group'] != ref[sid]['group']
            or int(records[sid]['fold']) != int(ref[sid]['fold'])
        )]
        if different:
            raise RuntimeError(
                f'{name}: label or fold mismatch for {different[:8]}')
    print('Aligned subjects, labels and fold assignments: PASS (46/46)')

    sorted_subjects = sorted(universe)
    y = np.asarray([int(ref[sid]['group'] == 'A') for sid in sorted_subjects])
    model_preds = {}
    for name, (records, predkey, probkey) in imported.items():
        pred = np.asarray([int(records[sid][predkey]) for sid in sorted_subjects])
        prob = np.asarray([float(records[sid][probkey]) for sid in sorted_subjects])
        if not np.isfinite(prob).all():
            raise RuntimeError(f'Invalid probabilities for {name}')
        model_preds[name] = pred
        print(f'\nMODEL: {name}')
        print(f'Accuracy: {accuracy_score(y, pred):.4f}')
        print(f'Balanced accuracy: {balanced_accuracy_score(y, pred):.4f}')
        print(f'ROC-AUC: {roc_auc_score(y, prob):.4f}')
        print(f'Probability range: {prob.min():.3f} to {prob.max():.3f}')
        print(f'Mean AD probability for AD subjects: {prob[y==1].mean():.3f}')
        print(f'Mean AD probability for healthy subjects: {prob[y==0].mean():.3f}')
        print(f'Subjects with probabilities 0.40 to 0.60: {int(((prob >= 0.40) & (prob <= 0.60)).sum())}/46')

    svm = model_preds['SVM']
    eegnet = model_preds['EEGNet']
    print('\nEEGNet vs SVM, paired subjects:')
    print('EEGNet correct, SVM wrong:', int(((eegnet == y) & (svm != y)).sum()))
    print('SVM correct, EEGNet wrong:', int(((svm == y) & (eegnet != y)).sum()))
    print('Both correct:', int(((svm == y) & (eegnet == y)).sum()))
    print('Both wrong:', int(((svm != y) & (eegnet != y)).sum()))
    return sorted_subjects


def check_fold1_learning():
    print('\n===== EEGNET FOLD-1 TRAINING DIAGNOSTIC =====')
    seed_everything(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = load_subjects(ROOT)
    y = np.asarray([int(r['group'] == 'A') for r in rows], dtype=int)
    outer_train, outer_test = next(iter(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        .split(np.zeros(len(y)), y)
    ))
    train_ids, val_ids = train_test_split(
        outer_train, test_size=0.25, stratify=y[outer_train],
        random_state=101
    )
    print('Device:', device)
    print('Train/validation/untouched-test subjects:',
          len(train_ids), len(val_ids), len(outer_test))
    print('No diagnostic metrics will be computed on held-out test subjects.')
    print('Preparing EEG segments...')
    windows, _ = prepare_windows(rows)

    seed_everything(SEED + 1)
    x_train = np.concatenate([windows[int(i)] for i in train_ids])
    y_train = np.concatenate([
        np.full(len(windows[int(i)]), y[int(i)], dtype=np.float32)
        for i in train_ids
    ])
    weights = np.concatenate([
        np.full(len(windows[int(i)]), 1.0/len(windows[int(i)]))
        for i in train_ids
    ])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32,
        sampler=WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True,
            generator=torch.Generator().manual_seed(SEED + 1)
        ), num_workers=0
    )
    model = CompactEEGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, MAX_DIAG_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        # Same evaluation mode for training and validation SUBJECT predictions.
        train_prob = predict_subjects(model, train_ids, windows, device)
        val_prob = predict_subjects(model, val_ids, windows, device)
        val_clip = np.clip(val_prob, 1e-6, 1-1e-6)
        vlabels = y[val_ids]
        val_loss = float(np.mean(
            -vlabels*np.log(val_clip) - (1-vlabels)*np.log(1-val_clip)
        ))
        train_acc = accuracy_score(y[train_ids], (train_prob >= 0.5).astype(int))
        val_acc = accuracy_score(vlabels, (val_prob >= 0.5).astype(int))
        val_auc = roc_auc_score(vlabels, val_prob)
        rec = {
            'epoch': epoch,
            'train_clip_loss': round(total_loss / steps, 6),
            'train_subject_accuracy': round(train_acc, 4),
            'val_subject_logloss': round(val_loss, 6),
            'val_subject_accuracy': round(val_acc, 4),
            'val_subject_auc': round(val_auc, 4),
            'val_mean_probability_ad': round(float(val_prob[vlabels==1].mean()), 5),
            'val_mean_probability_healthy': round(float(val_prob[vlabels==0].mean()), 5),
        }
        history.append(rec)
        if epoch <= 3 or epoch % 5 == 0 or epoch == MAX_DIAG_EPOCHS:
            print('Epoch {epoch:02d}: train loss={train_clip_loss:.3f}, '
                  'train subject acc={train_subject_accuracy:.3f}, '
                  'val loss={val_subject_logloss:.3f}, '
                  'val subject acc={val_subject_accuracy:.3f}, '
                  'val AUC={val_subject_auc:.3f}'.format(**rec))

    outfile = REPORTS / 'eegnet_fold1_learning_diagnostic.csv'
    with outfile.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best = min(history, key=lambda d: d['val_subject_logloss'])
    print('\n===== DIAGNOSTIC SUMMARY =====')
    print('Best validation-loss epoch through 35:', best['epoch'])
    print('Epoch 1 training loss:', history[0]['train_clip_loss'])
    print('Epoch 35 training loss:', history[-1]['train_clip_loss'])
    print('Epoch 1 train subject accuracy:', history[0]['train_subject_accuracy'])
    print('Epoch 35 train subject accuracy:', history[-1]['train_subject_accuracy'])
    print('Best validation loss:', best['val_subject_logloss'])
    print('Report:', outfile)
    print('No changes to original files or saved checkpoints.')
    print('This is a training diagnostic, NOT a new test-set result.')


if __name__ == '__main__':
    inspect_saved_predictions()
    check_fold1_learning()
