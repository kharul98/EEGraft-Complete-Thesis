#!/usr/bin/env python3

"""
EEGNet v2: Extended Training
Alzheimer's Disease vs Healthy Controls
OpenNeuro ds004504

Subject-level 5-fold cross-validation.
46 subjects: 23 AD and 23 Healthy.

Original EEG recordings are not modified.
"""

from pathlib import Path
from collections import Counter
import csv
import os
import random

import numpy as np
import mne

from scipy.signal import resample_poly

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score
)

import torch
from torch import nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler
)


# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================

SEED = 42

FS_ORIGINAL = 500
FS_MODEL = 125

N_CHANNELS = 19

WINDOW_SECONDS = 300
EPOCH_SECONDS = 2
CLIP_SECONDS = 10

N_FOLDS = 5

MAX_EPOCHS = 60
PATIENCE = 12
WARMUP_EPOCHS = 12
MIN_EPOCHS = 20


# ==========================================
# REPRODUCIBILITY
# ==========================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# LOAD DATASET MANIFEST
# ==========================================

def load_subjects(root):

    manifest = (
        root /
        "reports/openneuro_5min_manifest_v2.csv"
    )

    with manifest.open(newline="") as f:

        rows = [
            r for r in csv.DictReader(f)
            if r["status"] == "CANDIDATE"
            and r["group"] in ("A", "C")
        ]

    labels = Counter(
        r["group"] for r in rows
    )

    if (
        len(rows) != 46
        or labels != {"A": 23, "C": 23}
    ):
        raise RuntimeError(
            f"Expected 23 AD and 23 controls. "
            f"Found: {dict(labels)}"
        )

    subjects = [
        r["subject"] for r in rows
    ]

    if len(set(subjects)) != len(subjects):
        raise RuntimeError(
            "Duplicate subject IDs detected."
        )

    return rows


# ==========================================
# PREPARE EEG SEGMENTS
# ==========================================

def prepare_windows(rows):

    windows = []
    channel_order = None

    for i, row in enumerate(rows, 1):

        file = Path(row["file"])

        raw = mne.io.read_raw_eeglab(
            str(file),
            preload=False,
            verbose="ERROR"
        )

        sf = float(raw.info["sfreq"])

        if abs(sf - FS_ORIGINAL) > 0.01:

            raise RuntimeError(
                f"Unexpected sampling rate "
                f"for {row['subject']}: {sf}"
            )

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        names = tuple(
            raw.ch_names[p] for p in picks
        )

        if len(picks) != N_CHANNELS:

            raise RuntimeError(
                f"Unexpected EEG channel count "
                f"for {row['subject']}"
            )

        if channel_order is None:
            channel_order = names

        if names != channel_order:

            raise RuntimeError(
                f"Channel order mismatch: "
                f"{row['subject']}"
            )

        start = int(
            round(float(row["start_sec"]) * sf)
        )

        stop = (
            start +
            FS_ORIGINAL * WINDOW_SECONDS
        )

        if start < 0 or stop > raw.n_times:

            raise RuntimeError(
                f"Invalid 300-second window: "
                f"{row['subject']}"
            )

        data = raw.get_data(
            picks=picks,
            start=start,
            stop=stop
        )

        # Convert volts to microvolts.
        data = (
            data.astype(np.float32) * 1e6
        )

        if not np.isfinite(data).all():

            raise RuntimeError(
                f"Non-finite samples: "
                f"{row['subject']}"
            )

        # Split into 2-second epochs.
        samples_2s = (
            FS_ORIGINAL * EPOCH_SECONDS
        )

        two_second = data.reshape(
            N_CHANNELS,
            150,
            samples_2s
        )

        ptp = np.ptp(
            two_second,
            axis=2
        )

        sd = np.std(
            two_second,
            axis=2
        )

        # Preliminary artifact screening.
        bad_2s = (
            ((ptp > 200).sum(axis=0) >= 3)
            |
            ((ptp > 300).any(axis=0))
            |
            ((sd < 0.5).any(axis=0))
        )

        clips = []

        # Create 30 non-overlapping 10s clips.
        for k in range(
            WINDOW_SECONDS // CLIP_SECONDS
        ):

            left = k * 5
            right = (k + 1) * 5

            if bad_2s[left:right].any():
                continue

            clip = data[
                :,
                left * samples_2s:
                right * samples_2s
            ]

            # Resample 500 Hz -> 125 Hz.
            clip = resample_poly(
                clip,
                1,
                4,
                axis=1
            ).astype(np.float32)

            # Per-clip DC removal.
            clip -= clip.mean(
                axis=1,
                keepdims=True
            )

            # Fixed amplitude scaling.
            clip /= 100.0

            expected_shape = (
                N_CHANNELS,
                FS_MODEL * CLIP_SECONDS
            )

            if clip.shape != expected_shape:

                raise RuntimeError(
                    "Unexpected resampled EEG shape."
                )

            clips.append(clip)

        if len(clips) < 3:

            raise RuntimeError(
                f"Insufficient clean clips: "
                f"{row['subject']}"
            )

        windows.append(
            np.stack(clips)
        )

        if i % 10 == 0 or i == len(rows):

            print(
                f"Prepared {i}/{len(rows)} "
                "participants"
            )

    return windows, channel_order


# ==========================================
# EEGNET-STYLE ARCHITECTURE
# ==========================================

class CompactEEGNet(nn.Module):

    def __init__(
        self,
        channels=N_CHANNELS,
        samples=FS_MODEL * CLIP_SECONDS
    ):

        super().__init__()

        self.features = nn.Sequential(

            # Temporal convolution.
            nn.ConstantPad2d(
                (31, 32, 0, 0),
                0.0
            ),

            nn.Conv2d(
                1,
                8,
                kernel_size=(1, 64),
                bias=False
            ),

            nn.BatchNorm2d(
                8,
                eps=1e-3,
                momentum=0.01
            ),

            # Depthwise spatial convolution.
            nn.Conv2d(
                8,
                16,
                kernel_size=(channels, 1),
                groups=8,
                bias=False
            ),

            nn.BatchNorm2d(
                16,
                eps=1e-3,
                momentum=0.01
            ),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            nn.Dropout(p=0.5),

            # Depthwise separable temporal block.
            nn.ConstantPad2d(
                (7, 8, 0, 0),
                0.0
            ),

            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 16),
                groups=16,
                bias=False
            ),

            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(
                16,
                eps=1e-3,
                momentum=0.01
            ),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Linear(
            16 * (samples // 4 // 8),
            1
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.features(x)

        x = x.flatten(1)

        x = self.classifier(x)

        return x.squeeze(-1)


# ==========================================
# SUBJECT-LEVEL PREDICTION
# ==========================================

def predict_subjects(
    model,
    indices,
    windows,
    device
):

    model.eval()

    probabilities = []

    with torch.no_grad():

        for index in indices:

            clips = torch.from_numpy(
                windows[int(index)]
            ).to(device)

            logits = model(clips)

            probability = (
                torch.sigmoid(logits)
                .mean()
                .item()
            )

            probabilities.append(
                float(probability)
            )

    return np.asarray(
        probabilities,
        dtype=float
    )


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    seed_everything(SEED)

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = Path.cwd()

    report_dir = root / "reports"

    model_dir = (
        root / "models/eegnet_v2_cv"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = load_subjects(root)

    y = np.asarray(
        [
            int(r["group"] == "A")
            for r in rows
        ],
        dtype=int
    )

    print(
        "===== EEGNET EXTENDED-TRAINING "
        "EXPERIMENT ====="
    )

    print("Device:", device)
    print("Subjects:", len(rows))
    print("AD:", int(y.sum()))
    print("Healthy:", int((1-y).sum()))

    print("Maximum epochs:", MAX_EPOCHS)
    print("Warmup epochs:", WARMUP_EPOCHS)
    print("Minimum epochs:", MIN_EPOCHS)
    print("Patience:", PATIENCE)

    print(
        "\n===== PREPARING EEG SEGMENTS ====="
    )

    windows, channel_order = (
        prepare_windows(rows)
    )

    print(
        "Total nonoverlapping 10s clips:",
        sum(map(len, windows))
    )

    print(
        "Fewest clips per participant:",
        min(map(len, windows))
    )

    # Verify network output shape.
    probe = CompactEEGNet().to(device)

    with torch.no_grad():

        dummy = torch.zeros(
            2,
            N_CHANNELS,
            FS_MODEL * CLIP_SECONDS,
            device=device
        )

        output_shape = tuple(
            probe(dummy).shape
        )

    assert output_shape == (2,)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in probe.parameters()
        )
    )

    del probe

    # ======================================
    # CROSS-VALIDATION
    # ======================================

    oof_prob = np.full(
        len(rows),
        np.nan
    )

    folds = np.zeros(
        len(rows),
        dtype=int
    )

    histories = []

    outer = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    print(
        "\n===== FIVE-FOLD EEGNET V2 ====="
    )

    for fold, (
        outer_train,
        outer_test
    ) in enumerate(
        outer.split(
            np.zeros(len(y)),
            y
        ),
        1
    ):

        seed_everything(
            SEED + fold
        )

        train_ids, val_ids = (
            train_test_split(
                outer_train,
                test_size=0.25,
                stratify=y[outer_train],
                random_state=100 + fold
            )
        )

        # Ensure no participant overlap.
        assert not (
            set(train_ids) & set(val_ids)
        )

        assert not (
            set(outer_test)
            &
            (
                set(train_ids)
                | set(val_ids)
            )
        )

        # ==================================
        # TRAINING DATA
        # ==================================

        x_train = np.concatenate(
            [
                windows[int(i)]
                for i in train_ids
            ]
        )

        y_train = np.concatenate(
            [
                np.full(
                    len(windows[int(i)]),
                    y[int(i)],
                    dtype=np.float32
                )
                for i in train_ids
            ]
        )

        # Equal sampling contribution
        # for each participant.
        weights = np.concatenate(
            [
                np.full(
                    len(windows[int(i)]),
                    1.0 / len(windows[int(i)])
                )
                for i in train_ids
            ]
        )

        dataset = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train)
        )

        sampler = WeightedRandomSampler(
            weights=torch.tensor(
                weights,
                dtype=torch.double
            ),
            num_samples=len(weights),
            replacement=True,
            generator=(
                torch.Generator()
                .manual_seed(SEED + fold)
            )
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=0
        )

        # ==================================
        # INITIALIZE MODEL
        # ==================================

        model = CompactEEGNet().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.001
        )

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        best_loss = float("inf")

        best_state = None
        best_epoch = 0

        stale = 0

        # ==================================
        # TRAINING LOOP
        # ==================================

        for epoch in range(
            1,
            MAX_EPOCHS + 1
        ):

            model.train()

            for xb, yb in loader:

                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(xb)

                loss = criterion(
                    logits,
                    yb
                )

                loss.backward()

                optimizer.step()

            # ==================================
            # VALIDATION
            # ==================================

            val_probs = np.clip(
                predict_subjects(
                    model,
                    val_ids,
                    windows,
                    device
                ),
                1e-6,
                1 - 1e-6
            )

            val_labels = y[val_ids]

            val_loss = float(
                np.mean(
                    -val_labels
                    * np.log(val_probs)
                    -
                    (1 - val_labels)
                    * np.log(1 - val_probs)
                )
            )

            histories.append({
                "fold": fold,
                "epoch": epoch,
                "val_subject_logloss": round(
                    val_loss,
                    6
                )
            })

            # Allow sufficient learning before
            # selecting a checkpoint.
            if epoch >= WARMUP_EPOCHS:

                if (
                    val_loss <
                    best_loss - 1e-4
                ):

                    best_loss = val_loss
                    best_epoch = epoch
                    stale = 0

                    best_state = {
                        k: v.detach()
                        .cpu()
                        .clone()
                        for k, v
                        in model.state_dict().items()
                    }

                else:
                    stale += 1

            # Stop only after the minimum
            # number of training epochs.
            if (
                epoch >= MIN_EPOCHS
                and stale >= PATIENCE
            ):
                break

        if best_state is None:

            raise RuntimeError(
                f"No best checkpoint "
                f"for fold {fold}"
            )

        print(
            f"Fold {fold}: "
            f"trained {epoch} epochs; "
            f"best validation epoch="
            f"{best_epoch}; "
            f"validation loss="
            f"{best_loss:.4f}"
        )

        # ==================================
        # HELD-OUT FOLD EVALUATION
        # ==================================

        model.load_state_dict(
            best_state
        )

        test_probs = predict_subjects(
            model,
            outer_test,
            windows,
            device
        )

        oof_prob[outer_test] = (
            test_probs
        )

        folds[outer_test] = fold

        pred = (
            test_probs >= 0.5
        ).astype(int)

        fold_balanced = (
            balanced_accuracy_score(
                y[outer_test],
                pred
            )
        )

        fold_auc = roc_auc_score(
            y[outer_test],
            test_probs
        )

        print(
            f"Fold {fold}: "
            f"Balanced Accuracy="
            f"{fold_balanced:.3f}, "
            f"AUC={fold_auc:.3f}"
        )

        # Save fold checkpoint.
        torch.save(
            {
                "state_dict": best_state,
                "channels": channel_order,
                "sampling_rate_hz": FS_MODEL,
                "fold": fold,
                "best_epoch": best_epoch,
                "architecture":
                    "EEGNet-8,2-style"
            },
            model_dir / f"fold_{fold}.pt"
        )

    # ======================================
    # FINAL EVALUATION
    # ======================================

    if (
        not np.isfinite(oof_prob).all()
        or not (folds > 0).all()
    ):

        raise RuntimeError(
            "Missing participant predictions."
        )

    predictions = (
        oof_prob >= 0.5
    ).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    metrics = {
        "subjects": len(y),

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                predictions
            ),

        "roc_auc": roc_auc_score(
            y,
            oof_prob
        ),

        "sensitivity": (
            tp / (tp + fn)
        ),

        "specificity": (
            tn / (tn + fp)
        ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    print(
        "\n===== EEGNET OUT-OF-FOLD RESULTS ====="
    )

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        "Confusion matrix [Healthy, AD]:",
        [
            [tn, fp],
            [fn, tp]
        ]
    )

    # ======================================
    # SAVE PREDICTIONS
    # ======================================

    predictions_path = (
        report_dir /
        "eegnet_v2_cv_predictions.csv"
    )

    with predictions_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "subject",
            "group",
            "fold",
            "segments",
            "true_label",
            "predicted_label",
            "ad_probability"
        ])

        for i, row in enumerate(rows):

            writer.writerow([
                row["subject"],
                row["group"],
                int(folds[i]),
                len(windows[i]),
                int(y[i]),
                int(predictions[i]),
                round(
                    float(oof_prob[i]),
                    6
                )
            ])

    # ======================================
    # SAVE SUMMARY
    # ======================================

    summary_path = (
        report_dir /
        "eegnet_v2_cv_summary.csv"
    )

    with summary_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(metrics)
        )

        writer.writeheader()
        writer.writerow(metrics)

    # ======================================
    # SAVE TRAINING LOG
    # ======================================

    training_path = (
        report_dir /
        "eegnet_v2_training_log.csv"
    )

    with training_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "epoch",
                "val_subject_logloss"
            ]
        )

        writer.writeheader()
        writer.writerows(histories)

    print(
        "\n===== EXPERIMENT COMPLETE ====="
    )

    print("Predictions:", predictions_path)
    print("Summary:", summary_path)
    print("Training log:", training_path)
    print("Checkpoints:", model_dir)

    print(
        "Original EEG recordings unchanged."
    )

    print(
        "All evaluation performed at "
        "participant level."
    )

    print(
        "Results are exploratory, "
        "not external validation."
    )


if __name__ == "__main__":
    main()

