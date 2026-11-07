#!/usr/bin/env python3

"""Compact EEG-Conformer-inspired AD vs Healthy experiment."""

from pathlib import Path
from collections import Counter
import csv
import os
import random

import mne
import numpy as np
from scipy.signal import resample_poly

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

import torch
from torch import nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler
)


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42

MAX_EPOCHS = 55
MIN_EPOCHS = 20
WARMUP_EPOCHS = 10
PATIENCE = 12


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# LOAD SUBJECTS
# ==========================================

def load_manifest():

    path = Path(
        "reports/openneuro_5min_manifest_v2.csv"
    )

    with path.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["status"] == "CANDIDATE"
            and r["group"] in ("A", "C")
        ]

    groups = Counter(
        r["group"] for r in rows
    )

    if (
        len(rows) != 46
        or groups != {"A": 23, "C": 23}
    ):
        raise RuntimeError(
            "Expected exactly 23 AD and "
            "23 Healthy subjects."
        )

    if len({
        r["subject"] for r in rows
    }) != 46:
        raise RuntimeError(
            "Duplicate subjects detected."
        )

    return rows


# ==========================================
# PREPARE EEG CLIPS
# ==========================================

def prepare_clips(rows):

    windows = []
    reference_channels = None

    for idx, row in enumerate(rows, 1):

        raw = mne.io.read_raw_eeglab(
            row["file"],
            preload=False,
            verbose="ERROR"
        )

        sf = float(raw.info["sfreq"])

        if abs(sf - 500) > 0.01:
            raise RuntimeError(
                f"Unexpected sampling rate: "
                f"{row['subject']}"
            )

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        channels = tuple(
            raw.ch_names[p] for p in picks
        )

        if len(channels) != 19:
            raise RuntimeError(
                f"Wrong channel count: "
                f"{row['subject']}"
            )

        if reference_channels is None:
            reference_channels = channels

        if channels != reference_channels:
            raise RuntimeError(
                f"Channel mismatch: "
                f"{row['subject']}"
            )

        start = int(
            round(
                float(row["start_sec"]) * sf
            )
        )

        stop = start + 300 * 500

        if start < 0 or stop > raw.n_times:
            raise RuntimeError(
                f"Invalid EEG window: "
                f"{row['subject']}"
            )

        data = raw.get_data(
            picks=picks,
            start=start,
            stop=stop
        )

        data = (
            data * 1e6
        ).astype(np.float32)

        if (
            not np.isfinite(data).all()
            or data.shape != (19, 150000)
        ):
            raise RuntimeError(
                f"Invalid EEG samples: "
                f"{row['subject']}"
            )

        # Same artifact screening as
        # our previous CNN experiments.

        blocks = data.reshape(
            19, 150, 1000
        )

        ptp = np.ptp(
            blocks,
            axis=-1
        )

        sd = np.std(
            blocks,
            axis=-1
        )

        bad = (
            ((ptp > 200).sum(axis=0) >= 3)
            | ((ptp > 300).any(axis=0))
            | ((sd < 0.5).any(axis=0))
        )

        clean_clips = []

        for k in range(30):

            left = k * 5
            right = (k + 1) * 5

            if bad[left:right].any():
                continue

            clip = data[
                :,
                k * 5000:
                (k + 1) * 5000
            ]

            # Downsample 500 Hz to 125 Hz.

            clip = resample_poly(
                clip,
                up=1,
                down=4,
                axis=-1
            ).astype(np.float32)

            # Same normalization as EEGNet.

            clip -= clip.mean(
                axis=-1,
                keepdims=True
            )

            clip /= 100.0

            if clip.shape != (19, 1250):
                raise RuntimeError(
                    "Unexpected clip shape."
                )

            clean_clips.append(clip)

        if len(clean_clips) < 3:
            raise RuntimeError(
                f"Insufficient clean clips: "
                f"{row['subject']}"
            )

        windows.append(
            np.stack(clean_clips)
        )

        if (
            idx % 10 == 0
            or idx == len(rows)
        ):
            print(
                f"Prepared {idx}/{len(rows)} "
                "subjects"
            )

    return windows, reference_channels


# ==========================================
# COMPACT EEG-CONFORMER ARCHITECTURE
# ==========================================

class CompactEEGConformer(nn.Module):

    def __init__(
        self,
        n_channels=19,
        n_samples=1250
    ):

        super().__init__()

        # Temporal and spatial convolution.
        self.stem = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=(1, 25),
                stride=(1, 2),
                padding=(0, 12),
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.Conv2d(
                32,
                32,
                kernel_size=(n_channels, 1),
                groups=32,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            )
        )

        # Calculate number of tokens.
        with torch.no_grad():

            dummy = torch.zeros(
                1, 1,
                n_channels,
                n_samples
            )

            tokens = self.stem(
                dummy
            ).shape[-1]

        # Learnable positional embedding.
        self.position = nn.Parameter(
            torch.zeros(1, tokens, 32)
        )

        nn.init.normal_(
            self.position,
            std=0.02
        )

        # Transformer encoder.
        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=96,
                dropout=0.25,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
                enable_nested_tensor=False
            )
        )

        # Binary classification head.
        self.classifier = nn.Sequential(

            nn.LayerNorm(32),

            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.stem(x)

        x = x.squeeze(2).transpose(1, 2)

        x = (
            x + self.position[:, :x.shape[1]]
        )

        x = self.encoder(x)

        # Temporal token pooling.
        x = x.mean(dim=1)

        x = self.classifier(x)

        return x.squeeze(-1)


# ==========================================
# SUBJECT-LEVEL PREDICTIONS
# ==========================================

def subject_probabilities(
    model,
    indices,
    windows,
    device
):

    model.eval()

    result = []

    with torch.no_grad():

        for i in indices:

            clips = torch.from_numpy(
                windows[int(i)]
            ).to(device)

            logits = model(clips)

            prob = torch.sigmoid(
                logits
            ).mean().item()

            result.append(float(prob))

    return np.asarray(
        result,
        dtype=float
    )


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    seed_all(SEED)

    torch.set_num_threads(
        min(os.cpu_count() or 1, 4)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    Path("reports").mkdir(
        exist_ok=True
    )

    model_dir = Path(
        "models/eeg_conformer_cv"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = load_manifest()

    y = np.asarray([
        int(r["group"] == "A")
        for r in rows
    ])

    print(
        "===== COMPACT EEG-CONFORMER "
        "EXPERIMENT ====="
    )

    print("Device:", device)
    print("Subjects:", len(rows))
    print("AD:", int(y.sum()))
    print("Healthy:", int(len(y) - y.sum()))

    print(
        "Training configuration:",
        MAX_EPOCHS,
        "maximum epochs"
    )

    print(
        "\n===== PREPARING EEG SEGMENTS ====="
    )

    windows, channels = (
        prepare_clips(rows)
    )

    print(
        "Total 10s clips:",
        sum(len(w) for w in windows)
    )

    # Verify architecture.
    probe = CompactEEGConformer().to(
        device
    )

    with torch.no_grad():

        dummy = torch.zeros(
            2, 19, 1250,
            device=device
        )

        assert tuple(
            probe(dummy).shape
        ) == (2,)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in probe.parameters()
        )
    )

    del probe

    # Compare against the existing
    # subject-level fold assignments.
    reference_path = Path(
        "reports/baseline_predictions_v2.csv"
    )

    reference_folds = None

    if reference_path.exists():

        with reference_path.open(
            newline=""
        ) as f:

            reference_folds = {
                r["subject"]: int(r["fold"])
                for r in csv.DictReader(f)
            }

        if set(reference_folds) != {
            r["subject"] for r in rows
        }:
            raise RuntimeError(
                "Baseline subjects mismatch."
            )

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    oof = np.full(
        len(rows),
        np.nan
    )

    fold_ids = np.zeros(
        len(rows),
        dtype=int
    )

    training_log = []

    print(
        "\n===== FIVE-FOLD SUBJECT-LEVEL "
        "CONFORMER ====="
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

        seed_all(SEED + fold)

        if reference_folds is not None:

            for i in outer_test:

                sid = rows[int(i)]["subject"]

                if reference_folds[sid] != fold:
                    raise RuntimeError(
                        "Cross-validation fold mismatch."
                    )

        train_ids, val_ids = (
            train_test_split(
                outer_train,
                test_size=0.25,
                stratify=y[outer_train],
                random_state=100 + fold
            )
        )

        # Confirm subject separation.
        assert not (
            set(train_ids) & set(val_ids)
        )

        assert not (
            set(outer_test)
            & (
                set(train_ids)
                | set(val_ids)
            )
        )

        # Training segments.
        x_train = np.concatenate([
            windows[int(i)]
            for i in train_ids
        ])

        y_train = np.concatenate([
            np.full(
                len(windows[int(i)]),
                y[int(i)],
                dtype=np.float32
            )
            for i in train_ids
        ])

        # Equal contribution per subject.
        weights = np.concatenate([
            np.full(
                len(windows[int(i)]),
                1.0 / len(windows[int(i)])
            )
            for i in train_ids
        ])

        dataset = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train)
        )

        sampler = WeightedRandomSampler(
            torch.as_tensor(
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

        model = CompactEEGConformer().to(
            device
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.001
        )

        criterion = (
            nn.BCEWithLogitsLoss()
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        stale = 0

        # ==================================
        # TRAINING
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

                loss = criterion(
                    model(xb),
                    yb
                )

                loss.backward()

                optimizer.step()

            # Validation performed per subject.
            val_prob = np.clip(
                subject_probabilities(
                    model,
                    val_ids,
                    windows,
                    device
                ),
                1e-6,
                1 - 1e-6
            )

            val_y = y[val_ids]

            val_loss = float(
                np.mean(
                    -val_y * np.log(val_prob)
                    -
                    (1 - val_y)
                    * np.log(1 - val_prob)
                )
            )

            training_log.append({
                "fold": fold,
                "epoch": epoch,
                "validation_subject_logloss":
                    val_loss
            })

            if epoch >= WARMUP_EPOCHS:

                if (
                    val_loss
                    < best_loss - 1e-4
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

            if (
                epoch >= MIN_EPOCHS
                and stale >= PATIENCE
            ):
                break

        if best_state is None:
            raise RuntimeError(
                "No valid best checkpoint."
            )

        # ==================================
        # TEST FOLD
        # ==================================

        model.load_state_dict(
            best_state
        )

        test_prob = subject_probabilities(
            model,
            outer_test,
            windows,
            device
        )

        oof[outer_test] = test_prob

        fold_ids[outer_test] = fold

        test_pred = (
            test_prob >= 0.5
        ).astype(int)

        fold_bacc = (
            balanced_accuracy_score(
                y[outer_test],
                test_pred
            )
        )

        fold_auc = roc_auc_score(
            y[outer_test],
            test_prob
        )

        print(
            f"Fold {fold}: "
            f"trained={epoch}, "
            f"best_epoch={best_epoch}, "
            f"BACC={fold_bacc:.3f}, "
            f"AUC={fold_auc:.3f}"
        )

        torch.save(
            {
                "state_dict": best_state,
                "channels": channels,
                "sampling_rate": 125,
                "fold": fold,
                "best_epoch": best_epoch,
                "architecture":
                    "CompactEEGConformer"
            },
            model_dir / f"fold_{fold}.pt"
        )

    # ======================================
    # FINAL SUBJECT-LEVEL METRICS
    # ======================================

    if (
        not np.isfinite(oof).all()
        or (fold_ids == 0).any()
    ):
        raise RuntimeError(
            "Missing subject predictions."
        )

    pred = (
        oof >= 0.5
    ).astype(int)

    tn, fp, fn, tp = [
        int(v)
        for v in confusion_matrix(
            y,
            pred,
            labels=[0, 1]
        ).ravel()
    ]

    summary = {
        "subjects": len(y),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "roc_auc": float(
            roc_auc_score(y, oof)
        ),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    print(
        "\n===== EEG-CONFORMER "
        "OUT-OF-FOLD RESULTS ====="
    )

    for key, value in summary.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(
        "Confusion matrix [Healthy, AD]:",
        [[tn, fp], [fn, tp]]
    )

    # ======================================
    # SAVE RESULTS
    # ======================================

    with Path(
        "reports/eeg_conformer_cv_predictions.csv"
    ).open("w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "subject",
            "group",
            "fold",
            "clips",
            "true_label",
            "predicted_label",
            "ad_probability"
        ])

        for i, row in enumerate(rows):

            writer.writerow([
                row["subject"],
                row["group"],
                int(fold_ids[i]),
                len(windows[i]),
                int(y[i]),
                int(pred[i]),
                round(float(oof[i]), 6)
            ])

    with Path(
        "reports/eeg_conformer_cv_summary.csv"
    ).open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summary)
        )

        writer.writeheader()
        writer.writerow(summary)

    with Path(
        "reports/eeg_conformer_training_log.csv"
    ).open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(training_log[0])
        )

        writer.writeheader()
        writer.writerows(training_log)

    print("\n===== EXPERIMENT COMPLETE =====")

    print(
        "Predictions:",
        "reports/eeg_conformer_cv_predictions.csv"
    )

    print(
        "Summary:",
        "reports/eeg_conformer_cv_summary.csv"
    )

    print(
        "Training:",
        "reports/eeg_conformer_training_log.csv"
    )

    print("Checkpoints:", model_dir)

    print("Original EEG recordings unchanged.")

    print(
        "Evaluation is subject-level "
        "cross-validation."
    )

    print(
        "Results are exploratory, "
        "not external validation."
    )


if __name__ == "__main__":
    main()

