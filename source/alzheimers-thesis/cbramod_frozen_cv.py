#!/usr/bin/env python3

"""Frozen CBraMod embeddings with nested subject-level CV."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import random
import sys

import mne
import numpy as np
from scipy.signal import resample_poly

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
SF = 500
WINDOW_SEC = 300
OUT_SF = 200
BATCH_SIZE = 4

EXPECTED = [
    "Fp1", "Fp2", "F7", "F3", "Fz",
    "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz",
    "P4", "T6", "O1", "O2"
]

ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6"
}


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ==========================================
# PREPARE EEG CLIPS
# ==========================================

def get_clips(row):

    raw = mne.io.read_raw_eeglab(
        str(row["file"]),
        preload=False,
        verbose="ERROR"
    )

    fs = float(raw.info["sfreq"])

    if abs(fs - SF) > 0.01:
        raise ValueError(
            f"{row['subject']}: unexpected sampling rate"
        )

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[]
    )

    names = [
        ALIASES.get(raw.ch_names[i], raw.ch_names[i])
        for i in picks
    ]

    if (
        len(names) != 19
        or len(set(names)) != 19
        or set(names) != set(EXPECTED)
    ):
        raise ValueError(
            f"{row['subject']}: channel mismatch: {names}"
        )

    # Reorder channels to reference montage.
    order = [names.index(n) for n in EXPECTED]

    start = int(
        round(float(row["start_sec"]) * fs)
    )

    stop = start + WINDOW_SEC * SF

    if start < 0 or stop > raw.n_times:
        raise ValueError("Invalid five-minute window")

    data = raw.get_data(
        picks=picks,
        start=start,
        stop=stop
    )

    # MNE EEG output is in volts.
    data = (data * 1e6).astype(np.float32)
    data = data[order]

    if (
        data.shape != (19, 150000)
        or not np.isfinite(data).all()
    ):
        raise ValueError("Invalid EEG data")

    # Same artifact rules as previous experiments.
    blocks = data.reshape(19, 150, 1000)

    ptp = np.ptp(blocks, axis=-1)
    sd = np.std(blocks, axis=-1)

    bad = (
        ((ptp > 200).sum(axis=0) >= 3)
        | ((ptp > 300).any(axis=0))
        | ((sd < 0.5).any(axis=0))
    )

    clips = []

    for k in range(30):

        if bad[k*5:(k+1)*5].any():
            continue

        clip = data[
            :,
            k*5000:(k+1)*5000
        ]

        # 500 Hz to 200 Hz.
        clip = resample_poly(
            clip,
            up=2,
            down=5,
            axis=-1
        ).astype(np.float32)

        # Per-clip DC removal and fixed 100 uV scale.
        clip -= clip.mean(
            axis=-1,
            keepdims=True
        )

        clip /= 100.0

        if (
            clip.shape != (19, 2000)
            or not np.isfinite(clip).all()
        ):
            raise ValueError("Invalid resampled clip")

        # Channel x 10 one-second patches x 200 samples.
        clips.append(
            clip.reshape(19, 10, 200)
        )

    if len(clips) < 3:
        raise ValueError(
            f"{row['subject']}: insufficient clean clips"
        )

    return np.stack(clips), len(clips)


# ==========================================
# EXTRACT FROZEN EMBEDDINGS
# ==========================================

def extract_one(row, model, device):

    clips, nclips = get_clips(row)
    pooled = []

    with torch.inference_mode():

        for i in range(0, nclips, BATCH_SIZE):

            x = torch.from_numpy(
                clips[i:i+BATCH_SIZE]
            ).to(device)

            out = model(x)

            if tuple(out.shape[1:]) != (19, 10, 200):
                raise RuntimeError(
                    f"Unexpected embedding shape: {out.shape}"
                )

            # Summarize contextualized embeddings
            # across channels and temporal patches.
            mean = out.mean(dim=(1, 2))

            spread = out.std(
                dim=(1, 2),
                unbiased=False
            )

            features = torch.cat(
                [mean, spread],
                dim=1
            )

            pooled.append(
                features.cpu().numpy()
            )

    vectors = np.concatenate(
        pooled,
        axis=0
    )

    # One feature vector per participant.
    feature = vectors.mean(
        axis=0
    ).astype(np.float64)

    if (
        feature.shape != (400,)
        or not np.isfinite(feature).all()
    ):
        raise RuntimeError("Invalid subject embedding")

    return feature, nclips


# ==========================================
# MAIN
# ==========================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.set_num_threads(4)

    root = Path.cwd()

    weights = (
        root /
        "pretrained_weights/cbramod_official.pth"
    )

    repo = root / "third_party/CBraMod"

    manifest = (
        root /
        "reports/openneuro_5min_manifest_v2.csv"
    )

    baseline = (
        root /
        "reports/baseline_predictions_v2.csv"
    )

    cache_dir = (
        root /
        "reports/cbramod_subject_cache_v1"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in (
        weights,
        repo / "models/cbramod.py",
        manifest,
        baseline
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = [
        r for r in read_rows(manifest)
        if r["status"] == "CANDIDATE"
        and r["group"] in ("A", "C")
    ]

    if (
        len(rows) != 46
        or Counter(r["group"] for r in rows)
        != {"A": 23, "C": 23}
    ):
        raise RuntimeError(
            "Expected 23 AD and 23 Healthy subjects"
        )

    if len(set(r["subject"] for r in rows)) != 46:
        raise RuntimeError("Duplicate subjects")

    reference_folds = {
        r["subject"]: int(r["fold"])
        for r in read_rows(baseline)
    }

    if set(reference_folds) != {
        r["subject"] for r in rows
    }:
        raise RuntimeError(
            "Subject list differs from baseline"
        )

    # Import official CBraMod implementation.
    sys.path.insert(
        0,
        str(repo.resolve())
    )

    from models.cbramod import CBraMod

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CBraMod()

    state = torch.load(
        weights,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state,
        strict=True
    )

    # Remove reconstruction head for feature extraction.
    model.proj_out = nn.Identity()

    model.requires_grad_(False)
    model = model.to(device)
    model.eval()

    digest = hashlib.sha256(
        weights.read_bytes()
    ).hexdigest()

    print(
        "===== CBRAMOD FROZEN FEATURE EXPERIMENT =====",
        flush=True
    )

    print("Device:", device)
    print("Subjects:", len(rows))
    print("Pretrained SHA256:", digest)
    print("Feature dimensions per subject: 400")
    print("Classifier: PCA + Logistic Regression")

    # ======================================
    # FEATURE EXTRACTION WITH CACHE
    # ======================================

    features = []
    clip_counts = []

    print(
        "\n===== EXTRACTING PRETRAINED EMBEDDINGS =====",
        flush=True
    )

    for i, row in enumerate(rows, 1):

        path = Path(row["file"]).resolve()
        stat = path.stat()

        provenance = json.dumps(
            {
                "version": 1,
                "checkpoint": digest,
                "subject": row["subject"],
                "source": str(path),
                "start_sec": row["start_sec"],
                "file_size": stat.st_size,
                "file_mtime": stat.st_mtime_ns
            },
            sort_keys=True
        )

        cache_file = (
            cache_dir /
            (row["subject"] + ".npz")
        )

        loaded = False

        if cache_file.exists():

            with np.load(
                cache_file,
                allow_pickle=False
            ) as saved:

                if saved["provenance"].item() == provenance:

                    feature = saved["feature"].astype(
                        np.float64
                    )

                    nclips = int(
                        saved["clips"].item()
                    )

                    loaded = True

        if not loaded:

            feature, nclips = extract_one(
                row,
                model,
                device
            )

            np.savez_compressed(
                cache_file,
                provenance=provenance,
                feature=feature,
                clips=nclips
            )

        if (
            feature.shape != (400,)
            or not np.isfinite(feature).all()
        ):
            raise RuntimeError(
                f"Invalid cached feature: {row['subject']}"
            )

        features.append(feature)
        clip_counts.append(nclips)

        print(
            f"{i:02d}/46 "
            f"{row['subject']} "
            f"clips={nclips} "
            f"{'cached' if loaded else 'extracted'}",
            flush=True
        )

    X = np.asarray(
        features,
        dtype=np.float64
    )

    y = np.array([
        int(r["group"] == "A")
        for r in rows
    ])

    # ======================================
    # NESTED CROSS-VALIDATION
    # ======================================

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    oof = np.full(
        len(y),
        np.nan
    )

    fold_ids = np.zeros(
        len(y),
        dtype=int
    )

    selected_params = {}

    print(
        "\n===== FIVE-FOLD NESTED SUBJECT-LEVEL CV =====",
        flush=True
    )

    for fold, (train, test) in enumerate(
        outer.split(X, y),
        1
    ):

        # Verify identical outer folds to baselines.
        for j in test:

            sid = rows[int(j)]["subject"]

            if reference_folds[sid] != fold:
                raise RuntimeError(
                    f"Fold mismatch: {sid}"
                )

        # Scaling and PCA must be fitted only
        # inside the training partitions.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(svd_solver="full")),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=SEED
                )
            )
        ])

        inner = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=100 + fold
        )

        search = GridSearchCV(
            pipeline,
            {
                "pca__n_components": [5, 10, 15],
                "classifier__C": [
                    0.01, 0.1, 1.0, 10.0
                ]
            },
            cv=inner,
            scoring="balanced_accuracy",
            n_jobs=1,
            refit=True,
            error_score="raise"
        )

        search.fit(
            X[train],
            y[train]
        )

        prob = search.predict_proba(
            X[test]
        )[:, 1]

        pred = (prob >= 0.5).astype(int)

        oof[test] = prob
        fold_ids[test] = fold

        selected_params[fold] = (
            search.best_params_
        )

        bacc = balanced_accuracy_score(
            y[test],
            pred
        )

        auc = roc_auc_score(
            y[test],
            prob
        )

        print(
            f"Fold {fold}: "
            f"BACC={bacc:.3f}, "
            f"AUC={auc:.3f}, "
            f"best={search.best_params_}",
            flush=True
        )

    # ======================================
    # FINAL OUT-OF-FOLD METRICS
    # ======================================

    if (
        not np.isfinite(oof).all()
        or not np.all(fold_ids > 0)
    ):
        raise RuntimeError(
            "Missing out-of-fold predictions"
        )

    predictions = (
        oof >= 0.5
    ).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    summary = {
        "subjects": len(y),
        "accuracy": accuracy_score(
            y, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "roc_auc": roc_auc_score(
            y, oof
        ),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    print(
        "\n===== CBRAMOD OUT-OF-FOLD RESULTS ====="
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
    # SAVE REPORTS
    # ======================================

    predictions_path = (
        root /
        "reports/cbramod_frozen_cv_predictions.csv"
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
                int(clip_counts[i]),
                int(y[i]),
                int(predictions[i]),
                round(float(oof[i]), 6)
            ])

    summary_path = (
        root /
        "reports/cbramod_frozen_cv_summary.csv"
    )

    with summary_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summary)
        )

        writer.writeheader()
        writer.writerow(summary)

    params_path = (
        root /
        "reports/cbramod_frozen_cv_best_params.csv"
    )

    with params_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "fold",
            "pca_components",
            "logistic_C"
        ])

        for fold, params in selected_params.items():

            writer.writerow([
                fold,
                params["pca__n_components"],
                params["classifier__C"]
            ])

    print(
        "\n===== EXPERIMENT COMPLETE ====="
    )

    print("Predictions:", predictions_path)
    print("Summary:", summary_path)
    print("Parameters:", params_path)
    print("Embedding cache:", cache_dir)

    print("Original EEG and checkpoint unchanged.")
    print("Evaluation: subject-level nested CV.")
    print("Results are exploratory, not external validation.")


if __name__ == "__main__":
    main()

