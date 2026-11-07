#!/usr/bin/env python3
"""Compare augmentation against minute-matched specialist models."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()

BASE = (
    ROOT / "reports/thesis_final/temporal_repeated_v1"
    / "all_repeats_heldout_block_predictions.csv"
)

TRANSFER = (
    ROOT / "reports/thesis_final/temporal_transfer_matrix_v1"
    / "cross_block_held_out_predictions.csv"
)

AUGMENT = (
    ROOT / "reports/thesis_final/block_augmented_svm_v1"
    / "augmented_heldout_block_predictions.csv"
)

OUT = (
    ROOT / "reports/thesis_final"
    / "augmentation_specificity_audit_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

NREP = 5
NBLOCK = 5
NBOOT = 3000


# ==========================================
# LOCK ANALYSIS
# ==========================================

PROTOCOL = {
    "status":
        "Post-hoc exploratory audit; no new model fitting",

    "cohort":
        "46 AD/HC subjects from ds004504",

    "models": [
        "First-minute-trained SVM",
        "Position-matched minute-specific SVM",
        "Five-block-augmented SVM"
    ],

    "primary_check":
        "Augmented versus first-minute SVM; "
        "mean BACC across five test blocks",

    "mechanism_check":
        "Augmented versus fifth-minute specialist "
        "on held-out minute five",

    "uncertainty":
        "Stratified participant bootstrap; "
        "retain all repetitions and blocks together",

    "limitations": [
        "Previously examined participants",
        "Exploratory secondary comparisons",
        "Unequal training durations",
        "No external validation",
        "Full-session preprocessing"
    ]
}

lock = OUT / "analysis_protocol.json"

encoded = (
    json.dumps(
        PROTOCOL,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if lock.exists() and lock.read_text() != encoded:
    raise RuntimeError(
        "Locked protocol differs; refusing to overwrite."
    )

lock.write_text(encoded)


# ==========================================
# FILE UTILITIES
# ==========================================

def read(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def index(rows, key):

    result = {}

    for row in rows:

        k = key(row)

        if k in result:
            raise RuntimeError(
                f"Duplicate key: {k}"
            )

        result[k] = row

    return result


def save(path, rows):

    if not rows:
        raise RuntimeError(
            f"Empty output: {path}"
        )

    columns = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


print(
    "===== AUGMENTATION SPECIFICITY AUDIT =====",
    flush=True
)

paths = [
    BASE,
    TRANSFER,
    AUGMENT
]

save(
    OUT / "input_hashes.csv",
    [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        }
        for path in paths
    ]
)


# ==========================================
# LOAD THREE ORIGINAL EXPERIMENTS
# ==========================================

baseline_rows = read(BASE)
transfer_rows = read(TRANSFER)
augmented_rows = read(AUGMENT)

baseline_map = index(
    baseline_rows,
    lambda r: (
        int(r["repeat"]),
        r["subject"],
        int(r["block"])
    )
)

transfer_map = index(
    transfer_rows,
    lambda r: (
        int(r["repeat"]),
        r["subject"],
        int(r["train_block"]),
        int(r["test_block"])
    )
)

augmented_map = index(
    augmented_rows,
    lambda r: (
        int(r["repeat"]),
        r["subject"],
        int(r["test_block"])
    )
)


# ==========================================
# RECONCILE PARTICIPANTS
# ==========================================

ids = sorted([
    sid
    for rep, sid, block in baseline_map
    if rep == 1 and block == 1
])

if len(ids) != 46 or len(set(ids)) != 46:
    raise RuntimeError(
        "Expected 46 unique participants"
    )

y = np.array([
    int(
        baseline_map[
            (1, sid, 1)
        ]["true_label"]
    )
    for sid in ids
])

if Counter(y.tolist()) != {
    0: 23,
    1: 23
}:
    raise RuntimeError(
        "Unexpected diagnostic distribution"
    )

if (
    len(baseline_map) != 1150
    or len(augmented_map) != 1150
    or len(transfer_map) != 5750
):
    raise RuntimeError(
        "Incomplete source prediction tables"
    )


# ==========================================
# ALIGN HELD-OUT PREDICTIONS
# ==========================================

baseline = np.full(
    (NREP, 46, NBLOCK),
    -1,
    dtype=np.int8
)

augmented = np.full_like(
    baseline,
    -1
)

specialist = np.full_like(
    baseline,
    -1
)

for rep in range(1, NREP + 1):

    for i, sid in enumerate(ids):

        reference_fold = int(
            baseline_map[
                (rep, sid, 1)
            ]["fold"]
        )

        for block in range(1, NBLOCK + 1):

            old = baseline_map[
                (rep, sid, block)
            ]

            new = augmented_map[
                (rep, sid, block)
            ]

            old_transfer = transfer_map[
                (rep, sid, 1, block)
            ]

            matched = transfer_map[
                (rep, sid, block, block)
            ]

            for row, fold_field in (
                (old, "fold"),
                (new, "outer_fold"),
                (old_transfer, "outer_fold"),
                (matched, "outer_fold")
            ):

                if (
                    int(row["true_label"]) != y[i]
                    or int(row[fold_field]) != reference_fold
                ):
                    raise RuntimeError(
                        f"Label or fold mismatch: "
                        f"repeat={rep}, {sid}, block={block}"
                    )

            if (
                int(old["predicted_label"])
                != int(old_transfer["predicted_label"])
            ):
                raise RuntimeError(
                    f"Original prediction mismatch: {sid}"
                )

            if not np.isclose(
                float(old["decision_score"]),
                float(old_transfer["decision_score"]),
                atol=1e-7,
                rtol=0
            ):
                raise RuntimeError(
                    f"Original score mismatch: {sid}"
                )

            baseline[rep-1, i, block-1] = int(
                old["predicted_label"]
            )

            augmented[rep-1, i, block-1] = int(
                new["predicted_label"]
            )

            specialist[rep-1, i, block-1] = int(
                matched["predicted_label"]
            )

for name, arr in (
    ("baseline", baseline),
    ("augmented", augmented),
    ("specialist", specialist)
):

    if not np.isin(arr, [0, 1]).all():
        raise RuntimeError(
            f"Incomplete predictions: {name}"
        )

print(
    "PASS: 46 subjects x 5 repeats x 5 test blocks; "
    "all folds and source predictions reconciled."
)


# ==========================================
# CLASSIFICATION METRICS
# ==========================================

ad = np.flatnonzero(y == 1)
hc = np.flatnonzero(y == 0)

models = {
    "First-minute training": baseline,
    "Position-matched training": specialist,
    "Five-block training": augmented
}


def bacc(predictions, minute=None):

    if minute is not None:
        predictions = predictions[
            :, :, minute:minute+1
        ]

    sensitivity = (
        predictions[:, ad, :] == 1
    ).mean()

    specificity = (
        predictions[:, hc, :] == 0
    ).mean()

    return float(
        (sensitivity + specificity) / 2
    )


print(
    "\n===== THREE-MODEL BACC COMPARISON ====="
)

metrics = []

for name, predictions in models.items():

    print(
        f"{name}: "
        f"all-block BACC={bacc(predictions):.4f}"
    )

    for minute in range(NBLOCK):

        sensitivity = float(
            (
                predictions[:, ad, minute] == 1
            ).mean()
        )

        specificity = float(
            (
                predictions[:, hc, minute] == 0
            ).mean()
        )

        metrics.append({
            "model": name,
            "test_minute": minute + 1,
            "balanced_accuracy": bacc(
                predictions,
                minute
            ),
            "sensitivity_AD": sensitivity,
            "specificity_HC": specificity,
            "training_seconds_per_participant":
                300 if name == "Five-block training"
                else 60
        })

        print(
            f"  minute {minute+1}: "
            f"BACC={bacc(predictions,minute):.4f}, "
            f"AD sensitivity={sensitivity:.4f}, "
            f"HC specificity={specificity:.4f}"
        )

save(
    OUT / "classwise_minute_metrics.csv",
    metrics
)


# ==========================================
# SUBJECT-PAIRED UNCERTAINTY
# ==========================================

rng = np.random.default_rng(20261107)

sample_ad = rng.choice(
    ad,
    size=(NBOOT, len(ad)),
    replace=True
)

sample_hc = rng.choice(
    hc,
    size=(NBOOT, len(hc)),
    replace=True
)

comparisons = [
    (
        "PRIMARY: augmented minus baseline, all blocks",
        augmented, baseline, None
    ),
    (
        "Augmented minus position-matched, all blocks",
        augmented, specialist, None
    ),
    (
        "Augmented minus baseline, minute 5",
        augmented, baseline, 4
    ),
    (
        "Augmented minus minute-5 specialist, minute 5",
        augmented, specialist, 4
    ),
    (
        "Minute-5 specialist minus baseline, minute 5",
        specialist, baseline, 4
    ),
    (
        "Augmented minus baseline, minute 3",
        augmented, baseline, 2
    )
]

print(
    "\n===== SUBJECT-PAIRED CONDITIONAL 95% INTERVALS ====="
)

comparison_rows = []

for label, first_model, second_model, minute in comparisons:

    first = (
        first_model
        if minute is None
        else first_model[:, :, minute:minute+1]
    )

    second = (
        second_model
        if minute is None
        else second_model[:, :, minute:minute+1]
    )

    correct_first = (
        first == y[None, :, None]
    ).astype(np.int8)

    correct_second = (
        second == y[None, :, None]
    ).astype(np.int8)

    # Average within participant, retaining
    # all correlated CV repetitions together.
    gains = (
        correct_first - correct_second
    ).mean(axis=(0, 2))

    observed = 0.5 * (
        gains[ad].mean()
        + gains[hc].mean()
    )

    boot = 0.5 * (
        gains[sample_ad].mean(axis=1)
        + gains[sample_hc].mean(axis=1)
    )

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5]
    )

    comparison_rows.append({
        "contrast": label,
        "delta_balanced_accuracy": float(observed),
        "conditional_ci_low": float(lo),
        "conditional_ci_high": float(hi),
        "delta_AD_sensitivity":
            float(gains[ad].mean()),
        "delta_HC_specificity":
            float(gains[hc].mean()),
        "bootstrap_resamples": NBOOT
    })

    print(
        f"{label}: {observed:+.4f}, "
        f"CI [{lo:+.4f}, {hi:+.4f}], "
        f"AD change={gains[ad].mean():+.4f}, "
        f"HC change={gains[hc].mean():+.4f}"
    )

save(
    OUT / "matched_subject_contrasts.csv",
    comparison_rows
)


# ==========================================
# SUBJECT-LEVEL ERROR AUDIT
# ==========================================

subject_rows = []

for i, sid in enumerate(ids):

    subject_rows.append({
        "subject": sid,
        "group": "A" if y[i] else "C",

        "baseline_fifth_correct_repeats": int(
            (baseline[:, i, 4] == y[i]).sum()
        ),

        "specialist_fifth_correct_repeats": int(
            (specialist[:, i, 4] == y[i]).sum()
        ),

        "augmented_fifth_correct_repeats": int(
            (augmented[:, i, 4] == y[i]).sum()
        ),

        "baseline_flip_repeats": int(
            (
                baseline[:, i, :]
                != baseline[:, i, :1]
            ).any(axis=1).sum()
        ),

        "augmented_flip_repeats": int(
            (
                augmented[:, i, :]
                != augmented[:, i, :1]
            ).any(axis=1).sum()
        )
    })

save(
    OUT / "participant_error_audit.csv",
    subject_rows
)


# ==========================================
# FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(8, 4.7)
)

for name, predictions in models.items():

    values = [
        bacc(predictions, minute)
        for minute in range(NBLOCK)
    ]

    ax.plot(
        range(1, NBLOCK+1),
        values,
        marker="o",
        label=name
    )

ax.set(
    xlabel="Held-out 60-second recording block",
    ylabel="Participant-level balanced accuracy",
    title="Training exposure versus recording-position robustness",
    xticks=list(range(1, 6)),
    ylim=(0.45, 1.0)
)

ax.grid(
    axis="y",
    alpha=0.2
)

ax.legend()
fig.tight_layout()

fig.savefig(
    OUT / "three_training_strategies.png",
    dpi=180
)

plt.close(fig)


print(
    "\n===== AUGMENTATION SPECIFICITY AUDIT COMPLETE ====="
)

print("Outputs:", OUT)

print(
    "No retraining. Post-hoc conditional estimates; "
    "not independent external validation."
)

