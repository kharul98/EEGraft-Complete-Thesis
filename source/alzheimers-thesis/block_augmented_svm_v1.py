#!/usr/bin/env python3

"""Exploratory subject-grouped temporal-augmentation SVM evaluation."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()

SRC = ROOT / "reports/thesis_final/duration_locked_v1"

BASE = ROOT / "reports/thesis_final/temporal_repeated_v1"

OUT = ROOT / "reports/thesis_final/block_augmented_svm_v1"

OUT.mkdir(parents=True, exist_ok=True)


FEATURES = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
)

REPEATS = 5
N_BLOCKS = 5
N_SUBJECTS = 46
N_BOOT = 3000
SEED = 20261107

GRID = [
    (c, gamma)
    for c in (0.1, 1.0, 10.0)
    for gamma in ("scale", 0.01, 0.1)
]


# ==========================================
# 1. LOCK THE EXPERIMENT
# ==========================================

PROTOCOL = {
    "status":
        "Exploratory extension after earlier analyses "
        "on the same participants",

    "cohort":
        "23 AD and 23 HC; ds004504; "
        "fixed five-minute selections",

    "contrast":
        "Multi-block-trained SVM versus "
        "original first-minute-trained SVM",

    "training_examples":
        "All five nonoverlapping 60-second blocks "
        "per outer-training participant",

    "training_weight":
        "Every training participant contributes "
        "exactly five block observations",

    "test_examples":
        "Each held-out 60-second block "
        "evaluated independently",

    "primary_metric":
        "Mean participant-level BACC over five "
        "target blocks and five correlated CV repeats",

    "primary_comparison":
        "Multi-block-trained minus first-minute-trained "
        "on identical outer folds",

    "outer_cv":
        "Five stratified subject folds, five repeats",

    "inner_cv":
        "Three subject-stratified folds within "
        "each outer training split",

    "inner_selection":
        "Mean BACC across five block positions "
        "on inner-validation participants",

    "hyperparameters": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1]
    },

    "classifier":
        "StandardScaler and RBF-SVC, "
        "class_weight balanced",

    "secondary_metrics": [
        "First-block BACC",
        "Fifth-block BACC",
        "Five-block majority-vote BACC",
        "Within-recording class reversal rate"
    ],

    "uncertainty":
        "Stratified participant bootstrap "
        "preserving all repetitions and blocks",

    "interpretation":
        "Training uses 300 seconds per participant; "
        "single-block inference uses 60 seconds; "
        "training-data budgets are not matched",

    "limitations": [
        "Previously examined cohort",
        "No independent external validation",
        "Preprocessing may use full session",
        "No new architecture"
    ]
}

lock = OUT / "locked_protocol.json"

serialized = (
    json.dumps(
        PROTOCOL,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if lock.exists():

    if lock.read_text() != serialized:
        raise RuntimeError(
            "Locked protocol differs. "
            "Refusing to overwrite."
        )

else:
    lock.write_text(serialized)


# ==========================================
# 2. UTILITIES
# ==========================================

def read(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save(path, rows):

    if not rows:
        raise RuntimeError(
            "Empty table: " + str(path)
        )

    columns = list(dict.fromkeys(
        key
        for row in rows
        for key in row
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


def model_for(c, gamma):

    return Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "clf",
            SVC(
                C=c,
                gamma=gamma,
                kernel="rbf",
                class_weight="balanced"
            )
        )
    ])


def bacc(labels, predictions):

    if set(labels) != {0, 1}:
        raise RuntimeError(
            "Both diagnostic classes required"
        )

    sensitivity = (
        predictions[labels == 1] == 1
    ).mean()

    specificity = (
        predictions[labels == 0] == 0
    ).mean()

    return float(
        0.5 * (
            sensitivity + specificity
        )
    )


print(
    "===== BLOCK-AUGMENTED SVM TEST =====",
    flush=True
)


# ==========================================
# 3. LOAD PREVIOUS VERIFIED FEATURES
# ==========================================

source_paths = [
    SRC / "five_60s_blocks.csv",
    SRC / "duration_features.csv",
    SRC / "held_out_predictions.csv",
    BASE / "all_repeats_heldout_block_predictions.csv"
]

save(
    OUT / "input_sha256.csv",
    [
        {
            "source": str(
                path.relative_to(ROOT)
            ),
            "sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        }
        for path in source_paths
    ]
)

(
    block_rows,
    duration_rows,
    historical_rows,
    ref_rows
) = [
    read(path)
    for path in source_paths
]


# ==========================================
# 4. VERIFY PARTICIPANTS AND FEATURE ORDER
# ==========================================

first = [
    row
    for row in duration_rows
    if int(row["duration_sec"]) == 60
]

ids = [
    row["subject"]
    for row in first
]

y = np.array([
    int(row["group"] == "A")
    for row in first
], dtype=int)

if (
    len(ids) != N_SUBJECTS
    or len(set(ids)) != N_SUBJECTS
    or Counter(y) != {0: 23, 1: 23}
):
    raise RuntimeError(
        "Unexpected cohort or class mapping"
    )


feature_map = {}

for row in block_rows:

    key = (
        row["subject"],
        int(row["block_number"])
    )

    if key in feature_map:
        raise RuntimeError(
            "Duplicate block feature: " + str(key)
        )

    feature_map[key] = row


if len(feature_map) != N_SUBJECTS * N_BLOCKS:
    raise RuntimeError(
        "Expected 230 block feature records"
    )


X = np.array([
    [
        [
            float(
                feature_map[
                    (sid, block)
                ][feature]
            )
            for feature in FEATURES
        ]
        for block in range(1, N_BLOCKS + 1)
    ]
    for sid in ids
], dtype=float)


quality = np.array([
    [
        float(
            feature_map[
                (sid, block)
            ]["clean_fraction"]
        )
        for block in range(1, N_BLOCKS + 1)
    ]
    for sid in ids
])


if (
    not np.isfinite(X).all()
    or not np.isfinite(quality).all()
    or np.any(quality < 0.5)
):
    raise RuntimeError(
        "Invalid feature or block-quality values"
    )


first_features = {
    row["subject"]: row
    for row in first
}

for i, sid in enumerate(ids):

    original = np.array([
        float(
            first_features[sid][feature]
        )
        for feature in FEATURES
    ])

    if not np.allclose(
        X[i, 0],
        original,
        atol=1e-12,
        rtol=0
    ):
        raise RuntimeError(
            "First-minute feature mismatch: " + sid
        )


# ==========================================
# 5. VERIFY ORIGINAL HELD-OUT PREDICTIONS
# ==========================================

reference = {}

for row in ref_rows:

    key = (
        int(row["repeat"]),
        row["subject"],
        int(row["block"])
    )

    if key in reference:
        raise RuntimeError(
            "Duplicate reference prediction"
        )

    reference[key] = row


if len(reference) != REPEATS * N_SUBJECTS * N_BLOCKS:
    raise RuntimeError(
        "Incomplete original block predictions"
    )


orig_first = {}

for row in historical_rows:

    if (
        row["model"] == "RBF_SVM"
        and int(row["duration_sec"]) == 60
    ):

        key = (
            int(row["repeat"]),
            row["subject"]
        )

        if key in orig_first:
            raise RuntimeError(
                "Duplicate original SVM prediction"
            )

        orig_first[key] = row


if len(orig_first) != REPEATS * N_SUBJECTS:
    raise RuntimeError(
        "Incomplete original SVM predictions"
    )


old_pred = np.full(
    (REPEATS, N_SUBJECTS, N_BLOCKS),
    -1,
    dtype=int
)


for repeat in range(REPEATS):

    for i, sid in enumerate(ids):

        original = orig_first[
            (repeat + 1, sid)
        ]

        for block in range(1, N_BLOCKS + 1):

            row = reference[
                (repeat + 1, sid, block)
            ]

            if (
                int(row["true_label"]) != y[i]
                or int(row["predicted_label"])
                not in (0, 1)
            ):
                raise RuntimeError(
                    "Original prediction mismatch: " + sid
                )

            old_pred[
                repeat,
                i,
                block - 1
            ] = int(row["predicted_label"])

            if block == 1:

                if (
                    int(original["predicted_label"])
                    != int(row["predicted_label"])
                    or not np.isclose(
                        float(original["decision_score"]),
                        float(row["decision_score"]),
                        atol=1e-7,
                        rtol=0
                    )
                ):
                    raise RuntimeError(
                        "Original first-minute "
                        "predictions disagree"
                    )


print(
    "PASS: 46 subjects, all block features, "
    "and prior predictions reconciled.",
    flush=True
)


# ==========================================
# 6. NESTED SUBJECT-LEVEL CROSS-VALIDATION
# ==========================================

aug_pred = np.full_like(
    old_pred,
    -1
)

aug_scores = np.full(
    (REPEATS, N_SUBJECTS, N_BLOCKS),
    np.nan
)

fold_rows = []
prediction_rows = []


for repeat in range(REPEATS):

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42 + 100 * repeat
    )

    for fold, (train, test) in enumerate(
        outer.split(X[:, 0, :], y),
        1
    ):

        if set(train).intersection(test):
            raise RuntimeError(
                "Outer participant leakage"
            )

        # Confirm identical original outer folds.
        for i in test:

            sid = ids[i]

            original = orig_first[
                (repeat + 1, sid)
            ]

            if int(original["fold"]) != fold:
                raise RuntimeError(
                    f"Outer-fold mismatch: {sid}"
                )

            for block in range(1, N_BLOCKS + 1):

                row = reference[
                    (repeat + 1, sid, block)
                ]

                if int(row["fold"]) != fold:
                    raise RuntimeError(
                        f"Block-fold mismatch: {sid}"
                    )

        # Inner folds contain training subjects only.
        inner = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=(
                1000 + 100 * repeat + fold
            )
        )

        inner_splits = [
            (
                train[a],
                train[b]
            )
            for a, b in inner.split(
                X[train, 0, :],
                y[train]
            )
        ]

        best_score = -float("inf")
        best_combo = None

        for c, gamma in GRID:

            inner_scores = []

            for inner_train, inner_val in inner_splits:

                if set(inner_train).intersection(
                    inner_val
                ):
                    raise RuntimeError(
                        "Inner participant leakage"
                    )

                model = model_for(
                    c,
                    gamma
                )

                # Exactly five epochs-of-features
                # per training subject.
                model.fit(
                    X[inner_train].reshape(
                        -1,
                        len(FEATURES)
                    ),
                    np.repeat(
                        y[inner_train],
                        N_BLOCKS
                    )
                )

                # Score on unseen validation subjects,
                # separately for each 60s position.
                predictions = model.predict(
                    X[inner_val].reshape(
                        -1,
                        len(FEATURES)
                    )
                ).reshape(
                    len(inner_val),
                    N_BLOCKS
                )

                block_scores = [
                    bacc(
                        y[inner_val],
                        predictions[:, k]
                    )
                    for k in range(N_BLOCKS)
                ]

                inner_scores.append(
                    float(np.mean(block_scores))
                )

            mean_inner = float(
                np.mean(inner_scores)
            )

            if mean_inner > best_score + 1e-12:

                best_score = mean_inner
                best_combo = (
                    c,
                    gamma
                )

        c, gamma = best_combo

        # Refit on all OUTER-TRAINING subjects.
        fitted = model_for(
            c,
            gamma
        )

        fitted.fit(
            X[train].reshape(
                -1,
                len(FEATURES)
            ),
            np.repeat(
                y[train],
                N_BLOCKS
            )
        )

        # Every outer test participant remains unseen.
        test_features = X[test].reshape(
            -1,
            len(FEATURES)
        )

        predicted = fitted.predict(
            test_features
        ).reshape(
            len(test),
            N_BLOCKS
        )

        scores = fitted.decision_function(
            test_features
        ).reshape(
            len(test),
            N_BLOCKS
        )

        aug_pred[
            repeat,
            test,
            :
        ] = predicted

        aug_scores[
            repeat,
            test,
            :
        ] = scores

        fold_rows.append({
            "repeat": repeat + 1,
            "fold": fold,
            "C": c,
            "gamma": gamma,
            "inner_mean_block_BACC": best_score,
            "train_subjects": len(train),
            "test_subjects": len(test)
        })

        for local_i, subject_i in enumerate(test):

            for block in range(N_BLOCKS):

                prediction_rows.append({
                    "repeat": repeat + 1,
                    "outer_fold": fold,
                    "subject": ids[subject_i],
                    "true_label": int(
                        y[subject_i]
                    ),
                    "group":
                        "A" if y[subject_i] else "C",
                    "test_block": block + 1,
                    "predicted_label": int(
                        predicted[local_i, block]
                    ),
                    "decision_score": float(
                        scores[local_i, block]
                    )
                })

    print(
        f"Completed grouped nested CV "
        f"repeat {repeat + 1}/{REPEATS}",
        flush=True
    )


if (
    np.any(aug_pred < 0)
    or not np.isfinite(aug_scores).all()
    or len(prediction_rows) != 1150
):
    raise RuntimeError(
        "Incomplete augmented held-out predictions"
    )


save(
    OUT / "fold_hyperparameters.csv",
    fold_rows
)

save(
    OUT / "augmented_heldout_block_predictions.csv",
    prediction_rows
)

print(
    "PASS: All 1,150 held-out augmented "
    "predictions generated.",
    flush=True
)


# ==========================================
# 7. METRIC DEFINITIONS
# ==========================================

def mean_bacc(arr, ad_idx, hc_idx):

    sensitivity = (
        arr[:, ad_idx, :] == 1
    ).mean()

    specificity = (
        arr[:, hc_idx, :] == 0
    ).mean()

    return float(
        0.5 * (sensitivity + specificity)
    )


def reversal(arr):

    return (
        arr != arr[:, :, :1]
    ).any(axis=2)


def majority(arr):

    return (
        arr.sum(axis=2) >= 3
    ).astype(int)[:, :, None]


ad = np.flatnonzero(y == 1)
hc = np.flatnonzero(y == 0)

old_flip = reversal(old_pred)
new_flip = reversal(aug_pred)


# ==========================================
# 8. REPEAT-LEVEL RESULTS
# ==========================================

print(
    "\n===== REPEATED TEMPORAL "
    "AUGMENTATION RESULTS ====="
)

run_rows = []

for repeat in range(REPEATS):

    for block in range(N_BLOCKS):

        old_score = bacc(
            y,
            old_pred[repeat, :, block]
        )

        new_score = bacc(
            y,
            aug_pred[repeat, :, block]
        )

        print(
            f"Repeat {repeat + 1}, "
            f"block {block + 1}: "
            f"baseline BACC={old_score:.4f}, "
            f"augmented BACC={new_score:.4f}"
        )

    run_rows.append({
        "repeat": repeat + 1,

        "baseline_mean_block_bacc": np.mean([
            bacc(
                y,
                old_pred[repeat, :, k]
            )
            for k in range(N_BLOCKS)
        ]),

        "augmented_mean_block_bacc": np.mean([
            bacc(
                y,
                aug_pred[repeat, :, k]
            )
            for k in range(N_BLOCKS)
        ]),

        "baseline_flip_rate":
            old_flip[repeat].mean(),

        "augmented_flip_rate":
            new_flip[repeat].mean(),

        "baseline_vote5_bacc": bacc(
            y,
            majority(old_pred)[repeat, :, 0]
        ),

        "augmented_vote5_bacc": bacc(
            y,
            majority(aug_pred)[repeat, :, 0]
        )
    })


save(
    OUT / "repeat_summary.csv",
    run_rows
)


# ==========================================
# 9. PRIMARY AND SECONDARY COMPARISONS
# ==========================================

print(
    "\n===== PRIMARY AND SECONDARY COMPARISON ====="
)

old_overall = mean_bacc(
    old_pred, ad, hc
)

new_overall = mean_bacc(
    aug_pred, ad, hc
)

print(
    "Mean test-block BACC: "
    f"baseline={old_overall:.4f}, "
    f"augmented={new_overall:.4f}, "
    f"difference={new_overall-old_overall:+.4f}"
)


for block in (
    0, 2, 4
):

    old_score = mean_bacc(
        old_pred[:, :, block:block + 1],
        ad, hc
    )

    new_score = mean_bacc(
        aug_pred[:, :, block:block + 1],
        ad, hc
    )

    print(
        f"Test minute {block + 1}: "
        f"baseline BACC={old_score:.4f}, "
        f"augmented BACC={new_score:.4f}, "
        f"delta={new_score-old_score:+.4f}"
    )


old_vote = mean_bacc(
    majority(old_pred), ad, hc
)

new_vote = mean_bacc(
    majority(aug_pred), ad, hc
)

print(
    "Five-block vote BACC: "
    f"baseline={old_vote:.4f}, "
    f"augmented={new_vote:.4f}, "
    f"difference={new_vote-old_vote:+.4f}"
)

print(
    "Class reversal rate: "
    f"baseline={old_flip.mean():.4f}, "
    f"augmented={new_flip.mean():.4f}, "
    f"difference={new_flip.mean()-old_flip.mean():+.4f}"
)


# ==========================================
# 10. SUBJECT-PAIRED BOOTSTRAP
# ==========================================

rng = np.random.default_rng(SEED)

metrics = {

    "primary_all_blocks_BACC":
        lambda ai, hi:
        mean_bacc(aug_pred, ai, hi)
        - mean_bacc(old_pred, ai, hi),

    "first_block_BACC":
        lambda ai, hi:
        mean_bacc(
            aug_pred[:, :, :1], ai, hi
        )
        - mean_bacc(
            old_pred[:, :, :1], ai, hi
        ),

    "fifth_block_BACC":
        lambda ai, hi:
        mean_bacc(
            aug_pred[:, :, 4:], ai, hi
        )
        - mean_bacc(
            old_pred[:, :, 4:], ai, hi
        ),

    "five_block_vote_BACC":
        lambda ai, hi:
        mean_bacc(
            majority(aug_pred), ai, hi
        )
        - mean_bacc(
            majority(old_pred), ai, hi
        ),

    "class_reversal_rate":
        lambda ai, hi:
        (
            new_flip[
                :, np.r_[ai, hi]
            ].mean()
            -
            old_flip[
                :, np.r_[ai, hi]
            ].mean()
        )
}


observed = {
    name: function(ad, hc)
    for name, function in metrics.items()
}

bootstrap = {
    name: []
    for name in metrics
}


for _ in range(N_BOOT):

    sampled_ad = rng.choice(
        ad,
        len(ad),
        replace=True
    )

    sampled_hc = rng.choice(
        hc,
        len(hc),
        replace=True
    )

    for name, function in metrics.items():

        bootstrap[name].append(
            float(
                function(
                    sampled_ad,
                    sampled_hc
                )
            )
        )


ci_rows = []

print(
    "\n===== SUBJECT-PAIRED "
    "CONDITIONAL 95% CI ====="
)

for name, values in bootstrap.items():

    lo, hi = np.percentile(
        values,
        [2.5, 97.5]
    )

    ci_rows.append({
        "outcome": name,
        "difference_augmented_minus_baseline":
            observed[name],
        "conditional_95ci_low":
            float(lo),
        "conditional_95ci_high":
            float(hi),
        "bootstrap_resamples": N_BOOT
    })

    print(
        f"{name}: "
        f"{observed[name]:+.4f}, "
        f"conditional 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )


save(
    OUT / "paired_subject_bootstrap.csv",
    ci_rows
)


# ==========================================
# 11. GENERATE COMPARISON FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.5)
)

for label, arr, marker in (
    (
        "Original first-minute training",
        old_pred,
        "o"
    ),
    (
        "Five-block training augmentation",
        aug_pred,
        "s"
    )
):

    values = np.array([
        [
            bacc(
                y,
                arr[repeat, :, block]
            )
            for block in range(N_BLOCKS)
        ]
        for repeat in range(REPEATS)
    ])

    ax.errorbar(
        range(1, 6),
        values.mean(axis=0),
        yerr=values.std(
            axis=0,
            ddof=1
        ),
        marker=marker,
        capsize=4,
        label=label
    )


ax.set(
    xlabel="Held-out 60-second test block",
    ylabel="Participant-level balanced accuracy",
    title="Temporal augmentation: held-out comparison",
    xticks=range(1, 6),
    ylim=(0, 1)
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    OUT / "block_augmented_comparison.png",
    dpi=190
)

plt.close(fig)


# ==========================================
# 12. FINISH
# ==========================================

print(
    "\n===== BLOCK-AUGMENTATION TEST COMPLETE ====="
)

print(
    "Saved to:",
    OUT
)

print(
    "Exploratory analysis; "
    "no independent external validation."
)

print(
    "Original EEG and historical results unchanged."
)

