#!/usr/bin/env python3
"""Exploratory 5x5 source-to-target EEG minute generalization."""

from pathlib import Path
import csv
import json
import hashlib
from collections import Counter

import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()

SRC = ROOT / "reports/thesis_final/duration_locked_v1"
PRIOR = ROOT / "reports/thesis_final/temporal_repeated_v1"

OUT = ROOT / "reports/thesis_final/temporal_transfer_matrix_v1"
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
BLOCKS = 5
SUBJECTS = 46

N_BOOT = 2000
SEED = 20261107


# ========================================
# 1. LOCK ANALYSIS PROTOCOL
# ========================================

protocol = {
    "status": "post-hoc exploratory, not independently validated",
    "cohort": "46 previously selected AD / healthy participants",
    "source_block":
        "One of five nonoverlapping 60-second intervals "
        "from each training participant",
    "target_block":
        "One of five intervals from each held-out participant",
    "outer_cv":
        "Same five repeated five-fold participant splits "
        "as duration_locked_v1",
    "inner_cv":
        "Three stratified folds within outer training subjects",
    "model":
        "StandardScaler plus RBF SVC, class_weight=balanced",
    "grid": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1]
    },
    "primary_metric": "Participant-level balanced accuracy",
    "primary_comparison":
        "Mean matched-position minus cross-position BACC",
    "secondary_comparison":
        "Train block five / test block five minus "
        "train block one / test block five",
    "uncertainty":
        "Stratified participant bootstrap of held-out predictions "
        "keeping all repeated observations together",
    "limitations": [
        "Previously examined cohort",
        "Full-session preprocessing",
        "Repeated CV estimates are correlated",
        "No independent external validation"
    ]
}

protocol_path = OUT / "locked_protocol.json"

serialized = (
    json.dumps(
        protocol,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if protocol_path.exists():

    if protocol_path.read_text() != serialized:
        raise RuntimeError(
            "Locked protocol conflict. Refusing to overwrite."
        )

else:
    protocol_path.write_text(serialized)


# ========================================
# 2. FILE UTILITIES
# ========================================

def read(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save(path, records):

    if not records:
        raise RuntimeError("Empty output: " + str(path))

    columns = list(dict.fromkeys(
        key
        for record in records
        for key in record
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
        writer.writerows(records)


def unique_index(rows, key_fn, description):

    result = {}

    for row in rows:

        key = key_fn(row)

        if key in result:
            raise RuntimeError(
                f"Duplicate {description}: {key}"
            )

        result[key] = row

    return result


print(
    "===== FIVE-BY-FIVE TEMPORAL TRANSFER TEST =====",
    flush=True
)

print("Source:", SRC)


# ========================================
# 3. LOAD AND FINGERPRINT INPUT FILES
# ========================================

sources = [
    SRC / "duration_features.csv",
    SRC / "five_60s_blocks.csv",
    SRC / "held_out_predictions.csv",
    PRIOR / "all_repeats_heldout_block_predictions.csv"
]

save(
    OUT / "input_sha256.csv",
    [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        }
        for path in sources
    ]
)

historical_rows = read(sources[2])

initial = [
    row for row in historical_rows
    if row["model"] == "RBF_SVM"
    and int(row["repeat"]) == 1
    and int(row["duration_sec"]) == 60
]

ids = [
    row["subject"]
    for row in initial
]

if (
    len(ids) != SUBJECTS
    or len(set(ids)) != SUBJECTS
):
    raise RuntimeError(
        "Expected 46 unique participant IDs"
    )

y = np.array([
    int(row["true_label"])
    for row in initial
])

if Counter(y.tolist()) != {0: 23, 1: 23}:
    raise RuntimeError(
        "Unexpected AD/healthy distribution"
    )


# ========================================
# 4. VERIFY PREVIOUS HELD-OUT RESULTS
# ========================================

historical = unique_index(
    (
        row for row in historical_rows
        if row["model"] == "RBF_SVM"
        and int(row["duration_sec"]) == 60
    ),
    lambda row: (
        int(row["repeat"]),
        row["subject"]
    ),
    "historical SVM prediction"
)

if len(historical) != REPEATS * SUBJECTS:
    raise RuntimeError(
        "Historical 60-second predictions incomplete"
    )

reference = unique_index(
    read(sources[3]),
    lambda row: (
        int(row["repeat"]),
        row["subject"],
        int(row["block"])
    ),
    "previous block prediction"
)

if len(reference) != REPEATS * SUBJECTS * BLOCKS:
    raise RuntimeError(
        "Previous repeated block predictions incomplete"
    )


# ========================================
# 5. CONSTRUCT PARTICIPANT FEATURE ARRAYS
# ========================================

block_rows = unique_index(
    read(sources[1]),
    lambda row: (
        row["subject"],
        int(row["block_number"])
    ),
    "block feature"
)

if len(block_rows) != SUBJECTS * BLOCKS:
    raise RuntimeError(
        "Expected 46 x 5 block feature records"
    )

X = np.array([
    [
        [
            float(
                block_rows[(sid, block)][feature]
            )
            for feature in FEATURES
        ]
        for block in range(1, BLOCKS + 1)
    ]
    for sid in ids
], dtype=float)

quality = np.array([
    [
        float(
            block_rows[
                (sid, block)
            ]["clean_fraction"]
        )
        for block in range(1, BLOCKS + 1)
    ]
    for sid in ids
], dtype=float)

if not (
    np.isfinite(X).all()
    and np.isfinite(quality).all()
):
    raise RuntimeError(
        "Nonfinite features or quality values"
    )

if (quality < 0.5).any():
    raise RuntimeError(
        "A block violates minimum clean-fraction rule"
    )

first_minute = unique_index(
    (
        row for row in read(sources[0])
        if int(row["duration_sec"]) == 60
    ),
    lambda row: row["subject"],
    "first-minute feature"
)

if set(first_minute) != set(ids):
    raise RuntimeError(
        "First-minute participant mismatch"
    )

for i, sid in enumerate(ids):

    original_features = np.array([
        float(first_minute[sid][feature])
        for feature in FEATURES
    ])

    if not np.allclose(
        X[i, 0],
        original_features,
        atol=1e-12,
        rtol=0
    ):
        raise RuntimeError(
            f"First-block feature mismatch: {sid}"
        )

print(
    "PASS: Feature values, participants, "
    "and original first-minute inputs verified."
)


# ========================================
# 6. MATCHED NESTED CROSS-VALIDATION
# ========================================

# Axes:
# repetition, source block, target block, participant

pred = np.full(
    (
        REPEATS,
        BLOCKS,
        BLOCKS,
        SUBJECTS
    ),
    -1,
    dtype=np.int8
)

score = np.full(
    pred.shape,
    np.nan,
    dtype=float
)

fold_map = np.full(
    (REPEATS, SUBJECTS),
    -1,
    dtype=int
)

hyperparameters = []

for repeat in range(REPEATS):

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42 + 100 * repeat
    )

    splits = list(
        outer.split(
            X[:, 0, :],
            y
        )
    )

    for fold, (train, test) in enumerate(
        splits,
        1
    ):

        if set(train).intersection(test):
            raise RuntimeError(
                "Participant leakage detected"
            )

        fold_map[repeat, test] = fold

        for source_block in range(BLOCKS):

            pipeline = Pipeline([
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced"
                    )
                )
            ])

            inner = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=(
                    1000
                    + repeat * 100
                    + fold
                )
            )

            search = GridSearchCV(
                pipeline,
                {
                    "clf__C": [
                        0.1, 1.0, 10.0
                    ],
                    "clf__gamma": [
                        "scale", 0.01, 0.1
                    ]
                },
                cv=inner,
                scoring="balanced_accuracy",
                refit=True,
                n_jobs=1,
                error_score="raise"
            )

            # Fit only on training participants
            # using the chosen source block.

            search.fit(
                X[train, source_block, :],
                y[train]
            )

            fitted = search.best_estimator_

            # Evaluate on every recording block
            # from held-out participants.

            test_features = X[test].reshape(
                len(test) * BLOCKS,
                len(FEATURES)
            )

            predictions = fitted.predict(
                test_features
            ).reshape(
                len(test),
                BLOCKS
            )

            scores = (
                fitted.decision_function(
                    test_features
                )
            ).reshape(
                len(test),
                BLOCKS
            )

            for target_block in range(BLOCKS):

                pred[
                    repeat,
                    source_block,
                    target_block,
                    test
                ] = predictions[:, target_block]

                score[
                    repeat,
                    source_block,
                    target_block,
                    test
                ] = scores[:, target_block]

            hyperparameters.append({
                "repeat": repeat + 1,
                "outer_fold": fold,
                "train_block": source_block + 1,
                "C": search.best_params_["clf__C"],
                "gamma": search.best_params_["clf__gamma"]
            })

            # The first source block must reproduce
            # the historical saved predictions
            # for every repetition and target block.

            if source_block == 0:

                for local_i, subject_i in enumerate(test):

                    sid = ids[subject_i]

                    old = historical[
                        (repeat + 1, sid)
                    ]

                    if (
                        int(old["fold"]) != fold
                        or int(old["true_label"]) != y[subject_i]
                        or int(old["predicted_label"])
                            != int(predictions[local_i, 0])
                        or not np.isclose(
                            float(old["decision_score"]),
                            scores[local_i, 0],
                            atol=1e-7,
                            rtol=0
                        )
                    ):
                        raise RuntimeError(
                            "Original SVM reproduction failed: "
                            f"repeat={repeat+1}, {sid}"
                        )

                    for target_block in range(BLOCKS):

                        previous = reference[
                            (
                                repeat + 1,
                                sid,
                                target_block + 1
                            )
                        ]

                        if (
                            int(previous["fold"]) != fold
                            or int(previous["predicted_label"])
                                != int(
                                    predictions[
                                        local_i,
                                        target_block
                                    ]
                                )
                            or not np.isclose(
                                float(previous["decision_score"]),
                                scores[
                                    local_i,
                                    target_block
                                ],
                                atol=1e-7,
                                rtol=0
                            )
                        ):
                            raise RuntimeError(
                                "Prior block reproduction failed: "
                                f"repeat={repeat+1}, "
                                f"{sid}, "
                                f"block={target_block+1}"
                            )

    print(
        f"Repeat {repeat+1}/{REPEATS}: "
        "all training positions complete; "
        "historical reproduction PASS",
        flush=True
    )

if (
    (pred < 0).any()
    or not np.isfinite(score).all()
    or (fold_map < 1).any()
):
    raise RuntimeError(
        "Incomplete cross-validation predictions"
    )


# ========================================
# 7. SAVE CANONICAL HELD-OUT PREDICTIONS
# ========================================

save(
    OUT / "hyperparameters_by_fold.csv",
    hyperparameters
)

prediction_rows = []

for repeat in range(REPEATS):

    for source_block in range(BLOCKS):

        for target_block in range(BLOCKS):

            for i, sid in enumerate(ids):

                prediction_rows.append({
                    "repeat": repeat + 1,
                    "subject": sid,
                    "group": "A" if y[i] else "C",
                    "true_label": int(y[i]),
                    "outer_fold": int(
                        fold_map[repeat, i]
                    ),
                    "train_block": source_block + 1,
                    "test_block": target_block + 1,
                    "test_start_sec": target_block * 60,
                    "test_end_sec": (target_block + 1) * 60,
                    "predicted_label": int(
                        pred[
                            repeat,
                            source_block,
                            target_block,
                            i
                        ]
                    ),
                    "decision_score": float(
                        score[
                            repeat,
                            source_block,
                            target_block,
                            i
                        ]
                    )
                })

save(
    OUT / "cross_block_held_out_predictions.csv",
    prediction_rows
)


# ========================================
# 8. MEAN TEMPORAL GENERALIZATION MATRIX
# ========================================

ad = np.flatnonzero(y == 1)
hc = np.flatnonzero(y == 0)


def metric_matrix(ad_indices, hc_indices):

    correct = (
        pred
        ==
        y.reshape(1, 1, 1, -1)
    )

    sensitivity = correct[
        ..., ad_indices
    ].mean(axis=-1)

    specificity = correct[
        ..., hc_indices
    ].mean(axis=-1)

    return (
        sensitivity + specificity
    ) / 2


matrix = metric_matrix(
    ad,
    hc
)

mean_matrix = matrix.mean(axis=0)

sd_matrix = matrix.std(
    axis=0,
    ddof=1
)

per_repeat_rows = []
mean_rows = []

for source_block in range(BLOCKS):

    for target_block in range(BLOCKS):

        for repeat in range(REPEATS):

            p = pred[
                repeat,
                source_block,
                target_block
            ]

            per_repeat_rows.append({
                "repeat": repeat + 1,
                "train_block": source_block + 1,
                "test_block": target_block + 1,
                "balanced_accuracy": float(
                    matrix[
                        repeat,
                        source_block,
                        target_block
                    ]
                ),
                "accuracy": float(
                    np.mean(p == y)
                ),
                "sensitivity": float(
                    (p[ad] == 1).mean()
                ),
                "specificity": float(
                    (p[hc] == 0).mean()
                )
            })

        mean_rows.append({
            "train_block": source_block + 1,
            "test_block": target_block + 1,
            "mean_balanced_accuracy": float(
                mean_matrix[
                    source_block,
                    target_block
                ]
            ),
            "sd_across_correlated_repeats": float(
                sd_matrix[
                    source_block,
                    target_block
                ]
            ),
            "block_distance": abs(
                source_block - target_block
            ),
            "same_position": int(
                source_block == target_block
            )
        })

save(
    OUT / "transfer_matrix_per_repeat.csv",
    per_repeat_rows
)

save(
    OUT / "transfer_matrix_mean.csv",
    mean_rows
)

print(
    "\n===== MEAN CROSS-BLOCK BACC MATRIX ====="
)

print(
    "Rows = TRAIN block; "
    "columns = HELD-OUT TEST block"
)

print(
    "             test1   test2   test3   test4   test5"
)

for source_block in range(BLOCKS):

    values = " ".join(
        f"{mean_matrix[source_block, j]:.3f}"
        for j in range(BLOCKS)
    )

    print(
        f"train{source_block+1}       {values}"
    )


# ========================================
# 9. CONDITIONAL PARTICIPANT BOOTSTRAP
# ========================================

diagonal = np.eye(
    BLOCKS,
    dtype=bool
)

off_diagonal = ~diagonal

distance_grid = np.abs(
    np.subtract.outer(
        np.arange(BLOCKS),
        np.arange(BLOCKS)
    )
)


def aggregate_metrics(m):

    matched = float(
        m[:, diagonal].mean()
    )

    transferred = float(
        m[:, off_diagonal].mean()
    )

    result = {
        "matched_position": matched,
        "cross_position": transferred,
        "match_minus_cross": (
            matched - transferred
        ),
        "last_block_rescue": float(
            m[:, 4, 4].mean()
            -
            m[:, 0, 4].mean()
        ),
        "source1_target5_minus_source1_target1": float(
            m[:, 0, 4].mean()
            -
            m[:, 0, 0].mean()
        ),
        "source5_target5": float(
            m[:, 4, 4].mean()
        ),
        "source1_target5": float(
            m[:, 0, 4].mean()
        )
    }

    for distance in range(BLOCKS):

        result[
            f"block_distance_{distance}"
        ] = float(
            m[
                :,
                distance_grid == distance
            ].mean()
        )

    return result


observed = aggregate_metrics(
    matrix
)

rng = np.random.default_rng(SEED)

boot = {
    key: []
    for key in observed
}

for _ in range(N_BOOT):

    sample_ad = rng.choice(
        ad,
        size=len(ad),
        replace=True
    )

    sample_hc = rng.choice(
        hc,
        size=len(hc),
        replace=True
    )

    result = aggregate_metrics(
        metric_matrix(
            sample_ad,
            sample_hc
        )
    )

    for key, value in result.items():

        boot[key].append(
            value
        )


ci_rows = []

for key, value in observed.items():

    low, high = np.percentile(
        boot[key],
        [2.5, 97.5]
    )

    ci_rows.append({
        "outcome": key,
        "estimate": float(value),
        "conditional_95ci_low": float(low),
        "conditional_95ci_high": float(high),
        "bootstrap_subject_repetitions": N_BOOT
    })

save(
    OUT / "paired_subject_bootstrap.csv",
    ci_rows
)

print(
    "\n===== MATCHED VS CROSS-POSITION PERFORMANCE ====="
)

important = {
    "matched_position",
    "cross_position",
    "match_minus_cross",
    "last_block_rescue",
    "source1_target5_minus_source1_target1",
    "source5_target5",
    "source1_target5"
}

for row in ci_rows:

    if row["outcome"] not in important:
        continue

    print(
        f"{row['outcome']}: "
        f"{row['estimate']:+.4f}, "
        "conditional 95% CI "
        f"[{row['conditional_95ci_low']:+.4f}, "
        f"{row['conditional_95ci_high']:+.4f}]"
    )

print(
    "\n===== PERFORMANCE BY TEMPORAL DISTANCE ====="
)

for distance in range(BLOCKS):

    row = next(
        r for r in ci_rows
        if r["outcome"] == f"block_distance_{distance}"
    )

    print(
        f"Distance {distance}: "
        f"mean BACC={row['estimate']:.4f}, "
        "conditional 95% CI "
        f"[{row['conditional_95ci_low']:.4f}, "
        f"{row['conditional_95ci_high']:.4f}]"
    )


# ========================================
# 10. SIGNAL QUALITY AND FEATURE SHIFT
# ========================================

quality_rows = []
shift_rows = []

for label, group in [
    (1, "A"),
    (0, "C")
]:

    indices = np.flatnonzero(
        y == label
    )

    for block in range(BLOCKS):

        q = quality[
            indices,
            block
        ]

        quality_rows.append({
            "group": group,
            "block": block + 1,
            "n_subjects": len(indices),
            "median_clean_fraction": float(
                np.median(q)
            ),
            "mean_clean_fraction": float(
                q.mean()
            ),
            "min_clean_fraction": float(
                q.min()
            )
        })

    for k, feature in enumerate(FEATURES):

        difference = (
            X[indices, 4, k]
            -
            X[indices, 0, k]
        )

        first_sd = float(
            X[:, 0, k].std(ddof=1)
        )

        shift_rows.append({
            "group": group,
            "feature": feature,
            "median_block5_minus_block1": float(
                np.median(difference)
            ),
            "mean_block5_minus_block1": float(
                difference.mean()
            ),
            "median_change_in_first_block_sd_units": (
                float(np.median(difference) / first_sd)
                if first_sd > 0
                else ""
            ),
            "n_subjects": len(indices)
        })

save(
    OUT / "quality_by_block_and_group.csv",
    quality_rows
)

save(
    OUT / "feature_shift_block1_to_block5.csv",
    shift_rows
)

print(
    "\n===== QUALITY CHECK: BLOCK 1 VS BLOCK 5 ====="
)

for group in ("A", "C"):

    first_q = next(
        r for r in quality_rows
        if r["group"] == group
        and r["block"] == 1
    )

    last_q = next(
        r for r in quality_rows
        if r["group"] == group
        and r["block"] == 5
    )

    print(
        f"{group}: "
        "median clean fraction "
        f"block1={first_q['median_clean_fraction']:.3f}, "
        f"block5={last_q['median_clean_fraction']:.3f}"
    )


# ========================================
# 11. GENERATE TEMPORAL TRANSFER HEATMAP
# ========================================

fig, ax = plt.subplots(
    figsize=(7.0, 5.6)
)

image = ax.imshow(
    mean_matrix,
    vmin=0.45,
    vmax=0.95,
    cmap="viridis"
)

ax.set_xticks(
    range(BLOCKS),
    [
        f"Minute {i}"
        for i in range(1, BLOCKS + 1)
    ]
)

ax.set_yticks(
    range(BLOCKS),
    [
        f"Minute {i}"
        for i in range(1, BLOCKS + 1)
    ]
)

ax.set_xlabel(
    "Held-out test recording position"
)

ax.set_ylabel(
    "Training-subject recording position"
)

ax.set_title(
    "Participant-held-out temporal transfer "
    "(five CV repeats)"
)

for source_block in range(BLOCKS):

    for target_block in range(BLOCKS):

        value = mean_matrix[
            source_block,
            target_block
        ]

        ax.text(
            target_block,
            source_block,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=(
                "white"
                if value < 0.67
                else "black"
            ),
            fontsize=10
        )

fig.colorbar(
    image,
    ax=ax,
    label="Mean participant-level balanced accuracy"
)

fig.tight_layout()

fig.savefig(
    OUT / "temporal_transfer_matrix.png",
    dpi=200
)

plt.close(fig)


print(
    "\n===== TEMPORAL TRANSFER MATRIX COMPLETE ====="
)

print("Outputs:", OUT)

print(
    "No original EEG data or historical "
    "model results modified."
)

print(
    "Conditional intervals do not capture "
    "all model-retraining uncertainty."
)

print(
    "All results remain exploratory; "
    "no external validation was performed."
)

