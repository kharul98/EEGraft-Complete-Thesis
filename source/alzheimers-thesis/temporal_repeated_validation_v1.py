#!/usr/bin/env python3
"""Repeated-CV temporal transportability of the original 60-second SVM."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
SRC = ROOT / "reports/thesis_final/duration_locked_v1"
OUT = ROOT / "reports/thesis_final/temporal_repeated_v1"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = (
    "delta", "theta", "alpha",
    "beta", "gamma", "theta_alpha_ratio"
)

REPEATS = 5
N_SUBJECTS = 46
N_BLOCKS = 5
N_BOOT = 2500
SEED = 20261107


# ========================================
# LOCK THE ANALYSIS
# ========================================

PROTOCOL = {
    "version": "temporal_repeated_v1",
    "status": "exploratory; previously examined cohort",
    "cohort": "46 AD/healthy ds004504 participants",
    "training_input": "first 60 seconds of training subjects",
    "testing_input": "five nonoverlapping 60-second blocks",
    "outer_cv": "5 subject folds x 5 repeats",
    "inner_cv": "3 stratified training-only folds",
    "hyperparameters": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.1]
    },
    "primary_metric": "participant-level balanced accuracy",
    "main_outcomes": [
        "subject-level block reversal rate",
        "balanced accuracy by block",
        "three- and five-block majority voting",
        "agreement-based selective classification"
    ],
    "uncertainty":
        "stratified subject bootstrap retaining "
        "all repeated observations together",
    "limitations": [
        "exploratory post-hoc study",
        "same 46 participants reused",
        "full-session preprocessing",
        "no independent external validation"
    ]
}

protocol_file = OUT / "protocol.json"

protocol_text = (
    json.dumps(
        PROTOCOL,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if protocol_file.exists():
    if protocol_file.read_text() != protocol_text:
        raise RuntimeError(
            "Existing locked protocol differs. "
            "Refusing to overwrite."
        )
else:
    protocol_file.write_text(protocol_text)


# ========================================
# FILE UTILITIES
# ========================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise RuntimeError(
            "Cannot save empty results: " + str(path)
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


def bacc(y, pred):
    sensitivity = (
        pred[y == 1] == 1
    ).mean()

    specificity = (
        pred[y == 0] == 0
    ).mean()

    return float(
        0.5 * (sensitivity + specificity)
    )


def accuracy(y, pred):
    return float(
        (y == pred).mean()
    )


print(
    "===== REPEATED TEMPORAL TRANSPORTABILITY =====",
    flush=True
)

print("Source:", SRC)


# ========================================
# LOAD SAVED EVIDENCE
# ========================================

source_files = [
    "duration_features.csv",
    "five_60s_blocks.csv",
    "held_out_predictions.csv",
    "held_out_60s_blocks.csv"
]

save_csv(
    OUT / "input_checksums.csv",
    [
        {
            "file": name,
            "sha256": hashlib.sha256(
                (SRC / name).read_bytes()
            ).hexdigest()
        }
        for name in source_files
    ]
)

historical = load_csv(
    SRC / "held_out_predictions.csv"
)

duration = load_csv(
    SRC / "duration_features.csv"
)

block_features = load_csv(
    SRC / "five_60s_blocks.csv"
)

first_repeat_blocks = load_csv(
    SRC / "held_out_60s_blocks.csv"
)


# ========================================
# VERIFY EXACT PARTICIPANT ORDER
# ========================================

initial_rows = [
    row for row in historical
    if row["model"] == "RBF_SVM"
    and int(row["repeat"]) == 1
    and int(row["duration_sec"]) == 60
]

ids = [
    row["subject"]
    for row in initial_rows
]

if (
    len(ids) != 46
    or len(set(ids)) != 46
):
    raise RuntimeError(
        "Unexpected participant ordering"
    )

labels = {
    row["subject"]: int(row["true_label"])
    for row in initial_rows
}

y = np.array(
    [labels[sid] for sid in ids],
    dtype=int
)

if Counter(y.tolist()) != {0: 23, 1: 23}:
    raise RuntimeError(
        "Diagnosis distribution mismatch"
    )


# ========================================
# INDEX ORIGINAL PREDICTIONS
# ========================================

old = {}

for row in historical:

    if row["model"] != "RBF_SVM":
        continue

    key = (
        int(row["repeat"]),
        int(row["duration_sec"]),
        row["subject"]
    )

    if key in old:
        raise RuntimeError(
            "Duplicate prediction: " + str(key)
        )

    old[key] = row

if len(old) != REPEATS * 3 * N_SUBJECTS:
    raise RuntimeError(
        "Incomplete historical SVM predictions"
    )


# ========================================
# LOAD FIRST-MINUTE FEATURES
# ========================================

f60 = {}

for row in duration:

    if int(row["duration_sec"]) == 60:

        sid = row["subject"]

        if sid in f60:
            raise RuntimeError(
                "Duplicate 60-second features"
            )

        f60[sid] = row

if set(f60) != set(ids):
    raise RuntimeError(
        "First-minute participant mismatch"
    )


# ========================================
# LOAD NONOVERLAPPING BLOCK FEATURES
# ========================================

five = {}

for row in block_features:

    key = (
        row["subject"],
        int(row["block_number"])
    )

    if key in five:
        raise RuntimeError(
            "Duplicate block features"
        )

    five[key] = row

if len(five) != 46 * 5:
    raise RuntimeError(
        "Incomplete five-block feature table"
    )

X = np.array([
    [
        float(f60[sid][feature])
        for feature in FEATURES
    ]
    for sid in ids
])

XB = np.array([
    [
        [
            float(
                five[(sid, block)][feature]
            )
            for feature in FEATURES
        ]
        for block in range(1, 6)
    ]
    for sid in ids
])

if not (
    np.isfinite(X).all()
    and np.isfinite(XB).all()
):
    raise RuntimeError(
        "Nonfinite spectral features"
    )

if not np.allclose(
    X,
    XB[:, 0, :],
    atol=1e-12,
    rtol=0
):
    raise RuntimeError(
        "First block differs from 60-second features"
    )


# ========================================
# LOAD FIRST-REPEAT REFERENCE BLOCKS
# ========================================

old_blocks = {}

for row in first_repeat_blocks:

    key = (
        row["subject"],
        int(row["block"])
    )

    if key in old_blocks:
        raise RuntimeError(
            "Duplicate held-out block prediction"
        )

    old_blocks[key] = row

if len(old_blocks) != 230:
    raise RuntimeError(
        "Expected 230 original block predictions"
    )


# ========================================
# REPRODUCE REPEATED NESTED CV
# ========================================

bp = np.full(
    (REPEATS, N_SUBJECTS, N_BLOCKS),
    -1,
    dtype=int
)

bs = np.full(
    (REPEATS, N_SUBJECTS, N_BLOCKS),
    np.nan,
    dtype=float
)

folds = np.full(
    (REPEATS, N_SUBJECTS),
    -1,
    dtype=int
)

prediction_rows = []
hyperparameters = []

for repeat in range(REPEATS):

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42 + 100 * repeat
    )

    for fold, (train, test) in enumerate(
        outer.split(X, y),
        1
    ):

        pipeline = Pipeline([
            ("scale", StandardScaler()),
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
                1000 + repeat * 100 + fold
            )
        )

        search = GridSearchCV(
            pipeline,
            {
                "clf__C": [0.1, 1.0, 10.0],
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

        search.fit(
            X[train],
            y[train]
        )

        model = search.best_estimator_

        test_blocks = XB[test].reshape(
            -1,
            len(FEATURES)
        )

        block_pred = model.predict(
            test_blocks
        ).reshape(
            len(test),
            N_BLOCKS
        )

        block_score = (
            model.decision_function(
                test_blocks
            ).reshape(
                len(test),
                N_BLOCKS
            )
        )

        bp[repeat, test] = block_pred
        bs[repeat, test] = block_score
        folds[repeat, test] = fold

        hyperparameters.append({
            "repeat": repeat + 1,
            "fold": fold,
            "C": search.best_params_["clf__C"],
            "gamma": search.best_params_[
                "clf__gamma"
            ]
        })

        # Verify each held-out subject against
        # original saved model predictions.
        for local_index, subject_index in enumerate(test):

            sid = ids[subject_index]

            for seconds in (60, 180, 300):

                reference = old[
                    (repeat + 1, seconds, sid)
                ]

                if (
                    int(reference["fold"]) != fold
                    or int(reference["true_label"])
                    != int(y[subject_index])
                ):
                    raise RuntimeError(
                        f"Original fold/label mismatch: "
                        f"{sid}, repeat {repeat+1}"
                    )

            reference = old[
                (repeat + 1, 60, sid)
            ]

            if (
                int(reference["predicted_label"])
                != int(block_pred[local_index, 0])
                or not np.isclose(
                    float(reference["decision_score"]),
                    float(block_score[local_index, 0]),
                    atol=1e-7,
                    rtol=0
                )
            ):
                raise RuntimeError(
                    f"60s model reproduction FAILED: "
                    f"repeat={repeat+1}, {sid}"
                )

            # Entire first repeat must also
            # reproduce prior temporal analysis.
            if repeat == 0:

                for block in range(1, 6):

                    previous = old_blocks[
                        (sid, block)
                    ]

                    if (
                        int(previous["fold"]) != fold
                        or int(previous["predicted_ad"])
                        != int(
                            block_pred[
                                local_index, block-1
                            ]
                        )
                        or not np.isclose(
                            float(previous["decision_score"]),
                            float(
                                block_score[
                                    local_index, block-1
                                ]
                            ),
                            atol=1e-7,
                            rtol=0
                        )
                    ):
                        raise RuntimeError(
                            f"Block reproduction FAILED: "
                            f"{sid}, block {block}"
                        )

            for block in range(1, 6):

                prediction_rows.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "subject": sid,
                    "group":
                        "A" if y[subject_index] else "C",
                    "true_label":
                        int(y[subject_index]),
                    "block": block,
                    "start_sec":
                        (block - 1) * 60,
                    "end_sec": block * 60,
                    "predicted_label":
                        int(
                            block_pred[
                                local_index, block-1
                            ]
                        ),
                    "decision_score":
                        float(
                            block_score[
                                local_index, block-1
                            ]
                        ),
                    "clean_fraction":
                        float(
                            five[
                                (sid, block)
                            ]["clean_fraction"]
                        )
                })

    print(
        f"Reproduced repeat "
        f"{repeat+1}/{REPEATS}: "
        "saved 60s labels and scores match",
        flush=True
    )


if (
    (bp < 0).any()
    or not np.isfinite(bs).all()
    or (folds < 1).any()
):
    raise RuntimeError(
        "Incomplete cross-validation output"
    )

save_csv(
    OUT / "all_repeats_heldout_block_predictions.csv",
    prediction_rows
)

save_csv(
    OUT / "fold_hyperparameters.csv",
    hyperparameters
)

print(
    "VALIDATION PASS: "
    "5 repeats x 46 subjects x 5 blocks"
)


# ========================================
# DEFINE FIXED AGGREGATION METHODS
# ========================================

first = bp[:, :, 0]

majority3 = (
    bp[:, :, :3].sum(axis=2) >= 2
).astype(int)

majority5 = (
    bp.sum(axis=2) >= 3
).astype(int)

original180 = np.array([
    [
        int(
            old[
                (repeat + 1, 180, sid)
            ]["predicted_label"]
        )
        for sid in ids
    ]
    for repeat in range(REPEATS)
])

original300 = np.array([
    [
        int(
            old[
                (repeat + 1, 300, sid)
            ]["predicted_label"]
        )
        for sid in ids
    ]
    for repeat in range(REPEATS)
])

flip = (
    bp != bp[:, :, :1]
).any(axis=2)

votes = bp.sum(axis=2)

agreement = np.maximum(
    votes,
    N_BLOCKS - votes
)

METHODS = {
    "first60": first,
    "vote3": majority3,
    "vote5": majority5,
    "retrained180": original180,
    "retrained300": original300,
    "block2": bp[:, :, 1],
    "block3": bp[:, :, 2],
    "block4": bp[:, :, 3],
    "block5": bp[:, :, 4]
}


# ========================================
# REPORT REPEAT-LEVEL PERFORMANCE
# ========================================

print(
    "\n===== REPEAT-LEVEL SUMMARY ====="
)

run_rows = []

for repeat in range(REPEATS):

    result = {
        "repeat": repeat + 1,
        "class_reversals_n":
            int(flip[repeat].sum()),
        "class_reversal_rate":
            float(flip[repeat].mean())
    }

    for name, predictions in METHODS.items():

        result[name + "_bacc"] = bacc(
            y,
            predictions[repeat]
        )

        result[name + "_acc"] = accuracy(
            y,
            predictions[repeat]
        )

    for threshold in (4, 5):

        selected = (
            agreement[repeat] >= threshold
        )

        subset_y = y[selected]

        selected_count = int(
            selected.sum()
        )

        result[
            f"agree{threshold}_n"
        ] = selected_count

        result[
            f"agree{threshold}_coverage"
        ] = float(
            selected.mean()
        )

        result[
            f"agree{threshold}_ad_n"
        ] = int(
            (subset_y == 1).sum()
        )

        result[
            f"agree{threshold}_hc_n"
        ] = int(
            (subset_y == 0).sum()
        )

        if selected_count:

            result[
                f"agree{threshold}_acc"
            ] = accuracy(
                subset_y,
                majority5[repeat, selected]
            )

            result[
                f"agree{threshold}_first60_acc_same_subjects"
            ] = accuracy(
                subset_y,
                first[repeat, selected]
            )

            result[
                f"agree{threshold}_retrained300_acc_same_subjects"
            ] = accuracy(
                subset_y,
                original300[repeat, selected]
            )

        if set(subset_y) == {0, 1}:

            result[
                f"agree{threshold}_bacc"
            ] = bacc(
                subset_y,
                majority5[repeat, selected]
            )

            result[
                f"agree{threshold}_first60_bacc_same_subjects"
            ] = bacc(
                subset_y,
                first[repeat, selected]
            )

    run_rows.append(result)

    block_bacc = "/".join(
        f"{bacc(y, bp[repeat, :, j]):.3f}"
        for j in range(5)
    )

    print(
        f"Repeat {repeat+1}: "
        f"flips={int(flip[repeat].sum())}/46; "
        f"block BACC={block_bacc}; "
        f"vote3={result['vote3_bacc']:.3f}; "
        f"vote5={result['vote5_bacc']:.3f}; "
        f"agree5={result['agree5_n']}/46"
    )

save_csv(
    OUT / "repeat_level_results.csv",
    run_rows
)


# ========================================
# SUMMARIZE REPEATED PERFORMANCE
# ========================================

print(
    "\n===== MEAN ACROSS FIVE CORRELATED CV REPEATS ====="
)

summary_rows = []

for name, matrix in METHODS.items():

    scores = np.array([
        bacc(y, matrix[repeat])
        for repeat in range(REPEATS)
    ])

    row = {
        "method": name,
        "mean_bacc":
            float(scores.mean()),
        "repeat_sd_bacc":
            float(scores.std(ddof=1)),
        "min_bacc":
            float(scores.min()),
        "max_bacc":
            float(scores.max())
    }

    summary_rows.append(row)

    print(
        f"{name:14s} "
        f"mean BACC={scores.mean():.4f}, "
        f"repeat SD={scores.std(ddof=1):.4f}"
    )

save_csv(
    OUT / "performance_summary.csv",
    summary_rows
)

print(
    f"Mean class-reversal rate: "
    f"{flip.mean():.4f}"
)

print(
    f"Participants with a reversal "
    f"in at least one repeat: "
    f"{int(flip.any(axis=0).sum())}/46"
)

for threshold in (4, 5):

    coverage = np.array([
        row[f"agree{threshold}_coverage"]
        for row in run_rows
    ])

    selective_accuracy = np.array([
        row.get(
            f"agree{threshold}_acc",
            np.nan
        )
        for row in run_rows
    ])

    baseline_accuracy = np.array([
        row.get(
            f"agree{threshold}_first60_acc_same_subjects",
            np.nan
        )
        for row in run_rows
    ])

    selective_bacc = np.array([
        row.get(
            f"agree{threshold}_bacc",
            np.nan
        )
        for row in run_rows
    ])

    print(
        f"Agreement >= {threshold}/5: "
        f"coverage={coverage.mean():.3f}, "
        f"selected BACC={np.nanmean(selective_bacc):.4f}, "
        f"selected accuracy={np.nanmean(selective_accuracy):.4f}, "
        f"first-60s accuracy on SAME people="
        f"{np.nanmean(baseline_accuracy):.4f}"
    )


# ========================================
# SUBJECT-PAIRED UNCERTAINTY
# ========================================

rng = np.random.default_rng(SEED)

ad_indices = np.where(y == 1)[0]
hc_indices = np.where(y == 0)[0]

resamples = [
    np.r_[
        rng.choice(
            ad_indices,
            len(ad_indices),
            replace=True
        ),
        rng.choice(
            hc_indices,
            len(hc_indices),
            replace=True
        )
    ]
    for _ in range(N_BOOT)
]


def average_bacc(matrix, indices):

    return float(
        np.mean([
            bacc(
                y[indices],
                matrix[repeat, indices]
            )
            for repeat in range(REPEATS)
        ])
    )


comparisons = [
    (
        "block5_minus_block1",
        METHODS["block5"],
        first
    ),
    (
        "block3_minus_block1",
        METHODS["block3"],
        first
    ),
    (
        "vote3_minus_first60",
        majority3,
        first
    ),
    (
        "vote5_minus_first60",
        majority5,
        first
    ),
    (
        "vote3_minus_180s",
        majority3,
        original180
    ),
    (
        "vote5_minus_300s",
        majority5,
        original300
    ),
    (
        "retrained180_minus_first60",
        original180,
        first
    ),
    (
        "retrained300_minus_first60",
        original300,
        first
    )
]

print(
    "\n===== SUBJECT-PAIRED CONDITIONAL 95% BOOTSTRAP ====="
)

comparison_rows = []

all_indices = np.arange(N_SUBJECTS)

for name, method_a, method_b in comparisons:

    observed = (
        average_bacc(
            method_a,
            all_indices
        )
        -
        average_bacc(
            method_b,
            all_indices
        )
    )

    boot = np.array([
        average_bacc(method_a, indices)
        -
        average_bacc(method_b, indices)
        for indices in resamples
    ])

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5]
    )

    comparison_rows.append({
        "comparison": name,
        "mean_delta_bacc":
            float(observed),
        "conditional_ci_low":
            float(lo),
        "conditional_ci_high":
            float(hi)
    })

    print(
        f"{name}: "
        f"{observed:+.4f}, "
        f"conditional 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )

save_csv(
    OUT / "paired_bootstrap_comparisons.csv",
    comparison_rows
)


# ========================================
# INDIVIDUAL PARTICIPANT STABILITY
# ========================================

participant_rows = []

for index, sid in enumerate(ids):

    quality = np.array([
        float(
            five[(sid, block)]["clean_fraction"]
        )
        for block in range(1, 6)
    ])

    ratio = np.array([
        float(
            five[(sid, block)]["theta_alpha_ratio"]
        )
        for block in range(1, 6)
    ])

    if (
        not np.isfinite(ratio).all()
        or (ratio <= 0).any()
    ):
        raise RuntimeError(
            f"Invalid theta/alpha ratio: {sid}"
        )

    participant_rows.append({
        "subject": sid,
        "group":
            "A" if y[index] else "C",
        "flip_repetitions":
            int(flip[:, index].sum()),
        "flip_frequency_across_5_repeats":
            float(flip[:, index].mean()),
        "unanimous_repetitions":
            int(
                (agreement[:, index] == 5).sum()
            ),
        "mean_within_repeat_score_range":
            float(
                np.ptp(
                    bs[:, index, :],
                    axis=1
                ).mean()
            ),
        "clean_fraction_range":
            float(np.ptp(quality)),
        "sd_log_theta_alpha":
            float(
                np.log(ratio).std(ddof=1)
            ),
        "first60_correct_repeats":
            int(
                (first[:, index] == y[index]).sum()
            ),
        "vote5_correct_repeats":
            int(
                (majority5[:, index] == y[index]).sum()
            )
    })

save_csv(
    OUT / "participant_repeated_stability.csv",
    participant_rows
)

flip_boot = np.array([
    flip[:, indices].mean()
    for indices in resamples
])

flip_lo, flip_hi = np.percentile(
    flip_boot,
    [2.5, 97.5]
)

print(
    "\nMean participant reversal rate: "
    f"{flip.mean():.4f}"
)

print(
    "Conditional participant-bootstrap 95% CI: "
    f"[{flip_lo:.4f}, {flip_hi:.4f}]"
)


# ========================================
# GENERATE PUBLICATION FIGURE
# ========================================

block_means = []
block_deviations = []

for block in range(5):

    values = np.array([
        bacc(
            y,
            bp[repeat, :, block]
        )
        for repeat in range(REPEATS)
    ])

    block_means.append(
        float(values.mean())
    )

    block_deviations.append(
        float(values.std(ddof=1))
    )

fig, ax = plt.subplots(
    figsize=(7.2, 4.5)
)

ax.errorbar(
    np.arange(1, 6),
    block_means,
    yerr=block_deviations,
    marker="o",
    capsize=4
)

ax.set(
    xlabel="Nonoverlapping 60-second EEG block",
    ylabel="Participant-level balanced accuracy",
    title="Temporal transportability across repeated CV",
    xticks=np.arange(1, 6),
    ylim=(0, 1)
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUT / "block_position_repeated_cv.png",
    dpi=190
)

plt.close(fig)


print(
    "\n===== REPEATED TEMPORAL STUDY COMPLETE ====="
)

print("Saved results:", OUT)

print(
    "Training-only scaling and tuning; "
    "held-out blocks were never used for fitting."
)

print(
    "Repeated CV results are correlated. "
    "Bootstrap intervals are conditional "
    "on the evaluated protocol."
)

