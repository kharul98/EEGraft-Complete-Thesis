#!/usr/bin/env python3
"""
Milestone 1C: Risk-coverage and error-detection analysis.

Tasks:
    1. AD versus healthy
    2. FTD versus healthy

Uses:
    - Original first-minute SVM predictions
    - Training-only calibrated probabilities
    - Five-minute temporal consistency
    - Participant-level bootstrap

Does NOT retrain classifiers or modify earlier results.
"""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()

BASE = ROOT / "reports" / "thesis_final"

OUT = BASE / "milestone_1c_selective_prediction_v1"
OUT.mkdir(parents=True, exist_ok=True)

PROBABILITIES = (
    BASE / "milestone_1b_calibration_v1"
    / "calibrated_held_out_block_predictions.csv"
)

STATES = (
    BASE / "milestone_1a_reliability_v1"
    / "subject_repeat_states.csv"
)

TASK_INFO = {
    "AD_vs_HC": {
        "positive": "A",
        "n_pos": 23,
        "n_hc": 23
    },
    "FTD_vs_HC": {
        "positive": "F",
        "n_pos": 18,
        "n_hc": 23
    }
}

REPEATS = 5
BLOCKS = 5

COVERAGES = (
    1.00,
    0.90,
    0.80,
    0.70
)

PLOT_COVERAGES = (
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70
)

N_BOOT = 1500
SEED = 20261107

METHODS = {
    "first_minute_original_label_support": {
        "seconds_required": 60,
        "description":
            "Calibrated probability supporting the "
            "original first-minute SVM decision"
    },
    "five_minute_probability_consistency": {
        "seconds_required": 300,
        "description":
            "Lower probability variation across "
            "five minutes ranks as more reliable"
    },
    "five_minute_label_consensus": {
        "seconds_required": 300,
        "description":
            "Greater agreement among five "
            "original SVM predictions"
    }
}


# ==========================================
# 1. FILE UTILITIES
# ==========================================

def read_csv(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def save_csv(path, rows):

    if not rows:
        raise RuntimeError(
            "Cannot save empty table: "
            + str(path)
        )

    columns = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


def make_index(rows, key_function, description):

    result = {}

    for row in rows:

        key = key_function(row)

        if key in result:
            raise RuntimeError(
                f"Duplicate {description}: {key}"
            )

        result[key] = row

    return result


# ==========================================
# 2. SELECTIVE PREDICTION UTILITIES
# ==========================================

def coverage_weights_many(certainty, coverages):
    """
    Assign selection weights based on certainty.

    Higher certainty is preferred.

    Ties receive fractional inclusion so that
    arbitrary participant IDs do not determine
    which tied observations are retained.

    This estimates expected performance under
    random selection within tied certainty levels.
    """

    certainty = np.asarray(
        certainty,
        dtype=float
    )

    n = len(certainty)

    if (
        n == 0
        or not np.isfinite(certainty).all()
    ):
        raise RuntimeError(
            "Invalid certainty values"
        )

    coverages = np.asarray(
        coverages,
        dtype=float
    )

    if np.any(
        (coverages <= 0)
        | (coverages > 1)
    ):
        raise RuntimeError(
            "Invalid coverage values"
        )

    unique, inverse, counts = np.unique(
        certainty,
        return_inverse=True,
        return_counts=True
    )

    descending_counts = counts[::-1].astype(
        float
    )

    lower = (
        np.cumsum(descending_counts)
        - descending_counts
    )

    targets = coverages[:, None] * n

    fractions = np.clip(
        (
            targets - lower[None, :]
        )
        / descending_counts[None, :],
        0,
        1
    )

    weights = fractions[
        :,
        len(unique) - 1 - inverse
    ]

    if not np.allclose(
        weights.sum(axis=1),
        coverages * n,
        atol=1e-8,
        rtol=0
    ):
        raise RuntimeError(
            "Selected coverage does not match target"
        )

    return weights


def evaluate_coverage_batch(
    y,
    labels,
    certainty,
    coverages
):
    """
    Evaluate original first-minute predictions
    after ranking by uncertainty.

    Predictions are NOT changed.
    """

    coverages = [
        float(c)
        for c in coverages
    ]

    positive = y == 1
    negative = y == 0

    per_coverage = {
        coverage: []
        for coverage in coverages
    }

    for repeat in range(REPEATS):

        weights = coverage_weights_many(
            certainty[repeat],
            coverages
        )

        correct = (
            labels[repeat] == y
        ).astype(float)

        weighted_total = weights.sum(axis=1)

        positive_total = weights[
            :, positive
        ].sum(axis=1)

        negative_total = weights[
            :, negative
        ].sum(axis=1)

        if (
            np.any(positive_total <= 0)
            or np.any(negative_total <= 0)
        ):
            raise RuntimeError(
                "Retained subset lacks a class"
            )

        accuracy = (
            weights @ correct
        ) / weighted_total

        sensitivity = (
            weights[:, positive]
            @ correct[positive]
        ) / positive_total

        specificity = (
            weights[:, negative]
            @ correct[negative]
        ) / negative_total

        for j, coverage in enumerate(coverages):

            per_coverage[coverage].append({
                "bacc": float(
                    (
                        sensitivity[j]
                        + specificity[j]
                    ) / 2
                ),
                "accuracy": float(
                    accuracy[j]
                ),
                "risk": float(
                    1 - accuracy[j]
                ),
                "positive_coverage": float(
                    positive_total[j]
                    / positive.sum()
                ),
                "healthy_coverage": float(
                    negative_total[j]
                    / negative.sum()
                )
            })

    summary = {}

    for coverage in coverages:

        summary[coverage] = {
            key: float(
                np.mean([
                    row[key]
                    for row in per_coverage[coverage]
                ])
            )
            for key in per_coverage[coverage][0]
        }

    return summary, per_coverage


def safe_auc(errors, uncertainty, mask=None):
    """
    Error-detection AUROC.

    Higher uncertainty should identify
    incorrect original SVM predictions.

    Returns NaN if one error class is absent.
    """

    errors = np.asarray(
        errors,
        dtype=int
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float
    )

    if mask is not None:

        errors = errors[mask]
        uncertainty = uncertainty[mask]

    if (
        not np.isfinite(uncertainty).all()
        or not np.isin(errors, [0, 1]).all()
    ):
        raise RuntimeError(
            "Invalid AUROC inputs"
        )

    n_errors = int(
        errors.sum()
    )

    n_correct = (
        len(errors) - n_errors
    )

    if n_errors == 0 or n_correct == 0:
        return np.nan

    ranks = rankdata(
        uncertainty,
        method="average"
    )

    positive_rank_sum = ranks[
        errors == 1
    ].sum()

    return float(
        (
            positive_rank_sum
            - n_errors * (n_errors + 1) / 2
        )
        / (n_errors * n_correct)
    )


# ==========================================
# 3. LOCK ANALYSIS PROTOCOL
# ==========================================

protocol = {
    "version":
        "milestone_1c_selective_prediction_v1",

    "analysis_status":
        "post-hoc exploratory, no retraining",

    "diagnostic_labels":
        "original held-out first-minute "
        "RBF-SVM labels",

    "probabilities":
        "training-only cross-fitted "
        "sigmoid calibration",

    "error_target":
        "original first-minute SVM errors",

    "selection_methods":
        METHODS,

    "confidence_alignment":
        "probability supporting the original "
        "SVM label, not max probability "
        "when decisions disagree",

    "target_coverage":
        list(COVERAGES),

    "selection":
        "rank participants within each "
        "held-out CV repeat using certainty; "
        "fractional selection at ties",

    "bootstrap_resamples":
        N_BOOT,

    "uncertainty_note":
        "repeated CV correlated; bootstrap "
        "resamples participants",

    "acquisition_note":
        "five-minute methods require "
        "300 seconds of EEG",

    "deployment_note":
        "retention thresholds are cohort-relative "
        "and not training-derived clinical rules",

    "interpretation_note":
        "not independent external validation"
}

lock_path = (
    OUT / "locked_protocol.json"
)

lock_text = (
    json.dumps(
        protocol,
        sort_keys=True,
        indent=2
    )
    + "\n"
)

if lock_path.exists():

    if lock_path.read_text() != lock_text:
        raise RuntimeError(
            "Locked protocol changed. "
            "Refusing to overwrite."
        )

else:
    lock_path.write_text(
        lock_text
    )


# ==========================================
# 4. VERIFY SOURCE CHECKSUMS
# ==========================================

inputs = {
    "calibrated_predictions": PROBABILITIES,
    "milestone_1a_states": STATES
}

for path in inputs.values():

    if not path.is_file():
        raise FileNotFoundError(path)


hashes = {
    name: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for name, path in inputs.items()
}

hash_path = (
    OUT / "source_sha256.json"
)

hash_text = (
    json.dumps(
        hashes,
        sort_keys=True,
        indent=2
    )
    + "\n"
)

if hash_path.exists():

    if hash_path.read_text() != hash_text:
        raise RuntimeError(
            "Input files changed since "
            "the previous analysis."
        )

else:
    hash_path.write_text(
        hash_text
    )


# ==========================================
# 5. LOAD CALIBRATED PREDICTIONS
# ==========================================

print(
    "===== MILESTONE 1C: ERROR DETECTION "
    "AND SELECTIVE PREDICTION =====",
    flush=True
)

rows = read_csv(
    PROBABILITIES
)

state_index = make_index(
    read_csv(STATES),
    lambda row: (
        row["task"],
        row["subject"],
        int(row["repeat"])
    ),
    "Milestone 1A state"
)

probability_index = make_index(
    rows,
    lambda row: (
        row["task"],
        row["subject"],
        int(row["repeat"]),
        int(row["minute"])
    ),
    "calibrated prediction"
)

expected_total = (
    REPEATS * BLOCKS * (46 + 41)
)

if len(probability_index) != expected_total:

    raise RuntimeError(
        f"Expected {expected_total} "
        f"probabilities, found "
        f"{len(probability_index)}"
    )


rng = np.random.default_rng(SEED)

subject_rows = []
auroc_rows = []
coverage_rows = []
repeat_rows = []
contrast_rows = []
plot_data = {}


# ==========================================
# 6. PROCESS EACH DIAGNOSTIC TASK
# ==========================================

for task, meta in TASK_INFO.items():

    ids = sorted({
        sid
        for task_name, sid, rep, minute
        in probability_index
        if task_name == task
    })

    n = (
        meta["n_pos"]
        + meta["n_hc"]
    )

    if len(ids) != n:
        raise RuntimeError(
            f"{task}: participant count mismatch"
        )

    true = np.full(
        n,
        -1,
        dtype=int
    )

    groups = []

    original = np.full(
        (REPEATS, n, BLOCKS),
        -1,
        dtype=int
    )

    probability = np.full(
        (REPEATS, n, BLOCKS),
        np.nan,
        dtype=float
    )

    states = np.full(
        (REPEATS, n),
        -1,
        dtype=int
    )

    fold_map = np.full(
        (REPEATS, n),
        -1,
        dtype=int
    )

    state_names = (
        "stable_correct",
        "stable_wrong",
        "unstable"
    )


    # ======================================
    # 7. RECONCILE SUBJECT AND FOLD RECORDS
    # ======================================

    for i, sid in enumerate(ids):

        group = probability_index[
            (task, sid, 1, 1)
        ]["group"]

        groups.append(
            group
        )

        expected_label = int(
            group == meta["positive"]
        )

        if group not in (
            meta["positive"],
            "C"
        ):
            raise RuntimeError(
                "Unexpected group: " + group
            )

        true[i] = expected_label

        for rep in range(
            1,
            REPEATS + 1
        ):

            state_record = state_index[
                (task, sid, rep)
            ]

            if state_record["group"] != group:

                raise RuntimeError(
                    "State-group mismatch"
                )

            for minute in range(
                1,
                BLOCKS + 1
            ):

                record = probability_index[
                    (task, sid, rep, minute)
                ]

                label = int(
                    record["original_svm_label"]
                )

                p = float(
                    record[
                        "calibrated_positive_probability"
                    ]
                )

                fold = int(
                    record["fold"]
                )

                if (
                    record["group"] != group
                    or int(record["true_label"])
                       != expected_label
                    or label not in (0, 1)
                    or not np.isfinite(p)
                    or not (0 < p < 1)
                    or not (1 <= fold <= 5)
                ):
                    raise RuntimeError(
                        f"{task}: invalid entry: {sid}"
                    )

                expected_confidence = max(
                    p,
                    1 - p
                )

                saved_confidence = float(
                    record["calibrated_confidence"]
                )

                if not np.isclose(
                    expected_confidence,
                    saved_confidence,
                    atol=1e-10
                ):
                    raise RuntimeError(
                        "Calibrated confidence mismatch"
                    )

                previous_fold = fold_map[
                    rep - 1,
                    i
                ]

                if previous_fold not in (
                    -1,
                    fold
                ):
                    raise RuntimeError(
                        "Inconsistent outer fold"
                    )

                fold_map[
                    rep - 1,
                    i
                ] = fold

                original[
                    rep - 1,
                    i,
                    minute - 1
                ] = label

                probability[
                    rep - 1,
                    i,
                    minute - 1
                ] = p

            decisions = original[
                rep - 1,
                i
            ]

            unanimous = bool(
                np.all(
                    decisions == decisions[0]
                )
            )

            if not unanimous:

                inferred_state = "unstable"

            elif decisions[0] == expected_label:

                inferred_state = "stable_correct"

            else:

                inferred_state = "stable_wrong"

            if (
                inferred_state
                != state_record["state"]
            ):
                raise RuntimeError(
                    f"{task}: Milestone 1A "
                    f"state mismatch for {sid}"
                )

            states[
                rep - 1,
                i
            ] = state_names.index(
                inferred_state
            )

    expected_groups = {
        meta["positive"]: meta["n_pos"],
        "C": meta["n_hc"]
    }

    if Counter(groups) != expected_groups:

        raise RuntimeError(
            f"{task}: group counts changed"
        )

    if (
        not np.isfinite(probability).all()
        or np.any(original < 0)
        or np.any(states < 0)
    ):
        raise RuntimeError(
            "Incomplete prediction arrays"
        )

    print(
        f"PASS: {task}, {n} participants, "
        "all five repeats and all five "
        "blocks reconciled.",
        flush=True
    )


    # ======================================
    # 8. DEFINE UNCERTAINTY MEASURES
    # ======================================

    # Probability assigned to the original
    # first-minute SVM decision.
    #
    # This is deliberately not always
    # max(p, 1-p), because calibration can
    # change the label at threshold 0.5.

    first_support = np.where(
        original[:, :, 0] == 1,
        probability[:, :, 0],
        1.0 - probability[:, :, 0]
    )

    first_threshold_confidence = np.maximum(
        probability[:, :, 0],
        1.0 - probability[:, :, 0]
    )

    five_probability_sd = probability.std(
        axis=2,
        ddof=1
    )

    positive_votes = original.sum(
        axis=2
    )

    five_consensus = (
        np.maximum(
            positive_votes,
            BLOCKS - positive_votes
        )
        / BLOCKS
    )

    errors = (
        original[:, :, 0]
        != true[None, :]
    )

    certainty = {
        "first_minute_original_label_support":
            first_support,

        "five_minute_probability_consistency":
            -five_probability_sd,

        "five_minute_label_consensus":
            five_consensus
    }


    # ======================================
    # 9. SAVE SUBJECT-REPEAT UNCERTAINTY
    # ======================================

    for rep in range(REPEATS):

        for i, sid in enumerate(ids):

            p = float(
                probability[rep, i, 0]
            )

            subject_rows.append({
                "task": task,
                "subject": sid,
                "group": groups[i],
                "repeat": rep + 1,
                "fold": int(
                    fold_map[rep, i]
                ),
                "true_label": int(
                    true[i]
                ),
                "original_first_minute_label": int(
                    original[rep, i, 0]
                ),
                "first_minute_correct": int(
                    not errors[rep, i]
                ),
                "milestone_1a_state":
                    state_names[
                        states[rep, i]
                    ],
                "first_minute_probability": p,
                "first_minute_original_label_support":
                    float(
                        first_support[rep, i]
                    ),
                "first_minute_calibrated_threshold_confidence":
                    float(
                        first_threshold_confidence[rep, i]
                    ),
                "calibrated_threshold_changes_original_label":
                    int(
                        (p >= 0.5)
                        != (
                            original[rep, i, 0]
                            == 1
                        )
                    ),
                "five_minute_probability_sd":
                    float(
                        five_probability_sd[rep, i]
                    ),
                "five_minute_label_consensus":
                    float(
                        five_consensus[rep, i]
                    ),
                "five_minute_original_class_reversals":
                    int(
                        states[rep, i] == 2
                    )
            })


    # ======================================
    # 10. ERROR-DETECTION AUROC
    # ======================================

    positive_idx = np.flatnonzero(
        true == 1
    )

    healthy_idx = np.flatnonzero(
        true == 0
    )

    bootstrap_auroc = {
        key: []
        for key in certainty
    }

    bootstrap_stable_auroc = {
        key: []
        for key in certainty
    }

    boot_bacc = {
        (method, coverage):
            np.empty(N_BOOT)
        for method in certainty
        for coverage in COVERAGES
    }

    boot_gain = {
        (method, coverage):
            np.empty(N_BOOT)
        for method in certainty
        for coverage in COVERAGES
    }

    observed_auc = {}
    observed_stable_auc = {}

    full_result, _ = evaluate_coverage_batch(
        true,
        original[:, :, 0],
        first_support,
        (1.0,)
    )

    full_bacc = full_result[
        1.0
    ]["bacc"]

    changed_count = int(
        np.sum(
            (
                probability[:, :, 0] >= 0.5
            )
            != (
                original[:, :, 0] == 1
            )
        )
    )

    print(
        f"\n{task}: calibrated 0.5 threshold "
        f"changes {changed_count}/{REPEATS*n} "
        "original first-minute decisions; "
        "original labels remain unchanged."
    )

    print(
        f"\n===== {task}: ERROR DETECTION AUROC ====="
    )

    print(
        f"Participants={n}; "
        f"first-minute errors="
        f"{int(errors.sum())}/{REPEATS*n} "
        "correlated subject-repeats"
    )

    for method, values in certainty.items():

        uncertainty = -values

        auc = safe_auc(
            errors.ravel(),
            uncertainty.ravel()
        )

        stable_mask = (
            states != 2
        )

        stable_auc = safe_auc(
            errors.ravel(),
            uncertainty.ravel(),
            stable_mask.ravel()
        )

        observed_auc[method] = auc

        observed_stable_auc[method] = stable_auc

        print(
            f"{method}: "
            f"all-errors AUROC={auc:.4f}; "
            f"stable-only error AUROC="
            f"{stable_auc:.4f}"
        )


    # ======================================
    # 11. RISK-COVERAGE POINT ESTIMATES
    # ======================================

    point_results = {}

    for method, values in certainty.items():

        summaries, per_coverage = (
            evaluate_coverage_batch(
                true,
                original[:, :, 0],
                values,
                PLOT_COVERAGES
            )
        )

        for coverage in PLOT_COVERAGES:

            point_results[
                (method, coverage)
            ] = summaries[coverage]

            for rep, item in enumerate(
                per_coverage[coverage]
            ):

                repeat_rows.append({
                    "task": task,
                    "method": method,
                    "seconds_required":
                        METHODS[method][
                            "seconds_required"
                        ],
                    "repeat": rep + 1,
                    "target_coverage": coverage,
                    **item
                })


    # ======================================
    # 12. PARTICIPANT-CLUSTER BOOTSTRAP
    # ======================================

    for boot in range(N_BOOT):

        sampled = np.r_[
            rng.choice(
                positive_idx,
                len(positive_idx),
                replace=True
            ),
            rng.choice(
                healthy_idx,
                len(healthy_idx),
                replace=True
            )
        ]

        boot_y = true[sampled]

        boot_pred = original[
            :, sampled, 0
        ]

        boot_error = errors[
            :, sampled
        ]

        boot_state = states[
            :, sampled
        ]

        full_boot, _ = evaluate_coverage_batch(
            boot_y,
            boot_pred,
            first_support[:, sampled],
            (1.0,)
        )

        full_boot_bacc = full_boot[
            1.0
        ]["bacc"]

        for method, values in certainty.items():

            sampled_certainty = values[
                :, sampled
            ]

            auc = safe_auc(
                boot_error.ravel(),
                (
                    -sampled_certainty
                ).ravel()
            )

            bootstrap_auroc[
                method
            ].append(auc)

            stable_auc = safe_auc(
                boot_error.ravel(),
                (
                    -sampled_certainty
                ).ravel(),
                (
                    boot_state != 2
                ).ravel()
            )

            bootstrap_stable_auroc[
                method
            ].append(stable_auc)

            selected, _ = evaluate_coverage_batch(
                boot_y,
                boot_pred,
                sampled_certainty,
                COVERAGES
            )

            for coverage in COVERAGES:

                selected_bacc = selected[
                    coverage
                ]["bacc"]

                boot_bacc[
                    (method, coverage)
                ][boot] = selected_bacc

                boot_gain[
                    (method, coverage)
                ][boot] = (
                    selected_bacc
                    - full_boot_bacc
                )

        if (boot + 1) % 500 == 0:

            print(
                f"{task}: participant bootstrap "
                f"{boot+1}/{N_BOOT}",
                flush=True
            )


    # ======================================
    # 13. SAVE ERROR-DETECTION INTERVALS
    # ======================================

    for method in certainty:

        comparisons = (
            (
                "all_first_minute_errors",
                observed_auc[method],
                bootstrap_auroc[method]
            ),
            (
                "stable_wrong_among_unanimous",
                observed_stable_auc[method],
                bootstrap_stable_auroc[method]
            )
        )

        for target, observed, samples in comparisons:

            valid = np.asarray([
                x
                for x in samples
                if np.isfinite(x)
            ])

            if len(valid) >= 1000:

                lo, hi = np.percentile(
                    valid,
                    [2.5, 97.5]
                )

            else:

                lo, hi = np.nan, np.nan

            auroc_rows.append({
                "task": task,
                "method": method,
                "uncertainty_target": target,
                "auroc_higher_is_better":
                    float(observed),
                "conditional_ci_low": float(lo),
                "conditional_ci_high": float(hi),
                "valid_bootstrap_resamples":
                    len(valid),
                "bootstrap_resamples_requested":
                    N_BOOT
            })


    # ======================================
    # 14. SAVE RISK-COVERAGE INTERVALS
    # ======================================

    for method in certainty:

        for coverage in COVERAGES:

            result = point_results[
                (method, coverage)
            ]

            bacc_lo, bacc_hi = np.percentile(
                boot_bacc[
                    (method, coverage)
                ],
                [2.5, 97.5]
            )

            gain_lo, gain_hi = np.percentile(
                boot_gain[
                    (method, coverage)
                ],
                [2.5, 97.5]
            )

            coverage_rows.append({
                "task": task,
                "method": method,
                "seconds_required":
                    METHODS[method][
                        "seconds_required"
                    ],
                "target_coverage": coverage,
                "mean_selected_BACC":
                    result["bacc"],
                "conditional_BACC_ci_low":
                    float(bacc_lo),
                "conditional_BACC_ci_high":
                    float(bacc_hi),
                "mean_selected_accuracy":
                    result["accuracy"],
                "mean_selected_error_risk":
                    result["risk"],
                "positive_class_coverage":
                    result["positive_coverage"],
                "healthy_class_coverage":
                    result["healthy_coverage"],
                "BACC_gain_over_full_coverage":
                    result["bacc"] - full_bacc,
                "conditional_gain_ci_low":
                    float(gain_lo),
                "conditional_gain_ci_high":
                    float(gain_hi)
            })


    # ======================================
    # 15. PRINT SELECTIVE PERFORMANCE
    # ======================================

    print(
        f"\n===== {task}: "
        "SELECTIVE FIRST-MINUTE PREDICTIONS ====="
    )

    print(
        "All methods retain the SAME original "
        "first-minute SVM label."
    )

    for method in certainty:

        print(
            method
            + " ("
            + str(
                METHODS[method][
                    "seconds_required"
                ]
            )
            + "s of EEG required):"
        )

        for coverage in COVERAGES:

            row = next(
                r
                for r in coverage_rows
                if (
                    r["task"] == task
                    and r["method"] == method
                    and r["target_coverage"] == coverage
                )
            )

            print(
                f"  coverage={coverage:.0%}: "
                f"BACC="
                f"{row['mean_selected_BACC']:.4f}, "
                f"CI ["
                f"{row['conditional_BACC_ci_low']:.4f}, "
                f"{row['conditional_BACC_ci_high']:.4f}], "
                f"accuracy="
                f"{row['mean_selected_accuracy']:.4f}, "
                f"gain over full coverage="
                f"{row['BACC_gain_over_full_coverage']:+.4f}"
            )


    # ======================================
    # 16. COMPARE 60s AND 300s SELECTION
    # ======================================

    for method in (
        "five_minute_probability_consistency",
        "five_minute_label_consensus"
    ):

        for coverage in (
            0.90,
            0.80,
            0.70
        ):

            observed_delta = (
                point_results[
                    (method, coverage)
                ]["bacc"]
                -
                point_results[
                    (
                        "first_minute_original_label_support",
                        coverage
                    )
                ]["bacc"]
            )

            bootstrap_delta = (
                boot_bacc[
                    (method, coverage)
                ]
                -
                boot_bacc[
                    (
                        "first_minute_original_label_support",
                        coverage
                    )
                ]
            )

            lo, hi = np.percentile(
                bootstrap_delta,
                [2.5, 97.5]
            )

            contrast_rows.append({
                "task": task,
                "five_minute_method": method,
                "comparator":
                    "first_minute_original_label_support",
                "target_coverage": coverage,
                "BACC_difference":
                    float(observed_delta),
                "conditional_ci_low": float(lo),
                "conditional_ci_high": float(hi),
                "note":
                    "Comparison uses unequal "
                    "EEG acquisition durations"
            })


    # ======================================
    # 17. PREPARE RISK-COVERAGE PLOTS
    # ======================================

    plot_data[task] = {
        "coverages": [
            float(x)
            for x in reversed(
                PLOT_COVERAGES
            )
        ],
        "series": {
            method: [
                point_results[
                    (method, coverage)
                ]["risk"]
                for coverage in reversed(
                    PLOT_COVERAGES
                )
            ]
            for method in certainty
        }
    }

    print(
        f"\n===== {task}: "
        "UNCERTAINTY AUDIT COMPLETE =====",
        flush=True
    )


# ==========================================
# 18. SAVE ALL RESULTS
# ==========================================

save_csv(
    OUT / "subject_repeat_uncertainty.csv",
    subject_rows
)

save_csv(
    OUT / "error_detection_auroc.csv",
    auroc_rows
)

save_csv(
    OUT / "risk_coverage_summary.csv",
    coverage_rows
)

save_csv(
    OUT / "risk_coverage_repeat_level.csv",
    repeat_rows
)

save_csv(
    OUT / "five_vs_one_minute_paired_contrasts.csv",
    contrast_rows
)


# ==========================================
# 19. GENERATE RISK-COVERAGE FIGURES
# ==========================================

for task, data in plot_data.items():

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    for method, values in data["series"].items():

        ax.plot(
            data["coverages"],
            values,
            marker="o",
            label=method.replace(
                "_",
                " "
            )
        )

    ax.set(
        xlabel="Fraction of participants retained",
        ylabel="Error rate among retained participants",
        title=(
            task
            + ": retrospective risk-coverage"
        ),
        xlim=(0.69, 1.01),
        ylim=(0, 0.55)
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUT / (
            task + "_risk_coverage.png"
        ),
        dpi=190
    )

    plt.close(fig)


# ==========================================
# 20. FINAL VALIDATION
# ==========================================

print(
    "\n===== MILESTONE 1C COMPLETE ====="
)

print(
    "Saved results:",
    OUT
)

print(
    "No classifier retrained and no "
    "historical predictions modified."
)

print(
    "Subject-bootstrap intervals are conditional; "
    "five CV repetitions are correlated."
)

print(
    "Selection rankings are retrospective and "
    "use unlabeled held-out certainty scores."
)

print(
    "Five-minute methods require 300 seconds "
    "of EEG and cannot make 60-second decisions."
)

print(
    "Fractional tie weights avoid arbitrary "
    "selection among identical consensus scores."
)

