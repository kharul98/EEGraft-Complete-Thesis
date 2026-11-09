from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os

import numpy as np
from scipy.stats import wilcoxon, binomtest, spearmanr

HOME = Path.home()
WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

PRIVATE = DATA / "derived/caueeg_transfer_v1/analysis"

AUDIT_FILE = WORK / "results/21_event_alignment_summary.json"
INFERENCE_FILE = WORK / "results/11_train_val_inference_summary.json"
PLAN_FILE = WORK / "protocol/caueeg_final_test_plan_v1.json"

RECORDS_FILE = PRIVATE / "21_event_alignment_records.csv"
WINDOWS_FILE = PRIVATE / "21_event_aligned_windows.csv"

DETAIL_FILE = PRIVATE / "22_condition_paired_recordings.csv"
REPORT_FILE = WORK / "results/22_condition_analysis.json"

MODELS = {
    "DSRF": "dsrf_source_ad_score",
    "XGBoost": "xgb_source_ad_score"
}

EXPECTED = {
    "train": 404,
    "validation": 42
}

SEED = 20261109
BOOTSTRAPS = 5000
THRESHOLD = 0.5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4194304), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing result matches:", path.name)
            return True

        print("[PRESERVED] Different existing result:", path)
        return False

    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)

    print("[SAVED]", path)
    return True


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify():
    print("\n1. VERIFYING EVENT AUDIT AND MODEL PROVENANCE")

    required = [
        AUDIT_FILE, INFERENCE_FILE, PLAN_FILE,
        RECORDS_FILE, WINDOWS_FILE
    ]

    for path in required:
        require(path.is_file(), "Missing: " + str(path))

    audit = read_json(AUDIT_FILE)
    inference = read_json(INFERENCE_FILE)
    plan = read_json(PLAN_FILE)

    require(
        audit["status"] == "EVENT_ALIGNMENT_AUDIT_COMPLETED",
        "Code 21 did not finish"
    )

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Original development inference incomplete"
    )

    require(
        audit["development_recordings"] == 1032
        and audit["development_windows"] == 24768,
        "Unexpected audit population"
    )

    require(
        audit["test_waveforms_opened"] == 0
        and audit["test_event_files_opened"] == 0
        and audit["test_prediction_files_opened"] == 0,
        "Code 21 test-access status invalid"
    )

    require(
        audit["original_model_hashes"]
        == inference["model_hashes"]
        == plan["model_hashes"],
        "Original model hashes inconsistent"
    )

    require(
        audit["final_test_plan_sha256"] == sha256(PLAN_FILE),
        "Final test plan changed"
    )

    require(
        audit["timing"]["sampling_hz"] == 200
        and audit["timing"]["window_seconds"] == 4
        and audit["timing"]["event_boundary_margin_seconds"] == 2,
        "Unexpected event-alignment rules"
    )

    for split, count in EXPECTED.items():
        require(
            audit["paired_state_coverage"][split][
                "at_least_two_windows_each"
            ] == count,
            "Paired condition cohort count changed"
        )

    print("[OK] Code 21 audit verified")
    print("[OK] Original model identities verified")
    print("[OK] Frozen external evaluation plan unchanged")
    print("[OK] No test data required")

    return audit


def load_paired_cases(audit):
    print("\n2. VERIFYING EVENT-LABELED DEVELOPMENT WINDOWS")

    records = read_csv(RECORDS_FILE)
    windows = read_csv(WINDOWS_FILE)

    require(
        len(records) == 1032,
        "Expected 1032 recording audit rows"
    )

    require(
        len(windows) == 24768,
        "Expected 24768 window audit rows"
    )

    record_lookup = {}
    grouped = defaultdict(list)

    status_counts = Counter()
    condition_counts = Counter()
    class_condition_counts = Counter()

    for row in records:
        key = (row["split"], row["serial"])

        require(
            row["split"] in EXPECTED
            and row["class_name"] in (
                "normal", "mci", "dementia"
            ),
            "Unexpected development recording"
        )

        require(
            key not in record_lookup,
            "Duplicate recording audit row"
        )

        record_lookup[key] = row
        status_counts[
            (row["split"], row["event_status"])
        ] += 1

    for row in windows:
        split = row["split"]
        serial = row["serial"]
        key = (split, serial)

        require(
            key in record_lookup,
            "Window refers to unknown recording"
        )

        require(
            row["class_name"]
            == record_lookup[key]["class_name"],
            "Window diagnostic label mismatch"
        )

        condition = row["condition"]

        require(
            condition in (
                "eyes_open", "eyes_closed", "excluded"
            ),
            "Unexpected condition category"
        )

        if condition in ("eyes_open", "eyes_closed"):
            require(
                row["reason"] == "eligible"
                and record_lookup[key]["event_status"]
                == "aligned",
                "Unverified event-labeled window"
            )

        else:
            require(
                row["reason"] != "eligible",
                "Excluded window has eligible reason"
            )

        start = float(row["window_start_seconds"])
        end = float(row["window_end_seconds"])

        require(
            np.isfinite(start)
            and np.isfinite(end)
            and start >= 0
            and abs(end - start - 4) < 1e-6,
            "Invalid time coordinates"
        )

        item = {
            "index": int(row["window_index"]),
            "start": start,
            "condition": condition
        }

        require(
            0 <= item["index"] < 24,
            "Invalid window index"
        )

        for model, column in MODELS.items():
            value = float(row[column])

            require(
                np.isfinite(value)
                and 0 <= value <= 1,
                "Invalid model score"
            )

            item[model] = value

        grouped[key].append(item)

        condition_counts[(split, condition)] += 1
        class_condition_counts[
            (split, row["class_name"], condition)
        ] += 1

    require(
        len(grouped) == 1032,
        "Incomplete recording window groups"
    )

    for split in EXPECTED:
        for status, count in audit["event_quality"][split].items():
            require(
                status_counts[(split, status)] == count,
                "Event-quality summary inconsistency"
            )

        for condition, count in audit[
            "eligible_window_counts"
        ][split].items():
            require(
                condition_counts[(split, condition)] == count,
                "Condition-window count inconsistency"
            )

        for label in ("normal", "mci", "dementia"):
            for condition in ("eyes_open", "eyes_closed"):
                require(
                    class_condition_counts[
                        (split, label, condition)
                    ]
                    == audit["by_class_eligible_windows"][
                        split
                    ][label][condition],
                    "Class-condition summary inconsistency"
                )

    paired = {
        "train": [],
        "validation": []
    }

    for key, windows24 in grouped.items():
        split, serial = key
        record = record_lookup[key]

        require(
            len(windows24) == 24
            and sorted(w["index"] for w in windows24)
            == list(range(24)),
            "Incomplete or duplicate EEG windows"
        )

        opened = [
            w for w in windows24
            if w["condition"] == "eyes_open"
        ]

        closed = [
            w for w in windows24
            if w["condition"] == "eyes_closed"
        ]

        require(
            len(opened) == int(record["eyes_open_windows"])
            and len(closed)
            == int(record["eyes_closed_windows"]),
            "Per-recording condition coverage mismatch"
        )

        has_two = len(opened) >= 2 and len(closed) >= 2

        require(
            int(has_two)
            == int(record["has_two_windows_per_state"]),
            "Paired-state eligibility mismatch"
        )

        if not has_two:
            continue

        require(
            record["event_status"] == "aligned",
            "Unaligned recording entered paired analysis"
        )

        case = {
            "serial": serial,
            "split": split,
            "class_name": record["class_name"],
            "target": (
                None if record["class_name"] == "mci"
                else int(record["class_name"] == "dementia")
            ),
            "n_open": len(opened),
            "n_closed": len(closed),
            "mean_open_time": float(
                np.mean([w["start"] for w in opened])
            ),
            "mean_closed_time": float(
                np.mean([w["start"] for w in closed])
            )
        }

        case["time_offset"] = (
            case["mean_open_time"] - case["mean_closed_time"]
        )

        for model in MODELS:
            p_open = float(
                np.mean([w[model] for w in opened])
            )

            p_closed = float(
                np.mean([w[model] for w in closed])
            )

            pred_open = int(p_open >= THRESHOLD)
            pred_closed = int(p_closed >= THRESHOLD)

            case[model] = {
                "open_score": p_open,
                "closed_score": p_closed,
                "signed_difference": p_open - p_closed,
                "absolute_difference": abs(p_open - p_closed),
                "open_prediction": pred_open,
                "closed_prediction": pred_closed,
                "condition_flip": int(
                    pred_open != pred_closed
                ),
                "open_correct": (
                    None if case["target"] is None
                    else int(pred_open == case["target"])
                ),
                "closed_correct": (
                    None if case["target"] is None
                    else int(pred_closed == case["target"])
                )
            }

        paired[split].append(case)

    for split, expected in EXPECTED.items():
        require(
            len(paired[split]) == expected,
            "Expected paired-recording count mismatch"
        )

    print("[OK] All 24768 saved windows verified")
    print("[OK] Train paired recordings:", len(paired["train"]))
    print("[OK] Validation paired recordings:",
          len(paired["validation"]))
    print("Test recordings loaded: 0")

    return paired


def bootstrap_mean_ci(values, indices):
    values = np.asarray(values, dtype=float)

    replicated = np.mean(
        values[indices],
        axis=1
    )

    return [
        float(np.quantile(replicated, 0.025)),
        float(np.quantile(replicated, 0.975))
    ]


def robust_wilcoxon(values):
    values = np.asarray(values, dtype=float)

    if np.allclose(values, 0, atol=1e-12):
        return 1.0

    return float(
        wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        ).pvalue
    )


def holm_adjust(pvalues):
    ordered = sorted(
        pvalues,
        key=lambda name: pvalues[name]
    )

    adjusted = {}
    largest = 0.0
    n = len(ordered)

    for index, name in enumerate(ordered):
        value = min(
            1.0,
            (n - index) * pvalues[name]
        )

        largest = max(largest, value)
        adjusted[name] = float(largest)

    return adjusted


def binary_bacc(y, predicted):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    require(
        np.any(y == 0) and np.any(y == 1),
        "Both classes needed for balanced accuracy"
    )

    sensitivity = float(
        np.mean(predicted[y == 1] == 1)
    )

    specificity = float(
        np.mean(predicted[y == 0] == 0)
    )

    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": float(np.mean(predicted == y))
    }


def binary_condition_comparison(cases, model, seed):
    binary = [
        c for c in cases
        if c["target"] is not None
    ]

    labels = Counter(
        c["class_name"] for c in binary
    )

    if (
        labels["normal"] < 2
        or labels["dementia"] < 2
    ):
        return {
            "status": "insufficient_binary_class_coverage",
            "class_counts": dict(labels)
        }

    y = np.asarray(
        [c["target"] for c in binary],
        dtype=int
    )

    open_predictions = np.asarray(
        [c[model]["open_prediction"] for c in binary],
        dtype=int
    )

    closed_predictions = np.asarray(
        [c[model]["closed_prediction"] for c in binary],
        dtype=int
    )

    open_metrics = binary_bacc(
        y, open_predictions
    )

    closed_metrics = binary_bacc(
        y, closed_predictions
    )

    normal = np.flatnonzero(y == 0)
    dementia = np.flatnonzero(y == 1)

    rng = np.random.default_rng(seed)

    normal_draws = rng.choice(
        normal,
        size=(BOOTSTRAPS, len(normal)),
        replace=True
    )

    dementia_draws = rng.choice(
        dementia,
        size=(BOOTSTRAPS, len(dementia)),
        replace=True
    )

    open_bacc = (
        np.mean(
            open_predictions[normal_draws] == 0,
            axis=1
        )
        + np.mean(
            open_predictions[dementia_draws] == 1,
            axis=1
        )
    ) / 2

    closed_bacc = (
        np.mean(
            closed_predictions[normal_draws] == 0,
            axis=1
        )
        + np.mean(
            closed_predictions[dementia_draws] == 1,
            axis=1
        )
    ) / 2

    states = Counter()

    for case in binary:
        open_correct = case[model]["open_correct"]
        closed_correct = case[model]["closed_correct"]

        if open_correct and closed_correct:
            states["both_correct"] += 1
        elif open_correct:
            states["open_only_correct"] += 1
        elif closed_correct:
            states["closed_only_correct"] += 1
        else:
            states["both_wrong"] += 1

    return {
        "status": "computed",
        "recordings": len(binary),
        "class_counts": dict(labels),
        "open_metrics": open_metrics,
        "closed_metrics": closed_metrics,
        "open_minus_closed_bacc": float(
            open_metrics["balanced_accuracy"]
            - closed_metrics["balanced_accuracy"]
        ),
        "paired_stratified_recording_bootstrap_95ci": [
            float(np.quantile(
                open_bacc - closed_bacc, 0.025
            )),
            float(np.quantile(
                open_bacc - closed_bacc, 0.975
            ))
        ],
        "correctness_states": dict(states),
        "interpretation": (
            "Each recording contributes one mean score per "
            "condition. This is an observational subset, "
            "not an independent external test."
        )
    }


def interaction_summary(cases, model, seed):
    normal = np.asarray([
        c[model]["signed_difference"]
        for c in cases
        if c["class_name"] == "normal"
    ])

    dementia = np.asarray([
        c[model]["signed_difference"]
        for c in cases
        if c["class_name"] == "dementia"
    ])

    if len(normal) < 2 or len(dementia) < 2:
        return None

    estimate = float(
        np.mean(dementia) - np.mean(normal)
    )

    rng = np.random.default_rng(seed)

    sampled_normal = rng.choice(
        normal,
        size=(BOOTSTRAPS, len(normal)),
        replace=True
    )

    sampled_dementia = rng.choice(
        dementia,
        size=(BOOTSTRAPS, len(dementia)),
        replace=True
    )

    differences = (
        np.mean(sampled_dementia, axis=1)
        - np.mean(sampled_normal, axis=1)
    )

    return {
        "normal_recordings": len(normal),
        "dementia_recordings": len(dementia),
        "dementia_minus_normal_state_shift": estimate,
        "recording_bootstrap_95ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975))
        ],
        "caution": (
            "An observational group interaction. "
            "Age, time position, patient identity and "
            "acquisition differences may confound it."
        )
    }


def analyze_split(cases, split):
    print("\n3. ANALYSIS:", split.upper())

    n = len(cases)
    seed = SEED if split == "train" else SEED + 1

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, n, size=(BOOTSTRAPS, n)
    )

    class_counts = Counter(
        c["class_name"] for c in cases
    )

    offsets = np.asarray([
        c["time_offset"] for c in cases
    ])

    report = {
        "n_paired_recordings": n,
        "class_counts": dict(class_counts),
        "median_open_minus_closed_time_seconds": float(
            np.median(offsets)
        ),
        "fraction_open_sampled_later_than_closed": float(
            np.mean(offsets > 0)
        ),
        "models": {}
    }

    print("Paired recordings:", n)
    print("Class counts:", dict(class_counts))
    print(
        "Median eyes-open minus eyes-closed time offset:",
        f"{np.median(offsets):.1f} seconds"
    )

    for model in MODELS:
        signed = np.asarray([
            c[model]["signed_difference"]
            for c in cases
        ])

        absolute = np.abs(signed)

        flips = np.asarray([
            c[model]["condition_flip"]
            for c in cases
        ])

        by_class = {}

        for label in (
            "normal", "mci", "dementia"
        ):
            group = [
                c for c in cases
                if c["class_name"] == label
            ]

            if not group:
                continue

            deltas = [
                c[model]["signed_difference"]
                for c in group
            ]

            by_class[label] = {
                "recordings": len(group),
                "mean_open_minus_closed_score": float(
                    np.mean(deltas)
                ),
                "median_absolute_state_shift": float(
                    np.median(np.abs(deltas))
                ),
                "condition_label_flip_fraction": float(
                    np.mean([
                        c[model]["condition_flip"]
                        for c in group
                    ])
                )
            }

        time_correlation = spearmanr(
            offsets, signed
        )

        rho = float(time_correlation.statistic)

        if not np.isfinite(rho):
            rho = None

        report["models"][model] = {
            "mean_open_minus_closed_score": float(
                np.mean(signed)
            ),
            "mean_score_difference_bootstrap_95ci":
                bootstrap_mean_ci(signed, indices),
            "median_absolute_state_score_change": float(
                np.median(absolute)
            ),
            "mean_absolute_state_score_change": float(
                np.mean(absolute)
            ),
            "condition_label_flip_count": int(
                flips.sum()
            ),
            "condition_label_flip_fraction": float(
                np.mean(flips)
            ),
            "spearman_score_shift_vs_time_offset": rho,
            "by_class": by_class,
            "binary_correctness": binary_condition_comparison(
                cases, model, seed + 17
            ),
            "diagnosis_by_condition_interaction":
                interaction_summary(cases, model, seed + 29)
        }

        result = report["models"][model]

        print("\nModel:", model)
        print(
            "  Mean score shift (open - closed):",
            f"{result['mean_open_minus_closed_score']:+.4f}"
        )
        print(
            "  Median absolute score shift:",
            f"{result['median_absolute_state_score_change']:.4f}"
        )
        print(
            "  Recordings changing predicted class:",
            f"{result['condition_label_flip_count']}/{n}",
            f"({result['condition_label_flip_fraction']:.2%})"
        )

        binary = result["binary_correctness"]

        if binary["status"] == "computed":
            print(
                "  Eyes-open BAcc:",
                f"{binary['open_metrics']['balanced_accuracy']:.2%}"
            )
            print(
                "  Eyes-closed BAcc:",
                f"{binary['closed_metrics']['balanced_accuracy']:.2%}"
            )
            print(
                "  Open - closed BAcc:",
                f"{binary['open_minus_closed_bacc']:+.2%}"
            )

    return report


def exploratory_tests(cases):
    print("\n4. EXPLORATORY TRAINING-SET ASSOCIATION TESTS")

    d = np.asarray([
        c["DSRF"]["signed_difference"]
        for c in cases
    ])

    x = np.asarray([
        c["XGBoost"]["signed_difference"]
        for c in cases
    ])

    d_flip = np.asarray([
        c["DSRF"]["condition_flip"]
        for c in cases
    ])

    x_flip = np.asarray([
        c["XGBoost"]["condition_flip"]
        for c in cases
    ])

    dsrf_only = int(
        np.sum((d_flip == 1) & (x_flip == 0))
    )

    xgb_only = int(
        np.sum((d_flip == 0) & (x_flip == 1))
    )

    discordant = dsrf_only + xgb_only

    mcnemar_p = (
        float(
            binomtest(
                dsrf_only,
                discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue
        )
        if discordant > 0
        else 1.0
    )

    pvalues = {
        "DSRF_signed_state_score_shift":
            robust_wilcoxon(d),

        "XGBoost_signed_state_score_shift":
            robust_wilcoxon(x),

        "DSRF_minus_XGBoost_absolute_shift":
            robust_wilcoxon(
                np.abs(d) - np.abs(x)
            ),

        "model_difference_in_condition_flip":
            mcnemar_p
    }

    adjusted = holm_adjust(pvalues)

    results = {}

    for name in pvalues:
        results[name] = {
            "raw_exploratory_p": pvalues[name],
            "holm_adjusted_exploratory_p": adjusted[name]
        }

        print(
            name,
            "raw p =", f"{pvalues[name]:.6g}",
            "Holm p =", f"{adjusted[name]:.6g}"
        )

    print(
        "\nDSRF-only condition flips:", dsrf_only
    )
    print(
        "XGBoost-only condition flips:", xgb_only
    )

    return {
        "tests": results,
        "mcnemar_discordant_counts": {
            "DSRF_only": dsrf_only,
            "XGBoost_only": xgb_only
        },
        "multiplicity_method": "Holm",
        "test_family_size": 4,
        "test_status": "POST_HOC_EXPLORATORY",
        "caution": (
            "These tests were designed after inspection "
            "of the final external test results. "
            "Recording independence is not verified, "
            "and recording condition was not randomized. "
            "Adjusted p-values do not establish causality "
            "or prospective statistical confirmation."
        )
    }


def detail_csv(paired):
    fields = [
        "serial",
        "split",
        "class_name",
        "n_open",
        "n_closed",
        "mean_open_time",
        "mean_closed_time",
        "time_offset"
    ]

    for model in MODELS:
        fields += [
            model + "_open_score",
            model + "_closed_score",
            model + "_score_difference",
            model + "_absolute_difference",
            model + "_open_prediction",
            model + "_closed_prediction",
            model + "_condition_flip",
            model + "_open_correct",
            model + "_closed_correct"
        ]

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=fields
    )
    writer.writeheader()

    for split in ("train", "validation"):
        for case in paired[split]:
            row = {
                key: case[key]
                for key in (
                    "serial", "split", "class_name",
                    "n_open", "n_closed",
                    "mean_open_time",
                    "mean_closed_time", "time_offset"
                )
            }

            for model in MODELS:
                item = case[model]

                for key in (
                    "open_score",
                    "closed_score",
                    "signed_difference",
                    "absolute_difference",
                    "open_prediction",
                    "closed_prediction",
                    "condition_flip",
                    "open_correct",
                    "closed_correct"
                ):
                    field = (
                        "score_difference"
                        if key == "signed_difference"
                        else key
                    )

                    value = item[key]

                    row[model + "_" + field] = (
                        "" if value is None else value
                    )

            writer.writerow(row)

    return buffer.getvalue()


def main():
    audit = verify()

    paired = load_paired_cases(audit)

    training = analyze_split(
        paired["train"], "train"
    )

    validation = analyze_split(
        paired["validation"], "validation"
    )

    tests = exploratory_tests(
        paired["train"]
    )

    print("\n5. SAVING SCIENTIFIC EVIDENCE")

    report = {
        "status": "CONDITION_ANALYSIS_COMPLETED",
        "version": "caueeg_condition_analysis_v1",

        "research_question": (
            "Are changes in frozen EEG model predictions "
            "associated with annotated eyes-open versus "
            "eyes-closed recording conditions?"
        ),

        "analysis_type": "post_hoc_exploratory",

        "unit": "recording",

        "training": training,
        "validation": validation,

        "exploratory_inference": tests,

        "bootstrap": {
            "resamples": BOOTSTRAPS,
            "seed": SEED,
            "resampling_unit": "recording",
            "independence_verified": False
        },

        "selection": {
            "minimum_eligible_windows_per_state": 2,
            "event_status_required": "aligned",
            "exclude_event_boundaries_seconds": 2,
            "same_recordings_compared_across_conditions": True,
            "test_split_used": False
        },

        "provenance": {
            "code21_summary_sha256": sha256(AUDIT_FILE),
            "code21_record_csv_sha256": sha256(RECORDS_FILE),
            "code21_window_csv_sha256": sha256(WINDOWS_FILE),
            "final_test_plan_sha256": sha256(PLAN_FILE)
        },

        "scientific_boundaries": [
            "Recording conditions were not randomized.",
            "Eyes-open and eyes-closed occur at different recording times.",
            "Eye state, vigilance, elapsed time and artifacts may be confounded.",
            "The analysis excludes recordings with invalid event timestamps.",
            "The eligible recording subset may not represent the full benchmark.",
            "Per-window scores are correlated within recordings.",
            "Patient-level independence cannot be verified.",
            "Model probability ranges and calibration may differ.",
            "Exploratory adjusted p-values are not confirmatory evidence.",
            "Validation was previously viewed and is not a fresh independent replication.",
            "No clinical Alzheimer diagnosis or disease biomarker is inferred."
        ],

        "test_eeg_opened": 0,
        "test_event_files_opened": 0,
        "test_predictions_opened": 0,
        "models_retrained": False,
        "hyperparameters_modified": False
    }

    detail_ok = save_once(
        DETAIL_FILE,
        detail_csv(paired)
    )

    report_ok = save_once(
        REPORT_FILE,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False
        ) + "\n"
    )

    if detail_ok and report_ok:
        print(
            "\nCONDITION_ANALYSIS_STATUS: COMPLETED"
        )
    else:
        print(
            "\nCONDITION_ANALYSIS_STATUS: OUTPUT_CONFLICT"
        )

    print("Original models modified: 0")
    print("New training performed: 0")
    print("Test EEG/event/prediction files opened: 0")
    print("Private recording-level results remain local.")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )

    print("CONDITION_ANALYSIS_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 22 FINISHED ==========",
        flush=True
    )
