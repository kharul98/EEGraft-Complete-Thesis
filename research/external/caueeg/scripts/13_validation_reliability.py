from pathlib import Path
from collections import Counter
import csv
import io
import json
import os
import hashlib

import numpy as np
from scipy.stats import binomtest

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

PREDICTIONS = DATA / "derived/caueeg_transfer_v1"

DETAIL_INPUT = (
    PREDICTIONS / "analysis/12_recording_metrics.csv"
)

EVALUATION = (
    WORK / "results/12_development_evaluation.json"
)

INFERENCE = (
    WORK / "results/11_train_val_inference_summary.json"
)

TASK_LOCK = (
    WORK / "protocol/caueeg_transfer_task_v1.json"
)

SUMMARY_OUTPUT = (
    WORK / "results/13_validation_reliability.json"
)

DETAIL_OUTPUT = (
    PREDICTIONS
    / "analysis/13_validation_reliability_details.csv"
)

THRESHOLD = 0.5

MODELS = {
    "DSRF": "dsrf",
    "XGBoost": "xgb"
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def save_once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            print("[OK] Existing result matches:", path.name)
            return True

        print("[PRESERVED] Different result already exists:", path)
        return False

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(descriptor, "w", encoding="utf-8") as f:
        f.write(text)

    print("[SAVED]", path)
    return True


def validate_previous_results():
    print("\n1. VERIFYING COMPLETED DEVELOPMENT RESULTS")

    for path in (
        DETAIL_INPUT,
        EVALUATION,
        INFERENCE,
        TASK_LOCK
    ):
        require(path.is_file(), f"Missing: {path}")

    evaluation = read_json(EVALUATION)
    inference = read_json(INFERENCE)
    task = read_json(TASK_LOCK)

    require(
        evaluation["status"] == "DEVELOPMENT_EVALUATION_COMPLETED",
        "Code 12 did not finish"
    )

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Code 11 did not finish"
    )

    require(
        evaluation["provenance"] == inference["model_hashes"],
        "Model provenance mismatch"
    )

    require(
        inference["model_hashes"]["task_sha256"]
        == sha256(TASK_LOCK),
        "Locked external task has changed"
    )

    original_files = {
        "protocol_sha256":
            ROOT / "results/final_protocol_2026.json",
        "pretrained_sha256":
            ROOT / "checkpoints/cbramod_official.pth",
        "dsrf_sha256":
            ROOT / "results/final_development_models_2026/dsrf/final.pt",
        "xgboost_sha256":
            ROOT / "results/final_development_models_2026/xgboost/final.joblib",
        "inference_script_sha256":
            WORK / "scripts/11_train_val_inference.py"
    }

    for key, path in original_files.items():
        require(
            path.is_file()
            and sha256(path) == inference["model_hashes"][key],
            f"Original experiment artifact changed: {key}"
        )

    comparison = task["primary_target_comparison"]

    require(
        comparison["positive"] == "dementia"
        and comparison["negative"] == "normal"
        and comparison["decision_threshold"] == THRESHOLD
        and comparison["level"] == "recording",
        "External task definition changed"
    )

    require(
        evaluation["test_recordings_evaluated"] == 0
        and inference["test_recordings_processed"] == 0,
        "Unexpected external test use"
    )

    print("[OK] Original models unchanged")
    print("[OK] Locked 0.5 threshold unchanged")
    print("[OK] Previous development results verified")

    return evaluation, inference


def load_validation_cases(inference):
    print("\n2. LOADING VALIDATION PREDICTIONS ONLY")

    with DETAIL_INPUT.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    validation = [
        row for row in rows
        if row["split"] == "validation"
    ]

    counts = Counter(
        row["class_name"]
        for row in validation
    )

    require(
        counts == {
            "normal": 34,
            "mci": 28,
            "dementia": 20
        },
        "Unexpected validation class counts"
    )

    require(
        len(validation) == 82
        and len({row["serial"] for row in validation}) == 82,
        "Validation recording identifiers are inconsistent"
    )

    cases = []

    for row in sorted(
        validation,
        key=lambda r: r["serial"]
    ):
        serial = row["serial"]
        label = row["class_name"]

        require(
            serial.isascii()
            and serial.isdigit()
            and 5 <= len(serial) <= 8,
            "Invalid recording identifier"
        )

        prediction_path = (
            PREDICTIONS / "validation" / (serial + ".json")
        )

        require(
            prediction_path.is_file(),
            "Missing validation prediction file"
        )

        prediction = read_json(prediction_path)

        require(
            prediction["split"] == "validation"
            and prediction["serial"] == serial
            and prediction["class_name"] == label,
            "Validation prediction metadata mismatch"
        )

        require(
            prediction["model_hashes"]
            == inference["model_hashes"],
            "Inconsistent frozen model provenance"
        )

        case = {
            "serial": serial,
            "class_name": label,
            "binary_target": (
                1 if label == "dementia"
                else 0 if label == "normal"
                else None
            )
        }

        for model, prefix in MODELS.items():
            scores = np.asarray(
                prediction[prefix + "_window_p_ad"],
                dtype=float
            )

            require(
                scores.shape == (24,)
                and np.isfinite(scores).all()
                and np.all((scores >= 0) & (scores <= 1)),
                "Invalid stored window probabilities"
            )

            mean_score = float(scores.mean())

            require(
                np.isclose(
                    mean_score,
                    prediction[prefix + "_mean_p_ad"],
                    rtol=0,
                    atol=1e-7
                ),
                "Recording mean score mismatch"
            )

            # These are 24 windows sampled across the recording.
            # They are not 24 consecutive four-second windows.
            thirds = [
                float(np.mean(scores[0:8])),
                float(np.mean(scores[8:16])),
                float(np.mean(scores[16:24]))
            ]

            third_labels = [
                int(value >= THRESHOLD)
                for value in thirds
            ]

            window_labels = (
                scores >= THRESHOLD
            ).astype(int)

            full_label = int(
                mean_score >= THRESHOLD
            )

            case[model] = {
                "mean_score": mean_score,
                "predicted_positive": full_label,
                "window_sd": float(np.std(scores)),
                "window_label_disagreement": float(
                    np.mean(window_labels != full_label)
                ),
                "window_flip_fraction": float(
                    np.mean(
                        window_labels[1:]
                        != window_labels[:-1]
                    )
                ),
                "all_windows_same_label": bool(
                    np.all(window_labels == window_labels[0])
                ),
                "third_scores": thirds,
                "third_labels": third_labels,
                "thirds_all_agree": bool(
                    len(set(third_labels)) == 1
                ),
                "thirds_match_full": bool(
                    all(
                        value == full_label
                        for value in third_labels
                    )
                ),
                "absolute_early_late_difference": float(
                    abs(thirds[0] - thirds[2])
                )
            }

        cases.append(case)

    print("[OK] Validation recordings:", len(cases))
    print("[OK] Normal:", counts["normal"])
    print("[OK] MCI:", counts["mci"])
    print("[OK] Dementia:", counts["dementia"])
    print("Locked test prediction files opened: 0")

    return cases


def median_or_none(values):
    if not values:
        return None

    return float(np.median(values))


def reliability_summary(cases, model):
    all_stds = [
        case[model]["window_sd"]
        for case in cases
    ]

    all_flips = [
        case[model]["window_flip_fraction"]
        for case in cases
    ]

    early_late = [
        case[model]["absolute_early_late_difference"]
        for case in cases
    ]

    binary = [
        case for case in cases
        if case["binary_target"] is not None
    ]

    correct = [
        case for case in binary
        if case[model]["predicted_positive"]
        == case["binary_target"]
    ]

    incorrect = [
        case for case in binary
        if case[model]["predicted_positive"]
        != case["binary_target"]
    ]

    stable_correct = [
        case for case in correct
        if case[model]["all_windows_same_label"]
    ]

    stable_incorrect = [
        case for case in incorrect
        if case[model]["all_windows_same_label"]
    ]

    phase_agree = sum(
        case[model]["thirds_all_agree"]
        for case in cases
    )

    phase_match = sum(
        case[model]["thirds_match_full"]
        for case in cases
    )

    return {
        "n_all_validation_recordings": len(cases),
        "n_binary_evaluable_recordings": len(binary),
        "median_window_probability_sd": median_or_none(
            all_stds
        ),
        "mean_window_flip_fraction": float(
            np.mean(all_flips)
        ),
        "median_absolute_early_late_score_difference":
            median_or_none(early_late),
        "early_middle_late_prediction_agreement_fraction":
            phase_agree / len(cases),
        "all_three_phases_match_full_prediction_fraction":
            phase_match / len(cases),
        "correct_recordings": len(correct),
        "incorrect_recordings": len(incorrect),
        "all_24_windows_agree_and_correct":
            len(stable_correct),
        "all_24_windows_agree_but_wrong":
            len(stable_incorrect),
        "median_window_sd_when_correct": median_or_none([
            case[model]["window_sd"]
            for case in correct
        ]),
        "median_window_sd_when_incorrect": median_or_none([
            case[model]["window_sd"]
            for case in incorrect
        ]),
        "caution": (
            "Temporal consistency is not equivalent to accuracy. "
            "The 24 windows were sampled across each recording."
        )
    }


def comparison_and_controls(cases):
    print("\n4. COMPARING MODEL ERRORS")

    binary = [
        case for case in cases
        if case["binary_target"] is not None
    ]

    outcomes = Counter()

    for case in binary:
        y = case["binary_target"]

        d_correct = (
            case["DSRF"]["predicted_positive"] == y
        )

        x_correct = (
            case["XGBoost"]["predicted_positive"] == y
        )

        if d_correct and x_correct:
            outcomes["both_correct"] += 1
        elif d_correct:
            outcomes["DSRF_only_correct"] += 1
        elif x_correct:
            outcomes["XGBoost_only_correct"] += 1
        else:
            outcomes["both_wrong"] += 1

    discordant = (
        outcomes["DSRF_only_correct"]
        + outcomes["XGBoost_only_correct"]
    )

    if discordant:
        pvalue = float(
            binomtest(
                outcomes["DSRF_only_correct"],
                discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue
        )
    else:
        pvalue = 1.0

    normal_count = sum(
        case["binary_target"] == 0
        for case in binary
    )

    dementia_count = sum(
        case["binary_target"] == 1
        for case in binary
    )

    controls = {
        "always_normal": {
            "balanced_accuracy": 0.5,
            "accuracy": normal_count / len(binary)
        },
        "always_dementia": {
            "balanced_accuracy": 0.5,
            "accuracy": dementia_count / len(binary)
        }
    }

    print("Both correct:", outcomes["both_correct"])
    print("Only DSRF correct:", outcomes["DSRF_only_correct"])
    print("Only XGBoost correct:", outcomes["XGBoost_only_correct"])
    print("Both wrong:", outcomes["both_wrong"])
    print("Paired discordant cases:", discordant)
    print("Exact McNemar p-value:", round(pvalue, 5))

    print(
        "Always-normal baseline accuracy:",
        f"{controls['always_normal']['accuracy']:.2%}"
    )

    print(
        "Always-dementia baseline accuracy:",
        f"{controls['always_dementia']['accuracy']:.2%}"
    )

    print("Both constant baselines have 50% balanced accuracy.")

    return {
        "paired_correctness_counts": dict(outcomes),
        "discordant_recordings": discordant,
        "exact_mcnemar_pvalue_exploratory": pvalue,
        "constant_classifier_controls": controls,
        "caution": (
            "The exact McNemar test assumes independent cases. "
            "Repeated patients cannot be independently excluded, "
            "so this p-value is exploratory, not confirmatory."
        )
    }


def private_detail_csv(cases):
    fields = [
        "serial",
        "class_name",
        "binary_target"
    ]

    for model in MODELS:
        fields += [
            model + "_mean_score",
            model + "_prediction",
            model + "_window_sd",
            model + "_all_windows_same_label",
            model + "_thirds_all_agree",
            model + "_absolute_early_late_difference",
            model + "_correct_if_binary"
        ]

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=fields
    )

    writer.writeheader()

    for case in cases:
        output = {
            "serial": case["serial"],
            "class_name": case["class_name"],
            "binary_target": (
                ""
                if case["binary_target"] is None
                else case["binary_target"]
            )
        }

        for model in MODELS:
            item = case[model]

            output[model + "_mean_score"] = item["mean_score"]
            output[model + "_prediction"] = item["predicted_positive"]
            output[model + "_window_sd"] = item["window_sd"]
            output[model + "_all_windows_same_label"] = int(
                item["all_windows_same_label"]
            )
            output[model + "_thirds_all_agree"] = int(
                item["thirds_all_agree"]
            )
            output[
                model + "_absolute_early_late_difference"
            ] = item["absolute_early_late_difference"]

            output[model + "_correct_if_binary"] = (
                ""
                if case["binary_target"] is None
                else int(
                    item["predicted_positive"]
                    == case["binary_target"]
                )
            )

        writer.writerow(output)

    return buffer.getvalue()


def main():
    evaluation, inference = validate_previous_results()

    cases = load_validation_cases(inference)

    print("\n3. CALCULATING TEMPORAL RELIABILITY")

    temporal = {}

    for model in MODELS:
        temporal[model] = reliability_summary(
            cases, model
        )

        item = temporal[model]

        print("\nMODEL:", model)

        print(
            "Median window probability SD:",
            round(item["median_window_probability_sd"], 5)
        )

        print(
            "Mean sampled-window label flip fraction:",
            f"{item['mean_window_flip_fraction']:.2%}"
        )

        print(
            "Early/middle/late prediction agreement:",
            f"{item['early_middle_late_prediction_agreement_fraction']:.2%}"
        )

        print(
            "All windows agree but prediction is WRONG:",
            item["all_24_windows_agree_but_wrong"]
        )

        print(
            "All windows agree and prediction is CORRECT:",
            item["all_24_windows_agree_and_correct"]
        )

    comparison = comparison_and_controls(cases)

    print("\n5. EXISTING VALIDATION BOOTSTRAP INTERVALS")

    uncertainty = evaluation["validation_uncertainty"]

    intervals = uncertainty[
        "balanced_accuracy_95ci"
    ]

    for name in (
        "DSRF",
        "XGBoost",
        "DSRF_minus_XGBoost"
    ):
        lower, upper = intervals[name]

        print(
            name,
            "95% recording-bootstrap interval:",
            f"[{lower:.4f}, {upper:.4f}]"
        )

    print("\n6. GENERATING RELIABILITY REPORT")

    report = {
        "status": "VALIDATION_RELIABILITY_COMPLETED",
        "version": "caueeg_transfer_v1",
        "partition": "validation_only",
        "n_recordings": 82,
        "n_binary_evaluation_recordings": 54,
        "n_mci_descriptive_recordings": 28,
        "recording_level_only": True,
        "original_models_retrained": False,
        "test_prediction_files_read": 0,
        "threshold": THRESHOLD,
        "model_hashes": inference["model_hashes"],
        "temporal_reliability": temporal,
        "paired_error_analysis": comparison,
        "existing_validation_bootstrap": uncertainty,
        "scientific_limitations": [
            "The official labels represent a broader dementia task.",
            "An AD-trained probability is not automatically a calibrated dementia probability.",
            "Patient identities cannot be independently verified.",
            "Stable predictions may be consistently wrong.",
            "Temporal windows are sampled across recordings, not consecutively.",
            "Validation results must not be reported as locked-test performance.",
            "Exploratory comparisons are not evidence of clinical usefulness."
        ]
    }

    content = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False
    ) + "\n"

    details_ok = save_once(
        DETAIL_OUTPUT,
        private_detail_csv(cases)
    )

    summary_ok = save_once(
        SUMMARY_OUTPUT,
        content
    )

    if details_ok and summary_ok:
        print("\nVALIDATION_RELIABILITY_STATUS: COMPLETED")
    else:
        print("\nVALIDATION_RELIABILITY_STATUS: OUTPUT_CONFLICT")

    print("Locked test recordings accessed: 0")
    print("Original model checkpoints modified: 0")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc)
    )

    print(
        "VALIDATION_RELIABILITY_STATUS: FAILED"
    )

finally:
    print(
        "\n========== CODE 13 FINISHED =========="
    )
