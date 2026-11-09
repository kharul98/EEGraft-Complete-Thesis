from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import os

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, f1_score

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

PRIVATE = DATA / "derived/caueeg_transfer_v1"
VALIDATION = PRIVATE / "validation"

RESULTS = WORK / "results"
PROTOCOL = WORK / "protocol"

INFERENCE = RESULTS / "11_train_val_inference_summary.json"
EVALUATION = RESULTS / "12_development_evaluation.json"
TUNING = RESULTS / "14_train_only_threshold_tuning.json"
CALIBRATION = RESULTS / "15_train_only_calibration.json"

TASK = PROTOCOL / "caueeg_transfer_task_v1.json"
THRESHOLDS = PROTOCOL / "caueeg_adapted_thresholds_v1.json"
PLATT = PROTOCOL / "caueeg_platt_calibration_v1.json"

REPORT = RESULTS / "16_validation_variant_comparison.json"

PRIVATE_CSV = (
    PRIVATE / "analysis/16_validation_variant_details.csv"
)

MODELS = {
    "DSRF": (
        "dsrf_mean_p_ad",
        "dsrf_window_p_ad"
    ),
    "XGBoost": (
        "xgb_mean_p_ad",
        "xgb_window_p_ad"
    )
}

EPS = 1e-6


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_text(encoding="utf-8")

        if existing == text:
            print("[OK] Existing output matches:", path)
            return True

        print("[PRESERVED] Different existing output:", path)
        return False

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(
        descriptor,
        "w",
        encoding="utf-8"
    ) as handle:
        handle.write(text)

    print("[SAVED]", path)
    return True


def preflight():
    print("\n1. VERIFYING ALL FROZEN EXPERIMENT ARTIFACTS")

    required = [
        INFERENCE,
        EVALUATION,
        TUNING,
        CALIBRATION,
        TASK,
        THRESHOLDS,
        PLATT
    ]

    for path in required:
        require(
            path.is_file(),
            "Required prior result missing: " + str(path)
        )

    inference = read_json(INFERENCE)
    evaluation = read_json(EVALUATION)
    tuning = read_json(TUNING)
    calibration = read_json(CALIBRATION)

    task = read_json(TASK)
    thresholds = read_json(THRESHOLDS)
    platt = read_json(PLATT)

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Code 11 incomplete"
    )

    require(
        evaluation["status"]
        == "DEVELOPMENT_EVALUATION_COMPLETED",
        "Code 12 incomplete"
    )

    require(
        tuning["status"]
        == "TRAIN_ONLY_THRESHOLD_SEARCH_COMPLETED",
        "Code 14 incomplete"
    )

    require(
        calibration["status"]
        == "TRAIN_ONLY_CALIBRATION_COMPLETED",
        "Code 15 incomplete"
    )

    require(
        inference["test_recordings_processed"] == 0
        and evaluation["test_recordings_evaluated"] == 0,
        "Unexpected external test access status"
    )

    require(
        tuning["test_recordings_accessed"] == 0
        and calibration["test_predictions_accessed"] == 0,
        "Unexpected test tuning status"
    )

    require(
        calibration["validation_predictions_accessed"] == 0
        and tuning["validation_predictions_accessed"] == 0,
        "Training-only adaptation status invalid"
    )

    comparison = task["primary_target_comparison"]

    require(
        comparison["decision_threshold"] == 0.5
        and comparison["positive"] == "dementia"
        and comparison["negative"] == "normal"
        and comparison["level"] == "recording",
        "Original external task changed"
    )

    hashes = inference["model_hashes"]

    require(
        hashes
        == evaluation["provenance"]
        == tuning["source_model_hashes"]
        == calibration["source_model_hashes"]
        == thresholds["source_model_hashes"]
        == platt["source_model_hashes"],
        "Model provenance mismatch"
    )

    require(
        thresholds["adapted_track"]["selected_thresholds"]
        == tuning["selected_thresholds"],
        "Training threshold lock differs from report"
    )

    require(
        thresholds["adapted_track"]["test_used_for_tuning"] is False
        and platt["test_used_to_fit_calibration"] is False,
        "External test adaptation safety violation"
    )

    original_files = {
        "protocol_sha256":
            ROOT / "results/final_protocol_2026.json",

        "task_sha256": TASK,

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
            and sha256(path) == hashes[key],
            "Original artifact changed: " + key
        )

    for model in MODELS:
        require(
            model in thresholds["adapted_track"]["selected_thresholds"]
            and model in platt["calibration_parameters"],
            "Missing adapted configuration for " + model
        )

        selected_threshold = float(
            thresholds["adapted_track"]["selected_thresholds"][model]
        )

        require(
            0 < selected_threshold < 1,
            "Invalid training-selected threshold"
        )

        parameters = platt["calibration_parameters"][model]

        require(
            np.isfinite(float(parameters["coefficient"]))
            and float(parameters["coefficient"]) > 0
            and np.isfinite(float(parameters["intercept"])),
            "Invalid Platt calibration parameters"
        )

    print("[OK] Original frozen models verified")
    print("[OK] Original external task verified")
    print("[OK] Training threshold lock verified")
    print("[OK] Training calibration lock verified")

    return hashes, evaluation, thresholds, platt


def load_cases(hashes, platt):
    print("\n2. LOADING VALIDATION PREDICTIONS ONLY")

    require(
        VALIDATION.is_dir(),
        "Validation prediction directory missing"
    )

    paths = sorted(VALIDATION.glob("*.json"))

    require(
        len(paths) == 82,
        "Expected exactly 82 validation prediction files"
    )

    cases = []
    counts = Counter()
    seen = set()

    for path in paths:
        item = read_json(path)

        serial = item["serial"]
        label = item["class_name"]

        require(
            item["split"] == "validation"
            and item["version"] == "caueeg_transfer_v1"
            and item["model_hashes"] == hashes,
            "Validation prediction provenance mismatch"
        )

        require(
            path.stem == serial
            and serial not in seen,
            "Duplicate or mismatched recording identifier"
        )

        require(
            label in ("normal", "mci", "dementia")
            and item["window_count"] == 24,
            "Unexpected diagnosis or window count"
        )

        seen.add(serial)
        counts[label] += 1

        case = {
            "serial": serial,
            "class_name": label,
            "y": (
                None if label == "mci"
                else int(label == "dementia")
            )
        }

        for model, (mean_key, window_key) in MODELS.items():
            raw_score = float(item[mean_key])

            windows = np.asarray(
                item[window_key],
                dtype=float
            )

            require(
                windows.shape == (24,)
                and np.isfinite(windows).all()
                and np.all(
                    (windows >= 0)
                    & (windows <= 1)
                ),
                "Invalid window predictions"
            )

            require(
                0 <= raw_score <= 1
                and abs(
                    raw_score - float(windows.mean())
                ) <= 1e-7,
                "Stored mean probability is inconsistent"
            )

            parameters = platt["calibration_parameters"][model]

            coefficient = float(
                parameters["coefficient"]
            )

            intercept = float(
                parameters["intercept"]
            )

            clipped = np.clip(
                raw_score,
                EPS,
                1.0 - EPS
            )

            log_odds = float(
                np.log(clipped / (1.0 - clipped))
            )

            calibrated = float(
                expit(
                    coefficient * log_odds + intercept
                )
            )

            case[model + "_raw"] = raw_score
            case[model + "_platt"] = calibrated

        cases.append(case)

    require(
        counts == {
            "normal": 34,
            "mci": 28,
            "dementia": 20
        },
        "Official validation class distribution changed"
    )

    print("[OK] Normal:", counts["normal"])
    print("[OK] MCI:", counts["mci"])
    print("[OK] Dementia:", counts["dementia"])
    print("Total validation recordings:", len(cases))
    print("External test recordings accessed: 0")

    return cases


def calibration_error(y, probabilities):
    predicted = (
        probabilities >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    correctness = (
        predicted == y
    ).astype(float)

    ece = 0.0

    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10

        if index == 9:
            inside = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            inside = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if inside.any():
            ece += float(
                inside.mean()
                * abs(
                    confidence[inside].mean()
                    - correctness[inside].mean()
                )
            )

    return ece


def metrics(y, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    normal = y == 0
    dementia = y == 1

    tn = int(np.sum(normal & (predictions == 0)))
    fp = int(np.sum(normal & (predictions == 1)))
    fn = int(np.sum(dementia & (predictions == 0)))
    tp = int(np.sum(dementia & (predictions == 1)))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    clipped = np.clip(
        probabilities,
        EPS,
        1.0 - EPS
    )

    return {
        "n_normal": int(normal.sum()),
        "n_dementia": int(dementia.sum()),
        "threshold": float(threshold),

        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),

        "accuracy": float(
            np.mean(predictions == y)
        ),

        "dementia_sensitivity": float(sensitivity),
        "normal_specificity": float(specificity),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "auroc": float(
            roc_auc_score(y, probabilities)
        ),

        "brier_vs_dementia": float(
            np.mean((probabilities - y) ** 2)
        ),

        "log_loss_vs_dementia": float(
            -np.mean(
                y * np.log(clipped)
                + (1 - y) * np.log(1 - clipped)
            )
        ),

        # Standard calibration ECE assumes decisions at 0.5.
        # It is not directly applicable to a custom threshold.
        "ece_at_standard_0_5_decision": (
            float(calibration_error(y, probabilities))
            if threshold == 0.5
            else None
        ),

        "confusion_matrix_rows_normal_dementia": [
            [tn, fp],
            [fn, tp]
        ],

        "predicted_dementia_count": int(
            predictions.sum()
        )
    }


def bootstrap_difference(
    y,
    scores_a,
    threshold_a,
    scores_b,
    threshold_b,
    repeats=2000
):
    rng = np.random.default_rng(20261109)

    group_normal = np.flatnonzero(y == 0)
    group_dementia = np.flatnonzero(y == 1)

    predictions_a = (
        scores_a >= threshold_a
    )

    predictions_b = (
        scores_b >= threshold_b
    )

    differences = []

    for _ in range(repeats):
        sampled_normal = rng.choice(
            group_normal,
            size=len(group_normal),
            replace=True
        )

        sampled_dementia = rng.choice(
            group_dementia,
            size=len(group_dementia),
            replace=True
        )

        bacc_a = (
            np.mean(predictions_a[sampled_normal] == 0)
            + np.mean(predictions_a[sampled_dementia] == 1)
        ) / 2

        bacc_b = (
            np.mean(predictions_b[sampled_normal] == 0)
            + np.mean(predictions_b[sampled_dementia] == 1)
        ) / 2

        differences.append(
            float(bacc_a - bacc_b)
        )

    return [
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975))
    ]


def compare_variants(cases, evaluation, thresholds):
    print("\n3. COMPARING THREE FROZEN VARIANTS")

    binary = [
        case for case in cases
        if case["y"] is not None
    ]

    mci = [
        case for case in cases
        if case["y"] is None
    ]

    y = np.asarray(
        [case["y"] for case in binary],
        dtype=int
    )

    results = {}

    for model in MODELS:
        raw_scores = np.asarray([
            case[model + "_raw"]
            for case in binary
        ])

        calibrated_scores = np.asarray([
            case[model + "_platt"]
            for case in binary
        ])

        threshold = float(
            thresholds["adapted_track"]["selected_thresholds"][model]
        )

        frozen = metrics(
            y,
            raw_scores,
            0.5
        )

        tuned = metrics(
            y,
            raw_scores,
            threshold
        )

        calibrated = metrics(
            y,
            calibrated_scores,
            0.5
        )

        # Confirm that this script exactly reproduces Code 12
        # before analyzing the adapted variants.
        previous = (
            evaluation["models"][model]
            ["validation"]["binary_metrics"]
        )

        comparisons = [
            ("balanced_accuracy", "balanced_accuracy"),
            ("accuracy", "accuracy"),
            ("auroc", "auroc"),
            ("macro_f1", "macro_f1"),
            ("brier_vs_dementia", "brier_vs_dementia_label"),
            ("ece_at_standard_0_5_decision", "ece_10_bins")
        ]

        for current_key, previous_key in comparisons:
            require(
                np.isclose(
                    frozen[current_key],
                    previous[previous_key],
                    atol=1e-9
                ),
                "Code 12 baseline mismatch: "
                + model + "/" + current_key
            )

        require(
            frozen["confusion_matrix_rows_normal_dementia"]
            == previous["confusion_matrix"],
            "Original confusion matrix mismatch"
        )

        require(
            np.isclose(
                frozen["auroc"],
                calibrated["auroc"],
                atol=1e-10
            ),
            "Monotonic calibration unexpectedly changed AUROC"
        )

        mci_raw = np.asarray([
            case[model + "_raw"]
            for case in mci
        ])

        mci_calibrated = np.asarray([
            case[model + "_platt"]
            for case in mci
        ])

        results[model] = {
            "frozen_original_threshold_0_5": frozen,

            "train_selected_score_threshold": tuned,

            "train_fitted_platt_threshold_0_5": calibrated,

            "paired_recording_bootstrap_bacc_deltas_95ci": {
                "selected_threshold_minus_frozen":
                    bootstrap_difference(
                        y,
                        raw_scores,
                        threshold,
                        raw_scores,
                        0.5
                    ),

                "platt_minus_frozen":
                    bootstrap_difference(
                        y,
                        calibrated_scores,
                        0.5,
                        raw_scores,
                        0.5
                    )
            },

            "mci_descriptive_not_accuracy": {
                "count": len(mci),

                "median_original_source_ad_score": float(
                    np.median(mci_raw)
                ),

                "fraction_at_or_above_train_threshold": float(
                    np.mean(mci_raw >= threshold)
                ),

                "median_calibrated_dementia_score": float(
                    np.median(mci_calibrated)
                )
            }
        }

        print("\nMODEL:", model)

        variants = [
            ("Original frozen 0.50", frozen),
            ("Training-selected threshold", tuned),
            ("Training Platt calibration", calibrated)
        ]

        for name, result in variants:
            ece_value = result[
                "ece_at_standard_0_5_decision"
            ]

            ece_text = (
                "N/A"
                if ece_value is None
                else f"{ece_value:.4f}"
            )

            print("\n ", name)
            print("  Threshold:", result["threshold"])
            print(
                "  Balanced accuracy:",
                f"{result['balanced_accuracy']:.2%}"
            )
            print(
                "  Dementia sensitivity:",
                f"{result['dementia_sensitivity']:.2%}"
            )
            print(
                "  Normal specificity:",
                f"{result['normal_specificity']:.2%}"
            )
            print(
                "  AUROC:",
                f"{result['auroc']:.4f}"
            )
            print(
                "  Brier score:",
                f"{result['brier_vs_dementia']:.4f}"
            )
            print("  ECE:", ece_text)
            print(
                "  Confusion matrix:",
                result["confusion_matrix_rows_normal_dementia"]
            )

    return results


def make_private_csv(cases, thresholds):
    fields = [
        "serial",
        "class_name",
        "binary_target"
    ]

    for model in MODELS:
        fields += [
            model + "_raw_p_ad",
            model + "_calibrated_p_dementia",
            model + "_frozen_prediction",
            model + "_threshold_adapted_prediction",
            model + "_platt_prediction"
        ]

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=fields
    )

    writer.writeheader()

    for case in cases:
        row = {
            "serial": case["serial"],
            "class_name": case["class_name"],
            "binary_target": (
                ""
                if case["y"] is None
                else case["y"]
            )
        }

        for model in MODELS:
            original = case[model + "_raw"]
            calibrated = case[model + "_platt"]

            selected_threshold = float(
                thresholds["adapted_track"]
                ["selected_thresholds"][model]
            )

            row[model + "_raw_p_ad"] = original

            row[
                model + "_calibrated_p_dementia"
            ] = calibrated

            row[
                model + "_frozen_prediction"
            ] = int(original >= 0.5)

            row[
                model + "_threshold_adapted_prediction"
            ] = int(original >= selected_threshold)

            row[
                model + "_platt_prediction"
            ] = int(calibrated >= 0.5)

        writer.writerow(row)

    return buffer.getvalue()


def main():
    hashes, evaluation, thresholds, platt = preflight()

    cases = load_cases(hashes, platt)

    results = compare_variants(
        cases,
        evaluation,
        thresholds
    )

    print("\n4. SAVING VALIDATION COMPARISON")

    report = {
        "status": "VALIDATION_VARIANTS_COMPARED",
        "benchmark": "CAUEEG_dementia_no_overlap",
        "evaluation_level": "recording_not_verified_patient",

        "n_validation_recordings": 82,
        "n_binary_normal_dementia": 54,
        "n_mci_descriptive_only": 28,

        "test_recordings_accessed": 0,
        "original_models_retrained": False,

        "thresholds_selected_using_validation": False,
        "calibration_fit_using_validation": False,

        "previous_validation_results_were_already_viewed": True,

        "comparison_not_independent_for_new_experiment_design": True,

        "original_model_hashes": hashes,

        "training_threshold_lock_sha256": sha256(
            THRESHOLDS
        ),

        "training_calibration_lock_sha256": sha256(
            PLATT
        ),

        "per_model_variants": results,

        "interpretation": (
            "All three fixed variants are reported separately. "
            "No post-hoc winning variant is declared."
        ),

        "limitations": [
            "Dementia is not equivalent to confirmed Alzheimer's disease.",
            "The source AD score is not a clinical diagnosis.",
            "Patient identities are not independently verified.",
            "Bootstrap intervals resample recordings, not independent patients.",
            "Additional experiments were introduced after initial validation results were viewed.",
            "MCI is excluded from binary classification metrics."
        ]
    }

    private_ok = write_once(
        PRIVATE_CSV,
        make_private_csv(cases, thresholds)
    )

    report_ok = write_once(
        REPORT,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False
        ) + "\n"
    )

    if private_ok and report_ok:
        print(
            "\nCOMPARISON_STATUS: "
            "VALIDATION_VARIANTS_COMPARED"
        )
    else:
        print(
            "\nCOMPARISON_STATUS: OUTPUT_CONFLICT"
        )

    print("External test predictions read: 0")
    print("Original model checkpoints modified: 0")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc)
    )

    print("COMPARISON_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 16 FINISHED =========="
    )
