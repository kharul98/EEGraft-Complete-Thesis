from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import os
import re

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

PLAN_FILE = WORK / "protocol/caueeg_final_test_plan_v1.json"
PREFLIGHT_FILE = WORK / "results/17_final_test_preflight.json"
INFERENCE_FILE = WORK / "results/18_locked_test_inference_summary.json"
INFERENCE_SCRIPT = WORK / "scripts/18_locked_test_inference.py"

THRESHOLD_FILE = (
    WORK / "protocol/caueeg_adapted_thresholds_v1.json"
)

CALIBRATION_FILE = (
    WORK / "protocol/caueeg_platt_calibration_v1.json"
)

MANIFEST_FILE = DATA / "manifests/07_recording_audit.csv"

TEST_DIR = DATA / "derived/caueeg_transfer_v1/test"

DETAIL_FILE = (
    DATA / "derived/caueeg_transfer_v1"
    / "analysis/19_locked_test_recording_details.csv"
)

REPORT_FILE = WORK / "results/19_locked_test_evaluation.json"

MODEL_SCORE_KEYS = {
    "DSRF": "dsrf",
    "XGBoost": "xgb"
}

CLASS_COUNTS = {
    "normal": 35,
    "mci": 33,
    "dementia": 22
}

EPS = 1e-6


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def save_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing result matches:", path)
            return True

        print("[PRESERVED] Different existing result:", path)
        return False

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(
        descriptor, "w", encoding="utf-8"
    ) as stream:
        stream.write(content)

    print("[SAVED]", path)
    return True


def verify_freeze():
    print(
        "\n1. VERIFYING FINAL TEST PLAN AND PROVENANCE",
        flush=True
    )

    required = (
        PLAN_FILE,
        PREFLIGHT_FILE,
        INFERENCE_FILE,
        INFERENCE_SCRIPT,
        THRESHOLD_FILE,
        CALIBRATION_FILE,
        MANIFEST_FILE
    )

    for path in required:
        require(
            path.is_file(),
            "Required artifact missing: " + str(path)
        )

    plan = read_json(PLAN_FILE)
    preflight = read_json(PREFLIGHT_FILE)
    inference = read_json(INFERENCE_FILE)
    thresholds = read_json(THRESHOLD_FILE)
    calibration = read_json(CALIBRATION_FILE)

    plan_hash = sha256(PLAN_FILE)

    require(
        preflight["status"]
        == "READY_FOR_LOCKED_TEST_INFERENCE"
        and preflight["plan_sha256"] == plan_hash,
        "Code 17 final test lock mismatch"
    )

    require(
        inference["status"]
        == "LOCKED_TEST_INFERENCE_COMPLETED"
        and inference["final_test_plan_sha256"] == plan_hash,
        "Code 18 test inference is incomplete or inconsistent"
    )

    require(
        inference["inference_code_sha256"]
        == sha256(INFERENCE_SCRIPT),
        "Code 18 inference script changed"
    )

    require(
        inference["test_recordings"] == 90
        and inference["test_eeg_windows"] == 2160
        and inference["test_metrics_calculated"] is False,
        "Code 18 counts or evaluation status changed"
    )

    require(
        inference["original_models_retrained"] is False
        and inference["original_thresholds_changed"] is False,
        "Original models or thresholds were modified"
    )

    require(
        plan["version"] == "caueeg_final_test_plan_v1"
        and plan["official_task"] == "dementia-no-overlap"
        and plan["test_recordings"] == 90
        and plan["primary_binary_test_recordings"] == 57
        and plan["secondary_mci_recordings"] == 33,
        "Unexpected final test plan"
    )

    require(
        plan["test_class_counts"] == CLASS_COUNTS
        and plan["analysis_unit"] == "recording"
        and plan["patient_level_performance_claims_allowed"] is False,
        "Unexpected test population or evaluation unit"
    )

    primary = plan["primary_comparison"]

    require(
        primary["models"] == ["DSRF", "XGBoost"]
        and primary["source_classes"] == ["AD", "CN"]
        and primary["target_negative"] == "normal"
        and primary["target_positive"] == "dementia"
        and primary["excluded_from_primary_binary_metrics"] == "mci"
        and float(primary["DSRF_threshold"]) == 0.5
        and float(primary["XGBoost_threshold"]) == 0.5
        and primary["primary_metric"] == "balanced_accuracy",
        "Primary evaluation definition changed"
    )

    require(
        plan["uncertainty"]["method"]
        == "paired_stratified_recording_bootstrap"
        and int(plan["uncertainty"]["resamples"]) == 5000
        and int(plan["uncertainty"]["seed"]) == 20261109,
        "Bootstrap protocol changed"
    )

    require(
        plan["safety"][
            "no_further_hyperparameter_tuning_after_test"
        ] is True
        and plan["safety"]["no_variant_selection_using_test"] is True,
        "Locked test safety restrictions changed"
    )

    hashes = plan["model_hashes"]

    require(
        inference["model_hashes"] == hashes
        and thresholds["source_model_hashes"] == hashes
        and calibration["source_model_hashes"] == hashes,
        "Inconsistent original model hashes"
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

        "task_sha256":
            WORK / "protocol/caueeg_transfer_task_v1.json",

        "inference_script_sha256":
            WORK / "scripts/11_train_val_inference.py",
    }

    for key, path in original_files.items():
        require(
            path.is_file()
            and sha256(path) == hashes[key],
            "Original artifact changed: " + key
        )

    require(
        sha256(THRESHOLD_FILE)
        == plan["artifact_hashes"]["threshold_lock_sha256"]
        and sha256(CALIBRATION_FILE)
        == plan["artifact_hashes"]["calibration_lock_sha256"]
        and sha256(MANIFEST_FILE)
        == plan["artifact_hashes"]["recording_manifest_sha256"],
        "Dataset or adaptation artifact changed"
    )

    require(
        thresholds["adapted_track"]["selected_thresholds"]
        == plan["secondary_exploratory_variants"][
            "training_selected_thresholds"
        ],
        "Adapted thresholds differ from frozen plan"
    )

    require(
        calibration["calibration_parameters"]
        == plan["secondary_exploratory_variants"][
            "platt_calibration_parameters"
        ],
        "Calibration parameters differ from frozen plan"
    )

    require(
        plan["secondary_exploratory_variants"][
            "report_all_variants_without_test_selection"
        ] is True,
        "All frozen variants must be reported"
    )

    print("[OK] Code 17 final test plan verified")
    print("[OK] Code 18 completion verified")
    print("[OK] Original model checkpoints unchanged")
    print("[OK] All evaluation variants already frozen")

    return plan, inference, plan_hash


def load_test_records(plan, inference, plan_hash):
    print(
        "\n2. READING STORED TEST PREDICTIONS",
        flush=True
    )

    with MANIFEST_FILE.open(
        newline="", encoding="utf-8"
    ) as stream:
        manifest = list(csv.DictReader(stream))

    require(
        len(manifest) == 1122,
        "Unexpected official benchmark manifest size"
    )

    test_rows = [
        row for row in manifest
        if row["split"] == "test"
    ]

    require(
        len(test_rows) == 90
        and len({row["serial"] for row in test_rows}) == 90,
        "Expected 90 distinct official test recordings"
    )

    require(
        dict(Counter(
            row["class_name"] for row in test_rows
        )) == CLASS_COUNTS,
        "Official test class distribution changed"
    )

    require(
        TEST_DIR.is_dir(),
        "Test prediction directory missing"
    )

    paths = sorted(TEST_DIR.glob("*.json"))

    require(
        len(paths) == 90
        and {path.stem for path in paths}
        == {row["serial"] for row in test_rows},
        "Stored test predictions do not match official split"
    )

    cases = []
    prediction_hash_lines = []

    for row in sorted(
        test_rows, key=lambda item: item["serial"]
    ):
        serial = row["serial"]

        require(
            re.fullmatch(r"[0-9]{5,8}", serial) is not None,
            "Invalid recording serial"
        )

        path = TEST_DIR / (serial + ".json")
        item = read_json(path)

        require(
            item["version"] == "caueeg_transfer_v1"
            and item["split"] == "test"
            and item["serial"] == serial
            and item["class_name"] == row["class_name"]
            and item["benchmark"] == "dementia-no-overlap"
            and item["source_model_classes"] == ["AD", "CN"]
            and item["window_count"] == 24
            and item["patient_id_verified"] is False,
            "Stored test prediction metadata mismatch"
        )

        require(
            item["model_hashes"] == plan["model_hashes"]
            and item["final_test_plan_sha256"] == plan_hash
            and item["inference_code_sha256"]
            == inference["inference_code_sha256"],
            "Stored test prediction provenance mismatch"
        )

        require(
            re.fullmatch(
                r"[0-9a-f]{64}", item["edf_sha256"]
            ) is not None,
            "Missing original EDF checksum"
        )

        starts = item["window_start_seconds"]

        require(
            len(starts) == 24
            and all(
                isinstance(value, (int, float))
                and np.isfinite(value)
                for value in starts
            )
            and all(
                starts[i] < starts[i + 1]
                for i in range(23)
            ),
            "Invalid EEG window positions"
        )

        label = row["class_name"]

        case = {
            "serial": serial,
            "class_name": label,
            "y": (
                1 if label == "dementia"
                else 0 if label == "normal"
                else None
            )
        }

        for model, prefix in MODEL_SCORE_KEYS.items():
            windows = np.asarray(
                item[prefix + "_window_p_ad"],
                dtype=np.float64
            )

            mean_score = float(
                item[prefix + "_mean_p_ad"]
            )

            require(
                windows.shape == (24,)
                and np.isfinite(windows).all()
                and np.all(
                    (windows >= 0) & (windows <= 1)
                )
                and np.isfinite(mean_score)
                and 0 <= mean_score <= 1
                and np.isclose(
                    mean_score,
                    windows.mean(),
                    rtol=0,
                    atol=1e-7
                ),
                "Invalid model predictions for " + model
            )

            case[model] = {
                "score": mean_score,
                "windows": windows
            }

        cases.append(case)

        prediction_hash_lines.append(
            serial + ":" + sha256(path)
        )

    bundle_hash = hashlib.sha256(
        (
            "\n".join(prediction_hash_lines) + "\n"
        ).encode("utf-8")
    ).hexdigest()

    print("[OK] 90 prediction records verified")
    print("[OK] 2,160 window predictions per model")
    print("Test EEG signal files opened by Code 19: 0")

    return cases, bundle_hash


def calibrated_score(raw_score, parameters):
    clipped = float(
        np.clip(raw_score, EPS, 1.0 - EPS)
    )

    log_odds = np.log(
        clipped / (1.0 - clipped)
    )

    return float(
        expit(
            float(parameters["coefficient"]) * log_odds
            + float(parameters["intercept"])
        )
    )


def ece_10(y, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1 - probabilities
    )

    correct = (
        predictions == y
    ).astype(float)

    result = 0.0

    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10

        inside = (
            (confidence >= lower)
            & (
                (confidence <= upper)
                if index == 9
                else (confidence < upper)
            )
        )

        if inside.any():
            result += float(
                inside.mean()
                * abs(
                    confidence[inside].mean()
                    - correct[inside].mean()
                )
            )

    return result


def measure_binary(y, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

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
        "n_binary_recordings": int(len(y)),
        "n_normal": int(normal.sum()),
        "n_dementia": int(dementia.sum()),
        "threshold": float(threshold),

        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),

        "accuracy": float(
            np.mean(predictions == y)
        ),

        "sensitivity_dementia": float(sensitivity),
        "specificity_normal": float(specificity),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "dementia_f1": float(
            f1_score(
                y,
                predictions,
                pos_label=1,
                zero_division=0
            )
        ),

        "auroc": float(
            roc_auc_score(y, probabilities)
        ),

        "brier_vs_dementia_label": float(
            np.mean((probabilities - y) ** 2)
        ),

        "log_loss_vs_dementia_label": float(
            -np.mean(
                y * np.log(clipped)
                + (1 - y) * np.log(1 - clipped)
            )
        ),

        "ece_10_bins_at_standard_0_5_rule": (
            float(ece_10(y, probabilities))
            if threshold == 0.5
            else None
        ),

        "predicted_dementia_fraction": float(
            np.mean(predictions == 1)
        ),

        "confusion_matrix_rows_normal_dementia": [
            [tn, fp],
            [fn, tp]
        ]
    }


def summarize_distribution(values):
    scores = np.asarray(
        values, dtype=float
    )

    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "q25": float(np.quantile(scores, 0.25)),
        "q75": float(np.quantile(scores, 0.75)),
        "fraction_ge_0_5": float(
            np.mean(scores >= 0.5)
        )
    }


def temporal_reliability(cases, model):
    stds = []
    flips = []
    disagreements = []
    phase_agreement = []
    phase_vs_full = []

    stable_correct = 0
    stable_wrong = 0

    for case in cases:
        windows = case[model]["windows"]

        full_positive = (
            case[model]["score"] >= 0.5
        )

        window_labels = (
            windows >= 0.5
        )

        stds.append(
            float(np.std(windows))
        )

        flips.append(
            float(np.mean(
                window_labels[1:]
                != window_labels[:-1]
            ))
        )

        disagreements.append(
            float(np.mean(
                window_labels != full_positive
            ))
        )

        thirds = [
            float(np.mean(
                windows[start:start + 8]
            )) >= 0.5
            for start in (0, 8, 16)
        ]

        phase_agreement.append(
            len(set(thirds)) == 1
        )

        phase_vs_full.append(
            all(
                value == full_positive
                for value in thirds
            )
        )

        if (
            case["y"] is not None
            and np.all(
                window_labels == window_labels[0]
            )
        ):
            if int(full_positive) == case["y"]:
                stable_correct += 1
            else:
                stable_wrong += 1

    return {
        "all_test_recordings": len(cases),

        "binary_evaluable_recordings": sum(
            case["y"] is not None
            for case in cases
        ),

        "median_window_probability_sd": float(
            np.median(stds)
        ),

        "mean_sampled_window_label_flip_fraction": float(
            np.mean(flips)
        ),

        "mean_window_disagreement_with_recording_prediction": float(
            np.mean(disagreements)
        ),

        "early_middle_late_agreement_fraction": float(
            np.mean(phase_agreement)
        ),

        "three_phases_match_recording_prediction_fraction": float(
            np.mean(phase_vs_full)
        ),

        "all_24_windows_agree_and_correct_count":
            stable_correct,

        "all_24_windows_agree_but_wrong_count":
            stable_wrong,

        "note": (
            "The 24 windows were sampled across each recording, "
            "not consecutively. Stability is not correctness."
        )
    }


def paired_bootstrap(
    y,
    predictions_by_variant,
    resamples=5000,
    seed=20261109
):
    rng = np.random.default_rng(seed)

    normal_idx = np.flatnonzero(y == 0)
    dementia_idx = np.flatnonzero(y == 1)

    require(
        len(normal_idx) == 35
        and len(dementia_idx) == 22,
        "Invalid binary class counts"
    )

    draw_normal = rng.choice(
        normal_idx,
        size=(resamples, len(normal_idx)),
        replace=True
    )

    draw_dementia = rng.choice(
        dementia_idx,
        size=(resamples, len(dementia_idx)),
        replace=True
    )

    boot = {}

    for variant, predictions in predictions_by_variant.items():
        per_draw_bacc = (
            np.mean(
                predictions[draw_normal] == 0,
                axis=1
            )
            + np.mean(
                predictions[draw_dementia] == 1,
                axis=1
            )
        ) / 2

        boot[variant] = per_draw_bacc

    def interval(values):
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975))
        ]

    comparisons = {
        "DSRF_original_minus_XGBoost_original": (
            "DSRF/original",
            "XGBoost/original"
        ),

        "DSRF_calibrated_minus_DSRF_original": (
            "DSRF/calibrated",
            "DSRF/original"
        ),

        "XGBoost_threshold_minus_XGBoost_original": (
            "XGBoost/threshold",
            "XGBoost/original"
        ),

        "XGBoost_calibrated_minus_XGBoost_original": (
            "XGBoost/calibrated",
            "XGBoost/original"
        )
    }

    return {
        "method": "paired_stratified_recording_bootstrap",
        "resamples": resamples,
        "seed": seed,

        "balanced_accuracy_95ci_by_variant": {
            name: interval(values)
            for name, values in boot.items()
        },

        "paired_balanced_accuracy_difference_95ci": {
            name: interval(boot[a] - boot[b])
            for name, (a, b) in comparisons.items()
        },

        "caution": (
            "Recording-based intervals assume independent recordings. "
            "Patient identities are unverified, so uncertainty "
            "may be understated."
        )
    }


def evaluate(cases, plan):
    print(
        "\n3. CALCULATING FINAL TEST CLASSIFICATION METRICS",
        flush=True
    )

    binary = [
        case for case in cases
        if case["y"] is not None
    ]

    mci = [
        case for case in cases
        if case["y"] is None
    ]

    require(
        len(binary) == 57
        and len(mci) == 33,
        "Unexpected binary/MCI population"
    )

    y = np.asarray(
        [case["y"] for case in binary],
        dtype=np.int64
    )

    thresholds = plan[
        "secondary_exploratory_variants"
    ]["training_selected_thresholds"]

    platt = plan[
        "secondary_exploratory_variants"
    ]["platt_calibration_parameters"]

    variants = {}
    mci_results = {}
    reliability = {}
    prediction_vectors = {}

    for model in ("DSRF", "XGBoost"):
        raw = np.asarray([
            case[model]["score"]
            for case in binary
        ], dtype=float)

        calibrated = np.asarray([
            calibrated_score(
                score, platt[model]
            )
            for score in raw
        ], dtype=float)

        selected_threshold = float(
            thresholds[model]
        )

        designs = {
            "original": (raw, 0.5),

            "threshold": (
                raw, selected_threshold
            ),

            "calibrated": (
                calibrated, 0.5
            )
        }

        variants[model] = {}

        for name, (probabilities, cutoff) in designs.items():
            result = measure_binary(
                y,
                probabilities,
                cutoff
            )

            variants[model][name] = result

            prediction_vectors[
                model + "/" + name
            ] = (
                probabilities >= cutoff
            ).astype(int)

            print(
                f"\n{model} | {name} | threshold={cutoff:.2f}"
            )

            print(
                "  Balanced accuracy:",
                f"{result['balanced_accuracy']:.2%}"
            )

            print(
                "  Accuracy:",
                f"{result['accuracy']:.2%}"
            )

            print(
                "  Dementia sensitivity:",
                f"{result['sensitivity_dementia']:.2%}"
            )

            print(
                "  Normal specificity:",
                f"{result['specificity_normal']:.2%}"
            )

            print(
                "  Macro F1:",
                f"{result['macro_f1']:.4f}"
            )

            print(
                "  AUROC:",
                f"{result['auroc']:.4f}"
            )

            print(
                "  Brier score:",
                f"{result['brier_vs_dementia_label']:.4f}"
            )

            print(
                "  ECE at standard 0.5 rule:",
                result["ece_10_bins_at_standard_0_5_rule"]
            )

            print(
                "  Confusion matrix:",
                result[
                    "confusion_matrix_rows_normal_dementia"
                ]
            )

        require(
            np.isclose(
                variants[model]["original"]["auroc"],
                variants[model]["calibrated"]["auroc"],
                atol=1e-10
            ),
            "Monotonic calibration unexpectedly changed AUROC"
        )

        mci_raw = [
            case[model]["score"]
            for case in mci
        ]

        mci_calibrated = [
            calibrated_score(
                score, platt[model]
            )
            for score in mci_raw
        ]

        mci_results[model] = {
            "raw_source_ad_score":
                summarize_distribution(mci_raw),

            "calibrated_dementia_score":
                summarize_distribution(mci_calibrated),

            "raw_fraction_ge_training_selected_threshold":
                float(
                    np.mean(
                        np.asarray(mci_raw)
                        >= selected_threshold
                    )
                ),

            "no_binary_accuracy_claim_for_mci": True
        }

        reliability[model] = temporal_reliability(
            cases, model
        )

        print(
            f"\n{model} MCI descriptive analysis "
            "(33 recordings)"
        )

        print(
            "  Median original AD-score:",
            f"{np.median(mci_raw):.4f}"
        )

        print(
            "  Median calibrated dementia-score:",
            f"{np.median(mci_calibrated):.4f}"
        )

    print(
        "\n4. COMPUTING FINAL PAIRED BOOTSTRAP INTERVALS",
        flush=True
    )

    uncertainty = paired_bootstrap(
        y,
        prediction_vectors,
        resamples=int(
            plan["uncertainty"]["resamples"]
        ),
        seed=int(
            plan["uncertainty"]["seed"]
        )
    )

    for name in (
        "DSRF/original",
        "XGBoost/original"
    ):
        lower, upper = uncertainty[
            "balanced_accuracy_95ci_by_variant"
        ][name]

        print(
            f"{name} balanced accuracy 95% CI: "
            f"[{lower:.4f}, {upper:.4f}]"
        )

    lower, upper = uncertainty[
        "paired_balanced_accuracy_difference_95ci"
    ]["DSRF_original_minus_XGBoost_original"]

    print(
        "DSRF - XGBoost original balanced accuracy "
        f"95% CI: [{lower:.4f}, {upper:.4f}]"
    )

    print(
        "\n5. FINAL TEMPORAL RELIABILITY",
        flush=True
    )

    for model in ("DSRF", "XGBoost"):
        result = reliability[model]

        print("\nMODEL:", model)

        print(
            "Median window probability SD:",
            f"{result['median_window_probability_sd']:.4f}"
        )

        print(
            "Mean sampled-window label flip fraction:",
            f"{result['mean_sampled_window_label_flip_fraction']:.2%}"
        )

        print(
            "Stable and correct recordings:",
            result[
                "all_24_windows_agree_and_correct_count"
            ]
        )

        print(
            "Stable but WRONG recordings:",
            result[
                "all_24_windows_agree_but_wrong_count"
            ]
        )

    print(
        "\nConstant-classifier balanced accuracy baseline: 50.00%"
    )

    return {
        "model_variants": variants,

        "mci_descriptive": mci_results,

        "original_raw_score_temporal_reliability":
            reliability,

        "uncertainty": uncertainty,

        "constant_classifier_controls": {
            "always_normal_balanced_accuracy": 0.5,
            "always_normal_accuracy": 35 / 57,

            "always_dementia_balanced_accuracy": 0.5,
            "always_dementia_accuracy": 22 / 57
        }
    }


def private_csv(cases, plan):
    fieldnames = [
        "serial",
        "official_class",
        "binary_target"
    ]

    for model in ("DSRF", "XGBoost"):
        fieldnames += [
            model + suffix
            for suffix in (
                "_source_AD_score",
                "_original_0_5_prediction",
                "_training_threshold_prediction",
                "_calibrated_dementia_score",
                "_calibrated_0_5_prediction",
                "_window_score_sd",
                "_all_24_windows_same_label"
            )
        ]

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for case in cases:
        row = {
            "serial": case["serial"],
            "official_class": case["class_name"],

            "binary_target": (
                ""
                if case["y"] is None
                else case["y"]
            )
        }

        for model in ("DSRF", "XGBoost"):
            score = case[model]["score"]
            windows = case[model]["windows"]

            cutoff = float(
                plan["secondary_exploratory_variants"][
                    "training_selected_thresholds"
                ][model]
            )

            calibrated = calibrated_score(
                score,
                plan["secondary_exploratory_variants"][
                    "platt_calibration_parameters"
                ][model]
            )

            row.update({
                model + "_source_AD_score":
                    score,

                model + "_original_0_5_prediction":
                    int(score >= 0.5),

                model + "_training_threshold_prediction":
                    int(score >= cutoff),

                model + "_calibrated_dementia_score":
                    calibrated,

                model + "_calibrated_0_5_prediction":
                    int(calibrated >= 0.5),

                model + "_window_score_sd":
                    float(np.std(windows)),

                model + "_all_24_windows_same_label":
                    int(np.all(
                        (windows >= 0.5)
                        == (windows[0] >= 0.5)
                    ))
            })

        writer.writerow(row)

    return buffer.getvalue()


def main():
    plan, inference, plan_hash = verify_freeze()

    cases, bundle_hash = load_test_records(
        plan,
        inference,
        plan_hash
    )

    calculated = evaluate(
        cases,
        plan
    )

    report = {
        "status": "LOCKED_EXTERNAL_TEST_EVALUATION_COMPLETED",

        "final_test_plan_sha256": plan_hash,

        "code18_inference_script_sha256":
            inference["inference_code_sha256"],

        "locked_test_prediction_bundle_sha256":
            bundle_hash,

        "experiment": "CAUEEG_dementia-no-overlap",

        "analysis_level":
            "recording_only_not_verified_patients",

        "test_recordings_total": 90,

        "primary_binary_recordings": 57,
        "primary_normal_recordings": 35,
        "primary_dementia_recordings": 22,

        "secondary_mci_recordings": 33,

        "source_model_classes": ["AD", "CN"],
        "target_positive": "dementia",

        "model_weights_modified": False,
        "hyperparameters_reselected_after_test": False,

        "primary_variants": [
            "DSRF/original",
            "XGBoost/original"
        ],

        "secondary_variants_are_exploratory": True,

        "all_variants_reported_without_test_based_selection":
            True,

        "scientific_limitations": [
            "CAUEEG dementia is not equivalent to verified Alzheimer disease.",
            "The original AD score is not a validated clinical dementia probability.",
            "Patient identities are not independently verifiable; repeated recordings may reduce effective sample size.",
            "Confidence intervals bootstrap recordings, not independent people.",
            "Threshold adaptation and calibration were added after viewing initial validation outcomes.",
            "MCI recordings are excluded from binary accuracy calculations.",
            "Temporal stability does not imply prediction correctness.",
            "Test results must not be used for additional tuning or post-hoc variant selection."
        ],

        **calculated
    }

    print(
        "\n6. SAVING FINAL TEST RESULTS",
        flush=True
    )

    details_ok = save_once(
        DETAIL_FILE,
        private_csv(cases, plan)
    )

    summary_ok = save_once(
        REPORT_FILE,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False
        ) + "\n"
    )

    if details_ok and summary_ok:
        print(
            "\nFINAL_TEST_STATUS: EVALUATION_COMPLETED"
        )
    else:
        print(
            "\nFINAL_TEST_STATUS: OUTPUT_CONFLICT"
        )

    print("Original checkpoints unchanged.")
    print("No test-based model tuning performed.")
    print("Private recording details remain local.")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )

    print(
        "FINAL_TEST_STATUS: FAILED"
    )

finally:
    print(
        "\n========== CODE 19 FINISHED ==========",
        flush=True
    )
