from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import os

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

RESULTS = WORK / "results"
PROTOCOL = WORK / "protocol"

FILES = {
    "audit07": RESULTS / "07_metadata_audit_summary.json",
    "audit08": RESULTS / "08_edf_header_summary.json",
    "smoke09": RESULTS / "09_waveform_smoke_summary.json",
    "smoke10": RESULTS / "10_model_smoke_summary.json",
    "inference11": RESULTS / "11_train_val_inference_summary.json",
    "evaluation12": RESULTS / "12_development_evaluation.json",
    "reliability13": RESULTS / "13_validation_reliability.json",
    "tuning14": RESULTS / "14_train_only_threshold_tuning.json",
    "calibration15": RESULTS / "15_train_only_calibration.json",
    "comparison16": RESULTS / "16_validation_variant_comparison.json",
    "source_snapshot": PROTOCOL / "original_model_snapshot.json",
    "task": PROTOCOL / "caueeg_transfer_task_v1.json",
    "thresholds": PROTOCOL / "caueeg_adapted_thresholds_v1.json",
    "platt": PROTOCOL / "caueeg_platt_calibration_v1.json",
    "layout": PROTOCOL / "dataset_layout.json",
    "manifest": DATA / "manifests/07_recording_audit.csv",
}

FINAL_LOCK = PROTOCOL / "caueeg_final_test_plan_v1.json"

FINAL_SUMMARY = RESULTS / "17_final_test_preflight.json"

TEST_OUTPUT = DATA / "derived/caueeg_transfer_v1/test"

EXPECTED = {
    "train": {
        "normal": 367,
        "mci": 334,
        "dementia": 249
    },
    "validation": {
        "normal": 34,
        "mci": 28,
        "dementia": 20
    },
    "test": {
        "normal": 35,
        "mci": 33,
        "dementia": 22
    }
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data,
        indent=2,
        sort_keys=True,
        allow_nan=False
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing frozen file matches:", path.name)
            return True

        print("[BLOCKED] Existing frozen file differs:", path)
        print("The existing file was NOT overwritten.")
        return False

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(
        descriptor, "w", encoding="utf-8"
    ) as handle:
        handle.write(content)

    print("[FROZEN]", path)

    return True


def verify_development():
    print("\n1. VERIFYING CODES 07 THROUGH 16")

    for name, path in FILES.items():
        require(
            path.is_file(),
            f"Missing required artifact ({name}): {path}"
        )

    required_statuses = {
        "audit07": ("audit_status", "READY_FOR_EDF_QC"),
        "audit08": ("audit_status", "READY_FOR_WAVEFORM_SMOKE_TEST"),
        "smoke09": (
            "smoke_status",
            "READY_FOR_FROZEN_MODEL_INFERENCE_SETUP"
        ),
        "smoke10": ("status", "FROZEN_MODELS_VERIFIED"),
        "inference11": (
            "status",
            "TRAIN_VALIDATION_INFERENCE_COMPLETED"
        ),
        "evaluation12": (
            "status",
            "DEVELOPMENT_EVALUATION_COMPLETED"
        ),
        "reliability13": (
            "status",
            "VALIDATION_RELIABILITY_COMPLETED"
        ),
        "tuning14": (
            "status",
            "TRAIN_ONLY_THRESHOLD_SEARCH_COMPLETED"
        ),
        "calibration15": (
            "status",
            "TRAIN_ONLY_CALIBRATION_COMPLETED"
        ),
        "comparison16": (
            "status",
            "VALIDATION_VARIANTS_COMPARED"
        )
    }

    documents = {}

    for name in (
        "audit07", "audit08", "smoke09", "smoke10",
        "inference11", "evaluation12", "reliability13",
        "tuning14", "calibration15", "comparison16",
        "source_snapshot", "task", "thresholds",
        "platt", "layout"
    ):
        documents[name] = load_json(FILES[name])

    for name, (key, expected) in required_statuses.items():
        require(
            documents[name].get(key) == expected,
            f"Unexpected status in {name}"
        )

        print("[VERIFIED]", name)

    require(
        documents["inference11"]["test_recordings_processed"] == 0,
        "Code 11 reports test access"
    )

    require(
        documents["evaluation12"]["test_recordings_evaluated"] == 0,
        "Code 12 reports test access"
    )

    require(
        documents["reliability13"]["test_prediction_files_read"] == 0,
        "Code 13 reports test access"
    )

    require(
        documents["tuning14"]["test_recordings_accessed"] == 0,
        "Code 14 reports test access"
    )

    require(
        documents["calibration15"]["test_predictions_accessed"] == 0,
        "Code 15 reports test access"
    )

    require(
        documents["comparison16"]["test_recordings_accessed"] == 0,
        "Code 16 reports test access"
    )

    print("[OK] Previous experiments report no test inference")

    return documents


def verify_original_models(documents):
    print("\n2. VERIFYING ORIGINAL MODEL HASHES")

    model_hashes = documents["inference11"]["model_hashes"]

    model_paths = {
        "protocol_sha256":
            ROOT / "results/final_protocol_2026.json",

        "task_sha256":
            FILES["task"],

        "pretrained_sha256":
            ROOT / "checkpoints/cbramod_official.pth",

        "dsrf_sha256":
            ROOT / "results/final_development_models_2026/dsrf/final.pt",

        "xgboost_sha256":
            ROOT / "results/final_development_models_2026/xgboost/final.joblib",

        "inference_script_sha256":
            WORK / "scripts/11_train_val_inference.py"
    }

    for key, path in model_paths.items():
        require(path.is_file(), f"Missing: {path}")

        require(
            sha256(path) == model_hashes[key],
            f"Original model provenance mismatch: {key}"
        )

        print("[VERIFIED]", key)

    require(
        documents["source_snapshot"]["thesis_root"]
        == str(ROOT),
        "Original thesis root changed"
    )

    for relative, metadata in (
        documents["source_snapshot"]["files"].items()
    ):
        path = ROOT / relative

        require(path.is_file(), f"Missing: {relative}")

        require(
            sha256(path) == metadata["sha256"],
            f"Original source file changed: {relative}"
        )

    for name, key in (
        ("evaluation12", "provenance"),
        ("reliability13", "model_hashes"),
        ("tuning14", "source_model_hashes"),
        ("calibration15", "source_model_hashes"),
        ("comparison16", "original_model_hashes"),
        ("thresholds", "source_model_hashes"),
        ("platt", "source_model_hashes")
    ):
        require(
            documents[name][key] == model_hashes,
            f"Model provenance disagreement: {name}"
        )

    require(
        documents["comparison16"]["training_threshold_lock_sha256"]
        == sha256(FILES["thresholds"]),
        "Threshold lock has changed"
    )

    require(
        documents["comparison16"]["training_calibration_lock_sha256"]
        == sha256(FILES["platt"]),
        "Calibration lock has changed"
    )

    print("[OK] All original model and adaptation hashes match")

    return model_hashes


def verify_test_manifest(documents):
    print("\n3. VERIFYING OFFICIAL SPLITS")

    with FILES["manifest"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    require(
        len(rows) == 1122,
        "Expected 1122 benchmark recordings"
    )

    serials = set()
    counts = {}

    for row in rows:
        serial = row["serial"]

        require(
            serial not in serials,
            "Duplicate recording across official splits"
        )

        serials.add(serial)

        require(
            row["split"] in EXPECTED,
            "Unknown official split"
        )

        require(
            row["class_name"] in ("normal", "mci", "dementia"),
            "Unexpected diagnostic category"
        )

        require(
            row["edf_exists"] == "1"
            and row["annotation_exists"] == "1",
            "Missing official recording or annotation"
        )

    for split in ("train", "validation", "test"):
        counts[split] = dict(Counter(
            row["class_name"]
            for row in rows
            if row["split"] == split
        ))

        require(
            counts[split] == EXPECTED[split],
            f"Unexpected split counts for {split}"
        )

        print(split, counts[split])

    require(
        sum(counts["test"].values()) == 90,
        "Test cohort should contain 90 recordings"
    )

    require(
        counts["test"]["normal"]
        + counts["test"]["dementia"] == 57,
        "Primary test cohort must contain 57 recordings"
    )

    layout = documents["layout"]

    require(
        Path(layout["split_file"]).is_file()
        and Path(layout["annotation"]).is_file(),
        "Original CAUEEG metadata missing"
    )

    print("[OK] Official benchmark split counts verified")
    print("[OK] No overlapping recording serials")
    print("Primary test cases: 57")
    print("MCI descriptive test cases: 33")

    return counts


def verify_adaptation(documents):
    print("\n4. VERIFYING ADAPTED CONFIGURATIONS")

    task = documents["task"]
    threshold_lock = documents["thresholds"]
    calibration_lock = documents["platt"]

    primary = task["primary_target_comparison"]

    require(
        primary["negative"] == "normal"
        and primary["positive"] == "dementia"
        and primary["excluded_from_binary_metrics"] == "mci",
        "Primary external comparison changed"
    )

    require(
        primary["level"] == "recording"
        and primary["decision_threshold"] == 0.5,
        "Primary analysis unit or threshold changed"
    )

    selected = threshold_lock[
        "adapted_track"
    ]["selected_thresholds"]

    require(
        selected == documents["tuning14"]["selected_thresholds"],
        "Selected thresholds differ from training results"
    )

    require(
        math.isclose(float(selected["DSRF"]), 0.5)
        and math.isclose(float(selected["XGBoost"]), 0.7),
        "Unexpected final training-selected thresholds"
    )

    coefficients = calibration_lock[
        "calibration_parameters"
    ]

    for model in ("DSRF", "XGBoost"):
        require(
            model in coefficients,
            "Missing calibration for " + model
        )

        slope = float(
            coefficients[model]["coefficient"]
        )

        intercept = float(
            coefficients[model]["intercept"]
        )

        training_fit = (
            documents["calibration15"]["models"][model]
            ["final_training_fit"]
        )

        require(
            math.isfinite(slope)
            and slope > 0
            and math.isfinite(intercept),
            "Invalid calibration coefficients"
        )

        require(
            math.isclose(
                slope,
                float(training_fit["coefficient"]),
                abs_tol=1e-10
            )
            and math.isclose(
                intercept,
                float(training_fit["intercept"]),
                abs_tol=1e-10
            ),
            "Calibration parameters differ from training report"
        )

        print(
            f"{model}: threshold={selected[model]}, "
            f"Platt slope={slope:.6f}, intercept={intercept:.6f}"
        )

    print("[OK] Adapted configurations verified")

    return selected, coefficients


def main():
    documents = verify_development()

    model_hashes = verify_original_models(documents)

    counts = verify_test_manifest(documents)

    thresholds, calibration = verify_adaptation(documents)

    print("\n5. CHECKING FOR EARLIER TEST OUTPUTS")

    if TEST_OUTPUT.exists():
        existing = list(TEST_OUTPUT.glob("*.json"))

        require(
            not existing,
            "External test prediction files already exist. "
            "Cannot create a new pre-test freeze."
        )

    print("[OK] No existing test prediction files found")

    print("\n6. FREEZING FINAL EVALUATION PLAN")

    plan = {
        "version": "caueeg_final_test_plan_v1",

        "dataset": "CAUEEG",
        "official_task": "dementia-no-overlap",

        "test_recordings": 90,
        "test_class_counts": counts["test"],

        "primary_binary_test_recordings": 57,
        "secondary_mci_recordings": 33,

        "analysis_unit": "recording",

        "independent_patient_linkage_verified": False,
        "patient_level_performance_claims_allowed": False,

        "primary_comparison": {
            "models": ["DSRF", "XGBoost"],

            "model_weights": "original_frozen",

            "source_classes": ["AD", "CN"],

            "target_negative": "normal",
            "target_positive": "dementia",

            "excluded_from_primary_binary_metrics": "mci",

            "DSRF_threshold": 0.5,
            "XGBoost_threshold": 0.5,

            "decision_rule": "source_p_AD >= 0.5",

            "primary_metric": "balanced_accuracy",

            "secondary_metrics": [
                "accuracy",
                "macro_f1",
                "dementia_sensitivity",
                "normal_specificity",
                "auroc",
                "brier_score",
                "ece_10_bins",
                "confusion_matrix"
            ]
        },

        "secondary_exploratory_variants": {
            "training_selected_thresholds": thresholds,

            "platt_calibration_parameters": calibration,

            "calibration_input": (
                "mean recording-level source AD probability"
            ),

            "calibration_formula": (
                "sigmoid(coefficient * logit("
                "clip(p_AD, 1e-6, 1-1e-6)) + intercept)"
            ),

            "calibrated_probability_threshold": 0.5,

            "report_all_variants_without_test_selection": True,

            "adaptation_introduced_after_initial_validation_review":
                True
        },

        "temporal_reliability_analysis": {
            "windows_per_recording": 24,
            "window_seconds": 4,

            "window_selection": (
                "uniformly spaced nonoverlapping windows"
            ),

            "metrics": [
                "median_window_probability_sd",
                "sampled_window_label_flip_fraction",
                "stable_but_wrong_recording_count",
                "early_middle_late_prediction_agreement"
            ],

            "consistency_not_equivalent_to_correctness": True
        },

        "mci_analysis": {
            "type": "descriptive_only",
            "classification_accuracy_computed": False
        },

        "uncertainty": {
            "method": "paired_stratified_recording_bootstrap",
            "resamples": 5000,
            "seed": 20261109,
            "patient_independence_unverified": True
        },

        "safety": {
            "no_further_hyperparameter_tuning_after_test": True,
            "no_retraining_original_models": True,
            "no_threshold_selection_using_test": True,
            "no_variant_selection_using_test": True,
            "no_patient_level_claims": True,
            "no_restricted_data_uploads": True
        },

        "model_hashes": model_hashes,

        "artifact_hashes": {
            "official_split_json_sha256": sha256(
                Path(documents["layout"]["split_file"])
            ),

            "annotation_json_sha256": sha256(
                Path(documents["layout"]["annotation"])
            ),

            "recording_manifest_sha256": sha256(
                FILES["manifest"]
            ),

            "edf_header_audit_sha256": sha256(
                FILES["audit08"]
            ),

            "training_inference_summary_sha256": sha256(
                FILES["inference11"]
            ),

            "threshold_lock_sha256": sha256(
                FILES["thresholds"]
            ),

            "calibration_lock_sha256": sha256(
                FILES["platt"]
            ),

            "validation_comparison_sha256": sha256(
                FILES["comparison16"]
            )
        },

        "scientific_interpretation": (
            "Exploratory recording-level external transfer "
            "assessment of an original AD/CN classifier on "
            "broader CAUEEG Dementia and Normal categories. "
            "Not an Alzheimer's diagnostic validation."
        )
    }

    locked = save_once(FINAL_LOCK, plan)

    if not locked:
        print("FINAL_TEST_STATUS: LOCK_CONFLICT")
        return

    summary = {
        "status": "READY_FOR_LOCKED_TEST_INFERENCE",
        "plan_sha256": sha256(FINAL_LOCK),

        "total_test_recordings": 90,
        "binary_test_recordings": 57,
        "mci_descriptive_recordings": 33,

        "original_models_preserved": True,
        "frozen_thresholds_preserved": True,

        "test_waveforms_loaded_by_this_code": 0,
        "test_model_predictions_generated_by_this_code": 0
    }

    saved = save_once(FINAL_SUMMARY, summary)

    if not saved:
        print("FINAL_TEST_STATUS: SUMMARY_CONFLICT")
        return

    print("\n7. FINAL VERIFICATION")
    print("Original frozen model weights: unchanged")
    print("Primary threshold: 0.5 for both models")
    print("Exploratory XGBoost threshold: 0.7")
    print("Calibration: training-fitted parameters frozen")
    print("Test waveform files opened: 0")
    print("Test predictions generated: 0")

    print("\nFINAL_TEST_STATUS: READY_FOR_LOCKED_TEST_INFERENCE")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc)
    )

    print("FINAL_TEST_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 17 FINISHED =========="
    )
