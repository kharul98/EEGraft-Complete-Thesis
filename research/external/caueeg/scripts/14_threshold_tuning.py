from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import sys

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import StratifiedKFold

PREDICTION_DIR = (
    DATA / "derived/caueeg_transfer_v1/train"
)

INFERENCE_FILE = (
    WORK / "results/11_train_val_inference_summary.json"
)

TASK_FILE = (
    WORK / "protocol/caueeg_transfer_task_v1.json"
)

SUMMARY_FILE = (
    WORK / "results/14_train_only_threshold_tuning.json"
)

LOCK_FILE = (
    WORK / "protocol/caueeg_adapted_thresholds_v1.json"
)

GRID = np.round(
    np.arange(5, 96, dtype=float) / 100.0,
    2
)

SEED = 20261109
FOLDS = 5

MODEL_KEYS = {
    "DSRF": "dsrf_mean_p_ad",
    "XGBoost": "xgb_mean_p_ad"
}


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


def write_once(path, object_value):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        object_value,
        indent=2,
        sort_keys=True,
        allow_nan=False
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing file matches:", path.name)
            return True

        print("[PRESERVED] Different file already exists:", path)
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
        handle.write(content)

    print("[SAVED]", path)
    return True


def verify_protocol():
    print("\n1. VERIFYING FROZEN EXPERIMENT")

    require(
        INFERENCE_FILE.is_file()
        and TASK_FILE.is_file(),
        "Code 11 or frozen external task is missing"
    )

    inference = read_json(INFERENCE_FILE)
    task = read_json(TASK_FILE)

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Training inference is incomplete"
    )

    require(
        inference["test_recordings_processed"] == 0,
        "Unexpected test inference status"
    )

    require(
        task["primary_target_comparison"]["decision_threshold"]
        == 0.5,
        "Original frozen threshold has changed"
    )

    require(
        task["primary_target_comparison"]["positive"]
        == "dementia",
        "Unexpected target class"
    )

    expected_hashes = inference["model_hashes"]

    files = {
        "protocol_sha256":
            ROOT / "results/final_protocol_2026.json",
        "task_sha256": TASK_FILE,
        "pretrained_sha256":
            ROOT / "checkpoints/cbramod_official.pth",
        "dsrf_sha256":
            ROOT / "results/final_development_models_2026/dsrf/final.pt",
        "xgboost_sha256":
            ROOT / "results/final_development_models_2026/xgboost/final.joblib",
        "inference_script_sha256":
            WORK / "scripts/11_train_val_inference.py"
    }

    for key, path in files.items():
        require(
            path.is_file(),
            "Missing original experiment artifact: " + key
        )

        require(
            sha256(path) == expected_hashes[key],
            "Frozen experiment artifact has changed: " + key
        )

    print("[OK] Original model hashes match")
    print("[OK] Original zero-shot threshold remains 0.5")

    return expected_hashes


def load_training_data(expected_hashes):
    print("\n2. LOADING TRAINING PREDICTIONS ONLY")

    require(
        PREDICTION_DIR.is_dir(),
        "Training prediction directory missing"
    )

    files = sorted(
        PREDICTION_DIR.glob("*.json")
    )

    require(
        len(files) == 950,
        f"Expected 950 training files, found {len(files)}"
    )

    cases = []
    counts = Counter()
    serials = set()

    for path in files:
        item = read_json(path)

        require(
            item["split"] == "train",
            "Nontraining recording encountered"
        )

        require(
            item["version"] == "caueeg_transfer_v1",
            "Prediction version mismatch"
        )

        require(
            item["model_hashes"] == expected_hashes,
            "Prediction provenance mismatch"
        )

        serial = item["serial"]
        label = item["class_name"]

        require(
            label in ("normal", "mci", "dementia"),
            "Unexpected diagnostic category"
        )

        require(
            serial not in serials,
            "Duplicate training recording"
        )

        serials.add(serial)
        counts[label] += 1

        if label == "mci":
            continue

        case = {
            "label": int(label == "dementia")
        }

        for model, key in MODEL_KEYS.items():
            score = float(item[key])

            require(
                np.isfinite(score)
                and 0 <= score <= 1,
                "Invalid source model probability"
            )

            case[model] = score

        cases.append(case)

    require(
        counts == {
            "normal": 367,
            "mci": 334,
            "dementia": 249
        },
        "Unexpected training class distribution"
    )

    require(
        len(cases) == 616,
        "Incorrect binary training cohort"
    )

    print("Total training recordings:", 950)
    print("Normal used:", 367)
    print("Dementia used:", 249)
    print("MCI excluded from binary tuning:", 334)
    print("Validation predictions opened: 0")
    print("Test predictions opened: 0")

    y = np.asarray([
        case["label"] for case in cases
    ], dtype=np.int64)

    scores = {
        model: np.asarray([
            case[model] for case in cases
        ], dtype=float)
        for model in MODEL_KEYS
    }

    return y, scores


def score_threshold(y, scores, threshold):
    predictions = (
        scores >= threshold
    ).astype(int)

    normal = y == 0
    dementia = y == 1

    require(
        normal.any() and dementia.any(),
        "Both categories are necessary"
    )

    specificity = float(
        np.mean(predictions[normal] == 0)
    )

    sensitivity = float(
        np.mean(predictions[dementia] == 1)
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2.0

    return {
        "balanced_accuracy": float(balanced_accuracy),
        "dementia_sensitivity": sensitivity,
        "normal_specificity": specificity,
        "accuracy": float(
            np.mean(predictions == y)
        ),
        "positive_prediction_fraction": float(
            np.mean(predictions == 1)
        )
    }


def choose_threshold(y, scores):
    best_threshold = None
    best_score = -1.0

    for threshold in GRID:
        value = score_threshold(
            y, scores, float(threshold)
        )["balanced_accuracy"]

        if value > best_score + 1e-12:
            best_score = value
            best_threshold = float(threshold)

        elif abs(value - best_score) <= 1e-12:
            current_distance = abs(
                float(threshold) - 0.5
            )

            best_distance = abs(
                best_threshold - 0.5
            )

            if current_distance < best_distance - 1e-12:
                best_threshold = float(threshold)

    return best_threshold


def cross_validate(y, scores):
    splitter = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED
    )

    rows = []

    for fold_number, (train_idx, holdout_idx) in enumerate(
        splitter.split(scores, y),
        start=1
    ):
        threshold = choose_threshold(
            y[train_idx],
            scores[train_idx]
        )

        fold_metrics = score_threshold(
            y[holdout_idx],
            scores[holdout_idx],
            threshold
        )

        rows.append({
            "fold": fold_number,
            "selected_threshold": threshold,
            "heldout_recordings": int(len(holdout_idx)),
            "heldout_balanced_accuracy":
                fold_metrics["balanced_accuracy"],
            "heldout_sensitivity":
                fold_metrics["dementia_sensitivity"],
            "heldout_specificity":
                fold_metrics["normal_specificity"]
        })

    return {
        "folds": rows,
        "mean_balanced_accuracy": float(
            np.mean([
                row["heldout_balanced_accuracy"]
                for row in rows
            ])
        ),
        "sd_balanced_accuracy": float(
            np.std([
                row["heldout_balanced_accuracy"]
                for row in rows
            ])
        ),
        "fold_thresholds": [
            row["selected_threshold"]
            for row in rows
        ]
    }


def main():
    hashes = verify_protocol()

    y, model_scores = load_training_data(hashes)

    print("\n3. TRAINING-ONLY CROSS-VALIDATION")

    print("Folds:", FOLDS)
    print("Random seed:", SEED)
    print("Threshold candidate range: 0.05 to 0.95")
    print("Threshold increment: 0.01")
    print("Objective: balanced accuracy")
    print("Tie-break: threshold closest to 0.5")
    print("Original models are not being retrained.")

    results = {}
    chosen = {}

    for model, scores in model_scores.items():
        print("\nMODEL:", model)

        cv = cross_validate(y, scores)

        final_threshold = choose_threshold(
            y, scores
        )

        baseline = score_threshold(
            y, scores, 0.5
        )

        adapted = score_threshold(
            y, scores, final_threshold
        )

        chosen[model] = final_threshold

        results[model] = {
            "cross_validation": cv,
            "selected_full_training_threshold":
                final_threshold,
            "full_training_fixed_0_5": baseline,
            "full_training_adapted": adapted
        }

        print(
            "Training-CV mean BAcc:",
            f"{cv['mean_balanced_accuracy']:.2%}"
        )

        print(
            "Training-CV fold thresholds:",
            cv["fold_thresholds"]
        )

        print(
            "Original threshold:",
            0.5
        )

        print(
            "Adapted threshold:",
            final_threshold
        )

        print(
            "Full-training BAcc at 0.5:",
            f"{baseline['balanced_accuracy']:.2%}"
        )

        print(
            "Full-training BAcc adapted:",
            f"{adapted['balanced_accuracy']:.2%}"
        )

    print("\n4. SAVING SEPARATE ADAPTATION PROTOCOL")

    locked = {
        "version": "caueeg_adapted_thresholds_v1",
        "original_track": {
            "models": ["DSRF", "XGBoost"],
            "threshold": 0.5,
            "model_weights_modified": False
        },
        "adapted_track": {
            "method": "training_only_threshold_selection",
            "training_classes": ["normal", "dementia"],
            "training_recordings": 616,
            "selection_metric": "balanced_accuracy",
            "threshold_candidates": [
                float(value) for value in GRID
            ],
            "cross_validation_folds": FOLDS,
            "cross_validation_seed": SEED,
            "selected_thresholds": chosen,
            "source_models_retrained": False,
            "validation_used_for_tuning": False,
            "test_used_for_tuning": False
        },
        "source_model_hashes": hashes,
        "patient_level_claims_permitted": False,
        "interpretation": (
            "Supervised decision-threshold adaptation "
            "on CAUEEG training recordings. "
            "This is no longer zero-shot threshold selection."
        ),
        "limitations": [
            "The selected thresholds are not clinical cutoffs.",
            "No independently verified patient identifiers are available.",
            "Training cross-validation is recording-level.",
            "Repeated patients may make cross-validation optimistic.",
            "The decision to add this analysis came after initial validation results were viewed."
        ]
    }

    report = {
        "status": "TRAIN_ONLY_THRESHOLD_SEARCH_COMPLETED",
        "models": results,
        "selected_thresholds": chosen,
        "test_recordings_accessed": 0,
        "validation_predictions_accessed": 0,
        "source_model_hashes": hashes
    }

    ok1 = write_once(LOCK_FILE, locked)
    ok2 = write_once(SUMMARY_FILE, report)

    if ok1 and ok2:
        print(
            "\nTUNING_STATUS: "
            "TRAIN_ONLY_THRESHOLD_SEARCH_COMPLETED"
        )
    else:
        print(
            "\nTUNING_STATUS: OUTPUT_CONFLICT"
        )

    print("\nFrozen zero-shot models: unchanged")
    print("Original 0.5-threshold baseline: preserved")
    print("Validation predictions accessed: 0")
    print("Locked test predictions accessed: 0")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc)
    )

    print("TUNING_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 14 FINISHED =========="
    )
