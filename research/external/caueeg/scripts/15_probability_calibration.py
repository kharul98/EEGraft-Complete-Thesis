from pathlib import Path
from collections import Counter
import hashlib
import json
import os

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

TRAIN_DIR = DATA / "derived/caueeg_transfer_v1/train"

INFERENCE_FILE = (
    WORK / "results/11_train_val_inference_summary.json"
)

TUNING_FILE = (
    WORK / "results/14_train_only_threshold_tuning.json"
)

THRESHOLD_LOCK = (
    WORK / "protocol/caueeg_adapted_thresholds_v1.json"
)

TASK_FILE = (
    WORK / "protocol/caueeg_transfer_task_v1.json"
)

CALIBRATION_LOCK = (
    WORK / "protocol/caueeg_platt_calibration_v1.json"
)

REPORT_FILE = (
    WORK / "results/15_train_only_calibration.json"
)

MODEL_KEYS = {
    "DSRF": "dsrf_mean_p_ad",
    "XGBoost": "xgb_mean_p_ad"
}

SEED = 20261109
FOLDS = 5
REGULARIZATION_C = 1.0
EPSILON = 1e-6


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
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data,
        indent=2,
        sort_keys=True,
        allow_nan=False
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing output matches:", path.name)
            return True

        print("[PRESERVED] Different output already exists:", path)
        return False

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(descriptor, "w", encoding="utf-8") as f:
        f.write(content)

    print("[SAVED]", path)
    return True


def verify_experiment():
    print("\n1. VERIFYING EXISTING FROZEN EXPERIMENT")

    required = [
        INFERENCE_FILE,
        TUNING_FILE,
        THRESHOLD_LOCK,
        TASK_FILE
    ]

    for path in required:
        require(
            path.is_file(),
            f"Required file missing: {path}"
        )

    inference = read_json(INFERENCE_FILE)
    tuning = read_json(TUNING_FILE)
    thresholds = read_json(THRESHOLD_LOCK)
    task = read_json(TASK_FILE)

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Code 11 is incomplete"
    )

    require(
        tuning["status"]
        == "TRAIN_ONLY_THRESHOLD_SEARCH_COMPLETED",
        "Code 14 is incomplete"
    )

    require(
        tuning["validation_predictions_accessed"] == 0
        and tuning["test_recordings_accessed"] == 0,
        "Threshold selection was not training-only"
    )

    require(
        thresholds["adapted_track"]["selected_thresholds"]
        == tuning["selected_thresholds"],
        "Threshold lock and tuning report disagree"
    )

    require(
        thresholds["source_model_hashes"]
        == inference["model_hashes"]
        == tuning["source_model_hashes"],
        "Model provenance mismatch"
    )

    require(
        task["primary_target_comparison"]["positive"]
        == "dementia"
        and task["primary_target_comparison"]["negative"]
        == "normal"
        and task["primary_target_comparison"]["decision_threshold"]
        == 0.5,
        "Original task definition changed"
    )

    hashes = inference["model_hashes"]

    original_files = {
        "protocol_sha256":
            ROOT / "results/final_protocol_2026.json",
        "task_sha256":
            TASK_FILE,
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
            path.is_file() and sha256(path) == hashes[key],
            f"Frozen source artifact changed: {key}"
        )

    require(
        inference["test_recordings_processed"] == 0,
        "Unexpected external test use"
    )

    print("[OK] Frozen source models unchanged")
    print("[OK] Training-only threshold tuning verified")
    print("[OK] Original 0.5 threshold preserved")

    return hashes


def load_training_data(hashes):
    print("\n2. LOADING TRAINING-ONLY SCORES")

    require(
        TRAIN_DIR.is_dir(),
        "Training prediction folder is missing"
    )

    paths = sorted(TRAIN_DIR.glob("*.json"))

    require(
        len(paths) == 950,
        f"Expected 950 training files, found {len(paths)}"
    )

    counts = Counter()
    cases = []
    seen = set()

    for path in paths:
        item = read_json(path)

        require(
            item["split"] == "train",
            "Nontraining recording encountered"
        )

        require(
            item["version"] == "caueeg_transfer_v1",
            "Unexpected prediction version"
        )

        require(
            item["model_hashes"] == hashes,
            "Prediction provenance mismatch"
        )

        require(
            item["source_model_classes"] == ["AD", "CN"],
            "Unexpected source-model class order"
        )

        require(
            item["window_count"] == 24,
            "Unexpected window count"
        )

        serial = item["serial"]
        label = item["class_name"]

        require(
            path.stem == serial,
            "Prediction filename/serial mismatch"
        )

        require(
            serial not in seen,
            "Duplicate training recording"
        )

        require(
            label in ("normal", "mci", "dementia"),
            "Unexpected target class"
        )

        seen.add(serial)
        counts[label] += 1

        if label == "mci":
            continue

        record = {
            "target": int(label == "dementia")
        }

        for model, key in MODEL_KEYS.items():
            value = float(item[key])

            require(
                np.isfinite(value)
                and 0.0 <= value <= 1.0,
                "Invalid source score"
            )

            record[model] = value

        cases.append(record)

    require(
        counts == {
            "normal": 367,
            "mci": 334,
            "dementia": 249
        },
        "Training class distribution changed"
    )

    require(
        len(cases) == 616,
        "Unexpected calibration cohort size"
    )

    y = np.asarray(
        [case["target"] for case in cases],
        dtype=np.int64
    )

    scores = {
        model: np.asarray(
            [case[model] for case in cases],
            dtype=np.float64
        )
        for model in MODEL_KEYS
    }

    print("Training recordings available:", 950)
    print("Normal calibration cases:", 367)
    print("Dementia calibration cases:", 249)
    print("MCI excluded:", 334)
    print("Validation recordings accessed: 0")
    print("Test recordings accessed: 0")

    return y, scores


def logit_transform(scores):
    clipped = np.clip(
        np.asarray(scores, dtype=float),
        EPSILON,
        1.0 - EPSILON
    )

    return np.log(
        clipped / (1.0 - clipped)
    ).reshape(-1, 1)


def make_calibrator():
    return LogisticRegression(
        C=REGULARIZATION_C,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
        class_weight=None
    )


def metrics(y, probabilities):
    probabilities = np.asarray(
        probabilities, dtype=float
    )

    require(
        len(probabilities) == len(y)
        and np.isfinite(probabilities).all(),
        "Invalid evaluation probabilities"
    )

    require(
        np.all(
            (probabilities >= 0)
            & (probabilities <= 1)
        ),
        "Probabilities outside [0, 1]"
    )

    p = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    normal = y == 0
    dementia = y == 1

    sensitivity = float(
        np.mean(predictions[dementia] == 1)
    )

    specificity = float(
        np.mean(predictions[normal] == 0)
    )

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    correctness = (
        predictions == y
    ).astype(float)

    ece = 0.0

    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0

        if index == 9:
            in_bin = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            in_bin = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if in_bin.any():
            ece += float(
                in_bin.mean()
                * abs(
                    confidence[in_bin].mean()
                    - correctness[in_bin].mean()
                )
            )

    logloss = -np.mean(
        y * np.log(p)
        + (1 - y) * np.log(1.0 - p)
    )

    return {
        "n_recordings": int(len(y)),
        "brier_score": float(
            np.mean((probabilities - y) ** 2)
        ),
        "log_loss": float(logloss),
        "ece_10_bins": float(ece),
        "auroc": float(
            roc_auc_score(y, probabilities)
        ),
        "balanced_accuracy_at_0_5": float(
            (sensitivity + specificity) / 2.0
        ),
        "dementia_sensitivity_at_0_5": sensitivity,
        "normal_specificity_at_0_5": specificity,
        "mean_probability": float(
            np.mean(probabilities)
        )
    }


def cross_validate_calibration(y, scores):
    splitter = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED
    )

    x = logit_transform(scores)

    oof_calibrated = np.full(
        len(y), np.nan, dtype=float
    )

    oof_prevalence = np.full(
        len(y), np.nan, dtype=float
    )

    folds = []

    for number, (train_idx, heldout_idx) in enumerate(
        splitter.split(x, y),
        start=1
    ):
        model = make_calibrator()

        model.fit(
            x[train_idx],
            y[train_idx]
        )

        require(
            np.array_equal(model.classes_, [0, 1]),
            "Unexpected calibration class order"
        )

        calibrated = model.predict_proba(
            x[heldout_idx]
        )[:, 1]

        oof_calibrated[heldout_idx] = calibrated

        oof_prevalence[heldout_idx] = float(
            np.mean(y[train_idx])
        )

        fold_metrics = metrics(
            y[heldout_idx],
            calibrated
        )

        folds.append({
            "fold": number,
            "training_recordings": int(len(train_idx)),
            "heldout_recordings": int(len(heldout_idx)),
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "heldout_brier_score": fold_metrics["brier_score"],
            "heldout_log_loss": fold_metrics["log_loss"]
        })

    require(
        np.isfinite(oof_calibrated).all()
        and np.isfinite(oof_prevalence).all(),
        "Incomplete cross-validation predictions"
    )

    return {
        "fold_details": folds,
        "raw_source_scores": metrics(
            y, scores
        ),
        "out_of_fold_calibrated": metrics(
            y, oof_calibrated
        ),
        "out_of_fold_prevalence_baseline": metrics(
            y, oof_prevalence
        )
    }


def train_final_calibrator(y, scores):
    x = logit_transform(scores)

    model = make_calibrator()

    model.fit(x, y)

    require(
        np.array_equal(model.classes_, [0, 1]),
        "Unexpected fitted class order"
    )

    coefficient = float(
        model.coef_[0, 0]
    )

    intercept = float(
        model.intercept_[0]
    )

    require(
        np.isfinite(coefficient)
        and np.isfinite(intercept),
        "Nonfinite calibration parameters"
    )

    require(
        coefficient > 0,
        "Calibration slope is not positive; "
        "source-score direction requires investigation"
    )

    fitted_probabilities = expit(
        coefficient * x[:, 0] + intercept
    )

    return {
        "coefficient": coefficient,
        "intercept": intercept,
        "full_training_metrics_in_sample": metrics(
            y, fitted_probabilities
        )
    }


def main():
    hashes = verify_experiment()

    y, scores = load_training_data(hashes)

    print("\n3. FIXED CALIBRATION METHOD")

    print("Method: logistic regression on logit(source score)")
    print("Regularization: L2")
    print("C:", REGULARIZATION_C)
    print("Class weighting: none")
    print("Cross-validation folds:", FOLDS)
    print("Cross-validation seed:", SEED)
    print("Positive target: CAUEEG Dementia")
    print("Patient linkage: independently unverified")

    results = {}
    parameters = {}

    for model_name, source_scores in scores.items():
        print("\nMODEL:", model_name)

        cv = cross_validate_calibration(
            y, source_scores
        )

        final = train_final_calibrator(
            y, source_scores
        )

        raw = cv["raw_source_scores"]
        calibrated = cv["out_of_fold_calibrated"]
        baseline = cv["out_of_fold_prevalence_baseline"]

        print(
            "Raw source-score Brier:",
            f"{raw['brier_score']:.4f}"
        )

        print(
            "Cross-validated calibrated Brier:",
            f"{calibrated['brier_score']:.4f}"
        )

        print(
            "Cross-validated prevalence baseline Brier:",
            f"{baseline['brier_score']:.4f}"
        )

        print(
            "Raw source-score log loss:",
            f"{raw['log_loss']:.4f}"
        )

        print(
            "Cross-validated calibrated log loss:",
            f"{calibrated['log_loss']:.4f}"
        )

        print(
            "Raw score ECE:",
            f"{raw['ece_10_bins']:.4f}"
        )

        print(
            "Cross-validated calibrated ECE:",
            f"{calibrated['ece_10_bins']:.4f}"
        )

        print(
            "Cross-validated calibrated balanced accuracy:",
            f"{calibrated['balanced_accuracy_at_0_5']:.2%}"
        )

        print(
            "Final calibration coefficient:",
            round(final["coefficient"], 6)
        )

        print(
            "Final calibration intercept:",
            round(final["intercept"], 6)
        )

        results[model_name] = {
            "cross_validation": cv,
            "final_training_fit": final
        }

        parameters[model_name] = {
            "coefficient": final["coefficient"],
            "intercept": final["intercept"]
        }

    print("\n4. FREEZING CALIBRATION PARAMETERS")

    lock = {
        "version": "caueeg_platt_calibration_v1",
        "experiment_type": "exploratory_supervised_adaptation",
        "dataset": "CAUEEG",
        "target_negative_class": "normal",
        "target_positive_class": "dementia",
        "excluded_class": "mci",
        "source_model_classes": ["AD", "CN"],
        "method": "logistic_on_source_score_log_odds",
        "formula": (
            "p_dementia = sigmoid("
            "coefficient * logit(clipped_p_AD) + intercept)"
        ),
        "clipping_epsilon": EPSILON,
        "solver": "lbfgs",
        "regularization": "L2",
        "C": REGULARIZATION_C,
        "class_weight": None,
        "training_recordings": 616,
        "cv_folds": FOLDS,
        "cv_seed": SEED,
        "evaluation_level": "recording",
        "patient_identity_verified": False,
        "calibration_parameters": parameters,
        "source_model_hashes": hashes,
        "original_model_weights_modified": False,
        "original_threshold_track_modified": False,
        "validation_used_to_fit_calibration": False,
        "test_used_to_fit_calibration": False,
        "test_evaluated": False,
        "caution": (
            "This calibration adapts source AD scores to "
            "the CAUEEG Dementia-versus-Normal task. "
            "The adaptation was introduced after inspecting "
            "earlier validation results, so it must be "
            "identified as exploratory."
        )
    }

    report = {
        "status": "TRAIN_ONLY_CALIBRATION_COMPLETED",
        "version": "caueeg_platt_calibration_v1",
        "evaluation_unit": "recording",
        "training_recordings": 616,
        "validation_predictions_accessed": 0,
        "test_predictions_accessed": 0,
        "source_model_hashes": hashes,
        "models": results,
        "limitations": [
            "Calibration is for broad dementia, not verified AD.",
            "No patient IDs are available for independent grouping.",
            "Recording-level CV may be optimistic with repeated patients.",
            "Calibration does not demonstrate clinical reliability.",
            "Full-training fitted performance is in-sample.",
            "The calibration experiment was added after validation review."
        ]
    }

    lock_ok = write_once(
        CALIBRATION_LOCK, lock
    )

    report_ok = write_once(
        REPORT_FILE, report
    )

    if lock_ok and report_ok:
        print("\nCALIBRATION_STATUS: COMPLETED")
    else:
        print("\nCALIBRATION_STATUS: OUTPUT_CONFLICT")

    print("Original EEGraft model weights: unchanged")
    print("Original 0.5 threshold: unchanged")
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

    print("CALIBRATION_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 15 FINISHED =========="
    )
