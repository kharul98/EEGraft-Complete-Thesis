from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import os
import sys

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research" / "eegraft-caueeg"
DATA = HOME / "research" / "caueeg"

import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)

INFERENCE_SUMMARY = (
    WORK / "results/11_train_val_inference_summary.json"
)

TASK_LOCK = (
    WORK / "protocol/caueeg_transfer_task_v1.json"
)

MODEL_PROTOCOL = (
    ROOT / "results/final_protocol_2026.json"
)

MODEL_DSRF = (
    ROOT / "results/final_development_models_2026"
    / "dsrf/final.pt"
)

MODEL_XGB = (
    ROOT / "results/final_development_models_2026"
    / "xgboost/final.joblib"
)

MODEL_ENCODER = ROOT / "checkpoints/cbramod_official.pth"

INFERENCE_SCRIPT = (
    WORK / "scripts/11_train_val_inference.py"
)

MANIFEST = (
    DATA / "manifests/07_recording_audit.csv"
)

LAYOUT = WORK / "protocol/dataset_layout.json"

PRIVATE_ROOT = (
    DATA / "derived/caueeg_transfer_v1"
)

PRIVATE_REPORT = (
    DATA / "derived/caueeg_transfer_v1"
    / "analysis/12_recording_metrics.csv"
)

SUMMARY_REPORT = (
    WORK / "results/12_development_evaluation.json"
)

MODELS = {
    "DSRF": {
        "window_key": "dsrf_window_p_ad",
        "mean_key": "dsrf_mean_p_ad"
    },
    "XGBoost": {
        "window_key": "xgb_window_p_ad",
        "mean_key": "xgb_mean_p_ad"
    }
}

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
    }
}

THRESHOLD = 0.5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing output matches:", path)
            return True

        print("[PRESERVED] Existing output differs:", path)
        print("No existing file was overwritten.")
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

    print("[SAVED]", path)
    return True


def check_provenance():
    print("\n1. VERIFYING EXPERIMENT PROVENANCE")

    required = [
        INFERENCE_SUMMARY,
        TASK_LOCK,
        MODEL_PROTOCOL,
        MODEL_DSRF,
        MODEL_XGB,
        MODEL_ENCODER,
        INFERENCE_SCRIPT,
        MANIFEST,
        LAYOUT
    ]

    for path in required:
        require(
            path.is_file(),
            "Missing required file: " + str(path)
        )

    summary = read_json(INFERENCE_SUMMARY)
    task = read_json(TASK_LOCK)

    require(
        summary.get("status")
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Code 11 was not successfully completed"
    )

    require(
        summary["total_recordings"] == 1032
        and summary["total_windows"] == 24768
        and summary["test_recordings_processed"] == 0,
        "Unexpected Code 11 completion counts"
    )

    require(
        summary["completed_counts"] == EXPECTED,
        "Completed split counts do not match the audit"
    )

    require(
        task["version"] == "caueeg_transfer_v1",
        "Unknown external experiment version"
    )

    comparison = task["primary_target_comparison"]

    require(
        comparison["positive"] == "dementia"
        and comparison["negative"] == "normal"
        and comparison["excluded_from_binary_metrics"] == "mci"
        and comparison["level"] == "recording"
        and comparison["decision_threshold"] == THRESHOLD,
        "Primary task or threshold changed"
    )

    require(
        task["patient_identity_independently_verified"] is False
        and task["patient_level_performance_claims_allowed"] is False,
        "Unexpected patient identity claim"
    )

    expected_hashes = summary["model_hashes"]

    current_files = {
        "protocol_sha256": MODEL_PROTOCOL,
        "task_sha256": TASK_LOCK,
        "pretrained_sha256": MODEL_ENCODER,
        "dsrf_sha256": MODEL_DSRF,
        "xgboost_sha256": MODEL_XGB,
        "inference_script_sha256": INFERENCE_SCRIPT
    }

    for key, path in current_files.items():
        require(
            sha256(path) == expected_hashes[key],
            "Experiment provenance changed: " + key
        )

    print("[OK] Original models unchanged")
    print("[OK] Code 11 prediction provenance verified")
    print("[OK] Locked classification threshold: 0.5")
    print("[OK] External test predictions not accessed")

    return summary, task


def load_annotation_tags(layout):
    annotation = read_json(Path(layout["annotation"]))

    tagged = set()

    for item in annotation.get("data", []):
        if not isinstance(item, dict):
            continue

        serial = str(item.get("serial", "")).zfill(5)

        symptoms = item.get("symptom", [])

        if isinstance(symptoms, str):
            symptoms = [symptoms]

        if not isinstance(symptoms, list):
            continue

        tokens = {
            value.lower().strip()
            for value in symptoms
            if isinstance(value, str)
        }

        if "ad" in tokens:
            tagged.add(serial)

    return tagged


def calculate_stability(window_scores, recording_score):
    window_scores = np.asarray(
        window_scores,
        dtype=np.float64
    )

    window_predictions = (
        window_scores >= THRESHOLD
    ).astype(int)

    recording_prediction = int(
        recording_score >= THRESHOLD
    )

    return {
        "window_probability_sd": float(
            np.std(window_scores)
        ),
        "window_probability_range": float(
            np.ptp(window_scores)
        ),
        "window_disagreement_fraction": float(
            np.mean(
                window_predictions != recording_prediction
            )
        ),
        "sampled_window_flip_fraction": float(
            np.mean(
                window_predictions[1:]
                != window_predictions[:-1]
            )
        ),
        "mean_successive_score_change": float(
            np.mean(
                np.abs(np.diff(window_scores))
            )
        )
    }


def load_prediction_records(expected_hashes):
    print("\n2. LOADING EXISTING TRAIN/VALIDATION PREDICTIONS")

    layout = read_json(LAYOUT)
    ad_tags = load_annotation_tags(layout)

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as handle:
        manifest = list(csv.DictReader(handle))

    selected = [
        row for row in manifest
        if row["split"] in ("train", "validation")
    ]

    counts = {
        split: dict(Counter(
            row["class_name"]
            for row in selected
            if row["split"] == split
        ))
        for split in ("train", "validation")
    }

    require(
        counts == EXPECTED,
        "Manifest class counts have changed"
    )

    require(
        len(selected) == 1032,
        "Unexpected number of development recordings"
    )

    require(
        len({row["serial"] for row in selected}) == 1032,
        "Duplicate recording identifiers"
    )

    records = []

    for row in selected:
        serial = row["serial"]
        split = row["split"]
        label = row["class_name"]

        path = PRIVATE_ROOT / split / (
            serial + ".json"
        )

        require(
            path.is_file(),
            "Missing prediction file for a development recording"
        )

        result = read_json(path)

        require(
            result["serial"] == serial
            and result["split"] == split
            and result["class_name"] == label
            and result["version"] == "caueeg_transfer_v1",
            "Prediction/manifest mismatch"
        )

        require(
            result["model_hashes"] == expected_hashes,
            "Prediction file belongs to another model version"
        )

        require(
            result["window_count"] == 24
            and result["source_model_classes"] == ["AD", "CN"]
            and result["patient_id_verified"] is False,
            "Unexpected prediction format"
        )

        starts = result["window_start_seconds"]

        require(
            len(starts) == 24
            and all(
                starts[i] < starts[i + 1]
                for i in range(23)
            ),
            "Invalid sampled-window order"
        )

        item = {
            "serial": serial,
            "split": split,
            "class_name": label,
            "ad_symptom_tag": int(serial in ad_tags),
        }

        for model_name, keys in MODELS.items():
            values = np.asarray(
                result[keys["window_key"]],
                dtype=float
            )

            require(
                values.shape == (24,)
                and np.isfinite(values).all()
                and np.all(values >= 0)
                and np.all(values <= 1),
                "Invalid stored model probabilities"
            )

            mean_score = float(
                result[keys["mean_key"]]
            )

            require(
                np.isclose(
                    mean_score,
                    float(values.mean()),
                    rtol=0,
                    atol=1e-7
                ),
                "Stored average probability is inconsistent"
            )

            item[model_name + "_score"] = mean_score

            stability = calculate_stability(
                values, mean_score
            )

            for key, value in stability.items():
                item[model_name + "_" + key] = value

        records.append(item)

    print("[OK] 950 training prediction files verified")
    print("[OK] 82 validation prediction files verified")
    print("[OK] All 24 window scores per recording validated")
    print("Test prediction files read: 0")

    return records


def calibration_error(y, scores, bins=10):
    predicted = (scores >= THRESHOLD).astype(int)

    confidence = np.maximum(
        scores, 1.0 - scores
    )

    correctness = (
        predicted == y
    ).astype(float)

    ece = 0.0

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins

        if index == bins - 1:
            inside = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            inside = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if np.any(inside):
            ece += float(
                inside.mean()
                * abs(
                    confidence[inside].mean()
                    - correctness[inside].mean()
                )
            )

    return ece


def binary_metrics(rows, model_name):
    selected = [
        row for row in rows
        if row["class_name"] in ("normal", "dementia")
    ]

    require(
        len(selected) > 0,
        "No binary evaluation records"
    )

    y = np.asarray([
        int(row["class_name"] == "dementia")
        for row in selected
    ], dtype=int)

    scores = np.asarray([
        row[model_name + "_score"]
        for row in selected
    ], dtype=float)

    require(
        set(y.tolist()) == {0, 1},
        "Both binary classes are required"
    )

    predictions = (
        scores >= THRESHOLD
    ).astype(int)

    # Rows: Normal, Dementia.
    # Columns: Predicted Normal, Predicted Dementia.
    cm = confusion_matrix(
        y, predictions, labels=[0, 1]
    )

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    return {
        "n_recordings": int(len(y)),
        "n_normal": int(np.sum(y == 0)),
        "n_dementia": int(np.sum(y == 1)),
        "threshold": THRESHOLD,
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "sensitivity_dementia": float(
            tp / (tp + fn)
        ),
        "specificity_normal": float(
            tn / (tn + fp)
        ),
        "macro_f1": float(
            f1_score(
                y, predictions,
                average="macro",
                zero_division=0
            )
        ),
        "dementia_f1": float(
            f1_score(
                y, predictions,
                pos_label=1,
                zero_division=0
            )
        ),
        "matthews_correlation": float(
            matthews_corrcoef(y, predictions)
        ),
        "auroc": float(
            roc_auc_score(y, scores)
        ),
        "brier_vs_dementia_label": float(
            np.mean((scores - y) ** 2)
        ),
        "ece_10_bins": float(
            calibration_error(y, scores)
        ),
        "predicted_dementia_fraction": float(
            np.mean(predictions == 1)
        ),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_rows": [
            "normal",
            "dementia"
        ],
        "confusion_matrix_columns": [
            "predicted_normal",
            "predicted_dementia"
        ]
    }


def describe_scores(rows, model_name):
    scores = np.asarray([
        row[model_name + "_score"]
        for row in rows
    ], dtype=float)

    if len(scores) == 0:
        return None

    return {
        "n_recordings": len(scores),
        "mean_source_AD_score": float(
            np.mean(scores)
        ),
        "median_source_AD_score": float(
            np.median(scores)
        ),
        "q25_source_AD_score": float(
            np.quantile(scores, 0.25)
        ),
        "q75_source_AD_score": float(
            np.quantile(scores, 0.75)
        ),
        "fraction_above_fixed_threshold": float(
            np.mean(scores >= THRESHOLD)
        )
    }


def stability_summary(rows, model_name):
    if not rows:
        return None

    names = [
        "window_probability_sd",
        "window_probability_range",
        "window_disagreement_fraction",
        "sampled_window_flip_fraction",
        "mean_successive_score_change"
    ]

    output = {
        "n_recordings": len(rows)
    }

    for name in names:
        values = np.asarray([
            row[model_name + "_" + name]
            for row in rows
        ], dtype=float)

        output[name + "_mean"] = float(
            np.mean(values)
        )

        output[name + "_median"] = float(
            np.median(values)
        )

    return output


def paired_bootstrap(validation_rows, n=2000):
    print("\n4. COMPUTING PAIRED VALIDATION UNCERTAINTY")

    rows = [
        row for row in validation_rows
        if row["class_name"] in ("normal", "dementia")
    ]

    y = np.asarray([
        int(row["class_name"] == "dementia")
        for row in rows
    ], dtype=int)

    rng = np.random.default_rng(20261109)

    normal_idx = np.flatnonzero(y == 0)
    dementia_idx = np.flatnonzero(y == 1)

    require(
        len(normal_idx) >= 2
        and len(dementia_idx) >= 2,
        "Insufficient recordings for stratified bootstrap"
    )

    scores = {
        name: np.asarray([
            row[name + "_score"]
            for row in rows
        ], dtype=float)
        for name in MODELS
    }

    distributions = {
        "DSRF": [],
        "XGBoost": [],
        "DSRF_minus_XGBoost": []
    }

    for _ in range(n):
        sampled = np.concatenate([
            rng.choice(
                normal_idx,
                size=len(normal_idx),
                replace=True
            ),
            rng.choice(
                dementia_idx,
                size=len(dementia_idx),
                replace=True
            )
        ])

        boot_y = y[sampled]

        current = {}

        for name in MODELS:
            predictions = (
                scores[name][sampled] >= THRESHOLD
            ).astype(int)

            correct_normal = np.mean(
                predictions[boot_y == 0] == 0
            )

            correct_dementia = np.mean(
                predictions[boot_y == 1] == 1
            )

            value = float(
                (correct_normal + correct_dementia) / 2
            )

            distributions[name].append(value)
            current[name] = value

        distributions["DSRF_minus_XGBoost"].append(
            current["DSRF"] - current["XGBoost"]
        )

    return {
        "resamples": n,
        "seed": 20261109,
        "sampling_unit": "recording",
        "method": "paired_stratified_recording_bootstrap",
        "balanced_accuracy_95ci": {
            name: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975))
            ]
            for name, values in distributions.items()
        },
        "caution": (
            "Recordings are not independently verified as "
            "unique patients. Intervals may underestimate "
            "uncertainty if patients contributed multiple recordings."
        )
    }


def build_report(records, provenance):
    print("\n3. EVALUATING FROZEN MODELS")

    report = {
        "version": "caueeg_transfer_v1",
        "status": "DEVELOPMENT_EVALUATION_COMPLETED",
        "source_model_classes": ["AD", "CN"],
        "target_classes": ["normal", "mci", "dementia"],
        "binary_comparison": "dementia_vs_normal",
        "positive_target_class": "dementia",
        "threshold": THRESHOLD,
        "evaluation_unit": "recording",
        "patient_identity_verified": False,
        "patient_level_claims_permitted": False,
        "original_models_retrained": False,
        "test_recordings_evaluated": 0,
        "provenance": provenance,
        "models": {},
        "limitations": [
            "Dementia is not necessarily Alzheimer's disease.",
            "Source AD scores are not validated clinical probabilities.",
            "CAUEEG patient identities were not independently verified.",
            "Calibration scores describe source-score behavior on external labels.",
            "Within-recording prediction consistency does not establish correctness.",
            "Validation results are not external test results."
        ]
    }

    for model in MODELS:
        report["models"][model] = {}

        for split in ("train", "validation"):
            subset = [
                row for row in records
                if row["split"] == split
            ]

            binary = binary_metrics(
                subset, model
            )

            classes = {}

            temporal = {}

            for label in ("normal", "mci", "dementia"):
                group = [
                    row for row in subset
                    if row["class_name"] == label
                ]

                classes[label] = describe_scores(
                    group, model
                )

                temporal[label] = stability_summary(
                    group, model
                )

            dementia_cases = [
                row for row in subset
                if row["class_name"] == "dementia"
            ]

            ad_tagged = [
                row for row in dementia_cases
                if row["ad_symptom_tag"] == 1
            ]

            non_ad_tagged = [
                row for row in dementia_cases
                if row["ad_symptom_tag"] == 0
            ]

            report["models"][model][split] = {
                "binary_metrics": binary,
                "score_distributions": classes,
                "temporal_stability": temporal,
                "ad_tagged_dementia_exploratory": {
                    "ad_tagged": describe_scores(
                        ad_tagged, model
                    ),
                    "other_dementia": describe_scores(
                        non_ad_tagged, model
                    ),
                    "note": (
                        "AD is an annotation symptom tag, "
                        "not independently verified patient diagnosis."
                    )
                }
            }

            print(
                f"\n{model} | {split.upper()} | "
                f"Dementia vs Normal"
            )

            print(
                "Recording count:",
                binary["n_recordings"]
            )

            print(
                "Balanced accuracy:",
                f"{binary['balanced_accuracy']:.2%}"
            )

            print(
                "Accuracy:",
                f"{binary['accuracy']:.2%}"
            )

            print(
                "Dementia sensitivity:",
                f"{binary['sensitivity_dementia']:.2%}"
            )

            print(
                "Normal specificity:",
                f"{binary['specificity_normal']:.2%}"
            )

            print(
                "Macro F1:",
                f"{binary['macro_f1']:.4f}"
            )

            print(
                "AUROC:",
                f"{binary['auroc']:.4f}"
            )

            print(
                "Brier score:",
                f"{binary['brier_vs_dementia_label']:.4f}"
            )

            print(
                "ECE:",
                f"{binary['ece_10_bins']:.4f}"
            )

            print(
                "Confusion matrix:",
                binary["confusion_matrix"]
            )

            mci = classes["mci"]

            print(
                "MCI source AD-score median:",
                f"{mci['median_source_AD_score']:.4f}"
            )

            print(
                "MCI above 0.5:",
                f"{mci['fraction_above_fixed_threshold']:.2%}"
            )

    validation = [
        row for row in records
        if row["split"] == "validation"
    ]

    report["validation_uncertainty"] = paired_bootstrap(
        validation
    )

    return report


def save_recording_details(records):
    columns = [
        "serial",
        "split",
        "class_name",
        "ad_symptom_tag"
    ]

    for model in MODELS:
        columns += [
            model + "_score",
            model + "_window_probability_sd",
            model + "_window_probability_range",
            model + "_window_disagreement_fraction",
            model + "_sampled_window_flip_fraction",
            model + "_mean_successive_score_change"
        ]

    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(records)

    return write_once(
        PRIVATE_REPORT,
        buffer.getvalue()
    )


def main():
    summary, task = check_provenance()

    records = load_prediction_records(
        summary["model_hashes"]
    )

    report = build_report(
        records,
        summary["model_hashes"]
    )

    print("\n5. SAVING RESULTS")

    private_ok = save_recording_details(records)

    aggregate_ok = write_once(
        SUMMARY_REPORT,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False
        ) + "\n"
    )

    if private_ok and aggregate_ok:
        print("\nTRAINING RECORDINGS:", 950)
        print("VALIDATION RECORDINGS:", 82)
        print("TEST RECORDINGS EVALUATED:", 0)

        print(
            "\nDEVELOPMENT_STATUS: "
            "EVALUATION_COMPLETED"
        )

    else:
        print(
            "\nDEVELOPMENT_STATUS: "
            "OUTPUT_CONFLICT"
        )

    print(
        "\nIMPORTANT: The 0.5 threshold was not tuned "
        "on the validation or test results."
    )

    print(
        "All reported confidence intervals resample "
        "recordings, not independently verified patients."
    )


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc)
    )

    print(
        "DEVELOPMENT_STATUS: FAILED"
    )

finally:
    print(
        "\n========== CODE 12 FINISHED =========="
    )
