from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HOME = Path.home()
WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

PLAN = WORK / "protocol/caueeg_final_test_plan_v1.json"
INFERENCE = WORK / "results/11_train_val_inference_summary.json"
PREVIOUS = WORK / "results/24_age_label_audit.json"

PRIVATE_CSV = (
    DATA / "derived/caueeg_transfer_v1"
    / "analysis/24_age_label_development_details.csv"
)

OUTPUT = WORK / "results/25_age_incremental_information_audit.json"

SEED = 20261109
FOLDS = 5
BOOTSTRAPS = 5000

EXPECTED = {
    "train": {"normal": 367, "mci": 334, "dementia": 249},
    "validation": {"normal": 34, "mci": 28, "dementia": 20}
}

CONFIGURATIONS = {
    "linear_age": ("linear", None),
    "linear_age_DSRF": ("linear", "DSRF"),
    "linear_age_XGBoost": ("linear", "XGBoost"),
    "spline_age": ("spline", None),
    "spline_age_DSRF": ("spline", "DSRF"),
    "spline_age_XGBoost": ("spline", "XGBoost")
}

COMPARISONS = {
    "linear_DSRF_added_to_age": (
        "linear_age", "linear_age_DSRF"
    ),
    "linear_XGBoost_added_to_age": (
        "linear_age", "linear_age_XGBoost"
    ),
    "spline_DSRF_added_to_age": (
        "spline_age", "spline_age_DSRF"
    ),
    "spline_XGBoost_added_to_age": (
        "spline_age", "spline_age_XGBoost"
    )
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4194304), b""
        ):
            h.update(block)
    return h.hexdigest()


def save_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing output matches:", path)
            return True

        print("[PRESERVED] Existing output differs:", path)
        return False

    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)

    print("[SAVED]", path)
    return True


def verify():
    print("\n1. VERIFYING AGE AUDIT AND FROZEN PROVENANCE")

    for path in (PLAN, INFERENCE, PREVIOUS, PRIVATE_CSV):
        require(path.is_file(), "Missing artifact: " + str(path))

    plan = load_json(PLAN)
    inference = load_json(INFERENCE)
    previous = load_json(PREVIOUS)

    require(
        plan["version"] == "caueeg_final_test_plan_v1",
        "Final test plan mismatch"
    )

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Original development inference incomplete"
    )

    require(
        inference["model_hashes"] == plan["model_hashes"],
        "Original model provenance changed"
    )

    require(
        previous["status"] == "AGE_LABEL_AUDIT_COMPLETED",
        "Code 24 incomplete"
    )

    require(
        previous["provenance"]["final_test_plan_sha256"]
        == sha256(PLAN),
        "Final test plan changed since Code 24"
    )

    require(
        previous["provenance"]["development_inference_sha256"]
        == sha256(INFERENCE),
        "Development inference summary changed"
    )

    require(
        previous["test_records_analyzed"] == 0
        and previous["test_prediction_files_opened"] == 0
        and previous["original_models_modified"] is False,
        "Code 24 safety record mismatch"
    )

    print("[OK] Code 24 verified")
    print("[OK] Frozen model identities verified")
    print("[OK] Final test protocol unchanged")

    return previous


def load_cases():
    print("\n2. LOADING AGE AND EEG SCORES FROM DEVELOPMENT ONLY")

    with PRIVATE_CSV.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    require(
        len(rows) == 1032,
        "Expected 1032 development records"
    )

    grouped = {"train": [], "validation": []}
    identifiers = set()

    for row in rows:
        split = row["split"]
        label = row["class_name"]
        serial = row["serial"]

        require(
            split in EXPECTED
            and label in EXPECTED[split],
            "Unexpected split or diagnostic class"
        )

        key = (split, serial)

        require(
            key not in identifiers,
            "Duplicate development recording"
        )
        identifiers.add(key)

        age = float(row["age"])
        dsrf = float(row["dsrf_source_ad_score"])
        xgb = float(row["xgb_source_ad_score"])

        require(
            np.isfinite(age)
            and 0 <= age <= 120
            and np.isfinite(dsrf)
            and np.isfinite(xgb)
            and 0 <= dsrf <= 1
            and 0 <= xgb <= 1,
            "Invalid age or EEG score"
        )

        grouped[split].append({
            "class_name": label,
            "age": age,
            "DSRF": float(logit(
                np.clip(dsrf, 1e-6, 1 - 1e-6)
            )),
            "XGBoost": float(logit(
                np.clip(xgb, 1e-6, 1 - 1e-6)
            )),
            "target": (
                None if label == "mci"
                else int(label == "dementia")
            )
        })

    for split, expected in EXPECTED.items():
        require(
            dict(Counter(
                c["class_name"] for c in grouped[split]
            )) == expected,
            "Development cohort composition changed"
        )

    binary = {
        split: [
            c for c in grouped[split]
            if c["target"] is not None
        ]
        for split in EXPECTED
    }

    require(
        len(binary["train"]) == 616
        and len(binary["validation"]) == 54,
        "Incorrect binary cohort size"
    )

    print("Training binary recordings:", len(binary["train"]))
    print("Validation binary recordings:", len(binary["validation"]))
    print("MCI excluded from binary comparisons: 362")
    print("Locked test recordings accessed: 0")

    return binary


def build_features(cases, eeg_model):
    features = []

    for case in cases:
        values = [case["age"]]

        if eeg_model is not None:
            values.append(case[eeg_model])

        features.append(values)

    return np.asarray(features, dtype=float)


def make_estimator(kind, eeg_model):
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000 if kind == "spline" else 1000,
        class_weight=None
    )

    if kind == "linear":
        return make_pipeline(
            StandardScaler(),
            classifier
        )

    age_spline = SplineTransformer(
        n_knots=4,
        degree=3,
        knots="quantile",
        include_bias=False,
        extrapolation="linear"
    )

    components = [
        ("age_spline", age_spline, [0])
    ]

    if eeg_model is not None:
        components.append(
            ("source_eeg_score", "passthrough", [1])
        )

    preprocessing = ColumnTransformer(
        components,
        remainder="drop",
        sparse_threshold=0
    )

    return make_pipeline(
        preprocessing,
        StandardScaler(),
        classifier
    )


def measurements(y, probabilities):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    require(
        len(y) == len(p)
        and np.isfinite(p).all()
        and np.all((p >= 0) & (p <= 1)),
        "Invalid evaluation probabilities"
    )

    prediction = (p >= 0.5).astype(int)
    normal = y == 0
    dementia = y == 1

    sensitivity = float(
        np.mean(prediction[dementia] == 1)
    )

    specificity = float(
        np.mean(prediction[normal] == 0)
    )

    return {
        "recordings": int(len(y)),
        "auroc": float(roc_auc_score(y, p)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),
        "sensitivity": sensitivity,
        "specificity": specificity
    }


def calculate_predictions(cases):
    print("\n3. REPRODUCING LINEAR AND TESTING FLEXIBLE AGE CONTROLS")

    train = cases["train"]
    validation = cases["validation"]

    y_train = np.asarray(
        [c["target"] for c in train], dtype=int
    )

    y_validation = np.asarray(
        [c["target"] for c in validation], dtype=int
    )

    folds = list(StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED
    ).split(np.zeros((len(train), 1)), y_train))

    predictions = {
        "train_oof": {},
        "validation": {}
    }

    results = {}

    for name, (kind, eeg_model) in CONFIGURATIONS.items():
        x_train = build_features(train, eeg_model)
        x_validation = build_features(validation, eeg_model)

        oof = np.full(len(train), np.nan, dtype=float)

        for fit_idx, holdout_idx in folds:
            estimator = make_estimator(kind, eeg_model)

            estimator.fit(
                x_train[fit_idx],
                y_train[fit_idx]
            )

            oof[holdout_idx] = estimator.predict_proba(
                x_train[holdout_idx]
            )[:, 1]

        require(
            np.isfinite(oof).all(),
            "Incomplete cross-validation predictions"
        )

        final_model = make_estimator(kind, eeg_model)
        final_model.fit(x_train, y_train)

        val_probabilities = final_model.predict_proba(
            x_validation
        )[:, 1]

        predictions["train_oof"][name] = oof
        predictions["validation"][name] = val_probabilities

        results[name] = {
            "age_function": kind,
            "eeg_score_included": eeg_model,
            "train_five_fold_oof": measurements(
                y_train, oof
            ),
            "validation": measurements(
                y_validation, val_probabilities
            )
        }

        print("\nModel:", name)
        print(
            "  Train CV AUROC:",
            f"{results[name]['train_five_fold_oof']['auroc']:.4f}"
        )
        print(
            "  Validation AUROC:",
            f"{results[name]['validation']['auroc']:.4f}"
        )
        print(
            "  Validation Brier:",
            f"{results[name]['validation']['brier_score']:.4f}"
        )
        print(
            "  Validation BAcc:",
            f"{results[name]['validation']['balanced_accuracy']:.2%}"
        )

    return predictions, results, {
        "train_oof": y_train,
        "validation": y_validation
    }


def verify_linear_reproduction(results, previous):
    print("\n4. CHECKING CODE 24 REPRODUCIBILITY")

    previous_models = previous[
        "age_baseline_analysis"
    ]["model_comparisons"]

    mappings = {
        "linear_age": "Age only",
        "linear_age_DSRF": "Age + DSRF",
        "linear_age_XGBoost": "Age + XGBoost"
    }

    for current_name, previous_name in mappings.items():
        current = results[current_name]
        old = previous_models[previous_name]

        for current_split, old_split in (
            ("train_five_fold_oof", "training_5fold_out_of_fold"),
            ("validation", "validation")
        ):
            for metric in (
                "auroc", "brier_score", "balanced_accuracy"
            ):
                require(
                    np.isclose(
                        current[current_split][metric],
                        old[old_split][metric],
                        rtol=0,
                        atol=1e-8
                    ),
                    "Code 24 reproduction mismatch: "
                    + current_name + "/" + metric
                )

        print("[REPRODUCED]", previous_name)

    print("[OK] Code 24 results independently reproduced")


def bootstrap_auc(y, scores):
    ranks = rankdata(scores, method="average")

    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))

    return float(
        (
            np.sum(ranks[y == 1])
            - n_positive * (n_positive + 1) / 2
        ) / (n_positive * n_negative)
    )


def paired_bootstrap(y, base, added, seed):
    y = np.asarray(y, dtype=int)
    base = np.asarray(base, dtype=float)
    added = np.asarray(added, dtype=float)

    normal_idx = np.flatnonzero(y == 0)
    dementia_idx = np.flatnonzero(y == 1)

    rng = np.random.default_rng(seed)

    auc_differences = np.empty(BOOTSTRAPS)
    brier_improvements = np.empty(BOOTSTRAPS)

    point_auc = (
        roc_auc_score(y, added)
        - roc_auc_score(y, base)
    )

    base_errors = (base - y) ** 2
    added_errors = (added - y) ** 2

    point_brier_improvement = float(
        np.mean(base_errors - added_errors)
    )

    for index in range(BOOTSTRAPS):
        sample = np.concatenate([
            rng.choice(
                normal_idx,
                len(normal_idx),
                replace=True
            ),
            rng.choice(
                dementia_idx,
                len(dementia_idx),
                replace=True
            )
        ])

        sampled_y = y[sample]

        auc_differences[index] = (
            bootstrap_auc(
                sampled_y,
                added[sample]
            )
            - bootstrap_auc(
                sampled_y,
                base[sample]
            )
        )

        brier_improvements[index] = float(
            np.mean(
                base_errors[sample]
                - added_errors[sample]
            )
        )

    def interval(values):
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975))
        ]

    return {
        "auroc_improvement": float(point_auc),
        "auroc_difference_95ci": interval(auc_differences),
        "brier_improvement_positive_is_better":
            point_brier_improvement,
        "brier_improvement_95ci":
            interval(brier_improvements),
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_unit": "recording",
        "patient_independence_verified": False,
        "training_refit_uncertainty_included": False
    }


def uncertainty_analysis(predictions, labels):
    print("\n5. PAIRED INCREMENTAL-INFORMATION INTERVALS")

    results = {}

    for split in ("train_oof", "validation"):
        results[split] = {}
        y = labels[split]

        print("\nPartition:", split)

        for index, (name, pair) in enumerate(
            COMPARISONS.items()
        ):
            baseline, extended = pair

            result = paired_bootstrap(
                y,
                predictions[split][baseline],
                predictions[split][extended],
                seed=SEED + index + (
                    100 if split == "validation" else 0
                )
            )

            results[split][name] = result

            print("\n", name)
            print(
                "  AUROC improvement:",
                f"{result['auroc_improvement']:+.4f}"
            )
            print(
                "  AUROC 95% interval:",
                [
                    round(v, 4)
                    for v in result["auroc_difference_95ci"]
                ]
            )
            print(
                "  Brier improvement:",
                f"{result['brier_improvement_positive_is_better']:+.4f}"
            )
            print(
                "  Brier improvement 95% interval:",
                [
                    round(v, 4)
                    for v in result["brier_improvement_95ci"]
                ]
            )

    return results


def age_overlap(cases):
    print("\n6. CHECKING AGE-GROUP OVERLAP")

    bands = [
        (0, 50), (50, 60), (60, 70),
        (70, 80), (80, 90), (90, 121)
    ]

    result = {}

    for split, selected in cases.items():
        rows = []

        for lower, upper in bands:
            within = [
                c for c in selected
                if lower <= c["age"] < upper
            ]

            counts = Counter(
                c["class_name"] for c in within
            )

            rows.append({
                "age_band": f"{lower}-{upper - 1}",
                "normal_recordings": counts["normal"],
                "dementia_recordings": counts["dementia"]
            })

        result[split] = rows

        print("\n", split.upper())

        for row in rows:
            print(
                "  Age", row["age_band"],
                "Normal:", row["normal_recordings"],
                "Dementia:", row["dementia_recordings"]
            )

    return result


def main():
    previous = verify()

    cases = load_cases()

    predictions, metrics, labels = calculate_predictions(cases)

    verify_linear_reproduction(metrics, previous)

    intervals = uncertainty_analysis(
        predictions, labels
    )

    overlap = age_overlap(cases)

    print("\n7. SAVING FINAL DEMOGRAPHIC SENSITIVITY AUDIT")

    report = {
        "status": "AGE_INCREMENTAL_AUDIT_COMPLETED",
        "analysis_type": "post_hoc_exploratory",
        "dataset": "CAUEEG_dementia_no_overlap",
        "unit": "recording",

        "training_binary_recordings": 616,
        "validation_binary_recordings": 54,

        "fixed_settings": {
            "folds": FOLDS,
            "cv_seed": SEED,
            "logistic_C": 1.0,
            "spline_knots": 4,
            "spline_degree": 3,
            "bootstrap_resamples": BOOTSTRAPS
        },

        "model_results": metrics,
        "paired_incremental_information": intervals,
        "age_group_overlap": overlap,

        "source_hashes": {
            "code24_report_sha256": sha256(PREVIOUS),
            "code24_private_csv_sha256": sha256(PRIVATE_CSV),
            "final_test_plan_sha256": sha256(PLAN),
            "development_inference_sha256": sha256(INFERENCE)
        },

        "test_predictions_accessed": 0,
        "test_eeg_accessed": 0,
        "test_subjects_analyzed": 0,
        "original_models_retrained": False,
        "new_external_test_selection_performed": False,

        "limitations": [
            "The audit is exploratory and designed after locked-test evaluation.",
            "Validation was previously inspected and is not a fresh independent confirmation.",
            "Flexible age adjustment does not eliminate all demographic confounding.",
            "Age-plus-score models were trained on CAUEEG and are not the original frozen models.",
            "Cross-validated training predictions reuse development data across folds.",
            "Bootstrap intervals condition on fitted predictions and do not include training variability.",
            "The resampling unit is a recording, not a verified independent patient.",
            "Small incremental differences do not demonstrate a disease-specific EEG biomarker.",
            "No further inference or model selection is permitted on the previously evaluated locked test."
        ]
    }

    if save_once(OUTPUT, report):
        print("\nAGE_INCREMENTAL_STATUS: COMPLETED")
    else:
        print("\nAGE_INCREMENTAL_STATUS: OUTPUT_CONFLICT")

    print("Original EEG model weights modified: 0")
    print("Locked test records opened: 0")
    print("Individual participant metadata exported: 0")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )

    print("AGE_INCREMENTAL_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 25 FINISHED ==========",
        flush=True
    )
