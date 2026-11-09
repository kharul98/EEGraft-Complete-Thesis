from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import json
import os

import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HOME = Path.home()
WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

PLAN_FILE = WORK / "protocol/caueeg_final_test_plan_v1.json"
LAYOUT_FILE = WORK / "protocol/dataset_layout.json"
INFERENCE_FILE = WORK / "results/11_train_val_inference_summary.json"
MANIFEST_FILE = DATA / "manifests/07_recording_audit.csv"

PRIVATE = DATA / "derived/caueeg_transfer_v1"

DETAIL_FILE = (
    PRIVATE / "analysis/24_age_label_development_details.csv"
)

REPORT_FILE = WORK / "results/24_age_label_audit.json"

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

FEATURES = {
    "Age only": ["age"],
    "Age + DSRF": ["age", "dsrf_logit"],
    "Age + XGBoost": ["age", "xgb_logit"]
}

SEED = 20261109
CV_FOLDS = 5
BOOTSTRAPS = 5000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4194304), b""
        ):
            h.update(block)

    return h.hexdigest()


def save_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing output matches:", path.name)
            return True

        print("[PRESERVED] Different existing file:", path)
        return False

    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600
    )

    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)

    print("[SAVED]", path)
    return True


def verify_setup():
    print("\n1. VERIFYING FROZEN RESEARCH ARTIFACTS")

    for path in (
        PLAN_FILE, LAYOUT_FILE,
        INFERENCE_FILE, MANIFEST_FILE
    ):
        require(path.is_file(), "Missing: " + str(path))

    plan = load_json(PLAN_FILE)
    layout = load_json(LAYOUT_FILE)
    inference = load_json(INFERENCE_FILE)

    require(
        plan["version"] == "caueeg_final_test_plan_v1",
        "Unexpected final test plan"
    )

    require(
        inference["status"]
        == "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "Development inference incomplete"
    )

    require(
        inference["model_hashes"] == plan["model_hashes"],
        "Original model hashes disagree"
    )

    split_file = Path(layout["split_file"])

    require(
        split_file.is_file(),
        "Official benchmark split file missing"
    )

    require(
        sha256(split_file)
        == plan["artifact_hashes"]["official_split_json_sha256"],
        "Official benchmark annotations changed"
    )

    require(
        sha256(MANIFEST_FILE)
        == plan["artifact_hashes"]["recording_manifest_sha256"],
        "Original benchmark manifest changed"
    )

    print("[OK] Frozen test plan unchanged")
    print("[OK] Official CAUEEG annotation checksum verified")
    print("[OK] Original EEG models unchanged")
    print("[OK] No test waveform or prediction access required")

    return plan, split_file


def normalize_serial(value):
    serial = str(value).strip()

    require(
        serial.isdigit() and 1 <= len(serial) <= 8,
        "Invalid recording identifier"
    )

    return serial.zfill(5)


def optional_age(value):
    if value is None or str(value).strip() == "":
        return None

    try:
        age = float(value)
    except (ValueError, TypeError):
        return None

    if not np.isfinite(age) or not 0 <= age <= 120:
        return None

    return age


def load_development(plan, split_file):
    print("\n2. LOADING OFFICIAL TRAINING AND VALIDATION METADATA")

    # The official file contains all three splits.
    # Only train_split and validation_split are processed.
    task = load_json(split_file)

    with MANIFEST_FILE.open(
        newline="", encoding="utf-8"
    ) as stream:
        manifest = list(csv.DictReader(stream))

    lookup = {
        (row["split"], row["serial"]): row["class_name"]
        for row in manifest
        if row["split"] in EXPECTED
    }

    require(
        len(lookup) == 1032,
        "Development manifest does not contain 1032 records"
    )

    data = {"train": [], "validation": []}

    for split in ("train", "validation"):
        entries = task[split + "_split"]

        require(
            isinstance(entries, list)
            and len(entries) == sum(EXPECTED[split].values()),
            "Unexpected official " + split + " split"
        )

        seen = set()

        for entry in entries:
            serial = normalize_serial(entry["serial"])
            label = str(entry["class_name"]).lower().strip()

            require(
                serial not in seen
                and label in EXPECTED[split]
                and lookup.get((split, serial)) == label,
                "Official label or identifier mismatch"
            )

            seen.add(serial)

            symptoms = entry.get("symptom", [])

            if isinstance(symptoms, str):
                symptoms = [symptoms]

            require(
                isinstance(symptoms, list),
                "Unexpected symptom annotation format"
            )

            symptom_tags = [
                str(tag).lower().strip()
                for tag in symptoms
            ]

            prediction_file = (
                PRIVATE / split / (serial + ".json")
            )

            require(
                prediction_file.is_file(),
                "Missing training/validation prediction"
            )

            prediction = load_json(prediction_file)

            require(
                prediction["split"] == split
                and prediction["serial"] == serial
                and prediction["class_name"] == label
                and prediction["model_hashes"] == plan["model_hashes"]
                and prediction["window_count"] == 24,
                "Prediction provenance mismatch"
            )

            dsrf = float(prediction["dsrf_mean_p_ad"])
            xgb = float(prediction["xgb_mean_p_ad"])

            require(
                np.isfinite(dsrf)
                and np.isfinite(xgb)
                and 0 <= dsrf <= 1
                and 0 <= xgb <= 1,
                "Invalid frozen source scores"
            )

            age = optional_age(entry.get("age"))

            data[split].append({
                "serial": serial,
                "split": split,
                "class_name": label,
                "age": age,
                "symptom_tags": symptom_tags,
                "ad_tagged": int("ad" in symptom_tags),
                "dsrf_score": dsrf,
                "xgb_score": xgb,
                "dsrf_logit": float(
                    logit(np.clip(dsrf, 1e-6, 1 - 1e-6))
                ),
                "xgb_logit": float(
                    logit(np.clip(xgb, 1e-6, 1 - 1e-6))
                ),
                "target": (
                    None if label == "mci"
                    else int(label == "dementia")
                )
            })

        require(
            dict(Counter(
                item["class_name"]
                for item in data[split]
            )) == EXPECTED[split],
            "Development class counts changed"
        )

        print(
            split,
            "recordings:",
            len(data[split])
        )

    print("Test EEG files opened: 0")
    print("Test predictions opened: 0")
    print("Test records included in analysis: 0")

    return data


def summarize_age(cases):
    output = {}

    for label in ("normal", "mci", "dementia"):
        selected = [
            case for case in cases
            if case["class_name"] == label
        ]

        ages = np.asarray([
            case["age"]
            for case in selected
            if case["age"] is not None
        ], dtype=float)

        output[label] = {
            "total_recordings": len(selected),
            "age_available": len(ages),
            "age_missing": len(selected) - len(ages)
        }

        if len(ages):
            output[label].update({
                "mean_age": float(np.mean(ages)),
                "median_age": float(np.median(ages)),
                "age_sd": float(np.std(ages, ddof=1))
                    if len(ages) > 1 else None,
                "age_q25": float(np.quantile(ages, 0.25)),
                "age_q75": float(np.quantile(ages, 0.75))
            })

    return output


def age_difference(cases, seed):
    normal = np.asarray([
        case["age"]
        for case in cases
        if case["class_name"] == "normal"
        and case["age"] is not None
    ], dtype=float)

    dementia = np.asarray([
        case["age"]
        for case in cases
        if case["class_name"] == "dementia"
        and case["age"] is not None
    ], dtype=float)

    require(
        len(normal) >= 5 and len(dementia) >= 5,
        "Insufficient age observations"
    )

    difference = float(
        np.mean(dementia) - np.mean(normal)
    )

    pooled_variance = (
        (len(normal) - 1) * np.var(normal, ddof=1)
        + (len(dementia) - 1) * np.var(dementia, ddof=1)
    ) / (len(normal) + len(dementia) - 2)

    smd = (
        difference / np.sqrt(pooled_variance)
        if pooled_variance > 0 else None
    )

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

    differences = (
        dementia_draws.mean(axis=1)
        - normal_draws.mean(axis=1)
    )

    return {
        "dementia_minus_normal_mean_age_years": difference,
        "standardized_mean_difference": smd,
        "recording_bootstrap_95ci_years": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975))
        ],
        "independent_patient_sampling_verified": False
    }


def build_design(cases, feature_names):
    return np.asarray([
        [float(case[name]) for name in feature_names]
        for case in cases
    ], dtype=float)


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    pred = (p >= 0.5).astype(int)

    normal = y == 0
    dementia = y == 1

    sensitivity = float(
        np.mean(pred[dementia] == 1)
    )

    specificity = float(
        np.mean(pred[normal] == 0)
    )

    return {
        "n_recordings": int(len(y)),
        "auroc": float(roc_auc_score(y, p)),
        "balanced_accuracy": (
            sensitivity + specificity
        ) / 2,
        "accuracy": float(np.mean(pred == y)),
        "dementia_sensitivity": sensitivity,
        "normal_specificity": specificity,
        "brier_score": float(np.mean((p - y) ** 2))
    }


def make_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None
        )
    )


def evaluate_age_baselines(data):
    print("\n4. AGE-ONLY AND AGE-ADJUSTED COMPARISONS")

    train = [
        c for c in data["train"]
        if c["target"] is not None
        and c["age"] is not None
    ]

    validation = [
        c for c in data["validation"]
        if c["target"] is not None
        and c["age"] is not None
    ]

    require(
        len(train) >= 500 and len(validation) >= 40,
        "Too many missing ages for a reliable comparison"
    )

    y_train = np.asarray(
        [c["target"] for c in train], dtype=int
    )

    y_validation = np.asarray(
        [c["target"] for c in validation], dtype=int
    )

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = {}

    for name, feature_names in FEATURES.items():
        x_train = build_design(train, feature_names)
        x_validation = build_design(
            validation, feature_names
        )

        oof = np.full(
            len(train), np.nan, dtype=float
        )

        for fit_idx, heldout_idx in splitter.split(
            x_train, y_train
        ):
            model = make_classifier()

            model.fit(
                x_train[fit_idx],
                y_train[fit_idx]
            )

            oof[heldout_idx] = (
                model.predict_proba(
                    x_train[heldout_idx]
                )[:, 1]
            )

        require(
            np.isfinite(oof).all(),
            "Incomplete age-model CV predictions"
        )

        final_model = make_classifier()
        final_model.fit(x_train, y_train)

        validation_probabilities = (
            final_model.predict_proba(
                x_validation
            )[:, 1]
        )

        results[name] = {
            "features": feature_names,
            "training_5fold_out_of_fold": metrics(
                y_train, oof
            ),
            "validation": metrics(
                y_validation,
                validation_probabilities
            )
        }

        print("\nModel:", name)
        print(
            "  Train CV AUROC:",
            f"{results[name]['training_5fold_out_of_fold']['auroc']:.4f}"
        )
        print(
            "  Validation AUROC:",
            f"{results[name]['validation']['auroc']:.4f}"
        )
        print(
            "  Validation balanced accuracy:",
            f"{results[name]['validation']['balanced_accuracy']:.2%}"
        )
        print(
            "  Validation Brier:",
            f"{results[name]['validation']['brier_score']:.4f}"
        )

    baseline_auc = results["Age only"]["validation"]["auroc"]
    baseline_brier = results["Age only"]["validation"]["brier_score"]

    incremental = {}

    for model in ("DSRF", "XGBoost"):
        key = "Age + " + model
        result = results[key]["validation"]

        incremental[model] = {
            "validation_auroc_change_vs_age_only":
                result["auroc"] - baseline_auc,

            "validation_brier_change_vs_age_only":
                result["brier_score"] - baseline_brier
        }

        print(
            f"\n{model} incremental validation AUROC:",
            f"{incremental[model]['validation_auroc_change_vs_age_only']:+.4f}"
        )

    return {
        "training_complete_case_count": len(train),
        "validation_complete_case_count": len(validation),
        "fixed_logistic_C": 1.0,
        "fixed_cv_folds": 5,
        "seed": SEED,
        "model_comparisons": results,
        "incremental_eeg_score_associations": incremental,
        "scope": (
            "Exploratory age confounding assessment only. "
            "These are CAUEEG-trained diagnostic models, "
            "not the original frozen EEG models."
        )
    }


def score_age_correlations(cases):
    output = {}

    for label in ("normal", "dementia"):
        group = [
            c for c in cases
            if c["class_name"] == label
            and c["age"] is not None
        ]

        output[label] = {}

        ages = [
            c["age"] for c in group
        ]

        for model, column in (
            ("DSRF", "dsrf_score"),
            ("XGBoost", "xgb_score")
        ):
            scores = [
                c[column] for c in group
            ]

            rho = spearmanr(
                ages, scores
            ).statistic

            output[label][model] = (
                float(rho)
                if np.isfinite(rho)
                else None
            )

    return output


def diagnosis_ontology(cases):
    dementia = [
        case for case in cases
        if case["class_name"] == "dementia"
    ]

    tagged = [
        case for case in dementia
        if case["ad_tagged"] == 1
    ]

    tag_counts = Counter(
        tag
        for case in dementia
        for tag in set(case["symptom_tags"])
    )

    return {
        "broad_dementia_recordings": len(dementia),
        "explicit_ad_tag_recordings": len(tagged),
        "not_explicitly_ad_tagged": (
            len(dementia) - len(tagged)
        ),
        "common_symptom_tags_n_ge_10": {
            tag: count
            for tag, count in sorted(tag_counts.items())
            if count >= 10
        },
        "caution": (
            "A dataset symptom tag is not independent "
            "neuropathological confirmation of AD."
        )
    }


def private_csv(data):
    fields = [
        "serial", "split", "class_name",
        "age", "ad_tagged",
        "dsrf_source_ad_score",
        "xgb_source_ad_score"
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=fields
    )
    writer.writeheader()

    for split in ("train", "validation"):
        for c in data[split]:
            writer.writerow({
                "serial": c["serial"],
                "split": split,
                "class_name": c["class_name"],
                "age": (
                    "" if c["age"] is None else c["age"]
                ),
                "ad_tagged": c["ad_tagged"],
                "dsrf_source_ad_score": c["dsrf_score"],
                "xgb_source_ad_score": c["xgb_score"]
            })

    return buffer.getvalue()


def main():
    plan, split_file = verify_setup()

    data = load_development(
        plan, split_file
    )

    print("\n3. AGE DISTRIBUTIONS AND DIAGNOSIS LABELS")

    demographics = {}
    ontology = {}
    correlations = {}

    for split in ("train", "validation"):
        demographics[split] = {
            "by_class": summarize_age(data[split]),
            "normal_vs_dementia": age_difference(
                data[split],
                SEED + int(split == "validation")
            )
        }

        ontology[split] = diagnosis_ontology(
            data[split]
        )

        correlations[split] = score_age_correlations(
            data[split]
        )

        print("\nPartition:", split.upper())

        for label, result in demographics[split][
            "by_class"
        ].items():
            print(
                f"  {label}: n={result['total_recordings']}, "
                f"age available={result['age_available']}, "
                f"mean={result.get('mean_age', 'N/A')}"
            )

        age_gap = demographics[split][
            "normal_vs_dementia"
        ]

        print(
            "  Dementia minus Normal mean age:",
            f"{age_gap['dementia_minus_normal_mean_age_years']:+.2f} years"
        )

        print(
            "  Standardized age difference:",
            f"{age_gap['standardized_mean_difference']:+.3f}"
        )

        print(
            "  Dementia records explicitly tagged AD:",
            ontology[split]["explicit_ad_tag_recordings"],
            "/",
            ontology[split]["broad_dementia_recordings"]
        )

    baselines = evaluate_age_baselines(data)

    print("\n5. SAVING AGE-CONFOUNDING EVIDENCE")

    report = {
        "status": "AGE_LABEL_AUDIT_COMPLETED",
        "analysis_type": "post_hoc_exploratory",
        "analysis_unit": "recording",
        "dataset": "CAUEEG_dementia_no_overlap",

        "demographic_summary": demographics,
        "diagnostic_label_ontology": ontology,
        "within_class_age_score_correlations": correlations,
        "age_baseline_analysis": baselines,

        "provenance": {
            "final_test_plan_sha256": sha256(PLAN_FILE),
            "official_split_sha256": sha256(split_file),
            "development_inference_sha256":
                sha256(INFERENCE_FILE)
        },

        "test_eeg_files_opened": 0,
        "test_prediction_files_opened": 0,
        "test_records_analyzed": 0,
        "original_models_modified": False,

        "scientific_limitations": [
            "Designed after the locked test was evaluated.",
            "Age-only models are exploratory, fitted only on training.",
            "Validation has already been inspected in earlier work.",
            "Patient identities and recording independence are unverified.",
            "Age is not the only possible demographic confound.",
            "Diagnosis labels are clinical dataset annotations, not pathology confirmation.",
            "Age adjustment does not prove the EEG signal is disease-specific.",
            "Models fitted in this audit must not be evaluated on the previously used locked test.",
            "No future Alzheimer conversion risk is estimated."
        ]
    }

    details_ok = save_once(
        DETAIL_FILE,
        private_csv(data)
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

    if details_ok and report_ok:
        print("\nAGE_AUDIT_STATUS: COMPLETED")
    else:
        print("\nAGE_AUDIT_STATUS: OUTPUT_CONFLICT")

    print("Original EEG models modified: 0")
    print("Locked test predictions accessed: 0")
    print("Restricted individual ages remain local.")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )

    print("AGE_AUDIT_STATUS: FAILED")

finally:
    print(
        "\n========== CODE 24 FINISHED ==========",
        flush=True
    )
