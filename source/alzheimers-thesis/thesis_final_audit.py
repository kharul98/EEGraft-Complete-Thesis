#!/usr/bin/env python3

"""
Final thesis audit — Phase 1.

Checks:
1. Participant inclusion and exclusion.
2. EEG recording boundaries and channel counts.
3. Existing model predictions and accuracy.
4. Repeated nested-CV results.
5. Duration-study results, if available.
6. Feature consistency across 300-second analyses.

Original datasets and model outputs remain unchanged.
"""

from pathlib import Path
from collections import Counter
import csv
import json
import sys

import numpy as np
import mne

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)


ROOT = Path.cwd()
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "thesis_final" / "audit"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

LOG = []
ISSUES = []


def log(*parts):
    message = " ".join(str(part) for part in parts)
    print(message, flush=True)
    LOG.append(message)


def issue(message):
    ISSUES.append(message)
    log("CHECK: " + message)


def read_csv(path, delimiter=","):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(
            csv.DictReader(f, delimiter=delimiter)
        )


def save_csv(path, records):

    if not records:
        return

    fields = list(dict.fromkeys(
        key
        for record in records
        for key in record
    ))

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(records)


def indexed(rows):

    result = {
        r["subject"]: r
        for r in rows
    }

    if len(result) != len(rows):
        raise ValueError(
            "Duplicate participant identifiers"
        )

    return result


def safe_float(value):

    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


# ==========================================
# 1. PARTICIPANT MANIFEST
# ==========================================

log("===== THESIS FINAL AUDIT =====")

manifest_path = (
    REPORTS / "openneuro_5min_manifest_v2.csv"
)

manifest = read_csv(manifest_path)

if len(manifest) != 88:
    issue(
        f"Expected 88 participants; "
        f"manifest contains {len(manifest)}"
    )

if len({
    r["subject"] for r in manifest
}) != len(manifest):
    raise RuntimeError(
        "Duplicate subjects in manifest"
    )

log("\n===== DATASET COHORT =====")

groups = Counter(
    r["group"] for r in manifest
)

log(f"Total subjects: {len(manifest)}")
log(f"Group counts: {dict(groups)}")

statuses = Counter(
    (r["group"], r["status"])
    for r in manifest
)

for (group, status), count in sorted(
    statuses.items()
):
    log(
        f"{group} / {status}: {count}"
    )

included = [
    r for r in manifest
    if r["status"] == "CANDIDATE"
    and r["group"] in ("A", "C")
]

excluded_binary = [
    r for r in manifest
    if r["group"] in ("A", "C")
    and r["status"] != "CANDIDATE"
]

ids = [
    r["subject"] for r in included
]

y = np.array([
    int(r["group"] == "A")
    for r in included
])

log("\n===== PRIMARY COHORT =====")
log(f"Included subjects: {len(included)}")
log(f"Included AD: {int(y.sum())}")
log(f"Included Healthy: {int((1-y).sum())}")

log(
    "Excluded AD/Healthy:",
    if_false := len(excluded_binary)
)

if (
    len(included) != 46
    or Counter(r["group"] for r in included)
    != {"A": 23, "C": 23}
):
    raise RuntimeError(
        "Unexpected primary cohort. "
        "Do not continue until reviewed."
    )

log("\n===== EXCLUSION BREAKDOWN =====")

excluded_counts = Counter(
    (r["group"], r["status"])
    for r in excluded_binary
)

for key, count in sorted(
    excluded_counts.items()
):
    log(f"{key}: {count}")


# ==========================================
# 2. PARTICIPANT METADATA
# ==========================================

participant_file = (
    ROOT /
    "data/eeg/on004504_v1.0.0/participants.tsv"
)

metadata = {}

if participant_file.exists():

    for row in read_csv(
        participant_file,
        delimiter="\t"
    ):

        sid = row.get(
            "participant_id",
            ""
        )

        metadata[sid] = row

    log("\nParticipant metadata: FOUND")

else:
    issue(
        "participants.tsv not found "
        "at expected location"
    )


def metadata_field(subject, names):

    r = metadata.get(subject, {})

    normalized = {
        str(k).lower().replace("_", ""): v
        for k, v in r.items()
    }

    for name in names:
        key = name.lower().replace("_", "")

        if key in normalized:
            return normalized[key]

    return ""


cohort_records = []

for row in manifest:

    sid = row["subject"]

    selected = (
        row["group"] in ("A", "C")
        and row["status"] == "CANDIDATE"
    )

    cohort_records.append({
        "subject": sid,
        "group": row["group"],
        "age": metadata_field(
            sid, ["age", "age_years"]
        ),
        "sex": metadata_field(
            sid, ["sex", "gender"]
        ),
        "status": row["status"],
        "included_in_binary_analysis":
            int(selected),
        "selection_reason_or_exclusion_status":
            (
                "Selected candidate 300s window"
                if selected
                else row["status"]
            ),
        "source_file": row.get("file", ""),
        "start_sec": row.get("start_sec", ""),
        "end_sec": row.get("end_sec", ""),
        "multi200_pct":
            row.get("multi200_pct", "")
    })

save_csv(
    OUTPUT / "participant_audit.csv",
    cohort_records
)

log(
    "Participant audit saved:",
    OUTPUT / "participant_audit.csv"
)


# ==========================================
# 3. CONTINUITY AND CHANNEL AUDIT
# ==========================================

log("\n===== EEG RECORDING VALIDATION =====")

boundary_errors = []
recording_errors = []

for count, row in enumerate(included, 1):

    sid = row["subject"]
    source = Path(row["file"])

    if not source.is_absolute():
        source = ROOT / source

    try:

        if not source.exists():
            raise FileNotFoundError(source)

        raw = mne.io.read_raw_eeglab(
            str(source),
            preload=False,
            verbose="ERROR"
        )

        fs = float(raw.info["sfreq"])

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        if abs(fs - 500) > 0.01:
            raise ValueError(
                f"Unexpected sampling rate: {fs}"
            )

        if len(picks) != 19:
            raise ValueError(
                f"Expected 19 EEG channels; "
                f"found {len(picks)}"
            )

        start_sec = float(
            row["start_sec"]
        )

        end_sec = start_sec + 300.0

        start_sample = int(
            round(start_sec * fs)
        )

        end_sample = (
            start_sample + int(round(300 * fs))
        )

        if (
            start_sample < 0
            or end_sample > raw.n_times
        ):
            raise ValueError(
                "Selected window exceeds recording"
            )

        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description
        ):

            if (
                "boundary"
                not in str(description).lower()
            ):
                continue

            onset = float(onset)

            offset = onset + max(
                float(duration),
                1.0 / fs
            )

            if (
                onset < end_sec
                and offset > start_sec
            ):
                boundary_errors.append(sid)
                break

    except Exception as exc:

        recording_errors.append(
            (sid, str(exc))
        )

    if (
        count % 10 == 0
        or count == len(included)
    ):
        log(
            f"Checked {count}/{len(included)}"
        )

log(
    "Files or metadata with errors:",
    len(recording_errors)
)

log(
    "Selected windows crossing boundaries:",
    len(boundary_errors)
)

for sid, error in recording_errors:
    log(f"ERROR {sid}: {error}")

if boundary_errors:
    log(
        "BOUNDARY ERRORS:",
        ", ".join(boundary_errors)
    )

if recording_errors or boundary_errors:
    issue(
        "Recording validation needs review."
    )


# ==========================================
# 4. BASELINE FEATURES
# ==========================================

log("\n===== SPECTRAL FEATURE AUDIT =====")

feature_path = (
    REPORTS / "baseline_features_v2.csv"
)

feature_rows = indexed(
    read_csv(feature_path)
)

expected_features = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
]

if set(feature_rows) != set(ids):
    raise RuntimeError(
        "Baseline feature subjects mismatch."
    )

X = np.array([
    [
        float(feature_rows[sid][feature])
        for feature in expected_features
    ]
    for sid in ids
])

if (
    X.shape != (46, 6)
    or not np.isfinite(X).all()
):
    raise RuntimeError(
        "Invalid spectral feature matrix."
    )

log(
    "Spectral feature matrix:",
    X.shape
)

log(
    "All spectral features finite: PASS"
)


# ==========================================
# 5. MODEL PREDICTION AUDIT
# ==========================================

log("\n===== MODEL PREDICTION AUDIT =====")

baseline_path = (
    REPORTS / "baseline_predictions_v2.csv"
)

baseline = indexed(
    read_csv(baseline_path)
)

if set(baseline) != set(ids):
    raise RuntimeError(
        "Baseline prediction subjects mismatch."
    )

reference_folds = {
    sid: int(baseline[sid]["fold"])
    for sid in ids
}

model_results = []
canonical_predictions = []


def verify_model(
    name,
    filename,
    prediction_column,
    score_column
):

    path = REPORTS / filename

    if not path.exists():

        log(f"{name}: REPORT MISSING")

        model_results.append({
            "model": name,
            "status": "MISSING_REPORT"
        })

        return

    records = indexed(
        read_csv(path)
    )

    if set(records) != set(ids):

        issue(
            f"{name}: participant mismatch"
        )

        model_results.append({
            "model": name,
            "status": "SUBJECT_MISMATCH"
        })

        return

    predictions = []
    scores = []

    try:

        for sid, truth in zip(ids, y):

            row = records[sid]

            if row.get("group") not in (
                "A" if truth else "C",
                None,
                ""
            ):
                raise ValueError(
                    f"Diagnosis mismatch: {sid}"
                )

            if int(row["fold"]) != reference_folds[sid]:
                raise ValueError(
                    f"Outer-fold mismatch: {sid}"
                )

            if row.get("true_label") not in ("", None):

                if int(float(row["true_label"])) != truth:
                    raise ValueError(
                        f"True-label mismatch: {sid}"
                    )

            pred = int(
                float(row[prediction_column])
            )

            score = float(
                row[score_column]
            )

            if (
                pred not in (0, 1)
                or not np.isfinite(score)
                or not (0 <= score <= 1)
            ):
                raise ValueError(
                    f"Invalid prediction: {sid}"
                )

            predictions.append(pred)
            scores.append(score)

    except (KeyError, ValueError) as exc:

        issue(
            f"{name}: invalid report: {exc}"
        )

        model_results.append({
            "model": name,
            "status": "INVALID_REPORT"
        })

        return

    predictions = np.array(predictions)
    scores = np.array(scores)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    result = {
        "model": name,
        "status":
            "METRICS_VERIFIED_FROM_SAVED_OOF",
        "subjects": len(ids),
        "accuracy": round(
            accuracy_score(
                y, predictions
            ),
            6
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(
                y, predictions
            ),
            6
        ),
        "roc_auc": round(
            roc_auc_score(
                y, scores
            ),
            6
        ),
        "sensitivity": round(
            tp / (tp + fn), 6
        ),
        "specificity": round(
            tn / (tn + fp), 6
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    model_results.append(result)

    for i, sid in enumerate(ids):

        canonical_predictions.append({
            "subject": sid,
            "group": (
                "A" if y[i] else "C"
            ),
            "model": name,
            "duration_sec": 300,
            "repeat": 1,
            "outer_fold": reference_folds[sid],
            "true_label": int(y[i]),
            "predicted_label":
                int(predictions[i]),
            "ad_probability_estimate":
                float(scores[i]),
            "source_report": filename
        })

    log(
        f"{name}: "
        f"accuracy={result['accuracy']:.4f}, "
        f"BACC={result['balanced_accuracy']:.4f}, "
        f"AUC={result['roc_auc']:.4f}"
    )


# Classical baseline report.
classical = {
    "LogisticRegression":
        "LogisticRegression",
    "SVM": "SVM",
    "RandomForest": "RandomForest",
    "NestedSelection": "NestedSelection"
}

for name, prefix in classical.items():

    verify_model(
        name,
        "baseline_predictions_v2.csv",
        prefix + "_prediction",
        prefix + "_ad_probability"
    )


# Deep-learning and foundation-model reports.
advanced = {
    "BasicCNN":
        "eeg_cnn_cv_predictions.csv",
    "EEGNetV1":
        "eegnet_cv_predictions.csv",
    "EEGNetV2":
        "eegnet_v2_cv_predictions.csv",
    "EEGConformer":
        "eeg_conformer_cv_predictions.csv",
    "CBraModFrozen":
        "cbramod_frozen_cv_predictions.csv"
}

for name, filename in advanced.items():

    verify_model(
        name,
        filename,
        "predicted_label",
        "ad_probability"
    )

save_csv(
    OUTPUT / "verified_model_metrics.csv",
    model_results
)

save_csv(
    OUTPUT / "canonical_predictions.csv",
    canonical_predictions
)


# ==========================================
# 6. REPEATED NESTED CV
# ==========================================

log("\n===== REPEATED CV AUDIT =====")

repeated_path = (
    REPORTS /
    "robustness_repeated_nested_cv_runs.csv"
)

if repeated_path.exists():

    repeated = read_csv(repeated_path)

    for name in (
        "LogisticRegression",
        "RBF_SVM"
    ):

        records = [
            r for r in repeated
            if r["model"] == name
        ]

        if not records:
            issue(
                f"No repeated-CV runs for {name}"
            )
            continue

        values = np.array([
            float(r["balanced_accuracy"])
            for r in records
        ])

        log(
            f"{name}: "
            f"runs={len(values)}, "
            f"mean={values.mean():.4f}, "
            f"SD={values.std(ddof=1):.4f}, "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}"
        )

else:
    issue(
        "Repeated nested-CV report missing."
    )


# ==========================================
# 7. DURATION EXPERIMENT STATUS
# ==========================================

log("\n===== DURATION EXPERIMENT AUDIT =====")

duration_feature_path = (
    REPORTS /
    "duration_features_60_180_300.csv"
)

duration_run_path = (
    REPORTS /
    "duration_nested_cv_runs.csv"
)

if duration_feature_path.exists():

    duration_rows = read_csv(
        duration_feature_path
    )

    counts = Counter(
        int(float(r["duration_sec"]))
        for r in duration_rows
    )

    log(
        "Duration feature rows:",
        len(duration_rows)
    )

    log(
        "Per-duration counts:",
        dict(counts)
    )

    if counts != {
        60: 46,
        180: 46,
        300: 46
    }:
        issue(
            "Incomplete or duplicated "
            "duration-feature conditions."
        )

    duration_by_key = {
        (
            r["subject"],
            int(float(r["duration_sec"]))
        ): r
        for r in duration_rows
    }

    if len(duration_by_key) == 138:

        differences = {}

        for feature in expected_features:

            differences[feature] = max(
                abs(
                    float(
                        feature_rows[sid][feature]
                    )
                    -
                    float(
                        duration_by_key[
                            (sid, 300)
                        ][feature]
                    )
                )
                for sid in ids
            )

        log(
            "300-second baseline "
            "feature differences:"
        )

        for feature, diff in differences.items():

            log(
                f"  {feature}: {diff:.10f}"
            )

        worst = max(
            differences.values()
        )

        if worst <= 1e-6:

            log(
                "300-second feature "
                "reproduction: PASS"
            )

        else:

            issue(
                "300-second feature "
                "reproduction: MISMATCH. "
                "Do not use the duration "
                "classification results yet."
            )

else:

    log(
        "Duration feature report: "
        "NOT YET AVAILABLE"
    )


if duration_run_path.exists():

    duration_runs = read_csv(
        duration_run_path
    )

    log(
        "Duration CV rows:",
        len(duration_runs)
    )

    groups = {}

    for r in duration_runs:

        key = (
            r["model"],
            int(float(r["duration_sec"]))
        )

        groups.setdefault(
            key, []
        ).append(
            float(
                r["balanced_accuracy"]
            )
        )

    for key, values in sorted(
        groups.items()
    ):

        log(
            f"{key}: "
            f"runs={len(values)}, "
            f"mean BACC={np.mean(values):.4f}"
        )

else:

    log(
        "Duration CV report: NOT YET AVAILABLE"
    )


# ==========================================
# 8. SCRIPT AND REPORT INVENTORY
# ==========================================

log("\n===== PROJECT INVENTORY =====")

important = [
    "baseline_features_v2.csv",
    "baseline_predictions_v2.csv",
    "eeg_cnn_cv_predictions.csv",
    "eegnet_v2_cv_predictions.csv",
    "eeg_conformer_cv_predictions.csv",
    "cbramod_frozen_cv_predictions.csv",
    "fusion_ablation_cv_predictions.csv",
    "robustness_repeated_nested_cv_runs.csv",
    "robustness_subject_audit.csv",
    "spectral_explain_attributions.csv",
    "spectral_explain_permutation.csv",
    "duration_features_60_180_300.csv",
    "duration_nested_cv_runs.csv",
    "duration_theta_alpha_interaction.csv"
]

inventory = []

for name in important:

    path = REPORTS / name

    inventory.append({
        "file": str(path),
        "exists": int(path.exists()),
        "size_bytes":
            path.stat().st_size
            if path.exists() else 0
    })

save_csv(
    OUTPUT / "file_inventory.csv",
    inventory
)

for item in inventory:

    log(
        ("FOUND  " if item["exists"] else "MISSING")
        + " " + Path(item["file"]).name
    )


# ==========================================
# 9. FINAL AUDIT SUMMARY
# ==========================================

log("\n===== FINAL AUDIT SUMMARY =====")

log(
    "Participants in primary cohort:",
    len(included)
)

log(
    "Participant model predictions audited:",
    len(canonical_predictions)
)

log(
    "Audited model result rows:",
    len(model_results)
)

log(
    "Recording errors:",
    len(recording_errors)
)

log(
    "Boundary crossing errors:",
    len(boundary_errors)
)

log(
    "Issues requiring review:",
    len(ISSUES)
)

for item in ISSUES:
    log(" - " + item)

log("\nAudit files:", OUTPUT)

log(
    "Original EEG and existing "
    "experiment reports unchanged."
)

log(
    "Metrics verified from CSVs "
    "do not independently validate "
    "the original training implementations."
)

(OUTPUT / "audit_console_report.txt").write_text(
    "\n".join(LOG) + "\n",
    encoding="utf-8"
)

with (
    OUTPUT / "audit_metadata.json"
).open("w") as f:

    json.dump(
        {
            "cohort_subjects": ids,
            "included_n": len(ids),
            "model_results":
                len(model_results),
            "issues": ISSUES
        },
        f,
        indent=2
    )

log("\n===== TERMINAL AUDIT COMPLETE =====")


if __name__ == "__main__":
    pass

