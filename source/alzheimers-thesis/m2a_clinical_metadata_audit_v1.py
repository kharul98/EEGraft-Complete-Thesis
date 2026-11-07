#!/usr/bin/env python3
"""Milestone 2A: clinical metadata and saved-prediction feasibility audit."""

from pathlib import Path
from collections import Counter
from statistics import mean, median, stdev
import csv
import hashlib
import json
import urllib.request

import numpy as np


ROOT = Path.cwd()
BASE = ROOT / "reports" / "thesis_final"
OUT = BASE / "milestone_2a_clinical_metadata_v1"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "reports" / "openneuro_5min_manifest_v2.csv"

CAL = (
    BASE / "milestone_1b_calibration_v1"
    / "calibrated_held_out_block_predictions.csv"
)

STATES = (
    BASE / "milestone_1a_reliability_v1"
    / "subject_repeat_states.csv"
)

RISK = (
    BASE / "milestone_1c_selective_prediction_v1"
    / "risk_coverage_summary.csv"
)

ERROR_AUC = (
    BASE / "milestone_1c_selective_prediction_v1"
    / "error_detection_auroc.csv"
)

PUBLIC_URL = (
    "https://raw.githubusercontent.com/"
    "OpenNeuroDatasets/ds004504/main/participants.tsv"
)


def read_rows(path, sep=","):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f, delimiter=sep))


def save(name, rows):

    if not rows:
        raise RuntimeError("Empty output: " + name)

    cols = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with (OUT / name).open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=cols
        )

        writer.writeheader()
        writer.writerows(rows)


def unique(rows, key, label):

    out = {}

    for row in rows:

        k = key(row)

        if k in out:
            raise RuntimeError(
                "Duplicate " + label + ": " + str(k)
            )

        out[k] = row

    return out


def num(text, lo, hi, name, sid):

    s = str(text or "").strip()

    if s.lower() in (
        "", "n/a", "na", "nan", "none", "null"
    ):
        return None

    try:
        x = float(s)

    except ValueError:
        raise RuntimeError(
            f"Invalid {name} for {sid}: {s}"
        )

    if not np.isfinite(x) or not lo <= x <= hi:
        raise RuntimeError(
            f"Out-of-range {name} for {sid}: {x}"
        )

    return x


def values_summary(v):

    a = [
        float(x)
        for x in v
        if x is not None
    ]

    return {
        "available": len(a),
        "mean": round(mean(a), 3) if a else "",
        "sd": round(stdev(a), 3) if len(a) > 1 else "",
        "median": round(median(a), 3) if a else "",
        "minimum": min(a) if a else "",
        "maximum": max(a) if a else "",
        "unique_values": len(set(a)),
    }


print(
    "===== MILESTONE 2A: MMSE / CLINICAL METADATA PREFLIGHT =====",
    flush=True
)


# ==========================================
# 1. CHECK PREVIOUS RESULT FILES
# ==========================================

needed = (
    MANIFEST,
    CAL,
    STATES,
    RISK,
    ERROR_AUC
)

for p in needed:

    if not p.is_file():
        raise FileNotFoundError(p)


# ==========================================
# 2. LOCATE CLINICAL METADATA
# ==========================================

candidates = [
    ROOT / "data/eeg/on004504_v1.0.0/participants.tsv",
    ROOT / "data/eeg/ds004504/participants.tsv",
    ROOT / "data/ds004504/participants.tsv",
    ROOT / "participants.tsv",
]

local = [
    p for p in candidates
    if p.is_file()
]

if not local and (ROOT / "data").is_dir():

    local = sorted(
        p
        for p in (ROOT / "data").rglob(
            "participants.tsv"
        )
        if "004504" in str(p.parent)
    )


if local:

    metadata_path = local[0]
    origin = "local dataset file"

else:

    metadata_path = (
        OUT / "participants_from_public_ds004504_main.tsv"
    )

    if not metadata_path.is_file():

        print(
            "No local participants.tsv found; "
            "retrieving only the small public metadata table.",
            flush=True
        )

        try:

            with urllib.request.urlopen(
                PUBLIC_URL,
                timeout=20
            ) as response:

                content = response.read(100000)

        except Exception as exc:

            raise RuntimeError(
                "Cannot fetch public metadata. "
                "Obtain ds004504/participants.tsv locally "
                "and rerun."
            ) from exc

        if (
            not content.startswith(
                b"participant_id\t"
            )
            or len(content) > 10000
        ):
            raise RuntimeError(
                "Remote content is not the expected metadata TSV"
            )

        metadata_path.write_bytes(content)

    origin = (
        "public ds004504 GitHub main "
        "(version may differ from local NEMAR copy)"
    )


# ==========================================
# 3. VERIFY PARTICIPANT IDENTITIES
# ==========================================

manifest = unique(
    read_rows(MANIFEST),
    lambda r: r["subject"],
    "manifest subject"
)

meta_rows = read_rows(
    metadata_path,
    "\t"
)

required = {
    "participant_id",
    "Gender",
    "Age",
    "Group",
    "MMSE"
}

if (
    not meta_rows
    or not required.issubset(meta_rows[0])
):
    raise RuntimeError(
        "participants.tsv is missing expected columns: "
        + str(required)
    )

meta = unique(
    meta_rows,
    lambda r: r["participant_id"].strip(),
    "metadata participant"
)

if (
    len(meta) != 88
    or set(meta) != set(manifest)
):
    raise RuntimeError(
        f"Clinical metadata ID mismatch: "
        f"{len(meta)} metadata rows versus "
        f"{len(manifest)} manifest rows"
    )

if Counter(
    r["group"]
    for r in manifest.values()
) != {
    "A": 36,
    "F": 23,
    "C": 29
}:
    raise RuntimeError(
        "Original dataset group counts differ"
    )


clinical = {}

for sid, r in meta.items():

    group = r["Group"].strip()

    if group != manifest[sid]["group"]:
        raise RuntimeError(
            f"Group mismatch for {sid}: "
            f"{group} versus {manifest[sid]['group']}"
        )

    clinical[sid] = {
        "group": group,
        "status": manifest[sid]["status"],
        "age": num(
            r["Age"], 0, 120, "Age", sid
        ),
        "mmse": num(
            r["MMSE"], 0, 30, "MMSE", sid
        ),
        "gender": r["Gender"].strip()
    }


selected = {
    sid for sid, c in clinical.items()
    if c["status"] == "CANDIDATE"
}

if Counter(
    clinical[s]["group"]
    for s in selected
) != {
    "A": 23,
    "F": 18,
    "C": 23
}:
    raise RuntimeError(
        "Selected cohorts do not match previous experiments"
    )


# ==========================================
# 4. LOCK INPUT FINGERPRINTS
# ==========================================

fingerprints = {
    str(p.relative_to(ROOT)):
        hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    for p in needed
}

fingerprints[
    str(metadata_path.relative_to(ROOT))
] = hashlib.sha256(
    metadata_path.read_bytes()
).hexdigest()

lock = OUT / "input_sha256.json"

serialized = (
    json.dumps(
        fingerprints,
        sort_keys=True,
        indent=2
    ) + "\n"
)

if (
    lock.is_file()
    and lock.read_text() != serialized
):
    raise RuntimeError(
        "Clinical audit inputs changed since "
        "previous run; inspect source files."
    )

if not lock.is_file():
    lock.write_text(serialized)


print("Metadata source:", metadata_path)
print("Metadata origin:", origin)

print(
    "PASS: All 88 subject IDs and diagnostic "
    "labels match the frozen manifest."
)

print(
    "PASS: Selected subjects = "
    "23 AD + 18 FTD + 23 healthy "
    "(64 unique)."
)

print(
    "Available fields: Gender, Age, Group, MMSE. "
    "Per-subject disease duration: NOT in this TSV."
)


# ==========================================
# 5. ASSESS MMSE / AGE COMPLETENESS
# ==========================================

summary_rows = []

print(
    "\n===== SELECTED VS EXCLUDED "
    "MMSE / AGE AVAILABILITY ====="
)

for group in ("A", "F", "C"):

    for subset, selected_flag in (
        ("SELECTED", True),
        ("EXCLUDED", False)
    ):

        ids = [
            sid
            for sid, c in clinical.items()
            if (
                c["group"] == group
                and (sid in selected) == selected_flag
            )
        ]

        m = values_summary(
            clinical[s]["mmse"]
            for s in ids
        )

        a = values_summary(
            clinical[s]["age"]
            for s in ids
        )

        summary_rows.append({
            "group": group,
            "subset": subset,
            "n": len(ids),
            "mmse_available": m["available"],
            "mmse_mean": m["mean"],
            "mmse_sd": m["sd"],
            "mmse_median": m["median"],
            "mmse_min": m["minimum"],
            "mmse_max": m["maximum"],
            "mmse_unique_values": m["unique_values"],
            "age_available": a["available"],
            "age_mean": a["mean"],
            "age_sd": a["sd"],
            "age_median": a["median"],
        })

        print(
            f'{group} {subset}: '
            f'n={len(ids)}, '
            f'MMSE={m["available"]}/{len(ids)} '
            f'(median={m["median"]}, '
            f'range={m["minimum"]}-{m["maximum"]}, '
            f'unique={m["unique_values"]}); '
            f'Age={a["available"]}/{len(ids)}'
        )


save(
    "clinical_eligibility_summary.csv",
    summary_rows
)


# ==========================================
# 6. VERIFY PREVIOUS UNCERTAINTY GAINS
# ==========================================

risk = read_rows(RISK)
auc = read_rows(ERROR_AUC)

print(
    "\n===== MILESTONE 1C GAIN INTERVALS "
    "(SAVED RESULTS) ====="
)

for task in (
    "AD_vs_HC",
    "FTD_vs_HC"
):

    for method, cov in (
        (
            "first_minute_original_label_support",
            0.80
        ),
        (
            "five_minute_label_consensus",
            0.70
        )
    ):

        hits = [
            r for r in risk
            if (
                r["task"] == task
                and r["method"] == method
                and abs(
                    float(r["target_coverage"]) - cov
                ) < 1e-8
            )
        ]

        if len(hits) != 1:
            raise RuntimeError(
                f"Missing risk-coverage result: "
                f"{task}, {method}, {cov}"
            )

        r = hits[0]

        print(
            f"{task} {method} {cov:.0%}: "
            f'selected BACC='
            f'{float(r["mean_selected_BACC"]):.4f}, '
            f'gain='
            f'{float(r["BACC_gain_over_full_coverage"]):+.4f}, '
            f'conditional gain CI=['
            f'{float(r["conditional_gain_ci_low"]):+.4f}, '
            f'{float(r["conditional_gain_ci_high"]):+.4f}]'
        )

    hits = [
        r for r in auc
        if (
            r["task"] == task
            and r["method"]
            == "first_minute_original_label_support"
            and r["uncertainty_target"]
            == "stable_wrong_among_unanimous"
        )
    ]

    if len(hits) != 1:
        raise RuntimeError(
            "Missing stable-only error AUROC"
        )

    r = hits[0]

    print(
        f'{task} stable-only error AUROC='
        f'{float(r["auroc_higher_is_better"]):.4f}, '
        f'conditional CI=['
        f'{float(r["conditional_ci_low"]):.4f}, '
        f'{float(r["conditional_ci_high"]):.4f}]'
    )


# ==========================================
# 7. JOIN MMSE WITH SAVED PREDICTIONS
# ==========================================

preds = unique(
    read_rows(CAL),
    lambda r: (
        r["task"],
        r["subject"],
        int(r["repeat"]),
        int(r["minute"])
    ),
    "calibrated block probability"
)

state_rows = unique(
    read_rows(STATES),
    lambda r: (
        r["task"],
        r["subject"],
        int(r["repeat"])
    ),
    "stability state"
)

if (
    len(preds) != 5 * 5 * (46 + 41)
    or len(state_rows) != 5 * (46 + 41)
):
    raise RuntimeError(
        "Expected probability/state records are missing"
    )

joined = []

for task, patient in (
    ("AD_vs_HC", "A"),
    ("FTD_vs_HC", "F")
):

    ids = sorted(
        sid
        for sid in selected
        if clinical[sid]["group"] in (
            patient, "C"
        )
    )

    expected_n = (
        46 if task == "AD_vs_HC"
        else 41
    )

    if len(ids) != expected_n:
        raise RuntimeError(
            "Task participant count mismatch"
        )

    for sid in ids:

        c = clinical[sid]

        first_p = []
        first_margin = []
        error = []
        support = []
        prob_sd = []
        unstable = []
        stable_wrong = []

        for rep in range(1, 6):

            first = preds[
                (task, sid, rep, 1)
            ]

            state = state_rows[
                (task, sid, rep)
            ]

            if (
                first["group"] != c["group"]
                or state["group"] != c["group"]
            ):
                raise RuntimeError(
                    "Task diagnosis mismatch: "
                    + task + " " + sid
                )

            probs = np.array([
                float(
                    preds[
                        (task, sid, rep, minute)
                    ]["calibrated_positive_probability"]
                )
                for minute in range(1, 6)
            ])

            p = float(
                first["calibrated_positive_probability"]
            )

            y = int(
                c["group"] == patient
            )

            label = int(
                first["original_svm_label"]
            )

            if (
                int(first["true_label"]) != y
                or not np.all(
                    (probs > 0) & (probs < 1)
                )
            ):
                raise RuntimeError(
                    "Prediction consistency failure"
                )

            first_p.append(p)

            first_margin.append(
                float(
                    first[
                        "original_svm_decision_score"
                    ]
                )
            )

            support.append(
                p if label == 1 else 1 - p
            )

            error.append(
                int(label != y)
            )

            prob_sd.append(
                float(
                    np.std(
                        probs,
                        ddof=1
                    )
                )
            )

            unstable.append(
                int(
                    state["state"] == "unstable"
                )
            )

            stable_wrong.append(
                int(
                    state["state"] == "stable_wrong"
                )
            )

        joined.append({
            "task": task,
            "subject": sid,
            "group": c["group"],
            "mmse":
                "" if c["mmse"] is None
                else c["mmse"],
            "age":
                "" if c["age"] is None
                else c["age"],
            "gender": c["gender"],
            "mean_first_minute_disease_probability":
                float(np.mean(first_p)),
            "mean_first_minute_signed_svm_margin":
                float(np.mean(first_margin)),
            "mean_first_minute_original_label_support":
                float(np.mean(support)),
            "first_minute_error_rate_across_five_repeats":
                float(np.mean(error)),
            "five_minute_probability_sd_mean_across_repeats":
                float(np.mean(prob_sd)),
            "temporal_class_reversal_rate_across_repeats":
                float(np.mean(unstable)),
            "stable_wrong_rate_across_repeats":
                float(np.mean(stable_wrong)),
            "correlated_cv_repeats": 5,
        })


if len(joined) != 87:
    raise RuntimeError(
        "Expected 87 task-subject rows"
    )

save(
    "clinical_prediction_subject_join.csv",
    joined
)


# ==========================================
# 8. PRINT FEASIBILITY SUMMARY
# ==========================================

print(
    "\n===== CLINICAL ASSOCIATION FEASIBILITY ====="
)

for task, patient in (
    ("AD_vs_HC", "A"),
    ("FTD_vs_HC", "F")
):

    subset = [
        r for r in joined
        if (
            r["task"] == task
            and r["group"] == patient
        )
    ]

    mm = [
        float(r["mmse"])
        for r in subset
        if r["mmse"] != ""
    ]

    age = [
        float(r["age"])
        for r in subset
        if r["age"] != ""
    ]

    unique_mm = len(set(mm))

    print(
        f"{task} patient-only: "
        f"subjects={len(subset)}, "
        f"MMSE available={len(mm)}, "
        f"unique MMSE values={unique_mm}, "
        f"age available={len(age)}"
    )

    if (
        len(mm) < 10
        or unique_mm < 4
    ):
        print(
            "  CAUTION: Within-patient severity "
            "associations may be underpowered "
            "or unidentifiable."
        )


controls = [
    r for r in joined
    if (
        r["task"] == "AD_vs_HC"
        and r["group"] == "C"
    )
]

control_mm = [
    float(r["mmse"])
    for r in controls
    if r["mmse"] != ""
]

print(
    "Shared healthy controls: n=23, "
    f"MMSE unique values={len(set(control_mm))}; "
    "do not calculate a within-control "
    "correlation when MMSE is constant."
)

print(
    "The 87 task-subject rows represent "
    "64 unique participants because "
    "healthy controls are reused."
)

print(
    "No severity correlations or regression "
    "models have been fitted yet."
)

print(
    "\n===== MILESTONE 2A COMPLETE ====="
)

print("Saved:", OUT)

print(
    "Original EEG, classifiers, "
    "and prior reports unchanged."
)

