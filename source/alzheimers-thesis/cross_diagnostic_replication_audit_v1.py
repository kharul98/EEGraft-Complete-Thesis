#!/usr/bin/env python3
"""Independent audit of AD/HC and FTD/HC temporal reliability."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np

ROOT = Path.cwd()
R = ROOT / "reports"
S = R / "thesis_final"

OUT = S / "cross_diagnostic_replication_v1"
OUT.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "manifest": R / "openneuro_5min_manifest_v2.csv",
    "ad_duration":
        S / "duration_locked_v1/held_out_predictions.csv",
    "ad_blocks":
        S / "temporal_repeated_v1/"
        "all_repeats_heldout_block_predictions.csv",
    "ftd_duration":
        S / "ftd_extension_v1/locked_study/"
        "held_out_duration_predictions.csv",
    "ftd_blocks":
        S / "ftd_extension_v1/locked_study/"
        "held_out_60s_block_predictions.csv"
}

REPEATS = 5
NBOOT = 3000
DURATIONS = (60, 180, 300)
BLOCKS = (1, 2, 3, 4, 5)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    if not rows:
        raise RuntimeError("Empty output: " + name)

    columns = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with (OUT / name).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def make_index(rows, key_fn):
    result = {}

    for row in rows:
        key = key_fn(row)

        if key in result:
            raise RuntimeError(
                "Duplicate record: " + str(key)
            )

        result[key] = row

    return result


print(
    "===== CROSS-DIAGNOSTIC REPLICATION AUDIT =====",
    flush=True
)

hashes = {}

for name, path in INPUTS.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    hashes[name] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

hash_file = OUT / "source_hashes.json"
encoded = json.dumps(
    hashes,
    indent=2,
    sort_keys=True
) + "\n"

if hash_file.exists():
    if hash_file.read_text() != encoded:
        raise RuntimeError(
            "Input files changed since previous audit."
        )
else:
    hash_file.write_text(encoded)


# ==========================================
# VERIFY ORIGINAL COHORTS
# ==========================================

manifest = read_csv(INPUTS["manifest"])

ad_ids = sorted(
    r["subject"] for r in manifest
    if r["group"] in ("A", "C")
    and r["status"] == "CANDIDATE"
)

ftd_ids = sorted(
    r["subject"] for r in manifest
    if r["group"] in ("F", "C")
    and r["status"] == "CANDIDATE"
)

group = {
    r["subject"]: r["group"]
    for r in manifest
}

if (
    len(ad_ids) != 46
    or len(ftd_ids) != 41
    or len(set(ad_ids)) != 46
    or len(set(ftd_ids)) != 41
):
    raise RuntimeError(
        "Cohort eligibility mismatch"
    )

if Counter(
    group[sid] for sid in ad_ids
) != {"A": 23, "C": 23}:
    raise RuntimeError("AD cohort changed")

if Counter(
    group[sid] for sid in ftd_ids
) != {"F": 18, "C": 23}:
    raise RuntimeError("FTD cohort changed")

shared_hc = sorted(
    set(ad_ids).intersection(ftd_ids)
)

if (
    len(shared_hc) != 23
    or any(group[sid] != "C" for sid in shared_hc)
):
    raise RuntimeError(
        "Healthy-control overlap invalid"
    )


# ==========================================
# BUILD VERIFIED PREDICTION ARRAYS
# ==========================================

def load_task(name, ids, positive):

    dur_rows = read_csv(
        INPUTS[name + "_duration"]
    )

    if name == "ad":
        dur_rows = [
            r for r in dur_rows
            if r["model"] == "RBF_SVM"
        ]

    block_rows = read_csv(
        INPUTS[name + "_blocks"]
    )

    duration_index = make_index(
        dur_rows,
        lambda r: (
            int(r["repeat"]),
            r["subject"],
            int(r["duration_sec"])
        )
    )

    block_index = make_index(
        block_rows,
        lambda r: (
            int(r["repeat"]),
            r["subject"],
            int(r["block"])
        )
    )

    n = len(ids)

    if len(duration_index) != REPEATS*n*3:
        raise RuntimeError(
            f"{name}: duration rows incomplete"
        )

    if len(block_index) != REPEATS*n*5:
        raise RuntimeError(
            f"{name}: block rows incomplete"
        )

    labels = np.array([
        int(group[sid] == positive)
        for sid in ids
    ])

    duration = np.full(
        (REPEATS, n, 3),
        -1,
        dtype=int
    )

    blocks = np.full(
        (REPEATS, n, 5),
        -1,
        dtype=int
    )

    for rep in range(1, REPEATS+1):
        for i, sid in enumerate(ids):

            original = duration_index[
                (rep, sid, 60)
            ]

            fold = int(original["fold"])

            for d, seconds in enumerate(DURATIONS):
                row = duration_index[
                    (rep, sid, seconds)
                ]

                if (
                    int(row["fold"]) != fold
                    or int(row["true_label"]) != labels[i]
                ):
                    raise RuntimeError(
                        f"{name}: fold/label mismatch {sid}"
                    )

                duration[rep-1, i, d] = int(
                    row["predicted_label"]
                )

            for b in BLOCKS:
                row = block_index[
                    (rep, sid, b)
                ]

                if (
                    int(row["fold"]) != fold
                    or int(row["true_label"]) != labels[i]
                ):
                    raise RuntimeError(
                        f"{name}: block mismatch {sid}"
                    )

                blocks[rep-1, i, b-1] = int(
                    row["predicted_label"]
                )

            if (
                duration[rep-1, i, 0]
                != blocks[rep-1, i, 0]
            ):
                raise RuntimeError(
                    f"{name}: first-minute inconsistency {sid}"
                )

    if (
        not np.isin(duration, [0, 1]).all()
        or not np.isin(blocks, [0, 1]).all()
    ):
        raise RuntimeError(
            f"{name}: missing/invalid predictions"
        )

    return {
        "ids": ids,
        "index": {
            sid: i for i, sid in enumerate(ids)
        },
        "labels": labels,
        "duration": duration,
        "blocks": blocks,
        "flips": (
            blocks != blocks[:, :, :1]
        ).any(axis=2)
    }


AD = load_task("ad", ad_ids, "A")
FTD = load_task("ftd", ftd_ids, "F")

print(
    "PASS: 46 AD/HC and 41 FTD/HC participants; "
    "all five CV repetitions and all prediction "
    "positions reconciled."
)

print(
    "PASS: Same 23 healthy-control IDs appear "
    "in both tasks."
)


# ==========================================
# INFERENCE UTILITIES
# ==========================================

rng = np.random.default_rng(20261107)

idx_ad = np.array([
    AD["index"][sid]
    for sid in ad_ids
    if group[sid] == "A"
])

idx_ftd = np.array([
    FTD["index"][sid]
    for sid in ftd_ids
    if group[sid] == "F"
])

idx_ad_hc = np.array([
    AD["index"][sid]
    for sid in shared_hc
])

idx_ftd_hc = np.array([
    FTD["index"][sid]
    for sid in shared_hc
])


def task_bacc(matrix, pos_idx, hc_idx):

    sensitivity = (
        matrix[:, pos_idx, :] == 1
    ).mean()

    specificity = (
        matrix[:, hc_idx, :] == 0
    ).mean()

    return float(
        0.5 * (sensitivity + specificity)
    )


def group_flip(task, indices):
    return float(
        task["flips"][:, indices].mean()
    )


def calculate(a, f, hc_a, hc_f):

    ad_duration = AD["duration"]
    ftd_duration = FTD["duration"]

    ad_blocks = AD["blocks"]
    ftd_blocks = FTD["blocks"]

    results = {}

    for name, matrix, positive, healthy in (
        ("AD", ad_duration, a, hc_a),
        ("FTD", ftd_duration, f, hc_f)
    ):
        for d, seconds in enumerate(DURATIONS):
            results[
                f"{name}_{seconds}s_BACC"
            ] = task_bacc(
                matrix[:, :, d:d+1],
                positive,
                healthy
            )

    for name, matrix, positive, healthy in (
        ("AD", ad_blocks, a, hc_a),
        ("FTD", ftd_blocks, f, hc_f)
    ):
        for block in range(5):
            results[
                f"{name}_minute{block+1}_BACC"
            ] = task_bacc(
                matrix[:, :, block:block+1],
                positive,
                healthy
            )

    results["AD_patient_reversal"] = group_flip(
        AD, a
    )

    results["AD_HC_reversal"] = group_flip(
        AD, hc_a
    )

    results["FTD_patient_reversal"] = group_flip(
        FTD, f
    )

    results["FTD_HC_reversal"] = group_flip(
        FTD, hc_f
    )

    results["AD_class_balanced_reversal"] = (
        results["AD_patient_reversal"]
        + results["AD_HC_reversal"]
    ) / 2

    results["FTD_class_balanced_reversal"] = (
        results["FTD_patient_reversal"]
        + results["FTD_HC_reversal"]
    ) / 2

    results["FTD_minus_AD_reversal"] = (
        results["FTD_class_balanced_reversal"]
        - results["AD_class_balanced_reversal"]
    )

    for name in ("AD", "FTD"):

        results[f"{name}_minute5_minus_minute1"] = (
            results[f"{name}_minute5_BACC"]
            - results[f"{name}_minute1_BACC"]
        )

        results[f"{name}_180_minus_60"] = (
            results[f"{name}_180s_BACC"]
            - results[f"{name}_60s_BACC"]
        )

        results[f"{name}_300_minus_60"] = (
            results[f"{name}_300s_BACC"]
            - results[f"{name}_60s_BACC"]
        )

    results["FTD_minus_AD_minute1_BACC"] = (
        results["FTD_minute1_BACC"]
        - results["AD_minute1_BACC"]
    )

    results["FTD_minus_AD_minute5_BACC"] = (
        results["FTD_minute5_BACC"]
        - results["AD_minute5_BACC"]
    )

    results["shared_HC_reversal_difference"] = (
        results["FTD_HC_reversal"]
        - results["AD_HC_reversal"]
    )

    return results


observed = calculate(
    idx_ad,
    idx_ftd,
    idx_ad_hc,
    idx_ftd_hc
)


# ==========================================
# STRATIFIED SUBJECT-CLUSTER BOOTSTRAP
# ==========================================

bootstrap = {
    key: []
    for key in observed
}

for _ in range(NBOOT):

    sampled_ad = rng.choice(
        idx_ad,
        size=len(idx_ad),
        replace=True
    )

    sampled_ftd = rng.choice(
        idx_ftd,
        size=len(idx_ftd),
        replace=True
    )

    # Identical resampled healthy-control IDs
    # are applied to both classification tasks.
    shared_positions = rng.integers(
        0,
        len(shared_hc),
        size=len(shared_hc)
    )

    sampled_ad_hc = idx_ad_hc[
        shared_positions
    ]

    sampled_ftd_hc = idx_ftd_hc[
        shared_positions
    ]

    values = calculate(
        sampled_ad,
        sampled_ftd,
        sampled_ad_hc,
        sampled_ftd_hc
    )

    for key, value in values.items():
        bootstrap[key].append(value)


interval_rows = []

for key, value in observed.items():

    lo, hi = np.percentile(
        bootstrap[key],
        [2.5, 97.5]
    )

    interval_rows.append({
        "outcome": key,
        "estimate": value,
        "conditional_95CI_low": float(lo),
        "conditional_95CI_high": float(hi),
        "bootstrap_subject_resamples": NBOOT
    })

save_csv(
    "cross_diagnostic_claims.csv",
    interval_rows
)


# ==========================================
# SUBJECT-LEVEL REVERSAL PROFILES
# ==========================================

profiles = []

for task_name, task in (
    ("AD_vs_HC", AD),
    ("FTD_vs_HC", FTD)
):

    for i, sid in enumerate(task["ids"]):

        profiles.append({
            "task": task_name,
            "subject": sid,
            "group": group[sid],
            "reversal_repetitions": int(
                task["flips"][:, i].sum()
            ),
            "reversal_rate_across_repeats": float(
                task["flips"][:, i].mean()
            ),
            "minute1_correct_repetitions": int(
                (
                    task["blocks"][:, i, 0]
                    == task["labels"][i]
                ).sum()
            ),
            "minute5_correct_repetitions": int(
                (
                    task["blocks"][:, i, 4]
                    == task["labels"][i]
                ).sum()
            )
        })

save_csv(
    "participant_reversal_profiles.csv",
    profiles
)


# ==========================================
# DISPLAY CORE FINDINGS
# ==========================================

lookup = {
    r["outcome"]: r
    for r in interval_rows
}


def show(key):

    r = lookup[key]

    print(
        f"{key}: "
        f"{r['estimate']:+.4f}, "
        f"conditional 95% CI "
        f"[{r['conditional_95CI_low']:+.4f}, "
        f"{r['conditional_95CI_high']:+.4f}]"
    )


print(
    "\n===== CROSS-DIAGNOSTIC REVERSAL ANALYSIS ====="
)

for key in (
    "AD_patient_reversal",
    "AD_HC_reversal",
    "FTD_patient_reversal",
    "FTD_HC_reversal",
    "AD_class_balanced_reversal",
    "FTD_class_balanced_reversal",
    "FTD_minus_AD_reversal",
    "shared_HC_reversal_difference"
):
    show(key)


print(
    "\n===== WITHIN-TASK DURATION COMPARISONS ====="
)

for key in (
    "AD_180_minus_60",
    "AD_300_minus_60",
    "FTD_180_minus_60",
    "FTD_300_minus_60"
):
    show(key)


print(
    "\n===== RECORDING-POSITION COMPARISONS ====="
)

for key in (
    "AD_minute5_minus_minute1",
    "FTD_minute5_minus_minute1",
    "FTD_minus_AD_minute1_BACC",
    "FTD_minus_AD_minute5_BACC"
):
    show(key)


print(
    "\n===== SHARED-CONTROL SUBJECT AUDIT ====="
)

for sid in shared_hc:

    ia = AD["index"][sid]
    i_f = FTD["index"][sid]

    print(
        f"{sid}: "
        f"AD-model flip repetitions="
        f"{int(AD['flips'][:, ia].sum())}/5; "
        f"FTD-model flip repetitions="
        f"{int(FTD['flips'][:, i_f].sum())}/5"
    )


# ==========================================
# WRITE TRACEABLE SUMMARY
# ==========================================

notes = f"""# Cross-diagnostic temporal reliability audit

Status: exploratory, post-hoc analysis.

## Data and verification

- AD versus healthy: 23 AD + 23 controls.
- FTD versus healthy: 18 FTD + 23 controls.
- The exact same 23 controls are reused.
- Five repeated participant-level nested-CV evaluations.
- Five nonoverlapping held-out one-minute EEG blocks.
- No model was retrained for this audit.

## Main observations

AD/HC mean reversal rate:
{observed['AD_class_balanced_reversal']:.4f}

FTD/HC class-balanced reversal rate:
{observed['FTD_class_balanced_reversal']:.4f}

FTD minus AD reversal-rate difference:
{observed['FTD_minus_AD_reversal']:+.4f}

AD minute-5 minus minute-1 balanced accuracy:
{observed['AD_minute5_minus_minute1']:+.4f}

FTD minute-5 minus minute-1 balanced accuracy:
{observed['FTD_minute5_minus_minute1']:+.4f}

## Interpretation boundaries

The finding is a within-dataset replication
of the same methodology in another
diagnostic classification contrast.

It is not external-site validation,
proof of common disease biomarkers,
prospective early diagnosis, or
evidence of state-of-the-art accuracy.

The positive-class cohorts differ in size.
Healthy controls are shared across tasks.
Bootstrap comparisons resample those
shared controls jointly and preserve
within-participant CV repetitions.

Bootstrap intervals are conditional on
the existing fitted cross-validation
predictions; they do not fully represent
new-cohort or model-retraining uncertainty.

The original EEG window-selection
algorithm was not recovered from source.
FTD eligibility relies on the previously
frozen manifest. The quality and
representativeness limitations remain.
"""

(
    OUT / "CROSS_DIAGNOSTIC_EVIDENCE.md"
).write_text(
    notes,
    encoding="utf-8"
)

print(
    "\n===== CROSS-DIAGNOSTIC AUDIT COMPLETE ====="
)

print("Saved to:", OUT)

print(
    "Historical data and model predictions unchanged."
)

