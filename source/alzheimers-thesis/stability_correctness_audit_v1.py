#!/usr/bin/env python3

"""Audit stable-correct, stable-wrong and unstable EEG predictions."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np


ROOT = Path.cwd()

S = ROOT / "reports" / "thesis_final"

SOURCES = {
    "AD_vs_HC":
        S / "temporal_repeated_v1"
        / "all_repeats_heldout_block_predictions.csv",

    "FTD_vs_HC":
        S / "ftd_extension_v1"
        / "locked_study"
        / "held_out_60s_block_predictions.csv"
}

OUT = S / "stability_correctness_audit_v1"
OUT.mkdir(parents=True, exist_ok=True)

REPEATS = 5
BLOCKS = 5
BOOTSTRAPS = 5000


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("Empty result: " + str(path))

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


print(
    "===== STABILITY-VERSUS-CORRECTNESS AUDIT =====",
    flush=True
)


# =============================================
# 1. LOCK INPUTS
# =============================================

checksums = {
    name: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for name, path in SOURCES.items()
}

lock = OUT / "input_checksums.json"

encoded = json.dumps(
    checksums,
    indent=2,
    sort_keys=True
) + "\n"

if lock.exists():
    if lock.read_text() != encoded:
        raise RuntimeError(
            "Saved prediction inputs have changed."
        )
else:
    lock.write_text(encoded)


# =============================================
# 2. LOAD AND VERIFY TASK PREDICTIONS
# =============================================

def load_task(name, expected_group, expected_n):

    rows = load_csv(SOURCES[name])

    if len(rows) != expected_n * REPEATS * BLOCKS:
        raise RuntimeError(
            f"{name}: incorrect prediction count"
        )

    subjects = {}

    for row in rows:

        sid = row["subject"]

        if sid in subjects:
            continue

        subjects[sid] = {
            "group": row["group"],
            "label": int(row["true_label"])
        }

    ids = sorted(subjects)

    if len(ids) != expected_n:
        raise RuntimeError(
            f"{name}: unexpected unique participants"
        )

    groups = Counter(
        subjects[sid]["group"]
        for sid in ids
    )

    if name == "AD_vs_HC":
        expected = {"A": 23, "C": 23}
    else:
        expected = {"F": 18, "C": 23}

    if groups != expected:
        raise RuntimeError(
            f"{name}: unexpected group counts {groups}"
        )

    index = {
        sid: i
        for i, sid in enumerate(ids)
    }

    y = np.array([
        subjects[sid]["label"]
        for sid in ids
    ])

    for sid in ids:

        expected_label = int(
            subjects[sid]["group"] == expected_group
        )

        if subjects[sid]["label"] != expected_label:
            raise RuntimeError(
                f"{name}: label mismatch: {sid}"
            )

    predictions = np.full(
        (REPEATS, len(ids), BLOCKS),
        -1,
        dtype=int
    )

    fold_map = np.full(
        (REPEATS, len(ids)),
        -1,
        dtype=int
    )

    for row in rows:

        repeat = int(row["repeat"]) - 1
        block = int(row["block"]) - 1
        i = index[row["subject"]]

        if (
            repeat not in range(REPEATS)
            or block not in range(BLOCKS)
        ):
            raise RuntimeError(
                f"{name}: invalid repeat or block"
            )

        if predictions[repeat, i, block] != -1:
            raise RuntimeError(
                f"{name}: duplicate prediction"
            )

        prediction = int(
            row["predicted_label"]
        )

        fold = int(row["fold"])

        if prediction not in (0, 1):
            raise RuntimeError(
                f"{name}: invalid prediction"
            )

        if fold_map[repeat, i] == -1:
            fold_map[repeat, i] = fold

        if fold_map[repeat, i] != fold:
            raise RuntimeError(
                f"{name}: inconsistent subject fold"
            )

        if int(row["true_label"]) != y[i]:
            raise RuntimeError(
                f"{name}: diagnosis mismatch"
            )

        predictions[repeat, i, block] = prediction

    if (
        (predictions < 0).any()
        or (fold_map < 1).any()
    ):
        raise RuntimeError(
            f"{name}: incomplete prediction matrix"
        )

    unanimous = (
        predictions == predictions[:, :, :1]
    ).all(axis=2)

    stable_correct = (
        unanimous
        & (
            predictions[:, :, 0]
            == y[None, :]
        )
    )

    stable_wrong = (
        unanimous
        & (
            predictions[:, :, 0]
            != y[None, :]
        )
    )

    unstable = ~unanimous

    if not np.all(
        stable_correct.astype(int)
        + stable_wrong.astype(int)
        + unstable.astype(int)
        == 1
    ):
        raise RuntimeError(
            f"{name}: state partition failure"
        )

    return {
        "name": name,
        "ids": ids,
        "index": index,
        "groups": subjects,
        "y": y,
        "pred": predictions,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "unstable": unstable
    }


AD = load_task(
    "AD_vs_HC",
    "A",
    46
)

FTD = load_task(
    "FTD_vs_HC",
    "F",
    41
)

hc_ids = sorted(
    sid
    for sid in AD["ids"]
    if AD["groups"][sid]["group"] == "C"
)

ftd_hc_ids = sorted(
    sid
    for sid in FTD["ids"]
    if FTD["groups"][sid]["group"] == "C"
)

if hc_ids != ftd_hc_ids or len(hc_ids) != 23:
    raise RuntimeError(
        "Shared healthy controls are not identical"
    )

print(
    "PASS: All five blocks and five repeats "
    "reconciled in both tasks."
)

print(
    "PASS: Same 23 healthy participants "
    "in AD/HC and FTD/HC."
)


# =============================================
# 3. DECOMPOSE PREDICTION STABILITY
# =============================================

group_rows = []

print(
    "\n===== STABILITY AND CORRECTNESS BY GROUP ====="
)

for task in (AD, FTD):

    if task["name"] == "AD_vs_HC":
        group_names = ("A", "C")
    else:
        group_names = ("F", "C")

    for group in group_names:

        idx = np.array([
            task["index"][sid]
            for sid in task["ids"]
            if task["groups"][sid]["group"] == group
        ])

        correct = float(
            task["stable_correct"][:, idx].mean()
        )

        wrong = float(
            task["stable_wrong"][:, idx].mean()
        )

        unstable = float(
            task["unstable"][:, idx].mean()
        )

        block_accuracy = float(
            (
                task["pred"][:, idx, :]
                ==
                task["y"][idx][None, :, None]
            ).mean()
        )

        predicted_positive = float(
            task["pred"][:, idx, :].mean()
        )

        if not np.isclose(
            correct + wrong + unstable,
            1.0,
            atol=1e-12
        ):
            raise RuntimeError(
                "Stability proportions do not sum to one"
            )

        group_rows.append({
            "task": task["name"],
            "group": group,
            "independent_subjects": len(idx),
            "correlated_subject_repeat_evaluations":
                len(idx) * REPEATS,
            "stable_correct_fraction": correct,
            "stable_wrong_fraction": wrong,
            "unstable_fraction": unstable,
            "mean_block_accuracy": block_accuracy,
            "predicted_positive_fraction":
                predicted_positive
        })

        print(
            f"{task['name']} / {group}: "
            f"stable-correct={correct:.4f}, "
            f"stable-wrong={wrong:.4f}, "
            f"unstable={unstable:.4f}, "
            f"block accuracy={block_accuracy:.4f}, "
            f"positive prediction fraction="
            f"{predicted_positive:.4f}"
        )


write_csv(
    OUT / "group_stability_decomposition.csv",
    group_rows
)


# =============================================
# 4. COMPARE THE EXACT SAME HEALTHY EEGs
# =============================================

ad_indices = np.array([
    AD["index"][sid]
    for sid in hc_ids
])

ftd_indices = np.array([
    FTD["index"][sid]
    for sid in hc_ids
])

rng = np.random.default_rng(20261107)

comparison_rows = []

print(
    "\n===== PAIRED SHARED-CONTROL DIFFERENCES ====="
)

metrics = (
    "stable_correct",
    "stable_wrong",
    "unstable"
)

for metric in metrics:

    # Each element is one participant's average
    # rate across five correlated CV repetitions.

    ad_values = (
        AD[metric][:, ad_indices]
        .mean(axis=0)
    )

    ftd_values = (
        FTD[metric][:, ftd_indices]
        .mean(axis=0)
    )

    differences = (
        ftd_values - ad_values
    )

    estimate = float(
        differences.mean()
    )

    bootstrap_indices = rng.integers(
        0,
        len(hc_ids),
        size=(BOOTSTRAPS, len(hc_ids))
    )

    samples = differences[
        bootstrap_indices
    ].mean(axis=1)

    lo, hi = np.percentile(
        samples,
        [2.5, 97.5]
    )

    comparison_rows.append({
        "metric": metric,
        "AD_HC_fraction":
            float(ad_values.mean()),
        "FTD_HC_fraction":
            float(ftd_values.mean()),
        "FTD_minus_AD_difference":
            estimate,
        "conditional_95ci_low":
            float(lo),
        "conditional_95ci_high":
            float(hi)
    })

    print(
        f"{metric}: "
        f"AD={ad_values.mean():.4f}, "
        f"FTD={ftd_values.mean():.4f}, "
        f"FTD-AD={estimate:+.4f}, "
        f"conditional 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )


write_csv(
    OUT / "shared_control_paired_comparison.csv",
    comparison_rows
)


# =============================================
# 5. INDIVIDUAL PARTICIPANT PROFILES
# =============================================

subject_rows = []

for task in (AD, FTD):

    for i, sid in enumerate(task["ids"]):

        subject_rows.append({
            "task": task["name"],
            "subject": sid,
            "group": task["groups"][sid]["group"],
            "stable_correct_repeats": int(
                task["stable_correct"][:, i].sum()
            ),
            "stable_wrong_repeats": int(
                task["stable_wrong"][:, i].sum()
            ),
            "unstable_repeats": int(
                task["unstable"][:, i].sum()
            ),
            "mean_correct_fraction_across_blocks":
                float(
                    (
                        task["pred"][:, i, :]
                        == task["y"][i]
                    ).mean()
                ),
            "mean_positive_prediction_fraction":
                float(
                    task["pred"][:, i, :].mean()
                )
        })


write_csv(
    OUT / "participant_stability_profiles.csv",
    subject_rows
)


# =============================================
# 6. AUDIT CONSISTENCY WITH PRIOR RESULTS
# =============================================

ad_hc_reversals = float(
    AD["unstable"][:, ad_indices].mean()
)

ftd_hc_reversals = float(
    FTD["unstable"][:, ftd_indices].mean()
)

if not (
    abs(ad_hc_reversals - 0.48695652173913045) < 1e-8
    and abs(ftd_hc_reversals - 0.30434782608695654) < 1e-8
):
    raise RuntimeError(
        "This audit does not reproduce previous "
        "shared-control reversal rates"
    )


print(
    "\n===== FINAL RELIABILITY CHECK ====="
)

print(
    "AD/HC shared-control reversal rate:",
    f"{ad_hc_reversals:.4f}"
)

print(
    "FTD/HC shared-control reversal rate:",
    f"{ftd_hc_reversals:.4f}"
)

print(
    "Saved:",
    OUT
)

print(
    "Interpretation reminder: "
    "a prediction can be stable but wrong."
)

print(
    "All results are exploratory and "
    "conditional on fitted CV models."
)

print(
    "\n===== STABILITY-CORRECTNESS AUDIT COMPLETE ====="
)

