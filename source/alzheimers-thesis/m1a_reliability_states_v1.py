#!/usr/bin/env python3
"""Milestone 1A: stability and correctness of held-out EEG predictions."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE = ROOT / "reports" / "thesis_final"
OUT = BASE / "milestone_1a_reliability_v1"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "reports" / "openneuro_5min_manifest_v2.csv"

TASKS = {
    "AD_vs_HC": {
        "positive": "A",
        "count": 46,
        "blocks":
            BASE / "temporal_repeated_v1"
            / "all_repeats_heldout_block_predictions.csv",
        "duration":
            BASE / "duration_locked_v1"
            / "held_out_predictions.csv",
    },
    "FTD_vs_HC": {
        "positive": "F",
        "count": 41,
        "blocks":
            BASE / "ftd_extension_v1"
            / "locked_study"
            / "held_out_60s_block_predictions.csv",
        "duration":
            BASE / "ftd_extension_v1"
            / "locked_study"
            / "held_out_duration_predictions.csv",
    },
}

STATES = (
    "stable_correct",
    "stable_wrong",
    "unstable"
)

REPEATS = 5
BLOCKS = 5
N_BOOT = 4000


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save(name, records):
    if not records:
        raise RuntimeError(
            "Empty output: " + name
        )

    cols = list(dict.fromkeys(
        k for r in records for k in r
    ))

    with (OUT / name).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=cols
        )
        writer.writeheader()
        writer.writerows(records)


def indexed(rows, key_fn, label):
    result = {}

    for r in rows:
        key = key_fn(r)

        if key in result:
            raise RuntimeError(
                "Duplicate " + label + ": " + str(key)
            )

        result[key] = r

    return result


print(
    "===== MILESTONE 1A: STABILITY AND CORRECTNESS =====",
    flush=True
)


# ==========================================
# 1. LOCK INPUT FILES
# ==========================================

paths = {
    "manifest": MANIFEST
}

for name, spec in TASKS.items():
    paths[name + "_blocks"] = spec["blocks"]
    paths[name + "_first_minute"] = spec["duration"]

hashes = {
    key: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for key, path in paths.items()
    if path.is_file()
}

if len(hashes) != len(paths):
    raise FileNotFoundError(
        "One or more required saved result files are missing"
    )

lock = OUT / "locked_input_sha256.json"

text = (
    json.dumps(
        hashes,
        sort_keys=True,
        indent=2
    ) + "\n"
)

if lock.exists() and lock.read_text() != text:
    raise RuntimeError(
        "An input changed since the previous run. "
        "Review before proceeding."
    )

if not lock.exists():
    lock.write_text(text)


# ==========================================
# 2. LOAD AND VERIFY MANIFEST
# ==========================================

manifest = load(MANIFEST)

if (
    len(manifest) != 88
    or len({
        r["subject"] for r in manifest
    }) != 88
):
    raise RuntimeError(
        "Unexpected participant manifest"
    )

output = {}
subject_repeats = []
profiles = []
group_rows = []
margin_rows = []


# ==========================================
# 3. VERIFY EACH DIAGNOSTIC TASK
# ==========================================

for task_name, spec in TASKS.items():

    positive = spec["positive"]

    expected = (
        {"A": 23, "C": 23}
        if positive == "A"
        else {"F": 18, "C": 23}
    )

    chosen = {
        r["subject"]: r["group"]
        for r in manifest
        if (
            r["group"] in (positive, "C")
            and r["status"] == "CANDIDATE"
        )
    }

    if Counter(chosen.values()) != expected:
        raise RuntimeError(
            task_name + ": eligibility changed"
        )

    ids = sorted(chosen)

    if len(ids) != spec["count"]:
        raise RuntimeError(
            "Cohort size mismatch"
        )

    y = np.asarray([
        int(chosen[sid] == positive)
        for sid in ids
    ], dtype=int)

    block_rows = load(spec["blocks"])
    duration_rows = load(spec["duration"])

    if positive == "A":
        duration_rows = [
            r for r in duration_rows
            if r.get("model") == "RBF_SVM"
        ]

    duration_rows = [
        r for r in duration_rows
        if int(r["duration_sec"]) == 60
    ]

    bi = indexed(
        block_rows,
        lambda r: (
            int(r["repeat"]),
            r["subject"],
            int(r["block"])
        ),
        "block"
    )

    di = indexed(
        duration_rows,
        lambda r: (
            int(r["repeat"]),
            r["subject"]
        ),
        "60s prediction"
    )

    if (
        len(bi) != spec["count"] * REPEATS * BLOCKS
        or len(di) != spec["count"] * REPEATS
    ):
        raise RuntimeError(
            task_name + ": incomplete prediction tables"
        )

    p = np.full(
        (REPEATS, len(ids), BLOCKS),
        -1,
        dtype=int
    )

    scores = np.full(
        (REPEATS, len(ids), BLOCKS),
        np.nan
    )

    folds = np.full(
        (REPEATS, len(ids)),
        -1,
        dtype=int
    )

    for rep in range(1, REPEATS + 1):

        for i, sid in enumerate(ids):

            reference = di[(rep, sid)]
            fold = int(reference["fold"])

            if (
                int(reference["true_label"]) != y[i]
                or not 1 <= fold <= 5
            ):
                raise RuntimeError(
                    f"{task_name}: invalid fold/label "
                    f"for {sid}"
                )

            folds[rep-1, i] = fold

            for block in range(1, BLOCKS + 1):

                r = bi[(rep, sid, block)]

                if (
                    int(r["true_label"]) != y[i]
                    or r["group"] != chosen[sid]
                    or int(r["fold"]) != fold
                ):
                    raise RuntimeError(
                        f"{task_name}: block label/fold "
                        f"mismatch for {sid}"
                    )

                prediction = int(
                    r["predicted_label"]
                )

                score = float(
                    r["decision_score"]
                )

                if (
                    prediction not in (0, 1)
                    or not np.isfinite(score)
                ):
                    raise RuntimeError(
                        "Invalid prediction or margin"
                    )

                if (
                    (score > 1e-9 and prediction == 0)
                    or
                    (score < -1e-9 and prediction == 1)
                ):
                    raise RuntimeError(
                        "Score sign and predicted class disagree"
                    )

                p[
                    rep-1, i, block-1
                ] = prediction

                scores[
                    rep-1, i, block-1
                ] = score

            if (
                p[rep-1, i, 0]
                != int(reference["predicted_label"])
                or not np.isclose(
                    scores[rep-1, i, 0],
                    float(reference["decision_score"]),
                    rtol=0,
                    atol=1e-7
                )
            ):
                raise RuntimeError(
                    f"{task_name}: first-minute "
                    f"reproduction failed: {sid}"
                )

        if set(folds[rep-1]) != {1, 2, 3, 4, 5}:
            raise RuntimeError(
                "Missing an outer fold"
            )

    # ==========================================
    # 4. CLASSIFY STABILITY STATES
    # ==========================================

    unanimous = np.all(
        p == p[:, :, :1],
        axis=2
    )

    state = np.where(
        ~unanimous,
        2,
        np.where(
            p[:, :, 0] == y[None, :],
            0,
            1
        )
    )

    if (
        np.any(state < 0)
        or np.any(state > 2)
    ):
        raise RuntimeError(
            "Invalid state partition"
        )

    output[task_name] = {
        "ids": ids,
        "group": chosen,
        "y": y,
        "p": p,
        "scores": scores,
        "state": state,
        "index": {
            sid: i for i, sid in enumerate(ids)
        }
    }

    # ==========================================
    # 5. SUBJECT-REPEAT PROFILES
    # ==========================================

    for i, sid in enumerate(ids):

        for rep in range(REPEATS):

            decisions = p[rep, i]
            raw = scores[rep, i]

            vote_fraction = float(
                decisions.mean()
            )

            if vote_fraction in (0, 1):
                entropy = 0.0

            else:
                entropy = float(
                    -vote_fraction
                    * np.log2(vote_fraction)
                    -
                    (1 - vote_fraction)
                    * np.log2(1 - vote_fraction)
                )

            subject_repeats.append({
                "task": task_name,
                "subject": sid,
                "group": chosen[sid],
                "repeat": rep + 1,
                "outer_fold":
                    int(folds[rep, i]),
                "state":
                    STATES[int(state[rep, i])],
                "true_label": int(y[i]),
                "first_minute_pred":
                    int(decisions[0]),
                "correct_minutes":
                    int(np.sum(decisions == y[i])),
                "positive_votes":
                    int(np.sum(decisions)),
                "vote_entropy_bits_not_probability_entropy":
                    entropy,
                "first_minute_raw_margin_abs":
                    float(abs(raw[0])),
                "mean_raw_margin_abs":
                    float(np.abs(raw).mean()),
                "raw_score_range":
                    float(np.ptp(raw)),
            })

        profiles.append({
            "task": task_name,
            "subject": sid,
            "group": chosen[sid],
            "stable_correct_repeats":
                int(np.sum(state[:, i] == 0)),
            "stable_wrong_repeats":
                int(np.sum(state[:, i] == 1)),
            "unstable_repeats":
                int(np.sum(state[:, i] == 2)),
            "mean_correct_minute_fraction":
                float(
                    np.mean(
                        p[:, i, :] == y[i]
                    )
                ),
        })

    # ==========================================
    # 6. GROUP-LEVEL DECOMPOSITION
    # ==========================================

    for group_label in (
        positive,
        "C",
        "ALL"
    ):

        ix = [
            i for i, sid in enumerate(ids)
            if (
                chosen[sid] == group_label
                or group_label == "ALL"
            )
        ]

        values = state[:, ix]

        counts = [
            int(np.sum(values == k))
            for k in range(3)
        ]

        n = int(values.size)

        if sum(counts) != n:
            raise RuntimeError(
                "Partition count mismatch"
            )

        correct, wrong, unstable = [
            count / n
            for count in counts
        ]

        group_rows.append({
            "task": task_name,
            "group": group_label,
            "n_independent_subjects": len(ix),
            "n_correlated_subject_repeats": n,
            "stable_correct_n": counts[0],
            "stable_wrong_n": counts[1],
            "unstable_n": counts[2],
            "stable_correct_rate": correct,
            "stable_wrong_rate": wrong,
            "unstable_rate": unstable,
            "accuracy_among_unanimous":
                correct / (correct + wrong)
                if correct + wrong else "",
            "overall_60s_block_accuracy":
                float(
                    np.mean(
                        p[:, ix]
                        == y[ix][None, :, None]
                    )
                ),
        })

    for code, label in enumerate(STATES):

        mask = state == code

        raw_first = np.abs(
            scores[:, :, 0][mask]
        )

        margin_rows.append({
            "task": task_name,
            "state": label,
            "n_subject_repeats":
                int(mask.sum()),
            "median_first_minute_abs_svm_decision_score":
                float(np.median(raw_first))
                if len(raw_first) else "",
            "note":
                "Raw SVM scores are NOT calibrated probabilities",
        })

    print(
        f"PASS: {task_name}: {len(ids)} subjects, "
        "5 repeats, 5 blocks; "
        "first-minute predictions reconciled",
        flush=True
    )


# ==========================================
# 7. VERIFY SHARED CONTROLS
# ==========================================

ad = output["AD_vs_HC"]
ftd = output["FTD_vs_HC"]

shared = sorted(
    set(
        sid for sid in ad["ids"]
        if ad["group"][sid] == "C"
    )
    &
    set(
        sid for sid in ftd["ids"]
        if ftd["group"][sid] == "C"
    )
)

if len(shared) != 23:
    raise RuntimeError(
        "Expected exactly 23 shared healthy controls"
    )

save(
    "subject_repeat_states.csv",
    subject_repeats
)

save(
    "subject_profiles.csv",
    profiles
)

save(
    "group_state_decomposition.csv",
    group_rows
)

save(
    "raw_margin_descriptives.csv",
    margin_rows
)


# ==========================================
# 8. DISPLAY STABILITY AND CORRECTNESS
# ==========================================

print(
    "\n===== STABILITY-CORRECTNESS DECOMPOSITION ====="
)

for r in group_rows:

    print(
        f"{r['task']} / {r['group']}: "
        f"stable-correct={r['stable_correct_rate']:.4f}, "
        f"stable-wrong={r['stable_wrong_rate']:.4f}, "
        f"unstable={r['unstable_rate']:.4f}, "
        "accuracy among unanimous="
        f"{r['accuracy_among_unanimous']}"
    )


# ==========================================
# 9. SUBJECT-LEVEL BOOTSTRAP
# ==========================================

rng = np.random.default_rng(20261107)
interval_rows = []

for task_name, task in output.items():

    ix_pos = np.where(
        task["y"] == 1
    )[0]

    ix_hc = np.where(
        task["y"] == 0
    )[0]

    state = task["state"]

    for k, label in enumerate(STATES):

        observed = 0.5 * (
            np.mean(state[:, ix_pos] == k)
            +
            np.mean(state[:, ix_hc] == k)
        )

        boot = np.empty(N_BOOT)

        for b in range(N_BOOT):

            a = rng.choice(
                ix_pos,
                len(ix_pos),
                replace=True
            )

            h = rng.choice(
                ix_hc,
                len(ix_hc),
                replace=True
            )

            boot[b] = 0.5 * (
                np.mean(state[:, a] == k)
                +
                np.mean(state[:, h] == k)
            )

        lo, hi = np.percentile(
            boot,
            [2.5, 97.5]
        )

        interval_rows.append({
            "task": task_name,
            "state": label,
            "class_balanced_rate":
                float(observed),
            "conditional_ci_low":
                float(lo),
            "conditional_ci_high":
                float(hi)
        })

save(
    "task_state_conditional_intervals.csv",
    interval_rows
)


# ==========================================
# 10. PAIRED SHARED-CONTROL COMPARISON
# ==========================================

comparison = []

for k, label in enumerate(STATES):

    a = np.array([
        np.mean(
            ad["state"][
                :, ad["index"][sid]
            ] == k
        )
        for sid in shared
    ])

    f = np.array([
        np.mean(
            ftd["state"][
                :, ftd["index"][sid]
            ] == k
        )
        for sid in shared
    ])

    difference = f - a

    sample = rng.integers(
        0,
        len(shared),
        size=(N_BOOT, len(shared))
    )

    boot = difference[
        sample
    ].mean(axis=1)

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5]
    )

    comparison.append({
        "state": label,
        "n_same_healthy_participants": 23,
        "AD_HC_rate": float(a.mean()),
        "FTD_HC_rate": float(f.mean()),
        "FTD_minus_AD_rate":
            float(difference.mean()),
        "conditional_ci_low":
            float(lo),
        "conditional_ci_high":
            float(hi),
    })

save(
    "shared_control_paired_state_changes.csv",
    comparison
)

print(
    "\n===== SAME-23-CONTROL PAIRED COMPARISONS ====="
)

for r in comparison:

    print(
        f"{r['state']}: "
        f"AD={r['AD_HC_rate']:.4f}, "
        f"FTD={r['FTD_HC_rate']:.4f}, "
        f"FTD-AD={r['FTD_minus_AD_rate']:+.4f}, "
        "conditional 95% CI "
        f"[{r['conditional_ci_low']:+.4f}, "
        f"{r['conditional_ci_high']:+.4f}]"
    )


# ==========================================
# 11. DESCRIPTIVE DECISION-MARGIN AUDIT
# ==========================================

print(
    "\n===== RAW DECISION MARGIN AUDIT "
    "(NOT PROBABILITIES) ====="
)

for r in margin_rows:

    print(
        f"{r['task']} / {r['state']}: "
        f"n={r['n_subject_repeats']}, "
        "median absolute first-minute margin="
        f"{r['median_first_minute_abs_svm_decision_score']}"
    )


# ==========================================
# 12. FIGURE
# ==========================================

selected = [
    r for r in group_rows
    if r["group"] != "ALL"
]

labels = [
    f"{r['task']}\n{r['group']}"
    for r in selected
]

fig, ax = plt.subplots(
    figsize=(8.5, 4.7)
)

left = np.zeros(len(selected))

for state_label, field in (
    ("Stable and correct", "stable_correct_rate"),
    ("Stable but wrong", "stable_wrong_rate"),
    ("Changes class", "unstable_rate")
):

    values = np.array([
        float(r[field])
        for r in selected
    ])

    ax.barh(
        labels,
        values,
        left=left,
        label=state_label
    )

    left += values

ax.invert_yaxis()
ax.set_xlim(0, 1)

ax.set_xlabel(
    "Fraction of participant-CV repeats (correlated)"
)

ax.set_title(
    "EEG temporal decision stability versus correctness"
)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.35),
    ncol=3,
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    OUT / "stability_correctness_groups.png",
    dpi=190,
    bbox_inches="tight"
)

plt.close(fig)


# ==========================================
# 13. COMPLETE
# ==========================================

print(
    "\n===== MILESTONE 1A COMPLETE ====="
)

print(
    "Output directory:",
    OUT
)

print(
    "The analysis used only saved held-out "
    "predictions, with no model retraining."
)

print(
    "Raw SVM decision margins have NOT been "
    "calibrated; no probability claim is made."
)

print(
    "All intervals condition on existing "
    "cross-validation models and subject selection."
)

