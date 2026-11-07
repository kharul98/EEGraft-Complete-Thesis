#!/usr/bin/env python3

"""
Temporal stress test using existing held-out SVM predictions.

No model training.
No changes to historical results.
All findings are exploratory.
"""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np
from scipy.stats import beta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()

SRC = (
    ROOT / "reports"
    / "thesis_final"
    / "duration_locked_v1"
)

OUT = (
    ROOT / "reports"
    / "thesis_final"
    / "temporal_stress_v1"
)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

SEED = 20261107
N_BOOT = 3000

FEATURES = (
    "delta", "theta", "alpha",
    "beta", "gamma", "theta_alpha_ratio"
)


# =========================================
# LOCK ANALYSIS PROTOCOL
# =========================================

PROTOCOL = {
    "analysis_name":
        "Within-recording held-out 60-second "
        "block transportability",

    "version": "1.0",

    "status":
        "Post-hoc exploratory extension; "
        "not preregistered",

    "cohort":
        "46 ds004504 participants, "
        "23 AD and 23 healthy controls",

    "evaluation_unit": "participant",

    "primary_observation_already_known":
        "24 of 46 subject predictions changed "
        "across 60-second blocks",

    "model":
        "RBF_SVM trained and tuned on initial "
        "60 seconds of training subjects; "
        "repeat 1 outer folds",

    "fixed_splits":
        "Original repeat-1 matched "
        "5-fold CV from duration_locked_v1",

    "comparison_1":
        "One held-out model evaluated on "
        "five nonoverlapping 60-second blocks",

    "comparison_2":
        "Majority vote of first three and "
        "all five blocks without retraining",

    "comparison_3":
        "Agreement-based retention at "
        "4/5 and 5/5",

    "comparison_4":
        "Comparison with existing 180s "
        "and 300s SVM models",

    "reporting":
        "Subject-level BACC, accuracy, "
        "reversal rates, stratified bootstrap CIs",

    "exploratory_associations":
        "Class flips versus signal-quality "
        "and theta-alpha variability",

    "limitations": [
        "One outer-CV repetition",
        "Same retrospective cohort",
        "No external validation",
        "Full-session preprocessing",
        "No clinical deployment claim"
    ]
}

protocol_path = OUT / "locked_protocol.json"

serialized = (
    json.dumps(
        PROTOCOL,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if protocol_path.exists():

    if protocol_path.read_text() != serialized:
        raise RuntimeError(
            "Existing locked protocol differs. "
            "Review before changing the analysis."
        )

else:
    protocol_path.write_text(serialized)


# =========================================
# UTILITIES
# =========================================

def read_rows(name):

    path = SRC / name

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save_rows(name, rows):

    if not rows:
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with (OUT / name).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


def numeric(value):

    if value is None or str(value).strip() == "":
        return float("nan")

    return float(value)


def balanced_metric(ids, labels, predictions):

    if not ids:
        return None

    y = np.array([
        labels[s] for s in ids
    ])

    pred = np.array([
        predictions[s] for s in ids
    ])

    if set(y) != {0, 1}:
        return None

    sensitivity = (
        pred[y == 1] == 1
    ).mean()

    specificity = (
        pred[y == 0] == 0
    ).mean()

    return float(
        (sensitivity + specificity) / 2
    )


def metric_summary(ids, labels, predictions):

    y = np.array([
        labels[s] for s in ids
    ])

    pred = np.array([
        predictions[s] for s in ids
    ])

    if set(y) != {0, 1}:
        raise ValueError(
            "Both classes required for BACC"
        )

    return {
        "n": len(ids),
        "ad": int(y.sum()),
        "healthy": int(len(y) - y.sum()),

        "accuracy": float(
            (y == pred).mean()
        ),

        "balanced_accuracy":
            balanced_metric(
                ids, labels, predictions
            ),

        "sensitivity": float(
            (pred[y == 1] == 1).mean()
        ),

        "specificity": float(
            (pred[y == 0] == 0).mean()
        )
    }


def stratified_resample(ids, labels, rng):

    ad = [
        s for s in ids
        if labels[s] == 1
    ]

    hc = [
        s for s in ids
        if labels[s] == 0
    ]

    if not ad or not hc:
        raise ValueError(
            "Both classes required"
        )

    return (
        list(
            rng.choice(
                ad, size=len(ad), replace=True
            )
        )
        +
        list(
            rng.choice(
                hc, size=len(hc), replace=True
            )
        )
    )


def bacc_interval(ids, labels, predictions, rng):

    values = []

    for _ in range(N_BOOT):

        sample = stratified_resample(
            ids, labels, rng
        )

        values.append(
            balanced_metric(
                sample, labels, predictions
            )
        )

    return [
        float(v)
        for v in np.percentile(
            values, [2.5, 97.5]
        )
    ]


def paired_difference(
    ids,
    labels,
    first,
    second,
    rng
):

    observed = (
        balanced_metric(ids, labels, first)
        -
        balanced_metric(ids, labels, second)
    )

    boot = []

    for _ in range(N_BOOT):

        sample = stratified_resample(
            ids, labels, rng
        )

        boot.append(
            balanced_metric(
                sample, labels, first
            )
            -
            balanced_metric(
                sample, labels, second
            )
        )

    lo, hi = np.percentile(
        boot, [2.5, 97.5]
    )

    return {
        "delta_balanced_accuracy":
            float(observed),

        "conditional_ci_low":
            float(lo),

        "conditional_ci_high":
            float(hi)
    }


def binomial_interval(k, n):

    lo = (
        float(beta.ppf(.025, k, n-k+1))
        if k else 0.0
    )

    hi = (
        float(beta.ppf(.975, k+1, n-k))
        if k < n else 1.0
    )

    return lo, hi


# =========================================
# LOAD ORIGINAL SAVED RESULTS
# =========================================

print(
    "===== TEMPORAL STRESS TEST =====",
    flush=True
)

print("Input directory:", SRC)
print("Locked protocol:", protocol_path)

pred_rows = read_rows(
    "held_out_predictions.csv"
)

block_rows = read_rows(
    "held_out_60s_blocks.csv"
)

feature_rows = read_rows(
    "five_60s_blocks.csv"
)

duration_rows = read_rows(
    "duration_features.csv"
)


# Save input hashes for reproducibility.

fingerprints = []

for filename in (
    "held_out_predictions.csv",
    "held_out_60s_blocks.csv",
    "five_60s_blocks.csv",
    "duration_features.csv"
):

    path = SRC / filename

    fingerprints.append({
        "file": filename,
        "sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    })

save_rows(
    "source_sha256.csv",
    fingerprints
)


# =========================================
# VERIFY MATCHED SUBJECTS AND FOLDS
# =========================================

base = {}

for row in pred_rows:

    if (
        row.get("model") != "RBF_SVM"
        or int(row.get("repeat", -1)) != 1
    ):
        continue

    duration = int(
        row["duration_sec"]
    )

    if duration not in (60, 180, 300):
        continue

    key = (
        row["subject"],
        duration
    )

    if key in base:
        raise RuntimeError(
            "Duplicate held-out prediction: "
            + str(key)
        )

    base[key] = row


ids = sorted({
    sid
    for sid, duration in base
    if duration == 60
})

if len(ids) != 46:
    raise RuntimeError(
        "Expected 46 subjects"
    )

if any(
    (sid, duration) not in base
    for sid in ids
    for duration in (60, 180, 300)
):
    raise RuntimeError(
        "Duration participant mismatch"
    )


labels = {
    sid: int(
        base[(sid, 60)]["true_label"]
    )
    for sid in ids
}

if Counter(labels.values()) != {
    0: 23, 1: 23
}:
    raise RuntimeError(
        "Unexpected diagnostic distribution"
    )


for sid in ids:

    for duration in (60, 180, 300):

        r = base[(sid, duration)]

        if (
            int(r["true_label"]) != labels[sid]
            or int(r["fold"])
            != int(base[(sid, 60)]["fold"])
        ):
            raise RuntimeError(
                f"Fold or label mismatch: {sid}"
            )


# =========================================
# VERIFY NONOVERLAPPING BLOCKS
# =========================================

blocks = {}

for row in block_rows:

    key = (
        row["subject"],
        int(row["block"])
    )

    if key in blocks:
        raise RuntimeError(
            "Duplicate prediction block"
        )

    blocks[key] = row


features = {}

for row in feature_rows:

    key = (
        row["subject"],
        int(row["block_number"])
    )

    if key in features:
        raise RuntimeError(
            "Duplicate feature block"
        )

    features[key] = row


prefix = {}

for row in duration_rows:

    key = (
        row["subject"],
        int(row["duration_sec"])
    )

    if key in prefix:
        raise RuntimeError(
            "Duplicate duration feature row"
        )

    prefix[key] = row


if (
    len(blocks) != 230
    or len(features) != 230
):
    raise RuntimeError(
        "Expected 46 x 5 block records"
    )


for sid in ids:

    for j in range(1, 6):

        key = (sid, j)

        if (
            key not in blocks
            or key not in features
        ):
            raise RuntimeError(
                "Missing block " + str(key)
            )

        a = blocks[key]
        b = features[key]

        if (
            a["group"] != b["group"]
            or int(a["fold"])
            != int(base[(sid, 60)]["fold"])
        ):
            raise RuntimeError(
                "Group/fold mismatch " + str(key)
            )

        if (
            int(a["clean_epochs"])
            != int(b["clean_epochs"])
        ):
            raise RuntimeError(
                "Clean epoch mismatch " + str(key)
            )

    # The first block must equal the saved
    # 60-second feature representation.

    for feature in FEATURES:

        x = float(
            features[(sid, 1)][feature]
        )

        y = float(
            prefix[(sid, 60)][feature]
        )

        if not np.isclose(
            x, y,
            rtol=0,
            atol=1e-12
        ):
            raise RuntimeError(
                f"First-block feature mismatch: "
                f"{sid}, {feature}"
            )

    # The prediction from first block must
    # exactly reproduce original held-out
    # 60-second SVM prediction.

    first = blocks[(sid, 1)]
    original = base[(sid, 60)]

    if first["predicted_ad"] == "":
        raise RuntimeError(
            "Missing initial prediction: " + sid
        )

    if (
        int(first["predicted_ad"])
        != int(original["predicted_label"])
    ):
        raise RuntimeError(
            "Initial prediction mismatch: " + sid
        )

    if not np.isclose(
        float(first["decision_score"]),
        float(original["decision_score"]),
        rtol=0,
        atol=1e-7
    ):
        raise RuntimeError(
            "Initial decision-score mismatch: " + sid
        )


print(
    "VALIDATION PASS: 46 subjects, "
    "5 blocks, matched labels/folds, "
    "original 60s predictions reproduced."
)

rng = np.random.default_rng(SEED)


# =========================================
# HELD-OUT PERFORMANCE BY BLOCK POSITION
# =========================================

position_rows = []
choices = {}

for j in range(1, 6):

    present = [
        sid for sid in ids
        if blocks[(sid, j)]["predicted_ad"] != ""
    ]

    predicted = {
        sid: int(
            blocks[(sid, j)]["predicted_ad"]
        )
        for sid in present
    }

    choices[j] = predicted

    stats = metric_summary(
        present,
        labels,
        predicted
    )

    lo, hi = bacc_interval(
        present,
        labels,
        predicted,
        rng
    )

    position_rows.append({
        "block": j,
        "start_sec": (j-1)*60,
        "end_sec": j*60,
        **stats,
        "bacc_ci_low": lo,
        "bacc_ci_high": hi
    })


save_rows(
    "block_position_metrics.csv",
    position_rows
)

print(
    "\n===== HELD-OUT SEGMENT POSITION ====="
)

for r in position_rows:

    print(
        f"Block {r['block']} "
        f"({r['start_sec']}-{r['end_sec']}s): "
        f"n={r['n']}, "
        f"BACC={r['balanced_accuracy']:.4f}, "
        f"CI=[{r['bacc_ci_low']:.4f}, "
        f"{r['bacc_ci_high']:.4f}], "
        f"ACC={r['accuracy']:.4f}"
    )


# =========================================
# PARTICIPANT PREDICTION STABILITY
# =========================================

patient_rows = []
consensus3 = {}
consensus5 = {}

for sid in ids:

    present = [
        j for j in range(1, 6)
        if sid in choices[j]
    ]

    decisions = [
        choices[j][sid]
        for j in present
    ]

    quality = [
        numeric(
            features[(sid, j)]["clean_fraction"]
        )
        for j in range(1, 6)
    ]

    ratios = [
        numeric(
            features[(sid, j)]["theta_alpha_ratio"]
        )
        for j in range(1, 6)
    ]

    ratios = [
        np.log(x)
        for x in ratios
        if np.isfinite(x) and x > 0
    ]

    scores = [
        numeric(
            blocks[(sid, j)]["decision_score"]
        )
        for j in present
    ]

    flip = (
        int(len(set(decisions)) > 1)
        if len(present) >= 2
        else 0
    )

    # First three consecutive minutes.
    if all(
        sid in choices[j]
        for j in (1, 2, 3)
    ):

        consensus3[sid] = int(
            sum(
                choices[j][sid]
                for j in (1, 2, 3)
            ) >= 2
        )

    vote5 = ""
    agreement = ""

    if len(present) == 5:

        votes = sum(decisions)

        vote5 = int(
            votes >= 3
        )

        agreement = max(
            votes,
            5 - votes
        )

        consensus5[sid] = vote5

    patient_rows.append({
        "subject": sid,
        "group": "A" if labels[sid] else "C",
        "true_label": labels[sid],
        "fold": int(base[(sid, 60)]["fold"]),
        "usable_blocks": len(present),
        "any_class_flip": flip,
        "max_agreement_votes": agreement,
        "consensus5": vote5,

        "decision_score_range":
            float(np.ptp(scores))
            if len(scores) >= 2 else "",

        "mean_clean_fraction":
            float(np.mean(quality)),

        "range_clean_fraction":
            float(np.ptp(quality)),

        "sd_log_theta_alpha_ratio":
            float(np.std(ratios, ddof=1))
            if len(ratios) >= 2 else ""
    })


save_rows(
    "participant_stability.csv",
    patient_rows
)

eligible = [
    r for r in patient_rows
    if r["usable_blocks"] == 5
]

flips = sum(
    r["any_class_flip"]
    for r in eligible
)

lo, hi = binomial_interval(
    flips,
    len(eligible)
)

print(
    "\n===== WITHIN-RECORDING CLASS REVERSALS ====="
)

print(
    f"Participants changing class: "
    f"{flips}/{len(eligible)} "
    f"({flips / len(eligible):.3f})"
)

print(
    f"Exact binomial 95% CI: "
    f"[{lo:.3f}, {hi:.3f}]"
)


# =========================================
# PREDEFINED CONSENSUS STRATEGIES
# =========================================

methods = {
    "First 60s": choices[1],

    "Three-block majority": consensus3,

    "Five-block majority": consensus5,

    "180s retrained SVM": {
        sid: int(
            base[(sid, 180)]["predicted_label"]
        )
        for sid in ids
    },

    "300s retrained SVM": {
        sid: int(
            base[(sid, 300)]["predicted_label"]
        )
        for sid in ids
    }
}

method_rows = []

for name, predicted in methods.items():

    selected = [
        sid for sid in ids
        if sid in predicted
    ]

    stats = metric_summary(
        selected,
        labels,
        predicted
    )

    lo, hi = bacc_interval(
        selected,
        labels,
        predicted,
        rng
    )

    method_rows.append({
        "method": name,
        **stats,
        "bacc_ci_low": lo,
        "bacc_ci_high": hi
    })


save_rows(
    "consensus_and_duration_comparison.csv",
    method_rows
)

print(
    "\n===== LOCKED AGGREGATION COMPARISON ====="
)

for r in method_rows:

    print(
        f"{r['method']}: "
        f"N={r['n']}, "
        f"BACC={r['balanced_accuracy']:.4f}, "
        f"ACC={r['accuracy']:.4f}"
    )


# =========================================
# PAIRED PERFORMANCE COMPARISONS
# =========================================

paired_rows = []

for name in (
    "Three-block majority",
    "Five-block majority",
    "180s retrained SVM",
    "300s retrained SVM"
):

    selected = [
        sid for sid in ids
        if sid in choices[1]
        and sid in methods[name]
    ]

    result = paired_difference(
        selected,
        labels,
        methods[name],
        choices[1],
        rng
    )

    paired_rows.append({
        "comparison": name + " minus first 60s",
        "n": len(selected),
        **result
    })


save_rows(
    "paired_bacc_changes.csv",
    paired_rows
)

print(
    "\n===== PAIRED BALANCED ACCURACY CHANGES ====="
)

for r in paired_rows:

    print(
        f"{r['comparison']}: "
        f"{r['delta_balanced_accuracy']:+.4f}, "
        f"conditional 95% CI "
        f"[{r['conditional_ci_low']:+.4f}, "
        f"{r['conditional_ci_high']:+.4f}]"
    )


# =========================================
# AGREEMENT-BASED ABSTENTION
# =========================================

risk_rows = []

for threshold in (3, 4, 5):

    selected = [
        r for r in patient_rows
        if r["usable_blocks"] == 5
        and r["max_agreement_votes"] >= threshold
    ]

    selected_ids = [
        r["subject"]
        for r in selected
    ]

    predicted = {
        sid: consensus5[sid]
        for sid in selected_ids
    }

    if len({
        labels[sid]
        for sid in selected_ids
    }) == 2:

        stats = metric_summary(
            selected_ids,
            labels,
            predicted
        )

    else:
        stats = None

    risk_rows.append({
        "minimum_agreement": threshold,
        "retained_n": len(selected_ids),
        "coverage": len(selected_ids)/46,

        "accuracy_on_retained":
            stats["accuracy"]
            if stats else "",

        "bacc_on_retained":
            stats["balanced_accuracy"]
            if stats else "",

        "ad_retained": sum(
            labels[sid]
            for sid in selected_ids
        ),

        "healthy_retained": sum(
            labels[sid] == 0
            for sid in selected_ids
        )
    })


save_rows(
    "agreement_based_abstention.csv",
    risk_rows
)

print(
    "\n===== AGREEMENT-BASED ABSTENTION ====="
)

for r in risk_rows:

    print(
        f"Agreement >= {r['minimum_agreement']}/5: "
        f"retained={r['retained_n']}/46, "
        f"coverage={r['coverage']:.3f}, "
        f"BACC={r['bacc_on_retained']}, "
        f"ACC={r['accuracy_on_retained']}"
    )


# =========================================
# EXPLORATORY ASSOCIATIONS
# =========================================

association_rows = []

for field in (
    "range_clean_fraction",
    "sd_log_theta_alpha_ratio"
):

    complete = [
        r for r in patient_rows
        if r["usable_blocks"] == 5
        and np.isfinite(float(r[field]))
    ]

    unstable = np.array([
        float(r[field])
        for r in complete
        if r["any_class_flip"] == 1
    ])

    stable = np.array([
        float(r[field])
        for r in complete
        if r["any_class_flip"] == 0
    ])

    if not len(unstable) or not len(stable):
        continue

    observed = float(
        unstable.mean() - stable.mean()
    )

    values = np.array([
        float(r[field])
        for r in complete
    ])

    flips_array = np.array([
        int(r["any_class_flip"])
        for r in complete
    ])

    groups = np.array([
        r["group"]
        for r in complete
    ])

    extreme = 0

    # Preserve AD/control grouping.
    for _ in range(5000):

        shuffled = flips_array.copy()

        for group in ("A", "C"):

            idx = np.where(
                groups == group
            )[0]

            shuffled[idx] = rng.permutation(
                shuffled[idx]
            )

        difference = float(
            values[shuffled == 1].mean()
            -
            values[shuffled == 0].mean()
        )

        extreme += (
            abs(difference) >= abs(observed)
        )

    p = (extreme + 1) / 5001

    association_rows.append({
        "variable": field,
        "n": len(complete),
        "mean_unstable": float(unstable.mean()),
        "mean_stable": float(stable.mean()),
        "difference": observed,
        "exploratory_permutation_p": p
    })


save_rows(
    "exploratory_flip_associations.csv",
    association_rows
)

print(
    "\n===== EXPLORATORY QUALITY AND FEATURE ASSOCIATIONS ====="
)

for r in association_rows:

    print(
        f"{r['variable']}: "
        f"unstable minus stable="
        f"{r['difference']:+.5f}, "
        f"exploratory p="
        f"{r['exploratory_permutation_p']:.4f}"
    )


# =========================================
# FIGURE 1: BLOCK POSITION PERFORMANCE
# =========================================

fig, ax = plt.subplots(
    figsize=(7.2, 4.6)
)

x = np.array([
    r["block"]
    for r in position_rows
])

y = np.array([
    r["balanced_accuracy"]
    for r in position_rows
])

yerr = np.array([
    [
        r["balanced_accuracy"] - r["bacc_ci_low"]
        for r in position_rows
    ],
    [
        r["bacc_ci_high"] - r["balanced_accuracy"]
        for r in position_rows
    ]
])

ax.errorbar(
    x,
    y,
    yerr=yerr,
    marker="o",
    capsize=4
)

ax.set(
    xlabel="Nonoverlapping 60-second block",
    ylabel="Participant-level balanced accuracy",
    title="Fixed SVM performance across recording positions",
    xticks=x,
    ylim=(0, 1)
)

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    OUT / "figures/block_position_bacc.png",
    dpi=190
)

plt.close(fig)


# =========================================
# FIGURE 2: AGGREGATION COMPARISON
# =========================================

fig, ax = plt.subplots(
    figsize=(8.1, 4.7)
)

names = [
    r["method"]
    for r in method_rows
]

values = [
    r["balanced_accuracy"]
    for r in method_rows
]

ax.barh(
    names,
    values
)

ax.set(
    xlim=(0, 1),
    xlabel="Participant-level balanced accuracy",
    title="Short-block aggregation versus duration models"
)

ax.invert_yaxis()

for i, value in enumerate(values):

    ax.text(
        min(value + .012, .95),
        i,
        f"{value:.3f}",
        va="center",
        fontsize=9
    )

fig.tight_layout()

fig.savefig(
    OUT / "figures/aggregation_comparison.png",
    dpi=190
)

plt.close(fig)


print(
    "\n===== TEMPORAL STRESS TEST COMPLETE ====="
)

print("Outputs:", OUT)

print(
    "No new model training. "
    "Historical reports unchanged."
)

print(
    "Results are exploratory and based on "
    "one matched cross-validation repetition."
)

