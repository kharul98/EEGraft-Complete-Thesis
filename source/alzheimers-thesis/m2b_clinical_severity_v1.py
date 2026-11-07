#!/usr/bin/env python3
"""Milestone 2B: subject-level MMSE associations, no model retraining."""

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE = ROOT / "reports" / "thesis_final"

OUT = BASE / "milestone_2b_mmse_associations_v1"
OUT.mkdir(parents=True, exist_ok=True)

JOIN = (
    BASE / "milestone_2a_clinical_metadata_v1"
    / "clinical_prediction_subject_join.csv"
)

MANIFEST = ROOT / "reports" / "openneuro_5min_manifest_v2.csv"

METADATA = (
    ROOT / "data/eeg/on004504_v1.0.0"
    / "participants.tsv"
)

N_PERM = 5000
N_BOOT = 3000
BASE_SEED = 20261107

# Two primary exploratory endpoints, three secondary endpoints.

METRICS = [
    (
        "disease_probability",
        "mean_first_minute_disease_probability",
        "primary"
    ),
    (
        "class_reversal_rate",
        "temporal_class_reversal_rate_across_repeats",
        "primary"
    ),
    (
        "temporal_probability_sd",
        "five_minute_probability_sd_mean_across_repeats",
        "secondary"
    ),
    (
        "original_label_support",
        "mean_first_minute_original_label_support",
        "secondary"
    ),
    (
        "stable_wrong_rate",
        "stable_wrong_rate_across_repeats",
        "secondary"
    ),
]

TASKS = [
    ("AD_vs_HC", "A", 23),
    ("FTD_vs_HC", "F", 18),
]


# ==========================================
# 1. FILE UTILITIES
# ==========================================

def read_csv(path, delimiter=","):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(
            csv.DictReader(
                f,
                delimiter=delimiter
            )
        )


def index_rows(rows, field, label):

    result = {}

    for row in rows:

        key = row[field]

        if key in result:
            raise RuntimeError(
                f"Duplicate {label}: {key}"
            )

        result[key] = row

    return result


def write_csv(name, rows):

    if not rows:
        raise RuntimeError(
            f"Empty result: {name}"
        )

    columns = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with (OUT / name).open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 2. CORRELATION UTILITIES
# ==========================================

def correlation(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a - a.mean()
    b = b - b.mean()

    denom = float(
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom <= 1e-12:
        return float("nan")

    return float(
        np.dot(a, b) / denom
    )


def spearman(x, y):

    return correlation(
        rankdata(x),
        rankdata(y)
    )


def residualize(v, design):

    coefficients = np.linalg.lstsq(
        design,
        v,
        rcond=None
    )[0]

    return (
        v - design @ coefficients
    )


def age_partial_spearman(x, y, age):

    if len(np.unique(age)) < 3:
        return float("nan")

    design = np.column_stack((
        np.ones(len(age)),
        rankdata(age)
    ))

    rx = residualize(
        rankdata(x),
        design
    )

    ry = residualize(
        rankdata(y),
        design
    )

    return correlation(rx, ry)


# ==========================================
# 3. PERMUTATION INFERENCE
# ==========================================

def perm_test(x, y, age, seed):
    """
    Two-sided Spearman permutation test.

    The age-adjusted sensitivity test uses
    an approximate Freedman-Lane procedure
    applied to ranked variables.
    """

    rng = np.random.default_rng(seed)

    n = len(x)

    rx = rankdata(x)
    ry = rankdata(y)

    observed = correlation(rx, ry)

    permutations = np.argsort(
        rng.random((N_PERM, n)),
        axis=1
    )

    xc = rx - rx.mean()
    yc = ry - ry.mean()

    denom = (
        np.linalg.norm(xc)
        * np.linalg.norm(yc)
    )

    if denom <= 1e-12:

        return (
            observed,
            float("nan"),
            float("nan"),
            float("nan")
        )

    null_raw = (
        yc[permutations] @ xc
    ) / denom

    p_raw = float(
        (
            1 + np.count_nonzero(
                np.abs(null_raw)
                >= abs(observed) - 1e-12
            )
        )
        / (N_PERM + 1)
    )

    if len(np.unique(age)) < 3:

        return (
            observed,
            p_raw,
            float("nan"),
            float("nan")
        )

    design = np.column_stack((
        np.ones(n),
        rankdata(age)
    ))

    rx_res = residualize(
        rx,
        design
    )

    ry_res = residualize(
        ry,
        design
    )

    adjusted = correlation(
        rx_res,
        ry_res
    )

    if not np.isfinite(adjusted):

        return (
            observed,
            p_raw,
            float("nan"),
            float("nan")
        )

    y_fit = ry - ry_res

    randomized_y = (
        y_fit[None, :]
        + ry_res[permutations]
    )

    coefficients = np.linalg.lstsq(
        design,
        randomized_y.T,
        rcond=None
    )[0]

    residuals = (
        randomized_y
        - (design @ coefficients).T
    )

    denominator = (
        np.linalg.norm(
            residuals,
            axis=1
        )
        * np.linalg.norm(rx_res)
    )

    valid = denominator > 1e-12

    null_adjusted = np.full(
        N_PERM,
        np.nan
    )

    null_adjusted[valid] = (
        residuals[valid] @ rx_res
        / denominator[valid]
    )

    valid_null = null_adjusted[
        np.isfinite(null_adjusted)
    ]

    if len(valid_null) < N_PERM * 0.95:

        return (
            observed,
            p_raw,
            adjusted,
            float("nan")
        )

    p_adjusted = float(
        (
            1 + np.count_nonzero(
                np.abs(valid_null)
                >= abs(adjusted) - 1e-12
            )
        )
        / (len(valid_null) + 1)
    )

    return (
        observed,
        p_raw,
        adjusted,
        p_adjusted
    )


# ==========================================
# 4. PARTICIPANT BOOTSTRAP
# ==========================================

def boot_intervals(x, y, age, seed):

    rng = np.random.default_rng(seed)

    raw = []
    partial = []

    n = len(x)

    for _ in range(N_BOOT):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        r = spearman(
            x[indices],
            y[indices]
        )

        if np.isfinite(r):
            raw.append(r)

        adjusted = age_partial_spearman(
            x[indices],
            y[indices],
            age[indices]
        )

        if np.isfinite(adjusted):
            partial.append(adjusted)

    def summarize(values):

        if len(values) < N_BOOT * 0.75:
            return "", "", len(values)

        lo, hi = np.percentile(
            values,
            [2.5, 97.5]
        )

        return (
            float(lo),
            float(hi),
            len(values)
        )

    return (
        summarize(raw),
        summarize(partial)
    )


# ==========================================
# 5. MULTIPLE-COMPARISON CORRECTIONS
# ==========================================

def bh_qvalues(pvals):

    m = len(pvals)

    order = np.argsort(pvals)
    result = np.empty(m)

    previous = 1.0

    for i in reversed(range(m)):

        pos = int(order[i])

        previous = min(
            previous,
            float(pvals[pos]) * m / (i + 1)
        )

        result[pos] = min(
            1.0,
            previous
        )

    return result


def holm_adjust(pvals):

    m = len(pvals)

    order = np.argsort(pvals)
    result = np.empty(m)

    previous = 0.0

    for i, j in enumerate(order):

        previous = max(
            previous,
            min(
                1.0,
                (m - i) * float(pvals[j])
            )
        )

        result[j] = previous

    return result


# ==========================================
# 6. LOCK ANALYSIS PROTOCOL
# ==========================================

print(
    "===== MILESTONE 2B: "
    "WITHIN-DIAGNOSIS MMSE ASSOCIATIONS =====",
    flush=True
)

paths = {
    "subject_join": JOIN,
    "manifest": MANIFEST,
    "metadata": METADATA
}

for path in paths.values():

    if not path.is_file():
        raise FileNotFoundError(path)


protocol = {
    "version":
        "m2b_mmse_associations_v1",

    "status":
        "exploratory, post-hoc; "
        "no new classifier training",

    "cohorts":
        "23 selected AD and 18 selected FTD; "
        "no pooled healthy controls",

    "analysis_unit":
        "one participant; classifier estimates "
        "averaged across five correlated CV repeats",

    "primary_endpoints": [
        "disease_probability",
        "class_reversal_rate"
    ],

    "secondary_endpoints": [
        "temporal_probability_sd",
        "original_label_support",
        "stable_wrong_rate"
    ],

    "inference":
        f"{N_PERM} two-sided permutations, "
        f"{N_BOOT} participant bootstraps",

    "age_sensitivity":
        "partial rank correlation and approximate "
        "Freedman-Lane age-adjusted permutation",

    "multiplicity":
        "Holm over four primary raw p-values; "
        "BH FDR over all ten raw p-values",

    "limitations":
        "small selected cohorts, cross-sectional MMSE, "
        "correlated CV, no external validation"
}

protocol_file = OUT / "locked_protocol.json"

encoded = (
    json.dumps(
        protocol,
        sort_keys=True,
        indent=2
    ) + "\n"
)

if protocol_file.is_file():

    if protocol_file.read_text() != encoded:
        raise RuntimeError(
            "Locked protocol changed"
        )

else:
    protocol_file.write_text(encoded)


fingerprints = {
    name: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for name, path in paths.items()
}

source_file = OUT / "input_sha256.json"

encoded_hashes = (
    json.dumps(
        fingerprints,
        sort_keys=True,
        indent=2
    ) + "\n"
)

if source_file.is_file():

    if source_file.read_text() != encoded_hashes:
        raise RuntimeError(
            "Input files changed since previous execution"
        )

else:
    source_file.write_text(encoded_hashes)


# ==========================================
# 7. VERIFY ORIGINAL CLINICAL DATA
# ==========================================

manifest = index_rows(
    read_csv(MANIFEST),
    "subject",
    "manifest participant"
)

metadata = index_rows(
    read_csv(
        METADATA,
        "\t"
    ),
    "participant_id",
    "clinical metadata participant"
)

rows = read_csv(JOIN)

if (
    len(rows) != 87
    or len(manifest) != 88
    or len(metadata) != 88
):
    raise RuntimeError(
        "Unexpected clinical table sizes"
    )

joined = {}

for row in rows:

    key = (
        row["task"],
        row["subject"]
    )

    if key in joined:
        raise RuntimeError(
            "Duplicate task-subject record"
        )

    joined[key] = row

    sid = row["subject"]

    if (
        sid not in manifest
        or sid not in metadata
    ):
        raise RuntimeError(
            "Participant absent from source metadata"
        )

    if (
        manifest[sid]["status"] != "CANDIDATE"
        or row["group"] != manifest[sid]["group"]
        or row["group"] != metadata[sid]["Group"]
    ):
        raise RuntimeError(
            "Participant eligibility/group mismatch: "
            + sid
        )

    if not np.isclose(
        float(row["mmse"]),
        float(metadata[sid]["MMSE"]),
        atol=1e-10
    ):
        raise RuntimeError(
            "MMSE mismatch: " + sid
        )

    if not np.isclose(
        float(row["age"]),
        float(metadata[sid]["Age"]),
        atol=1e-10
    ):
        raise RuntimeError(
            "Age mismatch: " + sid
        )

    if int(row["correlated_cv_repeats"]) != 5:
        raise RuntimeError(
            "Unexpected cross-validation repetitions"
        )


if len(joined) != 87:
    raise RuntimeError(
        "Unexpected task-subject record count"
    )

controls_ad = {
    sid
    for (task, sid), row in joined.items()
    if (
        task == "AD_vs_HC"
        and row["group"] == "C"
    )
}

controls_ftd = {
    sid
    for (task, sid), row in joined.items()
    if (
        task == "FTD_vs_HC"
        and row["group"] == "C"
    )
}

if (
    controls_ad != controls_ftd
    or len(controls_ad) != 23
):
    raise RuntimeError(
        "Healthy controls differ between tasks"
    )

print(
    "PASS: 87 task-subject rows, 64 unique "
    "participants; shared controls and "
    "MMSE/age verified.",
    flush=True
)


# ==========================================
# 8. PATIENT-ONLY SEVERITY TESTS
# ==========================================

results = []
used_patients = []
plots = []

for task_i, (
    task,
    group,
    expected_n
) in enumerate(TASKS):

    selected = sorted(
        (
            row
            for (t, sid), row in joined.items()
            if (
                t == task
                and row["group"] == group
            )
        ),
        key=lambda row: row["subject"]
    )

    controls = [
        row
        for (t, sid), row in joined.items()
        if (
            t == task
            and row["group"] == "C"
        )
    ]

    if (
        len(selected) != expected_n
        or len(controls) != 23
    ):
        raise RuntimeError(
            "Unexpected task cohort: " + task
        )

    mmse = np.array([
        float(row["mmse"])
        for row in selected
    ])

    age = np.array([
        float(row["age"])
        for row in selected
    ])

    if not (
        np.isfinite(mmse).all()
        and np.isfinite(age).all()
    ):
        raise RuntimeError(
            "Nonfinite clinical values"
        )

    if len(np.unique(mmse)) < 4:
        raise RuntimeError(
            "Insufficient within-group MMSE variation"
        )

    print(
        f"\n===== {task}: "
        "PATIENT-ONLY SEVERITY ASSOCIATIONS ====="
    )

    print(
        f"n={expected_n}, "
        f"MMSE median={np.median(mmse):.1f}, "
        f"MMSE unique={len(np.unique(mmse))}, "
        f"age unique={len(np.unique(age))}, "
        f"age-MMSE rho={spearman(age,mmse):+.3f}"
    )

    for metric_i, (
        label,
        field,
        priority
    ) in enumerate(METRICS):

        values = np.array([
            float(row[field])
            for row in selected
        ])

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Nonfinite prediction measurement: "
                + field
            )

        if label in (
            "disease_probability",
            "class_reversal_rate",
            "original_label_support",
            "stable_wrong_rate"
        ):

            if np.any(
                (values < 0)
                | (values > 1)
            ):
                raise RuntimeError(
                    "Out-of-range measurement: " + field
                )

        if (
            label == "temporal_probability_sd"
            and np.any(values < 0)
        ):
            raise RuntimeError(
                "Negative temporal probability SD"
            )

        rho, p_raw, adjusted, p_adj = perm_test(
            values,
            mmse,
            age,
            BASE_SEED
            + 1000 * task_i
            + 10 * metric_i
        )

        (
            (low, high, valid_boot),
            (a_low, a_high, valid_age)
        ) = boot_intervals(
            values,
            mmse,
            age,
            BASE_SEED
            + 2000
            + 1000 * task_i
            + 10 * metric_i
        )

        results.append({
            "task": task,
            "patient_group": group,
            "metric": label,
            "priority": priority,
            "n_patients": expected_n,
            "n_unique_mmse": len(np.unique(mmse)),
            "n_unique_metric": len(np.unique(values)),
            "spearman_rho": rho,
            "permutation_p_two_sided": p_raw,
            "spearman_bootstrap_ci_low": low,
            "spearman_bootstrap_ci_high": high,
            "spearman_bootstrap_valid": valid_boot,
            "age_adjusted_partial_rank_rho": adjusted,
            "age_adjusted_permutation_p_approx": p_adj,
            "age_adjusted_bootstrap_ci_low": a_low,
            "age_adjusted_bootstrap_ci_high": a_high,
            "age_adjusted_bootstrap_valid": valid_age,
        })

        if np.isfinite(adjusted):

            print(
                f"{priority:9s} "
                f"{label:25s} "
                f"Spearman rho={rho:+.3f}, "
                f"perm p={p_raw:.4f}; "
                f"age-adjusted rho={adjusted:+.3f}, "
                f"approx p={p_adj:.4f}"
            )

        else:

            print(
                f"{priority:9s} "
                f"{label:25s} "
                f"Spearman rho={rho:+.3f}, "
                f"perm p={p_raw:.4f}; "
                "age adjustment unavailable"
            )

        if priority == "primary":

            plots.append((
                task,
                label,
                mmse.copy(),
                values.copy()
            ))

    for row in selected:

        used_patients.append({
            "task": task,
            "subject": row["subject"],
            "group": group,
            "MMSE": row["mmse"],
            "age": row["age"],
            **{
                label: row[field]
                for label, field, _ in METRICS
            }
        })


# ==========================================
# 9. MULTIPLE-TESTING CORRECTION
# ==========================================

raw_ps = np.array([
    float(r["permutation_p_two_sided"])
    for r in results
])

if not np.isfinite(raw_ps).all():
    raise RuntimeError(
        "At least one metric is constant; "
        "review before interpreting tests"
    )

q_values = bh_qvalues(raw_ps)

for row, q in zip(results, q_values):
    row["BH_FDR_q_across_10_raw_tests"] = float(q)


primary_indices = [
    i
    for i, row in enumerate(results)
    if row["priority"] == "primary"
]

if len(primary_indices) != 4:
    raise RuntimeError(
        "Expected exactly four primary contrasts"
    )

primary_holm = holm_adjust(
    raw_ps[primary_indices]
)

for row in results:
    row["Holm_adjusted_p_four_primary_tests"] = ""

for i, p in zip(
    primary_indices,
    primary_holm
):
    results[i][
        "Holm_adjusted_p_four_primary_tests"
    ] = float(p)


save_rows = used_patients

write_csv(
    "patient_only_clinical_inputs.csv",
    save_rows
)

write_csv(
    "mmse_association_results.csv",
    results
)


# ==========================================
# 10. DISPLAY CORRECTED RESULTS
# ==========================================

print(
    "\n===== MULTIPLICITY-ADJUSTED "
    "PRIMARY RESULTS ====="
)

for row in results:

    if row["priority"] != "primary":
        continue

    low = row["spearman_bootstrap_ci_low"]
    high = row["spearman_bootstrap_ci_high"]

    interval = (
        f"[{low:+.3f}, {high:+.3f}]"
        if low != "" and high != ""
        else "unavailable"
    )

    print(
        f"{row['task']} / {row['metric']}: "
        f"rho={row['spearman_rho']:+.3f}, "
        f"raw permutation p="
        f"{row['permutation_p_two_sided']:.4f}, "
        f"Holm p="
        f"{row['Holm_adjusted_p_four_primary_tests']:.4f}, "
        f"95% subject-bootstrap CI {interval}"
    )


print(
    "\n===== SECONDARY RESULTS "
    "AND MULTIPLE TESTING ====="
)

for row in results:

    if row["priority"] == "secondary":

        print(
            f"{row['task']} / {row['metric']}: "
            f"rho={row['spearman_rho']:+.3f}, "
            f"raw p="
            f"{row['permutation_p_two_sided']:.4f}, "
            f"BH q="
            f"{row['BH_FDR_q_across_10_raw_tests']:.4f}"
        )


# ==========================================
# 11. GENERATE FOUR PRIMARY SCATTERPLOTS
# ==========================================

for task, name, mmse, values in plots:

    fig, ax = plt.subplots(
        figsize=(6.2, 4.5)
    )

    ax.scatter(
        mmse,
        values,
        alpha=0.8
    )

    ax.set(
        xlabel="MMSE (higher = less impairment)",
        ylabel=name.replace("_", " "),
        title=(
            f"{task}: patient-only "
            f"MMSE association (n={len(mmse)})"
        )
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        OUT / f"{task}_{name}_vs_MMSE.png",
        dpi=180
    )

    plt.close(fig)


# ==========================================
# 12. SAVE METHODS AND LIMITATIONS
# ==========================================

notes = """# Milestone 2B: clinical severity evidence

- Analysis unit: individual patient.
- Cohorts: 23 selected AD and 18 selected FTD.
- Healthy controls excluded because MMSE is constant.
- Correlated CV predictions averaged once per participant.
- Primary exploratory outcomes: disease probability and
  temporal class-reversal rate.
- Secondary outcomes: temporal probability SD, original-label
  confidence support, and stable-wrong rate.
- Spearman correlations use within-group permutation tests
  and participant-bootstrap confidence intervals.
- Age adjustment is an approximate partial-rank sensitivity
  analysis using Freedman-Lane permutation.
- Holm correction: four primary raw tests.
- BH FDR correction: all ten raw tests.
- All tests remain exploratory because the dataset has
  already been examined during model development.
- MMSE is cross-sectional; associations are not evidence
  of causality, clinical diagnostic utility or prognosis.
- Window selection may limit severity representativeness.
- No classifiers were retrained.
"""

(
    OUT / "METHODS_AND_LIMITATIONS.md"
).write_text(
    notes,
    encoding="utf-8"
)


print(
    "\n===== MILESTONE 2B COMPLETE ====="
)

print(
    "Results:",
    OUT
)

print(
    "No EEG processing or model training performed; "
    "no pooled diagnostic-group correlations."
)

