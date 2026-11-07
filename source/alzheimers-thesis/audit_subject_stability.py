#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import re

import numpy as np
from scipy.stats import spearmanr

ROOT = Path.cwd()
R = ROOT / "reports"

def read_csv(path, delimiter=","):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))

def index_by_subject(rows, column="subject"):
    result = {r[column]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate subject identifiers")
    return result

def number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (ValueError, TypeError):
        return np.nan

def metadata_value(row, candidates):
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(k).lower()): v
        for k, v in row.items()
    }
    for key in candidates:
        if key in normalized:
            return normalized[key]
    return None

def describe(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "unavailable"
    return (
        f"median={np.median(v):.3f}, "
        f"min={np.min(v):.3f}, "
        f"max={np.max(v):.3f}"
    )

stability = index_by_subject(
    read_csv(R / "robustness_subject_stability.csv")
)

manifest = index_by_subject(
    read_csv(R / "openneuro_5min_manifest_v2.csv")
)

participants_path = (
    ROOT / "data/eeg/on004504_v1.0.0/participants.tsv"
)

participants = {}

if participants_path.exists():
    participants = index_by_subject(
        read_csv(participants_path, "\t"),
        "participant_id"
    )

expected = {
    sid for sid, r in manifest.items()
    if r["status"] == "CANDIDATE"
    and r["group"] in ("A", "C")
}

if set(stability) != expected or len(expected) != 46:
    raise RuntimeError(
        "The stability report does not match "
        "the 46 eligible AD/Healthy participants."
    )

results = []

print("===== SUBJECT STABILITY AUDIT =====")
print("Subjects:", len(expected))

for sid in sorted(expected):

    s = stability[sid]
    m = manifest[sid]
    p = participants.get(sid, {})

    lr = number(
        s["LogisticRegression_correct_fraction"]
    )

    svm = number(
        s["RBF_SVM_correct_fraction"]
    )

    if not (
        0 <= lr <= 1
        and 0 <= svm <= 1
    ):
        raise ValueError(
            f"Invalid stability score: {sid}"
        )

    age = number(
        metadata_value(
            p, ["age", "ageyears", "ageinyears"]
        )
    )

    sex = metadata_value(
        p, ["gender", "sex"]
    )

    artifact = number(
        m.get("multi200_pct")
    )

    mean_stability = (lr + svm) / 2

    if 0.4 <= mean_stability <= 0.6:
        category = "FRAGILE"
    elif mean_stability < 0.4:
        category = "OFTEN_INCORRECT"
    else:
        category = "USUALLY_CORRECT"

    results.append({
        "subject": sid,
        "group": m["group"],
        "age": age if np.isfinite(age) else "",
        "sex": sex or "",
        "selected_window_artifact_pct":
            artifact if np.isfinite(artifact) else "",
        "logistic_correct_fraction": lr,
        "svm_correct_fraction": svm,
        "mean_correct_fraction": mean_stability,
        "stability_category": category
    })

print("\n===== STABILITY BY GROUP =====")

for group in ("A", "C"):

    subset = [
        r for r in results if r["group"] == group
    ]

    print("\nGROUP:", group)
    print("Subjects:", len(subset))

    print(
        "Logistic correctness:",
        describe([
            r["logistic_correct_fraction"]
            for r in subset
        ])
    )

    print(
        "SVM correctness:",
        describe([
            r["svm_correct_fraction"]
            for r in subset
        ])
    )

    print(
        "Age:",
        describe([r["age"] for r in subset])
    )

    print(
        "Selected-window artifact percentage:",
        describe([
            r["selected_window_artifact_pct"]
            for r in subset
        ])
    )

    print(
        "Stability categories:",
        dict(Counter(
            r["stability_category"]
            for r in subset
        ))
    )

print("\n===== TEN LEAST CONSISTENT SUBJECTS =====")

ordered = sorted(
    results,
    key=lambda r: r["mean_correct_fraction"]
)

for row in ordered[:10]:
    print(
        row["subject"],
        "group=", row["group"],
        "LR=", row["logistic_correct_fraction"],
        "SVM=", row["svm_correct_fraction"],
        "age=", row["age"],
        "artifact%=",
        row["selected_window_artifact_pct"],
        "category=", row["stability_category"]
    )

print("\n===== EXPLORATORY CORRELATIONS =====")

def correlation(label, xkey, ykey, group=None):

    subset = [
        r for r in results
        if group is None or r["group"] == group
    ]

    pairs = [
        (number(r[xkey]), number(r[ykey]))
        for r in subset
    ]

    pairs = [
        (x, y) for x, y in pairs
        if np.isfinite(x) and np.isfinite(y)
    ]

    if (
        len(pairs) < 10
        or len(set(x for x, _ in pairs)) < 2
        or len(set(y for _, y in pairs)) < 2
    ):
        print(label, ": insufficient variation/data")
        return

    x, y = zip(*pairs)

    rho, p = spearmanr(x, y)

    print(
        label,
        "n=", len(pairs),
        "rho=", round(float(rho), 3),
        "p=", round(float(p), 4)
    )

for group in ("A", "C"):

    for model in (
        "logistic_correct_fraction",
        "svm_correct_fraction"
    ):

        correlation(
            f"{group} {model} vs age",
            model,
            "age",
            group
        )

        correlation(
            f"{group} {model} vs artifacts",
            model,
            "selected_window_artifact_pct",
            group
        )

out = R / "robustness_subject_audit.csv"

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0])
    )
    writer.writeheader()
    writer.writerows(results)

print("\n===== AUDIT COMPLETE =====")
print("Report:", out)
print("No subjects excluded.")
print("Original EEG unchanged.")
print(
    "Correlations are exploratory, unadjusted, "
    "and do not establish causality."
)
