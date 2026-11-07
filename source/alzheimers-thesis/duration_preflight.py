#!/usr/bin/env python3

from pathlib import Path
import csv
import os
import numpy as np

ROOT = Path.cwd()
REPORT = ROOT / "reports/thesis_final/audit"
REPORT.mkdir(parents=True, exist_ok=True)

output = []

def say(*parts):
    message = " ".join(str(x) for x in parts)
    print(message, flush=True)
    output.append(message)

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

FEATURES = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
]

say("===== DURATION PREFLIGHT =====")

# ------------------------------------------
# 1. Inspect existing spectral features
# ------------------------------------------

path = ROOT / "reports/baseline_features_v2.csv"
rows = read_csv(path)

say("\n===== BASELINE FEATURE DEFINITIONS =====")
say("Rows:", len(rows))
say("Columns:", list(rows[0]))

if len(rows) != 46:
    raise RuntimeError("Expected 46 participants")

for feature in FEATURES:

    values = np.array([
        float(row[feature])
        for row in rows
    ])

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Nonfinite values in {feature}"
        )

    say(
        feature,
        "median=", round(float(np.median(values)), 6),
        "min=", round(float(np.min(values)), 6),
        "max=", round(float(np.max(values)), 6)
    )

say("\nFirst three feature rows:")

for row in rows[:3]:
    say({
        "subject": row["subject"],
        "group": row["group"],
        **{
            name: row[name]
            for name in FEATURES
        }
    })

band_sum = np.array([
    sum(float(row[name]) for name in FEATURES[:5])
    for row in rows
])

say(
    "\nSum of five relative bands:",
    "median=", round(float(np.median(band_sum)), 6),
    "range=",
    (
        round(float(np.min(band_sum)), 6),
        round(float(np.max(band_sum)), 6)
    )
)

# ------------------------------------------
# 2. Inspect three review participants
# ------------------------------------------

manifest = read_csv(
    ROOT / "reports/openneuro_5min_manifest_v2.csv"
)

say("\n===== PARTICIPANTS REQUIRING REVIEW =====")
say("Manifest columns:", list(manifest[0]))

review = [
    row for row in manifest
    if row["group"] in ("A", "C")
    and row["status"] == "REVIEW"
]

say("Review participants:", len(review))

for row in review:

    details = {
        k: v
        for k, v in row.items()
        if k != "file"
        and v not in ("", None)
    }

    say(details)

# ------------------------------------------
# 3. Find the original extraction scripts
# ------------------------------------------

say("\n===== SEARCHING ORIGINAL PYTHON SCRIPTS =====")

skip_dirs = {
    ".venv",
    ".git",
    "__pycache__",
    "data",
    "models",
    "third_party",
    "reports",
    "node_modules"
}

skip_files = {
    "duration_preflight.py",
    "thesis_final_audit.py",
    "eeg_duration_study.py"
}

candidates = []

for directory, subdirs, files in os.walk(ROOT):

    subdirs[:] = [
        d for d in subdirs
        if d not in skip_dirs
    ]

    if len(Path(directory).relative_to(ROOT).parts) > 4:
        subdirs[:] = []
        continue

    for filename in files:

        if not filename.endswith(".py"):
            continue

        if filename in skip_files:
            continue

        file = Path(directory) / filename

        try:
            code = file.read_text(
                encoding="utf-8",
                errors="replace"
            )
        except OSError:
            continue

        score = (
            10 * ("baseline_features_v2.csv" in code)
            + 5 * ("theta_alpha_ratio" in code)
            + 3 * ("welch(" in code)
            + 2 * ("relative" in code.lower())
        )

        if score >= 5:
            candidates.append(
                (score, file, code)
            )

candidates.sort(
    key=lambda x: (-x[0], str(x[1]))
)

say(
    "Candidate scripts:",
    len(candidates)
)

keywords = [
    "baseline_features_v2",
    "theta_alpha_ratio",
    "welch(",
    "nperseg",
    "relative_power",
    "bandpower",
    "band_power",
    "good_epoch",
    "ptp",
    "BANDS",
    "bands ="
]

for score, file, code in candidates[:5]:

    say(
        "\n--- SCRIPT:",
        file.relative_to(ROOT),
        "score:", score,
        "---"
    )

    lines = code.splitlines()

    hits = [
        i for i, line in enumerate(lines)
        if any(
            key.lower() in line.lower()
            for key in keywords
        )
    ]

    selected = set()

    for i in hits[:12]:

        for j in range(
            max(0, i - 3),
            min(len(lines), i + 7)
        ):
            selected.add(j)

    for i in sorted(selected)[:110]:
        say(
            f"{i + 1:04d}: {lines[i]}"
        )

if not candidates:
    say(
        "No matching original extractor found. "
        "We must inspect the older baseline script "
        "or reconstruct and verify its feature formula."
    )

# ------------------------------------------
# 4. Existing duration reports
# ------------------------------------------

say("\n===== DURATION OUTPUT STATUS =====")

for name in [
    "duration_features_60_180_300.csv",
    "duration_nested_cv_runs.csv",
    "duration_nested_cv_predictions.csv",
    "duration_theta_alpha_interaction.csv"
]:

    file = ROOT / "reports" / name

    say(
        name,
        "FOUND" if file.exists() else "NOT CREATED"
    )

report_path = REPORT / "duration_preflight.txt"

report_path.write_text(
    "\n".join(output) + "\n",
    encoding="utf-8"
)

say("\n===== PREFLIGHT COMPLETE =====")
say("Saved report:", report_path)
say("Original EEG and previous results unchanged.")

