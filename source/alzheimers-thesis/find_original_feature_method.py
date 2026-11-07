#!/usr/bin/env python3

from pathlib import Path
import os

ROOT = Path.cwd()

SKIP = {
    ".venv",
    ".git",
    "data",
    "models",
    "third_party",
    "reports",
    "__pycache__",
    "node_modules"
}

files = []

for directory, subdirs, filenames in os.walk(ROOT):

    subdirs[:] = [
        name for name in subdirs
        if name not in SKIP
    ]

    if len(Path(directory).relative_to(ROOT).parts) > 4:
        subdirs[:] = []
        continue

    for name in filenames:

        if not name.endswith(".py"):
            continue

        path = Path(directory) / name

        try:
            code = path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        except OSError:
            continue

        score = (
            40 * (
                "baseline_features_v2.csv" in code
                and "writer" in code
            )
            + 30 * (
                "baseline_features.csv" in code
                and "writer" in code
            )
            + 20 * (
                "welch(" in code
                and "theta_alpha_ratio" in code
            )
            + 12 * ("clean_epochs" in code)
            + 10 * ("welch(" in code)
            + 5 * ("theta_alpha_ratio" in code)
        )

        if score:
            files.append((score, path, code))

files.sort(
    key=lambda item: (
        -item[0],
        str(item[1])
    )
)

print("===== FEATURE SOURCE CANDIDATES =====")

for score, path, code in files[:20]:
    print(
        f"score={score:3d} "
        f"{path.relative_to(ROOT)}"
    )

print("\n===== ORIGINAL EXTRACTION CODE =====")

# Prefer implementations that calculate features,
# rather than downstream classifiers reading CSVs.

extractors = [
    (score, path, code)
    for score, path, code in files
    if "welch(" in code
    and (
        "theta_alpha_ratio" in code
        or "clean_epochs" in code
    )
]

keywords = (
    "def ",
    "bands =",
    "bands:",
    "welch(",
    "nperseg",
    "noverlap",
    "relative",
    "theta_alpha_ratio",
    "clean_epochs",
    "np.median",
    "np.mean",
    "np.trapezoid",
    "np.trapz",
    "ptp",
    "start_sec",
    "writeheader",
    "writerow",
    "baseline_features"
)

for score, path, code in extractors[:5]:

    lines = code.splitlines()

    print(
        "\n========================================"
    )
    print("FILE:", path.relative_to(ROOT))
    print("TOTAL LINES:", len(lines))
    print("========================================")

    important = [
        i for i, line in enumerate(lines)
        if any(
            key.lower() in line.lower()
            for key in keywords
        )
    ]

    selected = set()

    for i in important:

        # Include enough surrounding lines to
        # identify parameters and aggregation.
        for j in range(
            max(0, i - 5),
            min(len(lines), i + 9)
        ):
            selected.add(j)

    # Limit printed source from very large scripts.
    chosen = sorted(selected)[:260]

    previous = -2

    for i in chosen:

        if i > previous + 1:
            print("    ...")

        print(
            f"{i + 1:04d}: {lines[i]}"
        )

        previous = i

if not extractors:
    print(
        "No original Welch feature extractor found."
    )

print("\n===== EXTRACTION SOURCE AUDIT COMPLETE =====")
print("No original files modified.")
