#!/usr/bin/env python3

from pathlib import Path
import os

ROOT = Path.cwd()

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "models",
    "third_party",
    "reports",
    "node_modules"
}

SKIP_FILES = {
    "identify_baseline_extractor.py",
    "find_original_feature_method.py",
    "duration_preflight.py",
    "thesis_final_audit.py",
    "eeg_duration_study.py",
    "spectral_explainability_cv.py",
    "eeg_hybrid_ablation_cv.py",
    "robustness_repeated_cv.py",
    "compare_all_models.py"
}

candidates = []

for directory, subdirs, filenames in os.walk(ROOT):

    subdirs[:] = [
        d for d in subdirs
        if d not in SKIP_DIRS
    ]

    if len(
        Path(directory).relative_to(ROOT).parts
    ) > 5:
        subdirs[:] = []
        continue

    for filename in filenames:

        if not filename.endswith(".py"):
            continue

        if filename in SKIP_FILES:
            continue

        path = Path(directory) / filename

        try:
            source = path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        except OSError:
            continue

        lower = source.lower()

        has_welch = (
            "welch(" in lower
            or "welch (" in lower
        )

        has_features = (
            "theta_alpha_ratio" in lower
            or "clean_epochs" in lower
        )

        writes_baseline = (
            "baseline_features_v2.csv" in lower
            or "baseline_features.csv" in lower
        )

        # Do not include scripts that merely
        # load existing feature CSV files.
        if not has_welch:
            continue

        score = (
            100 * (writes_baseline and has_welch)
            + 30 * has_features
            + 15 * ("clean_epochs" in lower)
            + 10 * ("theta_alpha_ratio" in lower)
            + 5 * ("np.median" in lower)
            + 5 * ("writerow(" in lower)
        )

        candidates.append(
            (score, path, source)
        )

candidates.sort(
    key=lambda item: (
        -item[0],
        str(item[1])
    )
)

print("===== ACTUAL EEG EXTRACTOR CANDIDATES =====")
print("Found:", len(candidates))

for score, path, source in candidates:

    print(
        f"Score {score:3d} | "
        f"{path.relative_to(ROOT)} | "
        f"{len(source.splitlines())} lines"
    )

print("\n===== EXTRACTION IMPLEMENTATIONS =====")

# Print the most relevant code, not source code
# belonging to the search scripts themselves.

keywords = [
    "def ",
    "welch(",
    "nperseg",
    "noverlap",
    "bands =",
    "bands:",
    "relative",
    "theta_alpha_ratio",
    "clean_epochs",
    "np.median",
    "np.mean",
    "np.trapezoid",
    "ptp",
    "reshape(",
    "start_sec",
    "writerow",
    "baseline_features"
]

for score, path, source in candidates[:4]:

    lines = source.splitlines()

    print("\n" + "=" * 65)
    print("FILE:", path.relative_to(ROOT))
    print("SCORE:", score)
    print("=" * 65)

    # For short implementations, print the
    # whole source to prevent missing logic.
    if len(lines) <= 260:

        for i, line in enumerate(lines, 1):
            print(f"{i:04d}: {line}")

        continue

    important = [
        i for i, line in enumerate(lines)
        if any(
            keyword.lower() in line.lower()
            for keyword in keywords
        )
    ]

    selected = set()

    for i in important:

        for j in range(
            max(0, i - 7),
            min(len(lines), i + 12)
        ):
            selected.add(j)

    previous = -2

    for i in sorted(selected)[:330]:

        if i > previous + 1:
            print("    ...")

        print(
            f"{i + 1:04d}: {lines[i]}"
        )

        previous = i

if not candidates:
    print(
        "Original feature extractor not found "
        "among saved Python scripts."
    )
    print(
        "We will reconstruct the method "
        "and verify it numerically against "
        "the existing 46-subject CSV."
    )

print("\n===== EXTRACTOR IDENTIFICATION COMPLETE =====")
print("No files modified.")
