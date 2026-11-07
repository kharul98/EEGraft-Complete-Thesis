#!/usr/bin/env python3
"""Read-only BrainLat physical EEG file readiness audit."""

from pathlib import Path
from collections import Counter
import os
import gc
import json

import numpy as np
import pandas as pd

ROOT = Path.home() / "alzheimers-thesis"

SEARCH_ROOTS = [
    Path("/mnt/d/user4"),
    Path("/mnt/c/Users/user4/Downloads")
]

def cohort(path):
    name = path.name.lower()
    parts = [
        part.lower()
        for part in path.parts
    ]

    if "1_ad" in parts or "sub-3" in name:
        return "AD"

    if "5_hc" in parts or "sub-1" in name:
        return "HC"

    return "UNKNOWN"


def find_files():
    found = set()

    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue

        for folder, dirs, names in os.walk(root):
            depth = len(
                Path(folder).relative_to(root).parts
            )

            dirs[:] = [
                d for d in dirs
                if depth < 12
                and d not in {
                    ".git", "__pycache__",
                    "$RECYCLE.BIN"
                }
            ]

            for name in names:
                if (
                    name.lower().endswith(".set")
                    and name.lower().startswith("s6_sub-")
                ):
                    found.add(Path(folder) / name)

    return sorted(found)


def main():
    print("===== BRAINLAT EEG READINESS AUDIT =====", flush=True)

    try:
        import mne
    except ImportError:
        print(
            "MNE is unavailable. Install using "
            "python -m pip install mne"
        )
        return

    files = find_files()

    print("Physical SET files found:", len(files), flush=True)

    rows = []

    for index, path in enumerate(files, start=1):
        group = cohort(path)

        record = {
            "cohort_candidate": group,
            "source_path": str(path),
            "size_mb": round(
                path.stat().st_size / 1024**2, 2
            ),
            "same_folder_fdt": path.with_suffix(
                ".fdt"
            ).is_file(),
            "readable": False,
            "sampling_rate": None,
            "channels": None,
            "duration_s": None,
            "eligible_300s_length_only": False,
            "error": ""
        }

        raw = None

        try:
            raw = mne.io.read_raw_eeglab(
                str(path),
                preload=False,
                verbose="ERROR"
            )

            fs = float(raw.info["sfreq"])
            duration = raw.n_times / fs

            # Test access to actual EEG samples.
            stop = min(
                raw.n_times,
                max(1, int(fs))
            )

            preview = raw.get_data(
                start=0,
                stop=stop
            )

            if not np.isfinite(preview).all():
                raise ValueError(
                    "Nonfinite signal in one-second preview"
                )

            record.update({
                "readable": True,
                "sampling_rate": fs,
                "channels": len(raw.ch_names),
                "duration_s": round(duration, 3),
                "eligible_300s_length_only": bool(
                    duration >= 300
                    and len(raw.ch_names) >= 19
                )
            })

        except Exception as error:
            record["error"] = (
                type(error).__name__
                + ": "
                + str(error)[:180]
            )

        finally:
            if raw is not None:
                raw.close()

            del raw
            gc.collect()

        rows.append(record)

        if index % 10 == 0 or index == len(files):
            print(
                f"Checked {index}/{len(files)} files",
                flush=True
            )

    df = pd.DataFrame(rows)

    output = (
        ROOT / "results/p3_brainlat_readiness.csv"
    )

    report = (
        ROOT / "docs/audit/p3_brainlat_readiness_summary.json"
    )

    df.to_csv(output, index=False)

    summary = {
        "files_found": len(df),
        "cohorts": {},
        "limitations": [
            "Filenames and folders provide provisional cohort labels.",
            "Readability does not establish preprocessing equivalence.",
            "Only first-second signal samples were checked.",
            "Length does not establish resting-condition continuity.",
            "Recordings inside unextracted ZIP archives are not scanned."
        ]
    }

    print()
    print("===== BRAINLAT READINESS SUMMARY =====")

    for group in ("AD", "HC", "UNKNOWN"):
        part = df[
            df.cohort_candidate == group
        ]

        readable = part[
            part.readable
        ]

        eligible = part[
            part.eligible_300s_length_only
        ]

        stats = {
            "SET_files": int(len(part)),
            "readable_EEG": int(len(readable)),
            "length_eligible_300s": int(len(eligible)),
            "readable_without_same_folder_FDT": int(
                (
                    readable.same_folder_fdt
                    .eq(False)
                ).sum()
            )
        }

        summary["cohorts"][group] = stats
        print(group, ":", stats)

    errors = df[
        df.error.fillna("").ne("")
    ]

    print()
    print("===== ERROR CATEGORIES =====")

    categories = Counter(
        text.split(":")[0]
        for text in errors.error
    )

    print(dict(categories))

    for text in errors.error.head(5):
        print("Example:", text[:150])

    report.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("Full inventory saved:", output)
    print("No EEG files were modified.")
    print("===== BRAINLAT READINESS COMPLETE =====")


if __name__ == "__main__":
    main()
