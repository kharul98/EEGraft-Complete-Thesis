#!/usr/bin/env python3

"""Audit FTD data before locking a second diagnostic experiment."""

from pathlib import Path
from collections import Counter
import csv
import json

import mne


ROOT = Path.cwd()
MANIFEST = ROOT / "reports/openneuro_5min_manifest_v2.csv"

OUT = (
    ROOT / "reports/thesis_final/ftd_extension_v1"
)
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise RuntimeError("No output rows")

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def shown(value):
    text = str(value or "").strip()
    return text if text else "-"


print(
    "===== FTD ELIGIBILITY PREFLIGHT =====",
    flush=True
)

if not MANIFEST.is_file():
    raise FileNotFoundError(MANIFEST)

rows = read_csv(MANIFEST)

if len(rows) != 88:
    raise RuntimeError(
        "Expected 88 manifest participants"
    )

if len({
    r["subject"] for r in rows
}) != 88:
    raise RuntimeError(
        "Duplicate participant ID"
    )

by_group = Counter(
    r["group"] for r in rows
)

if by_group != {
    "A": 36,
    "C": 29,
    "F": 23
}:
    raise RuntimeError(
        "Unexpected original dataset composition: "
        + str(by_group)
    )


ftd = sorted(
    [
        r for r in rows
        if r["group"] == "F"
    ],
    key=lambda r: r["subject"]
)

hc = sorted(
    [
        r for r in rows
        if r["group"] == "C"
        and r["status"] == "CANDIDATE"
    ],
    key=lambda r: r["subject"]
)

ad = [
    r for r in rows
    if r["group"] == "A"
    and r["status"] == "CANDIDATE"
]

if len(hc) != 23 or len(ad) != 23:
    raise RuntimeError(
        "Historical AD/HC cohort has changed"
    )

print(
    "Original dataset: "
    "36 AD, 23 FTD, 29 HC"
)

print(
    "Existing selected controls:",
    len(hc)
)

print(
    "Existing selected AD:",
    len(ad)
)

print(
    "\n===== EXISTING MANIFEST STATUS COUNTS ====="
)

for group in ("A", "C", "F"):
    counts = Counter(
        r["status"]
        for r in rows
        if r["group"] == group
    )

    print(group, dict(counts))


# ==========================================
# IDENTIFY EXISTING FTD DERIVATIVES
# ==========================================

DATA_ROOT = (
    ROOT / "data/eeg/on004504_v1.0.0"
)

fallback = {}

if DATA_ROOT.is_dir():

    # Search only if a manifest path is missing.
    missing_paths = False

    for r in ftd:

        raw_name = str(
            r.get("file", "")
        ).strip()

        if not raw_name:
            missing_paths = True
            break

        p = Path(raw_name)

        if not p.is_absolute():
            p = ROOT / p

        if not p.is_file():
            missing_paths = True
            break

    if missing_paths:

        for p in DATA_ROOT.rglob("*.set"):

            if "derivatives" not in p.parts:
                continue

            for sid in [
                x["subject"] for x in ftd
            ]:

                if sid in p.parts:
                    fallback.setdefault(
                        sid, []
                    ).append(p)


audit = []
annotation_counts = Counter()

print(
    "\n===== FTD RECORDING METADATA =====",
    flush=True
)

for r in ftd:

    sid = r["subject"]

    listed = str(
        r.get("file", "")
    ).strip()

    path = (
        Path(listed)
        if listed
        else Path("__missing__")
    )

    if listed and not path.is_absolute():
        path = ROOT / path

    resolution = "manifest"

    if not listed or not path.is_file():

        candidates = fallback.get(
            sid, []
        )

        if len(candidates) == 1:
            path = candidates[0]
            resolution = "unique_derivative_fallback"

        elif len(candidates) > 1:
            resolution = "AMBIGUOUS_DERIVATIVES"

        else:
            resolution = "MISSING_FILE"

    row = {
        "subject": sid,
        "group": "F",
        "manifest_status":
            shown(r.get("status")),
        "manifest_start_sec":
            shown(r.get("start_sec")),
        "manifest_end_sec":
            shown(r.get("end_sec")),
        "manifest_duration_sec":
            shown(r.get("duration_sec")),
        "manifest_multi200_pct":
            shown(r.get("multi200_pct")),
        "manifest_any300_pct":
            shown(r.get("any300_pct")),
        "manifest_flat_pct":
            shown(r.get("flat_pct")),
        "manifest_note":
            shown(r.get("note")),
        "file_resolution": resolution,
        "file_path":
            str(path) if path.is_file() else "",
        "header_status": "",
        "recording_seconds": "",
        "sampling_rate": "",
        "eeg_channels": "",
        "annotation_count": "",
        "annotation_types": "",
        "boundary_annotation_count": ""
    }

    if not path.is_file():

        row["header_status"] = (
            resolution
        )

    else:

        try:
            raw = mne.io.read_raw_eeglab(
                str(path),
                preload=False,
                verbose="ERROR"
            )

            eeg_picks = mne.pick_types(
                raw.info,
                eeg=True,
                exclude=[]
            )

            descriptions = [
                str(d)
                for d in raw.annotations.description
            ]

            annotation_counts.update(
                descriptions
            )

            boundaries = sum(
                (
                    "boundary" in d.lower()
                    or "edge" in d.lower()
                )
                for d in descriptions
            )

            row["recording_seconds"] = (
                f"{raw.n_times / raw.info['sfreq']:.3f}"
            )

            row["sampling_rate"] = (
                f"{raw.info['sfreq']:.3f}"
            )

            row["eeg_channels"] = (
                len(eeg_picks)
            )

            row["annotation_count"] = (
                len(descriptions)
            )

            row["boundary_annotation_count"] = (
                boundaries
            )

            row["annotation_types"] = (
                json.dumps(
                    dict(
                        Counter(descriptions)
                    ),
                    sort_keys=True
                )
            )

            valid = (
                abs(
                    raw.info["sfreq"] - 500
                ) < 0.01
                and len(eeg_picks) == 19
                and raw.n_times >= 150000
            )

            row["header_status"] = (
                "HEADER_COMPATIBLE"
                if valid
                else "HEADER_INCOMPATIBLE"
            )

            del raw

        except Exception as exc:

            row["header_status"] = (
                "READ_ERROR: "
                + str(exc)[:160]
            )

    audit.append(row)

    print(
        f"{sid}: "
        f"manifest={row['manifest_status']}; "
        f"EEG={row['header_status']}; "
        f"seconds={shown(row['recording_seconds'])}; "
        f"annotations="
        f"{shown(row['annotation_count'])}; "
        f"start={row['manifest_start_sec']}; "
        f"end={row['manifest_end_sec']}; "
        f"multi200={row['manifest_multi200_pct']}",
        flush=True
    )


save_csv(
    OUT / "ftd_recording_preflight.csv",
    audit
)


# ==========================================
# FIND ORIGINAL QUALITY-SELECTION SOURCE
# ==========================================

print(
    "\n===== ORIGINAL WINDOW-SELECTION CODE ====="
)

keywords = (
    "NO_UNINTERRUPTED_WINDOW",
    "multi200_pct",
    "any300_pct",
    "flat_pct"
)

source_paths = list(
    ROOT.glob("*.py")
)

scripts_dir = ROOT / "scripts"

if scripts_dir.is_dir():
    source_paths.extend(
        scripts_dir.rglob("*.py")
    )

matches = []

for path in source_paths:

    if path.resolve() == Path(
        __file__
    ).resolve():
        continue

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

    except OSError:
        continue

    hits = [
        (number, line.strip())
        for number, line in enumerate(
            lines, 1
        )
        if any(
            keyword in line
            for keyword in keywords
        )
    ]

    if hits:

        matches.append({
            "path": str(
                path.relative_to(ROOT)
            ),
            "matched_lines": len(hits)
        })

        print(
            str(
                path.relative_to(ROOT)
            ),
            "matching lines:",
            len(hits)
        )

        for number, text in hits[:4]:
            print(
                f"  L{number}: "
                f"{text[:130]}"
            )


if matches:
    save_csv(
        OUT / "selection_code_candidates.csv",
        matches
    )

else:
    print(
        "No selector code found by keyword."
    )


print(
    "\n===== FTD PREFLIGHT SUMMARY ====="
)

print(
    "FTD subjects in manifest:",
    len(ftd)
)

status_counts = Counter(
    r["header_status"]
    for r in audit
)

for status, count in status_counts.items():
    print(
        f"{status}: {count}"
    )

print(
    "FTD subjects with manifest start/end:",
    sum(
        r["manifest_start_sec"] != "-"
        and r["manifest_end_sec"] != "-"
        for r in audit
    )
)

print(
    "Most common annotation labels:",
    annotation_counts.most_common(8)
)

print(
    "Saved:",
    OUT / "ftd_recording_preflight.csv"
)

print(
    "No FTD participants selected yet. "
    "No models trained. "
    "Existing AD results unchanged."
)

print(
    "\n===== FTD PREFLIGHT COMPLETE ====="
)

