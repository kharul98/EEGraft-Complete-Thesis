#!/usr/bin/env python3
"""Milestone 3B: download pinned matched EO derivatives, choose earliest quality-passing 60s."""

from pathlib import Path, PurePosixPath
from collections import Counter
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import csv
import hashlib
import json
import os
import re
import time

import mne
import numpy as np


ROOT = Path.cwd()
BASE = ROOT / "reports" / "thesis_final"
PRE = BASE / "milestone_3a_cross_condition_preflight_v1"
OUT = BASE / "milestone_3b_eyes_open_quality_v1"

OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/eeg/on006036_v1.0.0"
DATA.mkdir(parents=True, exist_ok=True)

EC_MANIFEST = ROOT / "reports/openneuro_5min_manifest_v2.csv"
EC_META = ROOT / "data/eeg/on004504_v1.0.0/participants.tsv"

EO_META = PRE / "on006036_v1.0.0_participants.tsv"
REMOTE_MANIFEST = PRE / "nemar_v1.0.0_manifest.json"

BASE_URL = "https://data.nemar.org/on006036/v1.0.0/"

CHUNK = 2 * 1024 * 1024
WINDOW_S = 60
FS = 500
MIN_CLEAN_EPOCHS = 15
START_STEP_S = 10


# ==========================================
# FILE UTILITIES
# ==========================================

def read_rows(path, delimiter=","):

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as fh:

        return list(csv.DictReader(
            fh, delimiter=delimiter
        ))


def save_rows(path, rows):

    if not rows:
        return

    keys = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as fh:

        writer = csv.DictWriter(
            fh, fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(rows)


def digest(path, algorithm):

    h = hashlib.new(algorithm)

    with path.open("rb") as fh:

        while True:

            data = fh.read(CHUNK)

            if not data:
                break

            h.update(data)

    return h.hexdigest()


def expected_size(entry):

    val = entry.get("size")

    if val in ("", None):
        return None

    return int(val)


def valid_file(path, entry):

    if not path.is_file():
        return False

    size = expected_size(entry)

    if (
        size is not None
        and path.stat().st_size != size
    ):
        return False

    algo = str(
        entry.get("checksum_algorithm", "")
    ).lower()

    checksum = str(
        entry.get("checksum", "")
    ).lower()

    if algo in ("sha256", "md5") and checksum:

        return (
            digest(path, algo) == checksum
        )

    return path.stat().st_size > 1024


# ==========================================
# RESUMABLE DIRECT HTTPS DOWNLOAD
# ==========================================

def download_file(entry, destination):

    if valid_file(destination, entry):
        return "cached"

    if destination.exists():

        raise RuntimeError(
            "Existing file fails size/checksum: "
            + str(destination)
        )

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    partial = Path(
        str(destination) + ".part"
    )

    url = str(
        entry.get("url")
        or entry.get("bytes_url")
        or ""
    )

    if not url.startswith(BASE_URL):

        url = BASE_URL + entry["path"]

    size = expected_size(entry)

    last_error = None

    for attempt in range(1, 5):

        offset = (
            partial.stat().st_size
            if partial.exists()
            else 0
        )

        headers = {
            "User-Agent":
                "academic-eeg-quality-audit/1.0"
        }

        if offset:
            headers["Range"] = f"bytes={offset}-"

        try:

            request = Request(
                url, headers=headers
            )

            with urlopen(
                request, timeout=90
            ) as response:

                status = response.status

                if offset and status != 206:

                    partial.unlink()
                    offset = 0

                    raise RuntimeError(
                        "Server ignored range; "
                        "resetting partial download"
                    )

                with partial.open(
                    "ab" if offset else "wb"
                ) as fh:

                    while True:

                        chunk = response.read(CHUNK)

                        if not chunk:
                            break

                        fh.write(chunk)

            if (
                size is not None
                and partial.stat().st_size != size
            ):

                raise RuntimeError(
                    "Incomplete file: "
                    f"{partial.stat().st_size}/{size} bytes"
                )

            if not valid_file(partial, entry):

                raise RuntimeError(
                    "Downloaded file fails checksum/size"
                )

            os.replace(
                partial, destination
            )

            return "downloaded"

        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            RuntimeError
        ) as exc:

            last_error = exc

            print(
                f"  retry {attempt}/4 "
                f"for {destination.name}: "
                f"{str(exc)[:125]}",
                flush=True
            )

            time.sleep(
                min(2 * attempt, 7)
            )

    raise RuntimeError(
        f"Download failed after retries: "
        f"{destination}: {last_error}"
    )


def numeric(value):

    try:

        x = float(str(value).strip())

        if np.isfinite(x):
            return x

        return None

    except (TypeError, ValueError):
        return None


# ==========================================
# EEG QUALITY AND CONTINUITY AUDIT
# ==========================================

def audit_eeg(subject, path, reference_channels):

    row = {
        "subject": subject,
        "status": "UNASSESSED",
        "file": str(path.relative_to(ROOT)),
        "first_usable_start_sec": "",
        "first_usable_end_sec": "",
        "usable_60s_clean_epochs": "",
        "usable_60s_clean_fraction": "",
        "recording_duration_sec": "",
        "eeg_channels": "",
        "sfreq": "",
        "boundary_count": "",
        "tested_starts": 0,
        "reason": ""
    }

    try:

        raw = mne.io.read_raw_eeglab(
            str(path),
            preload=False,
            verbose="ERROR"
        )

        sf = float(
            raw.info["sfreq"]
        )

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        channels = {
            raw.ch_names[i].strip().lower()
            for i in picks
        }

        row["sfreq"] = sf
        row["eeg_channels"] = len(picks)

        row["recording_duration_sec"] = round(
            raw.n_times / sf, 3
        )

        if (
            abs(sf - FS) > 1e-6
            or len(picks) != 19
            or channels != reference_channels
        ):

            row["status"] = "INCOMPATIBLE_HEADER"

            row["reason"] = (
                "EEG sampling rate, channel count "
                "or labels differ from EC reference"
            )

            return row

        window_n = WINDOW_S * FS

        if raw.n_times < window_n:

            row["status"] = "TOO_SHORT"

            row["reason"] = (
                "Recording has fewer than 60 seconds"
            )

            return row

        boundaries = [
            float(onset)
            for onset, label in zip(
                raw.annotations.onset,
                raw.annotations.description
            )
            if "boundary" in str(label).lower()
        ]

        row["boundary_count"] = len(boundaries)

        best_clean = -1

        latest_start_s = int(
            (raw.n_times - window_n) / FS
        )

        for start_s in range(
            0,
            latest_start_s + 1,
            START_STEP_S
        ):

            row["tested_starts"] += 1

            end_s = start_s + WINDOW_S

            # Photic stimulation markers are
            # not automatically discontinuities.
            # Only boundary annotations invalidate
            # a continuous candidate window.

            if any(
                start_s + 0.002 < t < end_s - 0.002
                for t in boundaries
            ):
                continue

            start_sample = start_s * FS

            signal = raw.get_data(
                picks=picks,
                start=start_sample,
                stop=start_sample + window_n
            ) * 1e6

            if (
                signal.shape != (19, window_n)
                or not np.isfinite(signal).all()
            ):
                continue

            # 30 nonoverlapping 2-second epochs.

            epochs = signal.reshape(
                19, 30, 1000
            ).transpose(
                1, 0, 2
            )

            ptp = np.ptp(
                epochs, axis=2
            )

            std = np.std(
                epochs, axis=2
            )

            clean = ~(
                ((ptp > 200).sum(axis=1) >= 3)
                | (ptp > 300).any(axis=1)
                | (std < 0.5).any(axis=1)
            )

            n_clean = int(
                clean.sum()
            )

            best_clean = max(
                best_clean, n_clean
            )

            if n_clean >= MIN_CLEAN_EPOCHS:

                row["status"] = "ELIGIBLE_60S"

                row["first_usable_start_sec"] = (
                    start_s
                )

                row["first_usable_end_sec"] = (
                    end_s
                )

                row["usable_60s_clean_epochs"] = (
                    n_clean
                )

                row["usable_60s_clean_fraction"] = (
                    round(n_clean / 30, 4)
                )

                row["reason"] = (
                    "Earliest 10s-grid 60s window "
                    "passing frozen epoch screen"
                )

                return row

        row["status"] = "NO_ELIGIBLE_60S_WINDOW"

        row["reason"] = (
            "No continuity/quality passing window; "
            f"max clean epochs in tested windows="
            f"{best_clean}"
        )

        return row

    except Exception as exc:

        row["status"] = "READ_ERROR"

        row["reason"] = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:260]}"
        )

        return row


# ==========================================
# VERIFY FROZEN INPUTS
# ==========================================

print(
    "===== MILESTONE 3B: "
    "PINNED EO DERIVATIVE DOWNLOAD =====",
    flush=True
)

for path in (
    EC_MANIFEST,
    EC_META,
    EO_META,
    REMOTE_MANIFEST
):

    if not path.is_file():
        raise FileNotFoundError(path)


manifest = {
    row["subject"]: row
    for row in read_rows(EC_MANIFEST)
}

selected = {
    sid: row
    for sid, row in manifest.items()
    if (
        row["status"] == "CANDIDATE"
        and row["group"] in ("A", "F", "C")
    )
}

if Counter(
    row["group"]
    for row in selected.values()
) != {"A": 23, "F": 18, "C": 23}:

    raise RuntimeError(
        "Frozen eligibility changed"
    )


# ==========================================
# RECONCILE NUMERICAL MMSE METADATA
# ==========================================

ecm = {
    row["participant_id"]: row
    for row in read_rows(EC_META, "\t")
}

eom = {
    row["participant_id"]: row
    for row in read_rows(EO_META, "\t")
}

if (
    len(ecm) != 88
    or len(eom) != 88
    or set(ecm) != set(eom)
):

    raise RuntimeError(
        "Participant IDs do not match"
    )

if any(
    eom[sid]["Group"].strip()
    != manifest[sid]["group"]
    for sid in manifest
):

    raise RuntimeError(
        "Diagnostic groups disagree"
    )


same_numeric = 0
real_differences = 0
missing = 0

for sid in sorted(ecm):

    a = numeric(
        ecm[sid].get("MMSE")
    )

    b = numeric(
        eom[sid].get("MMSE")
    )

    if a is None or b is None:

        missing += 1

    elif np.isclose(a, b):

        same_numeric += 1

    else:

        real_differences += 1


print(
    "\n===== MMSE TEXT-DIFFERENCE RECONCILIATION ====="
)

print(
    f"Numerically equal: {same_numeric}/88; "
    f"numeric differences: {real_differences}; "
    f"missing or nonnumeric: {missing}"
)

for group in ("A", "F", "C"):

    sid = next(
        s for s in sorted(selected)
        if selected[s]["group"] == group
    )

    print(
        f"{group} example {sid}: "
        f"EC-MMSE={ecm[sid].get('MMSE')!r}, "
        f"EO-MMSE={eom[sid].get('MMSE')!r}"
    )

print(
    "Clinical severity analyses remain "
    "based exclusively on EC metadata."
)


# ==========================================
# READ PINNED NEMAR MANIFEST
# ==========================================

entries = json.loads(
    REMOTE_MANIFEST.read_text(
        encoding="utf-8"
    )
)

if not isinstance(entries, list):

    raise RuntimeError(
        "Expected NEMAR manifest JSON array"
    )


files = {}

for entry in entries:

    if not isinstance(entry, dict):
        continue

    raw_path = str(
        entry.get("path", "")
    )

    path = PurePosixPath(raw_path)

    if (
        not raw_path
        or path.is_absolute()
        or ".." in path.parts
    ):
        continue

    if not (
        raw_path.startswith("derivatives/eeglab/")
        and raw_path.endswith(".set")
    ):
        continue

    subjects = [
        part
        for part in path.parts
        if re.fullmatch(
            r"sub-\d{3}", part
        )
    ]

    if (
        len(subjects) != 1
        or subjects[0] not in selected
    ):
        continue

    sid = subjects[0]

    if sid in files:

        raise RuntimeError(
            f"More than one derivative SET for {sid}"
        )

    files[sid] = entry


if set(files) != set(selected):

    raise RuntimeError(
        "Derivative coverage mismatch; missing="
        + str(
            sorted(
                set(selected) - set(files)
            )
        )
    )


size_total = sum(
    expected_size(entry) or 0
    for entry in files.values()
)

print(
    f"Preprocessed derivative SET files: "
    f"{len(files)}/64; "
    f"total listed size="
    f"{size_total / 2**30:.3f} GiB"
)

print(
    "Destination:", DATA
)


# ==========================================
# VERIFY EYES-CLOSED REFERENCE MONTAGE
# ==========================================

ref_sid = next(
    sid for sid in sorted(selected)
    if selected[sid]["group"] == "A"
)

ref_path = Path(
    manifest[ref_sid]["file"]
)

if not ref_path.is_absolute():

    ref_path = ROOT / ref_path


ref = mne.io.read_raw_eeglab(
    str(ref_path),
    preload=False,
    verbose="ERROR"
)

ref_picks = mne.pick_types(
    ref.info,
    eeg=True,
    exclude=[]
)

ref_channels = {
    ref.ch_names[i].strip().lower()
    for i in ref_picks
}

if (
    len(ref_picks) != 19
    or len(ref_channels) != 19
):

    raise RuntimeError(
        "Existing EC reference montage unexpected"
    )


# ==========================================
# DOWNLOAD PILOT FIRST, THEN REMAINING FILES
# ==========================================

ordered = []

for group in ("A", "F", "C"):

    ordered.append(
        next(
            sid for sid in sorted(files)
            if selected[sid]["group"] == group
        )
    )

ordered += [
    sid
    for sid in sorted(files)
    if sid not in ordered
]

all_rows = []
source_rows = []

for i, sid in enumerate(
    ordered, 1
):

    entry = files[sid]

    destination = (
        DATA / entry["path"]
    )

    action = download_file(
        entry, destination
    )

    row = audit_eeg(
        sid,
        destination,
        ref_channels
    )

    row["group"] = (
        selected[sid]["group"]
    )

    row["download_action"] = action

    row["bytes"] = (
        destination.stat().st_size
    )

    row["manifest_checksum_algorithm"] = (
        entry.get("checksum_algorithm", "")
    )

    row["manifest_checksum"] = (
        entry.get("checksum", "")
    )

    all_rows.append(row)

    source_rows.append({
        "subject": sid,
        "group": selected[sid]["group"],
        "path": entry["path"],
        "size": entry.get("size", ""),
        "checksum_algorithm":
            entry.get("checksum_algorithm", ""),
        "checksum": entry.get("checksum", ""),
        "dataset": "on006036",
        "version": "v1.0.0"
    })

    if (
        i <= 3
        or i % 10 == 0
        or i == len(ordered)
    ):

        print(
            f"  {i}/64 {sid} "
            f"({selected[sid]['group']}) "
            f"{action}: {row['status']}; "
            f"start={row['first_usable_start_sec']}; "
            f"clean={row['usable_60s_clean_epochs']}/30",
            flush=True
        )

        if row["status"] != "ELIGIBLE_60S":

            print(
                "    ", row["reason"],
                flush=True
            )

    if i == 3:

        print(
            "\n===== THREE-GROUP PILOT QUALITY ====="
        )

        for pilot in all_rows:

            print(
                f"{pilot['group']} "
                f"{pilot['subject']}: "
                f"{pilot['status']}; "
                f"seconds="
                f"{pilot['recording_duration_sec']}; "
                f"starts_tested="
                f"{pilot['tested_starts']}; "
                f"detail={pilot['reason']}"
            )

        if any(
            pilot["status"] in (
                "READ_ERROR",
                "INCOMPATIBLE_HEADER"
            )
            for pilot in all_rows
        ):

            save_rows(
                OUT / "eo_quality_inventory_partial.csv",
                all_rows
            )

            raise RuntimeError(
                "Pilot file/header problem. "
                "Stop before downloading remainder."
            )

    if i % 10 == 0:

        save_rows(
            OUT / "eo_quality_inventory_partial.csv",
            all_rows
        )


# ==========================================
# SAVE COMPLETE RECORDING ELIGIBILITY AUDIT
# ==========================================

save_rows(
    OUT / "eo_quality_inventory.csv",
    sorted(
        all_rows,
        key=lambda row: row["subject"]
    )
)

save_rows(
    OUT / "pinned_derivative_checksums.csv",
    sorted(
        source_rows,
        key=lambda row: row["subject"]
    )
)


print(
    "\n===== FULL COHORT QUALITY SUMMARY ====="
)

for group in ("A", "F", "C"):

    subset = [
        row for row in all_rows
        if row["group"] == group
    ]

    counts = Counter(
        row["status"]
        for row in subset
    )

    print(
        f"{group} ({len(subset)}): "
        f"{dict(counts)}"
    )

    eligible = [
        row for row in subset
        if row["status"] == "ELIGIBLE_60S"
    ]

    if eligible:

        print(
            "  chosen 60s window starts: "
            f"min="
            f"{min(row['first_usable_start_sec'] for row in eligible)}s, "
            f"max="
            f"{max(row['first_usable_start_sec'] for row in eligible)}s; "
            "minimum clean epochs="
            f"{min(row['usable_60s_clean_epochs'] for row in eligible)}/30"
        )


print(
    "Unusable subjects (technical/quality):"
)

for row in sorted(
    all_rows,
    key=lambda item: item["subject"]
):

    if row["status"] != "ELIGIBLE_60S":

        print(
            " ", row["subject"],
            row["group"],
            row["status"],
            row["reason"]
        )


print(
    "\n===== MILESTONE 3B COMPLETE ====="
)

print(
    "Quality manifest:",
    OUT / "eo_quality_inventory.csv"
)

print(
    "EEG root:", DATA
)

print(
    "No models trained; original EC features "
    "and results unchanged."
)

print(
    "Photic stimulation events are not treated "
    "as recording discontinuities; only "
    "boundary annotations invalidate windows."
)

