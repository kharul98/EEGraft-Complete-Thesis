#!/usr/bin/env python3
"""Milestone 3C: boundary-aware 30-second EEG feasibility audit."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

import mne
import numpy as np


ROOT = Path.cwd()
BASE = ROOT / "reports" / "thesis_final"

OUT = BASE / "milestone_3c_boundary_feasibility_v1"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = (
    ROOT / "reports/openneuro_5min_manifest_v2.csv"
)

QC60 = (
    BASE / "milestone_3b_eyes_open_quality_v1"
    / "eo_quality_inventory.csv"
)

SF = 500
WIN = 30
STEP = 10

# 30 seconds = 15 two-second epochs.
# Require at least 50% retained, rounded upward.
MIN_CLEAN = 8


def load(path):

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save(path, rows):

    if not rows:
        raise RuntimeError("No results to save")

    columns = list(dict.fromkeys(
        key for r in rows for key in r
    ))

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


def unique(rows, column):

    result = {}

    for row in rows:

        key = row[column]

        if key in result:
            raise RuntimeError(
                "Duplicate participant: " + key
            )

        result[key] = row

    return result


# ==========================================
# 1. LOCK THE AUDIT PROTOCOL
# ==========================================

protocol = {
    "version": "m3c_boundary_feasibility_v1",

    "aim":
        "Assess uninterrupted shorter windows "
        "before zero-shot prediction",

    "original_60s_eligibility":
        "Preserved as recorded in Milestone 3B",

    "new_window_seconds": WIN,

    "candidate_start_grid_seconds": STEP,

    "epoch_seconds": 2,

    "min_clean_epochs": MIN_CLEAN,

    "screen":
        "Reject >=3 EEG channels ptp>200uV; "
        "any ptp>300uV; any std<0.5uV",

    "selection":
        "Earliest continuous quality-eligible "
        "30-second window",

    "continuity":
        "Exclude candidates containing "
        "an EEGLAB boundary annotation",

    "endpoint":
        "Group-specific eligibility only; "
        "no classification or label-based "
        "window choice"
}

lock = OUT / "locked_protocol.json"

encoded = (
    json.dumps(
        protocol,
        sort_keys=True,
        indent=2
    ) + "\n"
)

if (
    lock.exists()
    and lock.read_text() != encoded
):
    raise RuntimeError(
        "Protocol changed since prior run"
    )

if not lock.exists():
    lock.write_text(encoded)


for p in (MANIFEST, QC60):

    if not p.is_file():
        raise FileNotFoundError(p)


fingerprints = {
    str(p.relative_to(ROOT)):
        hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    for p in (MANIFEST, QC60)
}

source_lock = OUT / "input_sha256.json"

encoded = (
    json.dumps(
        fingerprints,
        sort_keys=True,
        indent=2
    ) + "\n"
)

if (
    source_lock.exists()
    and source_lock.read_text() != encoded
):
    raise RuntimeError(
        "Existing source changed since prior run"
    )

if not source_lock.exists():
    source_lock.write_text(encoded)


# ==========================================
# 2. VERIFY ORIGINAL COHORT AND QUALITY
# ==========================================

manifest = unique(
    load(MANIFEST),
    "subject"
)

qc = unique(
    load(QC60),
    "subject"
)

selected = {
    sid: row
    for sid, row in manifest.items()
    if (
        row["status"] == "CANDIDATE"
        and row["group"] in ("A", "F", "C")
    )
}

if (
    len(selected) != 64
    or set(qc) != set(selected)
):
    raise RuntimeError(
        "Selected cohort or 3B inventory mismatch"
    )

if Counter(
    r["group"]
    for r in selected.values()
) != {
    "A": 23,
    "F": 18,
    "C": 23
}:
    raise RuntimeError(
        "Wrong diagnostic-group counts"
    )

if Counter(
    r["status"]
    for r in qc.values()
) != {
    "ELIGIBLE_60S": 40,
    "NO_ELIGIBLE_60S_WINDOW": 24
}:
    raise RuntimeError(
        "Unexpected original 60-second status counts"
    )


# ==========================================
# 3. VERIFY EYES-CLOSED CHANNEL MONTAGE
# ==========================================

ref_sid = sorted(
    s for s in selected
    if selected[s]["group"] == "A"
)[0]

ref_file = Path(
    manifest[ref_sid]["file"]
)

if not ref_file.is_absolute():
    ref_file = ROOT / ref_file

ref = mne.io.read_raw_eeglab(
    str(ref_file),
    preload=False,
    verbose="ERROR"
)

ref_picks = mne.pick_types(
    ref.info,
    eeg=True,
    exclude=[]
)

reference_channels = {
    ref.ch_names[i].strip().lower()
    for i in ref_picks
}

if (
    len(ref_picks) != 19
    or len(reference_channels) != 19
):
    raise RuntimeError(
        "Eyes-closed reference montage "
        "differs from 19 EEG channels"
    )


# ==========================================
# 4. AUDIT CONTINUITY AND 30S QUALITY
# ==========================================

print(
    "===== MILESTONE 3C: "
    "BOUNDARY-AWARE DURATION FEASIBILITY =====",
    flush=True
)

print(
    "Original 60s eligibility frozen: 40/64; "
    "trying one predefined 30s fallback."
)

results = []

for idx, sid in enumerate(
    sorted(selected), 1
):

    prior = qc[sid]

    filepath = Path(
        prior["file"]
    )

    if not filepath.is_absolute():
        filepath = ROOT / filepath

    if not filepath.is_file():
        raise FileNotFoundError(filepath)

    raw = mne.io.read_raw_eeglab(
        str(filepath),
        preload=False,
        verbose="ERROR"
    )

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[]
    )

    chan = {
        raw.ch_names[i].strip().lower()
        for i in picks
    }

    if (
        abs(
            float(raw.info["sfreq"]) - SF
        ) > 1e-6
        or len(picks) != 19
        or chan != reference_channels
    ):
        raise RuntimeError(
            "Montage or sampling mismatch: " + sid
        )

    total = raw.n_times / SF

    boundaries = sorted(set(
        float(t)
        for t, desc in zip(
            raw.annotations.onset,
            raw.annotations.description
        )
        if (
            "boundary" in str(desc).lower()
            and 0 < float(t) < total
        )
    ))

    cuts = (
        [0.0]
        + boundaries
        + [float(total)]
    )

    spans = np.diff(
        np.asarray(
            cuts,
            dtype=float
        )
    )

    longest = float(
        spans.max()
    )

    latest30 = int(
        (raw.n_times - WIN * SF) // SF
    )

    latest60 = int(
        (raw.n_times - 60 * SF) // SF
    )

    # Candidate windows must not cross
    # any EEGLAB discontinuity.

    def candidates(length, latest):

        if latest < 0:
            return []

        return [
            start
            for start in range(
                0,
                latest + 1,
                STEP
            )
            if not any(
                start + 0.002
                < t
                < start + length - 0.002
                for t in boundaries
            )
        ]

    starts60 = candidates(
        60,
        latest60
    )

    starts30 = candidates(
        WIN,
        latest30
    )

    # Cross-check exact previous screening.
    # Do not silently change prior eligibility.

    if (
        prior["status"] == "ELIGIBLE_60S"
        and not starts60
    ):
        raise RuntimeError(
            "Previous 60s window lacks a "
            "continuous grid window: " + sid
        )

    if (
        prior["status"] != "ELIGIBLE_60S"
        and starts60
    ):
        raise RuntimeError(
            "Previous 60s no-window report "
            "is inconsistent: " + sid
        )

    state = "NO_CONTINUOUS_30S_WINDOW"

    first_start = ""
    clean_count = ""

    tried = 0
    best_clean = -1
    invalid_data = 0

    if starts30:

        state = "FAILED_30S_EPOCH_SCREEN"

        for start in starts30:

            tried += 1

            sig = raw.get_data(
                picks=picks,
                start=start * SF,
                stop=(start + WIN) * SF
            ) * 1e6

            if (
                sig.shape != (
                    19,
                    WIN * SF
                )
                or not np.isfinite(sig).all()
            ):
                invalid_data += 1
                continue

            # 15 epochs, each two seconds long.

            ep = sig.reshape(
                19,
                WIN // 2,
                2 * SF
            ).transpose(
                1, 0, 2
            )

            ptp = np.ptp(
                ep,
                axis=2
            )

            sd = np.std(
                ep,
                axis=2
            )

            good = ~(
                (
                    np.sum(
                        ptp > 200,
                        axis=1
                    ) >= 3
                )
                | np.any(
                    ptp > 300,
                    axis=1
                )
                | np.any(
                    sd < 0.5,
                    axis=1
                )
            )

            valid_epochs = int(
                good.sum()
            )

            best_clean = max(
                best_clean,
                valid_epochs
            )

            if valid_epochs >= MIN_CLEAN:

                state = "ELIGIBLE_30S"

                first_start = start
                clean_count = valid_epochs

                break

    row = {
        "subject": sid,

        "group":
            selected[sid]["group"],

        "original_60s_status":
            prior["status"],

        "original_60s_start_sec":
            prior["first_usable_start_sec"],

        "recording_seconds":
            round(total, 3),

        "boundary_count":
            len(boundaries),

        "max_uninterrupted_span_sec":
            round(longest, 3),

        "valid_60s_starts_on_original_10s_grid":
            len(starts60),

        "60s_span_exists_but_10s_grid_misses_it":
            int(
                longest >= 60
                and not starts60
            ),

        "valid_30s_starts_on_10s_grid":
            len(starts30),

        "30s_status":
            state,

        "30s_selected_start_sec":
            first_start,

        "30s_selected_end_sec":
            first_start + WIN
            if state == "ELIGIBLE_30S"
            else "",

        "30s_clean_epochs":
            clean_count,

        "30s_clean_fraction":
            clean_count / (WIN // 2)
            if state == "ELIGIBLE_30S"
            else "",

        "30s_quality_starts_tested":
            tried,

        "best_tested_30s_clean_epochs":
            best_clean,

        "30s_nonfinite_candidate_count":
            invalid_data,

        "source_file":
            str(
                filepath.relative_to(ROOT)
            )
    }

    results.append(row)

    if (
        idx % 10 == 0
        or idx == 64
    ):

        print(
            f"  Continuity/quality audit: "
            f"{idx}/64",
            flush=True
        )

        save(
            OUT / "30s_feasibility_partial.csv",
            results
        )


save(
    OUT / "duration_continuity_and_30s_qc.csv",
    results
)


# ==========================================
# 5. PRINT CONTINUITY DIAGNOSIS
# ==========================================

print(
    "\n===== BOUNDARY CONTINUITY DIAGNOSIS ====="
)

for group in ("A", "F", "C"):

    r = [
        z for z in results
        if z["group"] == group
    ]

    gaps = np.array([
        z["max_uninterrupted_span_sec"]
        for z in r
    ])

    excluded = [
        z for z in r
        if (
            z["original_60s_status"]
            != "ELIGIBLE_60S"
        )
    ]

    median_boundaries = np.median([
        z["boundary_count"]
        for z in r
    ])

    misses = sum(
        z["60s_span_exists_but_10s_grid_misses_it"]
        for z in excluded
    )

    print(
        f"{group}: "
        f"median longest uninterrupted "
        f"span={np.median(gaps):.1f}s; "
        f"median boundaries="
        f"{median_boundaries:.0f}; "
        f"potential 60s grid misses="
        f"{misses}"
    )


# ==========================================
# 6. PRINT 30S ELIGIBILITY
# ==========================================

print(
    "\n===== 30S QUALITY ELIGIBILITY BY GROUP ====="
)

for group in ("A", "F", "C"):

    r = [
        z for z in results
        if z["group"] == group
    ]

    freq = Counter(
        z["30s_status"]
        for z in r
    )

    old = sum(
        z["original_60s_status"]
        == "ELIGIBLE_60S"
        for z in r
    )

    now = sum(
        z["30s_status"]
        == "ELIGIBLE_30S"
        for z in r
    )

    gained = sum(
        (
            z["30s_status"] == "ELIGIBLE_30S"
            and z["original_60s_status"]
            != "ELIGIBLE_60S"
        )
        for z in r
    )

    print(
        f"{group}: "
        f"original 60s={old}/{len(r)}; "
        f"new 30s={now}/{len(r)}; "
        f"newly eligible={gained}; "
        f"30s statuses={dict(freq)}"
    )


# ==========================================
# 7. REMAINING EXCLUSIONS
# ==========================================

print(
    "\n===== REMAINING 30S EXCLUSIONS ====="
)

excluded = [
    z for z in results
    if z["30s_status"] != "ELIGIBLE_30S"
]

print(
    f"Total still ineligible: "
    f"{len(excluded)}/64"
)

for z in excluded:

    print(
        f"  {z['subject']} ({z['group']}): "
        f"{z['30s_status']}; "
        f"longest span="
        f"{z['max_uninterrupted_span_sec']:.1f}s; "
        f"boundaries={z['boundary_count']}; "
        f"best clean="
        f"{z['best_tested_30s_clean_epochs']}"
    )


# ==========================================
# 8. FINISH
# ==========================================

print(
    "\n===== MILESTONE 3C COMPLETE ====="
)

print(
    "Saved:",
    OUT / "duration_continuity_and_30s_qc.csv"
)

print(
    "No classifier trained. No results "
    "or eligibility from Milestone 3B "
    "were overwritten."
)

