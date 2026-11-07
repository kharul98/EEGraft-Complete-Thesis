from __future__ import annotations

from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
import mne


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    ROOT / "data" / "eeg" / "on004504_v1.0.0"
)
AUDIT_DIR = ROOT / "docs" / "audit"
RESULT_DIR = ROOT / "results" / "eeg_feasibility"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def find_subject(text: str) -> str | None:
    match = re.search(r"(sub-[A-Za-z0-9]+)", text)
    return match.group(1) if match else None


print("==================================================")
print("EEG DATASET FEASIBILITY AUDIT")
print("==================================================")

print("Python:", sys.version.split()[0])
print("MNE:", mne.__version__)
print("Data root:", DATA_ROOT)

participants_files = sorted(DATA_ROOT.rglob("participants.tsv"))

if not participants_files:
    raise RuntimeError("participants.tsv was not downloaded.")

participants_path = min(participants_files, key=lambda p: len(p.parts))
participants = pd.read_csv(participants_path, sep="\t")

print()
print("----- PARTICIPANTS -----")
print("participants.tsv:", participants_path)
print("Shape:", participants.shape)
print("Columns:", list(participants.columns))
print()
print(participants.head(10).to_string(index=False))

print()
print("----- LOW-CARDINALITY PARTICIPANT VARIABLES -----")

for col in participants.columns:
    unique_count = participants[col].nunique(dropna=False)
    if unique_count <= 12:
        print()
        print(f"{col}:")
        print(participants[col].value_counts(dropna=False).to_string())

if "participant_id" not in participants.columns:
    raise RuntimeError("participant_id column is missing.")

if participants["participant_id"].duplicated().any():
    print("WARNING: duplicate participant IDs detected.")
else:
    print()
    print("Participant IDs are unique.")

set_files = sorted(
    p for p in DATA_ROOT.rglob("*.set")
    if "derivatives" in p.parts
)

print()
print("----- DERIVATIVE EEG FILES -----")
print("Derivative .set file count:", len(set_files))

records = []
errors = []

for i, path in enumerate(set_files, start=1):
    subject = find_subject(str(path))

    try:
        raw = mne.io.read_raw_eeglab(
            path,
            preload=False,
            verbose="ERROR",
        )

        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        duration_seconds = n_times / sfreq

        records.append(
            {
                "participant_id": subject,
                "file": str(path.relative_to(ROOT)),
                "sfreq_hz": sfreq,
                "n_channels": len(raw.ch_names),
                "n_samples": n_times,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60.0,
                "channels": "|".join(raw.ch_names),
            }
        )

        if i % 10 == 0 or i == len(set_files):
            print(f"Read {i}/{len(set_files)} files")

    except Exception as exc:
        errors.append(
            {
                "file": str(path.relative_to(ROOT)),
                "error": repr(exc),
            }
        )

recordings = pd.DataFrame(records)

if recordings.empty:
    print()
    print("No derivative EEG files could be read.")
    print("Errors:")
    for item in errors[:20]:
        print(item)
    raise RuntimeError("EEG derivative audit failed.")

merged = recordings.merge(
    participants,
    how="left",
    on="participant_id",
    validate="many_to_one",
)

print()
print("----- RECORDING STRUCTURE -----")
print("Readable recordings:", len(recordings))
print("Read errors:", len(errors))
print("Unique subjects in recordings:", recordings["participant_id"].nunique())
print("Sampling frequencies:", sorted(recordings["sfreq_hz"].unique().tolist()))
print("Channel counts:", sorted(recordings["n_channels"].unique().tolist()))
print("Unique channel layouts:", recordings["channels"].nunique())

print()
print("----- DURATION SUMMARY -----")
print(
    recordings["duration_seconds"]
    .describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    .to_string()
)

if "Group" in merged.columns:
    print()
    print("----- GROUP COUNTS -----")
    print(
        merged[["participant_id", "Group"]]
        .drop_duplicates()["Group"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("----- DURATION BY GROUP (SECONDS) -----")
    print(
        merged.groupby("Group")["duration_seconds"]
        .agg(["count", "min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )

print()
print("----- FIXED-DURATION ELIGIBILITY -----")

duration_grid = [30, 60, 120, 180, 240, 300, 360, 480, 600]

eligibility_rows = []

for seconds in duration_grid:
    eligible = merged[merged["duration_seconds"] >= seconds].copy()

    row = {
        "duration_seconds": seconds,
        "eligible_recordings": int(len(eligible)),
        "eligible_subjects": int(eligible["participant_id"].nunique()),
    }

    if "Group" in eligible.columns:
        counts = (
            eligible[["participant_id", "Group"]]
            .drop_duplicates()["Group"]
            .value_counts()
            .to_dict()
        )

        for group, count in counts.items():
            row[f"group_{group}"] = int(count)

    eligibility_rows.append(row)

eligibility = pd.DataFrame(eligibility_rows)
print(eligibility.fillna(0).to_string(index=False))

print()
print("----- DEMOGRAPHIC SUMMARY -----")

for col in ["Age", "Gender", "MMSE", "Group"]:
    if col in participants.columns:
        print()
        print(f"{col}:")
        if pd.api.types.is_numeric_dtype(participants[col]):
            print(participants.groupby("Group")[col].describe().round(2).to_string()
                  if "Group" in participants.columns
                  else participants[col].describe().round(2).to_string())
        else:
            print(participants[col].value_counts(dropna=False).to_string())

recordings_path = RESULT_DIR / "recording_inventory.csv"
eligibility_path = RESULT_DIR / "duration_eligibility.csv"
errors_path = RESULT_DIR / "read_errors.json"
summary_path = AUDIT_DIR / "eeg_feasibility_summary.json"

merged.to_csv(recordings_path, index=False)
eligibility.to_csv(eligibility_path, index=False)

with errors_path.open("w", encoding="utf-8") as f:
    json.dump(errors, f, indent=2)

summary = {
    "participant_rows": int(len(participants)),
    "unique_participant_ids": int(participants["participant_id"].nunique()),
    "derivative_set_files": int(len(set_files)),
    "readable_recordings": int(len(recordings)),
    "read_errors": int(len(errors)),
    "unique_recording_subjects": int(recordings["participant_id"].nunique()),
    "sampling_frequencies": sorted(
        float(x) for x in recordings["sfreq_hz"].unique()
    ),
    "channel_counts": sorted(
        int(x) for x in recordings["n_channels"].unique()
    ),
    "unique_channel_layouts": int(recordings["channels"].nunique()),
    "min_duration_seconds": float(recordings["duration_seconds"].min()),
    "median_duration_seconds": float(recordings["duration_seconds"].median()),
    "max_duration_seconds": float(recordings["duration_seconds"].max()),
}

if "Group" in participants.columns:
    summary["participant_group_counts"] = {
        str(k): int(v)
        for k, v in participants["Group"].value_counts(dropna=False).items()
    }

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print()
print("----- SAVED AUDIT ARTIFACTS -----")
print(recordings_path)
print(eligibility_path)
print(errors_path)
print(summary_path)

print()
print("==================================================")
print("EEG FEASIBILITY SUMMARY")
print("==================================================")
print(json.dumps(summary, indent=2))
print("==================================================")
