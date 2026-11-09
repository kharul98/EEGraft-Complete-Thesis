import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

D = Path.home() / "research/caueeg"
M = D / "manifests"

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5",
    "P3", "Pz", "P4", "T6", "O1", "O2"
]

ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

def norm_channel(name):
    s = str(name).strip()
    s = re.sub(r"^EEG[\s_-]*", "", s, flags=re.I)
    s = re.sub(r"[\s_-]+(?:AVG|REF|LE)$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return ALIASES.get(s, s)

def norm_unit(unit):
    return str(unit).replace("µ", "u").replace("μ", "u").strip().lower()

df = pd.read_csv(M / "recordings_audit.csv", dtype={"serial": str})
results = []

for n, row in enumerate(df.itertuples(index=False), 1):
    out = {
        "serial": row.serial,
        "status": "error",
        "fs_hz": np.nan,
        "duration_s": np.nan,
        "n_channels": 0,
        "missing_channels": "",
        "eeg_units": "",
    }

    path = Path(row.edf_path)
    if not path.is_file():
        out["status"] = "missing_edf"
        results.append(out)
        continue

    reader = None
    try:
        reader = pyedflib.EdfReader(str(path))
        labels = reader.getSignalLabels()
        frequencies = reader.getSampleFrequencies()
        samples = reader.getNSamples()

        lookup = {}
        for i, name in enumerate(labels):
            key = norm_channel(name)
            lookup.setdefault(key, []).append(i)

        missing = [
            ch for ch in CHANNELS
            if norm_channel(ch) not in lookup
        ]
        duplicate = [
            ch for ch in CHANNELS
            if len(lookup.get(norm_channel(ch), [])) > 1
        ]

        out["n_channels"] = len(labels)
        out["duration_s"] = float(reader.getFileDuration())
        out["missing_channels"] = "|".join(missing + duplicate)

        if missing or duplicate:
            out["status"] = "channel_mismatch"
        else:
            indices = [lookup[norm_channel(ch)][0] for ch in CHANNELS]
            eeg_fs = np.array(
                [frequencies[i] for i in indices], dtype=float
            )
            eeg_units = [
                norm_unit(reader.getPhysicalDimension(i))
                for i in indices
            ]
            out["fs_hz"] = float(eeg_fs[0])
            out["eeg_units"] = "|".join(sorted(set(eeg_units)))

            reasons = []
            if not np.allclose(eeg_fs, eeg_fs[0], atol=0.001):
                reasons.append("mixed_eeg_sampling_rates")
            if eeg_fs[0] <= 90:
                reasons.append("sampling_rate_too_low")
            if out["duration_s"] < 4:
                reasons.append("recording_too_short")
            if any(int(samples[i]) < 4 * eeg_fs[0] for i in indices):
                reasons.append("insufficient_samples")
            if any(unit != "uv" for unit in eeg_units):
                reasons.append("review_physical_units")

            out["status"] = "ok" if not reasons else "|".join(reasons)

    except Exception as exc:
        out["status"] = f"read_error:{type(exc).__name__}"
    finally:
        if reader is not None:
            reader.close()

    results.append(out)
    if n % 100 == 0:
        print(f"Inspected {n}/{len(df)} EDF headers", flush=True)

qc = pd.DataFrame(results)
qc.to_csv(M / "edf_qc.csv", index=False)

summary = {
    "n_recordings": len(qc),
    "status_counts": {
        str(k): int(v)
        for k, v in Counter(qc.status).items()
    },
    "sampling_rates_hz": sorted(
        qc.loc[qc.status == "ok", "fs_hz"].dropna().unique().tolist()
    ),
}

(M / "edf_qc_summary.json").write_text(
    json.dumps(summary, indent=2)
)
print(json.dumps(summary, indent=2))
