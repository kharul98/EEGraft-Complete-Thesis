#!/usr/bin/env python3
"""Metadata-only Dryad AD recording identity audit."""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ROOT / "data/dryad_padic/alz_c1_new.mat",
    Path("/mnt/c/Users/user4/Downloads/alz_c1_new.mat")
]

OUT_CSV = (
    ROOT / "results/p3_dryad_ad_candidate_recordings.csv"
)

OUT_JSON = (
    ROOT / "docs/audit/p3_dryad_ad_identity_summary.json"
)

FS_ASSUMED = 500


def metadata_value(h5, ref, depth=0):
    if not ref:
        return ["empty"]

    if depth > 4:
        return ["depth_limit"]

    obj = h5[ref]

    if not isinstance(obj, h5py.Dataset):
        return ["other", obj.name]

    if obj.size > 2048:
        return ["large_not_loaded", list(obj.shape)]

    if h5py.check_dtype(ref=obj.dtype) is not None:
        return [
            "nested",
            [
                metadata_value(h5, item, depth + 1)
                for item in np.asarray(
                    obj[()]
                ).ravel(order="F")
            ]
        ]

    values = np.asarray(
        obj[()]
    ).ravel(order="F")

    if obj.dtype.kind in "uifb":
        cleaned = [
            float(v) if obj.dtype.kind == "f"
            else int(v)
            for v in values
        ]

        return ["numeric", list(obj.shape), cleaned]

    if obj.dtype.kind in "SU":
        return [
            "text",
            list(obj.shape),
            [str(v) for v in values]
        ]

    return ["unsupported", str(obj.dtype)]


def fingerprint(h5, ref):
    if not ref:
        return ""

    content = metadata_value(h5, ref)

    if content == ["empty"]:
        return ""

    serialized = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        serialized.encode()
    ).hexdigest()[:20]


def main():
    source = next(
        (
            p for p in CANDIDATES
            if p.is_file()
            and p.stat().st_size > 100_000_000
            and h5py.is_hdf5(str(p))
        ),
        None
    )

    if source is None:
        print("ERROR: Valid AD file not found.")
        print("No data was modified.")
        return

    rows = []

    private_checks = defaultdict(
        lambda: defaultdict(list)
    )

    with h5py.File(source, "r") as h5:
        group = h5["alz_r"]
        matrix = group["G"]

        if matrix.shape != (3, 43):
            print(
                "NOTICE: Unexpected reference grid:",
                matrix.shape
            )

        if h5py.check_dtype(ref=matrix.dtype) is None:
            print("ERROR: G is not a reference matrix.")
            return

        fields = [
            name
            for name in ("eeg_files", "birth", "sex", "n")
            if name in group
            and group[name].shape == matrix.shape
        ]

        refs = matrix[()]

        metadata = {
            name: group[name][()]
            for name in fields
        }

        for (r, c), ref in np.ndenumerate(refs):
            if not ref:
                continue

            obj = h5[ref]

            shape = tuple(
                int(value)
                for value in getattr(obj, "shape", ())
            )

            kind = getattr(
                getattr(obj, "dtype", None),
                "kind",
                ""
            )

            is_eeg = (
                isinstance(obj, h5py.Dataset)
                and len(shape) == 2
                and 19 in shape
                and max(shape) > 19
                and kind in "fiu"
            )

            samples = max(shape) if is_eeg else 0

            eligible = bool(
                is_eeg
                and samples >= 300 * FS_ASSUMED
            )

            rows.append({
                "slot_row": r + 1,
                "slot_column": c + 1,
                "reference_path": obj.name,
                "signal_shape": str(shape),
                "eeg_19channel_array": bool(is_eeg),
                "samples": samples,
                "seconds_if_500hz": (
                    round(samples / FS_ASSUMED, 3)
                    if is_eeg else None
                ),
                "eligible_by_length_only": eligible
            })

            if is_eeg:
                for field in fields:
                    digest = fingerprint(
                        h5,
                        metadata[field][r, c]
                    )

                    if digest:
                        private_checks[c + 1][field].append(
                            digest
                        )

    df = pd.DataFrame(rows)

    eligible = df.loc[
        df.eligible_by_length_only
    ].copy()

    per_column = (
        eligible.groupby("slot_column")
        .size()
        .to_dict()
    )

    distribution = Counter(
        per_column.values()
    )

    consistency = {}

    for field in ("birth", "sex", "n", "eeg_files"):
        multiple = [
            values[field]
            for values in private_checks.values()
            if len(values.get(field, [])) >= 2
        ]

        consistency[field] = {
            "multi_recording_columns_with_metadata":
                len(multiple),
            "columns_with_matching_values":
                sum(
                    len(set(values)) == 1
                    for values in multiple
                )
        }

    if len(eligible):
        durations = eligible[
            "seconds_if_500hz"
        ].astype(float)

        length_summary = [
            float(durations.min()),
            float(durations.median()),
            float(durations.max())
        ]
    else:
        length_summary = [None, None, None]

    summary = {
        "status":
            "METADATA_ONLY; patients not verified",

        "source_name": source.name,

        "reference_slots": len(df),

        "unique_referenced_objects":
            int(df.reference_path.nunique()),

        "long_19channel_recording_slots":
            len(eligible),

        "unique_long_signal_references":
            int(eligible.reference_path.nunique()),

        "columns_with_long_recordings":
            len(per_column),

        "eligible_recordings_per_column": {
            str(k): int(v)
            for k, v in sorted(distribution.items())
        },

        "long_recordings_by_row": {
            str(k): int(v)
            for k, v in eligible.groupby(
                "slot_row"
            ).size().items()
        },

        "duration_seconds_if_500hz_min_median_max":
            length_summary,

        "within_column_metadata_consistency":
            consistency,

        "limitations": [
            "Columns are candidate participant clusters.",
            "Independent patient identities remain unverified.",
            "Sampling frequency assumed from publication.",
            "Recording length does not prove eyes-closed continuity.",
            "Personal metadata values are not exported."
        ]
    }

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUT_CSV,
        index=False
    )

    OUT_JSON.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("===== AD RECORDING IDENTITY AUDIT =====")

    for key, value in summary.items():
        if key != "limitations":
            print(key, ":", value)

    print()
    print("===== RECORDINGS PER COLUMN =====")

    for count, number in sorted(distribution.items()):
        print(
            f"{count} recording(s): "
            f"{number} column(s)"
        )

    print()
    print("No complete EEG signal arrays were loaded.")
    print("Participant independence remains unverified.")
    print("AD METADATA AUDIT COMPLETE")


if __name__ == "__main__":
    main()
