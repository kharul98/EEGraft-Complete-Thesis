#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
from collections import Counter

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ROOT / "data/dryad_padic/alz_c1_new.mat",
    Path("/mnt/c/Users/user4/Downloads/alz_c1_new.mat")
]

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
    print(
        "ERROR: No valid nonempty AD HDF5 file found."
    )
    print("Original files were not changed.")

else:
    print("===== DRYAD AD STRUCTURE AUDIT =====")
    print("Reading:", source)

    rows = []

    with h5py.File(source, "r") as h5:
        group = h5["alz_r"]
        signals = group["G"]

        assert signals.ndim == 2

        signal_grid = signals[()]
        filenames = group["eeg_files"][()]

        assert filenames.shape == signal_grid.shape

        for (r, c), ref in np.ndenumerate(
            signal_grid
        ):
            if not ref:
                continue

            target = h5[ref]

            if isinstance(target, h5py.Dataset):
                shape = tuple(
                    int(n) for n in target.shape
                )
            else:
                shape = ()

            channels = (
                19
                if len(shape) == 2 and 19 in shape
                else None
            )

            samples = (
                max(shape)
                if channels == 19
                else None
            )

            filename_hash = ""

            file_ref = filenames[r, c]

            if file_ref:
                file_obj = h5[file_ref]

                if (
                    isinstance(file_obj, h5py.Dataset)
                    and file_obj.size <= 1024
                    and file_obj.dtype.kind in ("u", "i")
                ):
                    codes = np.asarray(
                        file_obj[()]
                    ).reshape(-1, order="F")

                    filename = "".join(
                        chr(int(v))
                        for v in codes
                        if 32 <= int(v) < 127
                    )

                    if filename:
                        filename_hash = (
                            hashlib.sha256(
                                filename.encode()
                            )
                            .hexdigest()[:12]
                        )

            rows.append({
                "slot_row": r + 1,
                "slot_column": c + 1,
                "hdf5_signal_path": target.name,
                "signal_shape": str(shape),
                "dtype": str(
                    getattr(target, "dtype", "")
                ),
                "channels_inferred": channels,
                "samples_inferred": samples,
                "candidate_300s_at_500hz": bool(
                    samples is not None
                    and samples >= 150000
                ),
                "filename_fingerprint":
                    filename_hash
            })

        layout = signal_grid.shape

        n_ref = group["n"][0, 0]

        if n_ref:
            n_obj = h5[n_ref]
            n_shape = str(n_obj.shape)
            n_dtype = str(
                getattr(n_obj, "dtype", "")
            )
        else:
            n_shape = "null"
            n_dtype = "null"

    df = pd.DataFrame(rows)

    column_counts = Counter(
        df.groupby("slot_column").size()
    )

    summary = {
        "source_file": source.name,
        "source_size_bytes": source.stat().st_size,
        "reference_grid_shape": list(layout),
        "occupied_signal_slots": len(df),
        "unique_signal_references": int(
            df.hdf5_signal_path.nunique()
        ),
        "unique_filename_fingerprints": int(
            df.loc[
                df.filename_fingerprint.ne(""),
                "filename_fingerprint"
            ].nunique()
        ),
        "potential_300s_slots_assuming_500hz": int(
            df.candidate_300s_at_500hz.sum()
        ),
        "slots_by_row": {
            str(k): int(v)
            for k, v in df.groupby(
                "slot_row"
            ).size().items()
        },
        "column_occupancy_distribution": {
            str(k): int(v)
            for k, v in column_counts.items()
        },
        "n_first_ref_shape": n_shape,
        "n_first_ref_dtype": n_dtype,
        "limitations": [
            "Storage slots are not independent patients.",
            "Sampling rate is assumed from publication.",
            "Signal length does not establish continuity.",
            "Participant mapping remains unverified."
        ]
    }

    output = (
        ROOT
        / "results/p3_dryad_ad_reference_inventory.csv"
    )

    audit = (
        ROOT
        / "docs/audit/p3_dryad_ad_reference_summary.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    audit.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(
        output, index=False
    )

    audit.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("===== DRYAD REFERENCE INVENTORY =====")

    for key, value in summary.items():
        if key != "limitations":
            print(key, ":", value)

    print()
    print("===== SIGNAL SHAPE EXAMPLES =====")

    print(
        df[
            [
                "slot_row",
                "slot_column",
                "signal_shape",
                "dtype",
                "candidate_300s_at_500hz"
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    print()
    print("Metadata audit saved.")
    print("No complete EEG arrays were loaded.")
    print("===== DRYAD STRUCTURE AUDIT COMPLETE =====")
