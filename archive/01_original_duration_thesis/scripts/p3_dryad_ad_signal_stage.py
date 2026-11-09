#!/usr/bin/env python3
"""Exploratory Dryad AD spectral reliability, recording slots only."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
from scipy.signal import butter, sosfilt, welch

ROOT = Path(__file__).resolve().parents[1]

SRC = [
    ROOT / "data/dryad_padic/alz_c1_new.mat",
    Path("/mnt/c/Users/user4/Downloads/alz_c1_new.mat")
]

OUT = ROOT / "results/p3_dryad_ad_signal_slots.csv"
CLUS = ROOT / "results/p3_dryad_ad_column_sensitivity.csv"
FAIL = ROOT / "results/p3_dryad_ad_signal_exclusions.csv"
REPORT = ROOT / "docs/audit/p3_dryad_ad_signal_summary.json"

FS = 500
NEEDED = 300 * FS

SOS = butter(
    4, (1, 40),
    btype="bandpass",
    fs=FS,
    output="sos"
)

BANDS = (
    (1, 4), (4, 8), (8, 13),
    (13, 30), (30, 40)
)


def norm(v):
    v = np.asarray(v, dtype=float)
    return v / max(float(v.sum()), 1e-20)


def tv(v, r):
    return float(
        np.abs(norm(v) - norm(r)).sum() / 2
    )


def epoch_bands(data):
    freq, p = welch(
        data.T,
        fs=FS,
        nperseg=4 * FS,
        noverlap=2 * FS,
        axis=-1
    )

    mask = (freq >= 1) & (freq <= 40)

    total = np.trapezoid(
        p[:, mask],
        freq[mask],
        axis=-1
    )

    if (
        not np.isfinite(total).all()
        or (total <= 0).any()
    ):
        raise ValueError(
            "Nonpositive spectral power in one or more channels"
        )

    values = []

    for low, high in BANDS:
        mask = (
            (freq >= low)
            & (freq <= high)
        )

        power = np.trapezoid(
            p[:, mask],
            freq[mask],
            axis=-1
        )

        values.append(power / total)

    return np.stack(values, axis=-1)


def analyze(signal):
    if (
        signal.shape != (NEEDED, 19)
        or not np.isfinite(signal).all()
    ):
        raise ValueError(
            "Nonfinite or unexpected EEG array"
        )

    # Causal processing; no EEG after 300s is read.
    filtered = sosfilt(
        SOS,
        signal,
        axis=0
    )

    epochs = np.stack([
        epoch_bands(
            filtered[i*5000:(i+1)*5000]
        )
        for i in range(30)
    ])

    def med(start, stop):
        return np.median(
            epochs[start:stop],
            axis=(0, 1)
        )

    reference = med(24, 30)

    return {
        "prefix_60_tv": tv(
            med(0, 6), reference
        ),
        "prefix_120_tv": tv(
            med(0, 12), reference
        ),
        "prefix_180_tv": tv(
            med(0, 18), reference
        ),
        "fixed_end_60_tv": tv(
            med(12, 18), reference
        ),
        "fixed_end_120_tv": tv(
            med(6, 18), reference
        ),
        "fixed_end_180_tv": tv(
            med(0, 18), reference
        ),
        "late_halves_30_tv": tv(
            med(24, 27),
            med(27, 30)
        )
    }


def main():
    source = next(
        (
            p for p in SRC
            if p.is_file()
            and p.stat().st_size > 1_000_000
            and h5py.is_hdf5(p)
        ),
        None
    )

    if source is None:
        raise FileNotFoundError(
            "Valid Dryad AD HDF5 file not found"
        )

    print(
        "===== REAL DRYAD AD SIGNAL STAGE =====",
        flush=True
    )

    print("Source:", source)
    print("Sampling frequency: 500 Hz (from publication)")
    print("Analysis unit: recording slot, not verified patient")

    results = []
    exclusions = []

    with h5py.File(source, "r") as h5:
        grid = h5["alz_r"]["G"][()]

        if grid.shape != (3, 43):
            raise ValueError(
                f"Unexpected reference grid: {grid.shape}"
            )

        for column in range(grid.shape[1]):
            for row in range(grid.shape[0]):
                ref = grid[row, column]

                if not ref:
                    continue

                obj = h5[ref]

                if (
                    not isinstance(obj, h5py.Dataset)
                    or obj.ndim != 2
                ):
                    continue

                shape = tuple(
                    int(v) for v in obj.shape
                )

                if (
                    shape[1] == 19
                    and shape[0] >= NEEDED
                ):
                    samples = shape[0]
                    signal = lambda: obj[:NEEDED, :]

                elif (
                    shape[0] == 19
                    and shape[1] >= NEEDED
                ):
                    samples = shape[1]
                    signal = lambda: obj[:, :NEEDED].T

                else:
                    continue

                try:
                    x = np.asarray(
                        signal(),
                        dtype=np.float64
                    )

                    metrics = analyze(x)

                    metrics.update({
                        "slot_column": column + 1,
                        "slot_row": row + 1,
                        "signal_reference": obj.name,
                        "recording_seconds_assuming_500hz":
                            round(samples / FS, 3)
                    })

                    metrics[
                        "primary_delta_180_minus_60"
                    ] = (
                        metrics["fixed_end_180_tv"]
                        - metrics["fixed_end_60_tv"]
                    )

                    results.append(metrics)

                except (
                    ValueError,
                    OSError,
                    RuntimeError
                ) as error:
                    exclusions.append({
                        "slot_column": column + 1,
                        "slot_row": row + 1,
                        "reason": (
                            type(error).__name__
                            + ": "
                            + str(error)[:150]
                        )
                    })

            if (column + 1) % 10 == 0:
                print(
                    f"Processed {column+1}/43 columns",
                    flush=True
                )

    df = pd.DataFrame(results)

    if df.empty:
        raise ValueError(
            "No analyzable long EEG arrays"
        )

    if df.duplicated([
        "slot_column",
        "slot_row"
    ]).any():
        raise ValueError("Duplicate recording slots")

    first = (
        df.sort_values(
            ["slot_column", "slot_row"]
        )
        .drop_duplicates(
            "slot_column",
            keep="first"
        )
    )

    last = (
        df.sort_values(
            ["slot_column", "slot_row"]
        )
        .drop_duplicates(
            "slot_column",
            keep="last"
        )
    )

    paired = first[
        [
            "slot_column",
            "slot_row",
            "primary_delta_180_minus_60"
        ]
    ].merge(
        last[
            [
                "slot_column",
                "slot_row",
                "primary_delta_180_minus_60"
            ]
        ],
        on="slot_column",
        validate="one_to_one",
        suffixes=("_first", "_last")
    )

    if not np.isfinite(
        df.select_dtypes(
            include="number"
        ).to_numpy()
    ).all():
        raise ValueError("Nonfinite results")

    def stats(values):
        a = np.asarray(values, dtype=float)

        return {
            "n": len(a),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "fraction_negative": float(
                np.mean(a < 0)
            )
        }

    report = {
        "status":
            "Exploratory recording-slot feasibility",
        "source_filename": source.name,
        "source_bytes": source.stat().st_size,
        "reference_time_s": [240, 300],
        "fixed_60_s": [120, 180],
        "fixed_180_s": [0, 180],
        "records_with_successful_300s_analysis":
            int(len(df)),
        "candidate_columns_analyzed":
            int(first.slot_column.nunique()),
        "excluded_long_array_slots":
            len(exclusions),
        "primary_first_recording_per_column":
            stats(
                first.primary_delta_180_minus_60
            ),
        "last_recording_per_column_sensitivity":
            stats(
                last.primary_delta_180_minus_60
            ),
        "no_participant_identity_claim": True,
        "limitations": [
            "Array length does not prove uninterrupted eyes-closed EEG.",
            "Sampling frequency comes from publication.",
            "MATLAB columns are not verified patient IDs.",
            "No independent-patient confidence intervals are calculated.",
            "No diagnosis classification is performed."
        ]
    }

    for path in (OUT, CLUS, FAIL, REPORT):
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    df.sort_values(
        ["slot_column", "slot_row"]
    ).to_csv(
        OUT,
        index=False,
        float_format="%.12g"
    )

    paired.to_csv(
        CLUS,
        index=False,
        float_format="%.12g"
    )

    pd.DataFrame(
        exclusions,
        columns=[
            "slot_column",
            "slot_row",
            "reason"
        ]
    ).to_csv(
        FAIL,
        index=False
    )

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("===== DRYAD SIGNAL EXPERIMENT RESULTS =====")
    print("Analyzed recording slots:", len(df))
    print("Candidate columns:", len(first))
    print("Excluded long arrays:", len(exclusions))

    print()
    print("Median disagreement by duration:")

    print(
        first[
            [
                "prefix_60_tv",
                "prefix_120_tv",
                "prefix_180_tv",
                "fixed_end_60_tv",
                "fixed_end_120_tv",
                "fixed_end_180_tv"
            ]
        ]
        .median()
        .round(6)
        .to_string()
    )

    print()
    print("===== PRIMARY EFFECT, DESCRIPTIVE ONLY =====")

    print(
        "First per column:",
        report["primary_first_recording_per_column"]
    )

    print(
        "Last per column:",
        report["last_recording_per_column_sensitivity"]
    )

    print()
    print("Patient independence and condition continuity unverified.")
    print("DRYAD AD SIGNAL STAGE COMPLETE")


if __name__ == "__main__":
    main()
