#!/usr/bin/env python3
"""Dryad control nested-reference resolution and signal analysis."""

from pathlib import Path
import importlib.util
import json

import h5py
import numpy as np
import pandas as pd

ROOT = Path.home() / "alzheimers-thesis"
BASE = Path("/mnt/c/Users/user4/Downloads")
FS = 500
NEEDED = 300 * FS


def resolve_signal(h5, initial_ref):
    """Follow single-element HDF5 reference containers."""
    ref = initial_ref
    chain = []
    visited = set()

    for _ in range(8):
        if not ref:
            return None, chain

        obj = h5[ref]

        if obj.name in visited:
            return None, chain

        visited.add(obj.name)
        chain.append({
            "path": obj.name,
            "shape": tuple(obj.shape)
                if hasattr(obj, "shape") else None
        })

        if not isinstance(obj, h5py.Dataset):
            return None, chain

        ref_type = h5py.check_dtype(ref=obj.dtype)

        if ref_type is None:
            return obj, chain

        if obj.size != 1:
            return None, chain

        next_ref = obj[()].reshape(-1)[0]

        if not isinstance(next_ref, h5py.Reference):
            return None, chain

        ref = next_ref

    return None, chain


def main():
    print("===== DRYAD CONTROL NESTED-REFERENCE AUDIT =====")

    candidates = [
        p for p in BASE.rglob("controls_c1_new.mat")
        if p.stat().st_size > 1_000_000_000
    ]

    if not candidates:
        print("ERROR: Large Dryad control file not found.")
        return

    source = max(candidates, key=lambda p: p.stat().st_size)

    if not h5py.is_hdf5(str(source)):
        print("ERROR: Control file is not valid HDF5.")
        return

    print("Source:", source)
    print("Size GiB:", round(source.stat().st_size / 1024**3, 3))

    inventory = []
    eligible = []
    seen_signals = set()

    with h5py.File(source, "r") as h5:
        grid = h5["controls_r"]["G"]

        print("Reference grid:", grid.shape)

        for index, ref in enumerate(grid[()].flat, start=1):
            obj, chain = resolve_signal(h5, ref)

            if index <= 3:
                print()
                print("REFERENCE CHAIN", index)

                for item in chain:
                    print(
                        " ", item["path"],
                        "shape:", item["shape"]
                    )

            if obj is None:
                inventory.append({
                    "slot": index,
                    "status": "unresolved"
                })
                continue

            shape = tuple(int(x) for x in obj.shape)

            is_eeg = (
                len(shape) == 2
                and 19 in shape
                and max(shape) > 19
                and obj.dtype.kind in "fiu"
            )

            samples = max(shape) if is_eeg else 0

            long_enough = (
                is_eeg and samples >= NEEDED
            )

            duplicate = obj.name in seen_signals

            inventory.append({
                "slot": index,
                "signal_path": obj.name,
                "shape": str(shape),
                "samples": samples,
                "seconds_if_500hz": (
                    samples / FS if is_eeg else None
                ),
                "eligible_by_length": bool(long_enough),
                "duplicate_signal_reference": duplicate
            })

            if long_enough and not duplicate:
                eligible.append({
                    "slot": index,
                    "path": obj.name,
                    "shape": shape
                })
                seen_signals.add(obj.name)

        print()
        print("===== CONTROL ARRAY ELIGIBILITY =====")

        table = pd.DataFrame(inventory)

        out_inventory = (
            ROOT / "results/p3_dryad_hc_reference_inventory.csv"
        )

        table.to_csv(out_inventory, index=False)

        print("Total slots:", len(inventory))
        print("Unique eligible signal arrays:", len(eligible))

        lengths = [
            entry["samples"]
            for entry in inventory
            if entry.get("eligible_by_length", False)
        ]

        if lengths:
            print(
                "Eligible duration min/median/max (s):",
                [
                    round(v / FS, 2)
                    for v in (
                        min(lengths),
                        np.median(lengths),
                        max(lengths)
                    )
                ]
            )

        if not eligible:
            print(
                "No eligible signal arrays. "
                "Analysis stopped without changing prior results."
            )
            return

        module_path = (
            ROOT / "scripts/p3_dryad_ad_signal_stage.py"
        )

        spec = importlib.util.spec_from_file_location(
            "dryad_ad_stage", str(module_path)
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if module.FS != FS or module.NEEDED != NEEDED:
            raise ValueError(
                "AD and HC protocols do not match"
            )

        print()
        print("===== REAL DRYAD CONTROL EEG ANALYSIS =====")

        results = []
        exclusions = []

        for position, record in enumerate(eligible, start=1):
            obj = h5[record["path"]]

            try:
                if obj.shape[1] == 19:
                    data = np.asarray(
                        obj[:NEEDED, :],
                        dtype=np.float64
                    )
                else:
                    data = np.asarray(
                        obj[:, :NEEDED].T,
                        dtype=np.float64
                    )

                metrics = module.analyze(data)

                metrics["slot"] = record["slot"]
                metrics["signal_path"] = record["path"]

                metrics["primary_delta_180_minus_60"] = (
                    metrics["fixed_end_180_tv"]
                    - metrics["fixed_end_60_tv"]
                )

                results.append(metrics)

            except Exception as error:
                exclusions.append({
                    "slot": record["slot"],
                    "reason": (
                        type(error).__name__
                        + ": " + str(error)[:180]
                    )
                })

            if position % 15 == 0:
                print(
                    f"Processed {position}/{len(eligible)} "
                    "unique eligible arrays",
                    flush=True
                )

    df = pd.DataFrame(results)

    pd.DataFrame(
        exclusions,
        columns=["slot", "reason"]
    ).to_csv(
        ROOT / "results/p3_dryad_hc_signal_exclusions.csv",
        index=False
    )

    if df.empty:
        print("No control EEG measurements completed.")
        return

    df.to_csv(
        ROOT / "results/p3_dryad_hc_signal_slots.csv",
        index=False,
        float_format="%.12g"
    )

    delta = df["primary_delta_180_minus_60"]

    summary = {
        "status": "Exploratory control recording-level analysis",
        "successful_unique_recordings": int(len(df)),
        "excluded_recordings": int(len(exclusions)),
        "mean_180_minus_60": float(delta.mean()),
        "median_180_minus_60": float(delta.median()),
        "fraction_negative": float((delta < 0).mean()),
        "assumed_sampling_rate_hz": FS,
        "reference_window_seconds": [240, 300],
        "participant_independence_verified": False,
        "recording_condition_verified": False,
        "same_analysis_function_as_dryad_ad": True
    }

    report = (
        ROOT / "docs/audit/p3_dryad_hc_signal_summary.json"
    )

    report.write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print()
    print("===== DRYAD CONTROL SIGNAL RESULTS =====")
    print("Analyzed recordings:", len(df))
    print("Excluded recordings:", len(exclusions))

    print()
    print("Median disagreement:")
    print(
        df[
            [
                "fixed_end_60_tv",
                "fixed_end_120_tv",
                "fixed_end_180_tv"
            ]
        ].median().round(6).to_string()
    )

    print()
    print("===== PRIMARY CONTROL EFFECT =====")

    for key, value in summary.items():
        if key not in (
            "status",
            "reference_window_seconds"
        ):
            print(key, ":", value)

    print()
    print("DRYAD CONTROL SIGNAL STAGE COMPLETE")


if __name__ == "__main__":
    main()
