#!/usr/bin/env python3
"""Exploratory Dryad age-overlap and age-matching sensitivity."""

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path.home() / "alzheimers-thesis"
DOWNLOADS = Path("/mnt/c/Users/user4/Downloads")

AD_FILE = ROOT / "data/dryad_padic/alz_c1_new.mat"
AD_RESULTS = ROOT / "results/p3_dryad_ad_signal_slots.csv"
HC_RESULTS = ROOT / "results/p3_dryad_hc_signal_slots.csv"
OUTPUT = ROOT / "docs/audit/p3_dryad_age_sensitivity.json"

DELTA = "primary_delta_180_minus_60"
CALIPER_YEARS = 5.0


def age_components(h5, ref):
    """Follow nested references, reading only small metadata."""
    visited = set()

    for _ in range(8):
        if not ref:
            raise ValueError("Empty age reference")

        obj = h5[ref]

        if obj.name in visited:
            raise ValueError("Cyclic age reference")

        visited.add(obj.name)

        if not isinstance(obj, h5py.Dataset):
            raise ValueError("Age reference is not a dataset")

        if h5py.check_dtype(ref=obj.dtype) is not None:
            if obj.size != 1:
                raise ValueError("Ambiguous nested age references")

            ref = np.asarray(obj[()]).reshape(-1)[0]
            continue

        if obj.size not in (1, 2) or obj.dtype.kind not in "iuf":
            raise ValueError("Unexpected age structure")

        values = np.asarray(
            obj[()], dtype=float
        ).ravel(order="F")

        year_component = float(values[0])

        if not (18 <= year_component <= 110):
            raise ValueError("Implausible age-year component")

        if len(values) == 2 and not (0 <= values[1] <= 11):
            raise ValueError("Unexpected second age component")

        return year_component

    raise ValueError("Age reference nesting too deep")


def describe(df):
    if df.empty:
        return {"n": 0}

    return {
        "n": int(len(df)),
        "median_age_component": float(
            df.age_year_component.median()
        ),
        "mean_age_component": float(
            df.age_year_component.mean()
        ),
        "mean_duration_effect": float(
            df[DELTA].mean()
        ),
        "median_duration_effect": float(
            df[DELTA].median()
        ),
        "fraction_improving": float(
            (df[DELTA] < 0).mean()
        ),
    }


def comparison(ad, hc):
    if ad.empty or hc.empty:
        return {"status": "Insufficient overlap"}

    return {
        "AD": describe(ad),
        "HC": describe(hc),
        "AD_minus_HC_mean_effect": float(
            ad[DELTA].mean() - hc[DELTA].mean()
        )
    }


def match_ages(ad, hc):
    """Maximize eligible pairs, then minimize age differences."""
    a = ad.age_year_component.to_numpy()
    h = hc.age_year_component.to_numpy()

    gaps = np.abs(a[:, None] - h[None, :])

    # Real matches outside the caliper are prohibited.
    cost = np.where(
        gaps <= CALIPER_YEARS,
        gaps,
        10000.0
    )

    # Dummy columns allow unmatched AD candidates.
    dummies = np.full((len(a), len(a)), 100.0)
    full_cost = np.concatenate(
        [cost, dummies], axis=1
    )

    rows, cols = linear_sum_assignment(full_cost)

    valid = cols < len(h)
    rows, cols = rows[valid], cols[valid]

    valid = gaps[rows, cols] <= CALIPER_YEARS
    rows, cols = rows[valid], cols[valid]

    if len(rows) == 0:
        return {
            "matched_pairs": 0,
            "status": "No eligible age matches"
        }

    matched_ad = ad.iloc[rows].reset_index(drop=True)
    matched_hc = hc.iloc[cols].reset_index(drop=True)

    return {
        "matched_pairs": int(len(rows)),
        "mean_absolute_age_gap": float(
            gaps[rows, cols].mean()
        ),
        "max_absolute_age_gap": float(
            gaps[rows, cols].max()
        ),
        "AD": describe(matched_ad),
        "HC": describe(matched_hc),
        "AD_minus_HC_mean_effect": float(
            (matched_ad[DELTA].to_numpy()
             - matched_hc[DELTA].to_numpy()).mean()
        ),
        "status": "Exploratory matched sensitivity only"
    }


def main():
    print("===== DRYAD AGE-OVERLAP SENSITIVITY =====")

    hc_files = [
        p for p in DOWNLOADS.rglob("controls_c1_new.mat")
        if p.is_file() and p.stat().st_size > 1_000_000_000
    ]

    if not AD_FILE.is_file():
        raise FileNotFoundError("Dryad AD file unavailable")

    if not hc_files:
        raise FileNotFoundError("Dryad control file unavailable")

    hc_file = max(
        hc_files,
        key=lambda p: p.stat().st_size
    )

    ad = pd.read_csv(AD_RESULTS)
    hc = pd.read_csv(HC_RESULTS)

    required_ad = {"slot_column", "slot_row", DELTA}
    required_hc = {"slot", DELTA}

    if not required_ad.issubset(ad.columns):
        raise ValueError("AD result columns missing")

    if not required_hc.issubset(hc.columns):
        raise ValueError("Control result columns missing")

    ad = (
        ad.sort_values(["slot_column", "slot_row"])
        .drop_duplicates("slot_column", keep="first")
        .copy()
    )

    hc = hc.sort_values("slot").copy()

    if ad.slot_column.duplicated().any():
        raise ValueError("Duplicate AD columns")

    if hc.slot.duplicated().any():
        raise ValueError("Duplicate control slots")

    if len(ad) != 43 or len(hc) != 96:
        raise ValueError(
            f"Unexpected result counts: AD={len(ad)}, HC={len(hc)}"
        )

    with h5py.File(AD_FILE, "r") as f:
        age_grid = f["alz_r"]["age"]

        ad["age_year_component"] = [
            age_components(
                f,
                age_grid[
                    int(row.slot_row) - 1,
                    int(row.slot_column) - 1
                ]
            )
            for row in ad.itertuples()
        ]

    with h5py.File(hc_file, "r") as f:
        age_grid = f["controls_r"]["age"]

        hc["age_year_component"] = [
            age_components(
                f,
                age_grid[0, int(row.slot) - 1]
            )
            for row in hc.itertuples()
        ]

    for group in (ad, hc):
        if not np.isfinite(
            group[[DELTA, "age_year_component"]]
            .to_numpy(dtype=float)
        ).all():
            raise ValueError("Nonfinite age or outcome")

    low = max(
        ad.age_year_component.min(),
        hc.age_year_component.min()
    )

    high = min(
        ad.age_year_component.max(),
        hc.age_year_component.max()
    )

    ad_overlap = ad[
        ad.age_year_component.between(low, high)
    ].copy()

    hc_overlap = hc[
        hc.age_year_component.between(low, high)
    ].copy()

    report = {
        "status": "Exploratory age sensitivity",
        "age_interpretation":
            "First MATLAB age component, provisionally years",
        "all_available": comparison(ad, hc),
        "overlap_range": [float(low), float(high)],
        "common_age_range": comparison(
            ad_overlap, hc_overlap
        ),
        "nearest_age_matching_without_replacement":
            match_ages(ad, hc),
        "matching_caliper_years": CALIPER_YEARS,
        "participant_identities_verified": False,
        "eye_condition_verified": False,
        "notes": [
            "One AD recording per candidate column.",
            "Control observation unit is unique recording array.",
            "Matching uses ages only, not EEG outcomes.",
            "No p-values or patient-level confidence intervals.",
            "Age is not the only possible confound.",
            "Individual age values are not exported."
        ]
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    for name in (
        "all_available",
        "common_age_range",
        "nearest_age_matching_without_replacement"
    ):
        print()
        print("=====", name.upper(), "=====")

        for key, value in report[name].items():
            print(key, ":", value)

    print()
    print("Shared age range:", report["overlap_range"])
    print("Age matching caliper: 5 years")
    print("Saved aggregate report:", OUTPUT)
    print("No raw EEG signals loaded.")
    print("No individual ages exported.")
    print("===== AGE SENSITIVITY COMPLETE =====")


if __name__ == "__main__":
    main()
