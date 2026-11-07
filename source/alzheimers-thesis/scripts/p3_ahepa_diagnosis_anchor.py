#!/usr/bin/env python3
"""AHEPA diagnosis-specific fixed-end duration reliability audit."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "alzheimers-thesis"

SOURCE = ROOT / "results/p2_duration_anchor_control.csv"
OUTPUT = ROOT / "results/p3_ahepa_diagnosis_anchor_summary.csv"
PAIRS = ROOT / "results/p3_ahepa_diagnosis_anchor_participants.csv"

print("===== AHEPA FIXED-END DIAGNOSIS AUDIT =====")

df = pd.read_csv(SOURCE)

required = {
    "participant_id", "group",
    "start_s", "end_s", "duration_s",
    "tv_to_late_240_300"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for col in (
    "start_s", "end_s",
    "duration_s", "tv_to_late_240_300"
):
    df[col] = pd.to_numeric(df[col], errors="raise")

df["group"] = (
    df["group"].astype(str).str.strip().str.upper()
)

print("Source rows:", len(df))
print(
    "Available groups:",
    df.groupby("group").participant_id.nunique().to_dict()
)


def window(start, end, duration, name):
    mask = (
        np.isclose(df.start_s, start)
        & np.isclose(df.end_s, end)
        & np.isclose(df.duration_s, duration)
    )

    selected = df.loc[
        mask,
        ["participant_id", "group", "tv_to_late_240_300"]
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No exact {start}-{end}s window found"
        )

    values = selected["tv_to_late_240_300"]

    if (
        not np.isfinite(values).all()
        or not values.between(0, 1).all()
    ):
        raise ValueError(f"Invalid TV distances: {name}")

    groups = selected.groupby("participant_id")

    if (groups.group.nunique() != 1).any():
        raise ValueError("Conflicting participant labels")

    spread = groups.tv_to_late_240_300.agg(
        lambda x: x.max() - x.min()
    )

    if (spread > 1e-9).any():
        raise ValueError(
            f"Conflicting duplicate measurements: {name}"
        )

    print(
        name,
        "| source rows:", len(selected),
        "| participants:",
        selected.participant_id.nunique()
    )

    return (
        selected
        .drop_duplicates("participant_id")
        .rename(
            columns={"tv_to_late_240_300": name}
        )
    )


short = window(
    120, 180, 60,
    "fixed_end_60_tv"
)

long = window(
    0, 180, 180,
    "fixed_end_180_tv"
)

paired = short.merge(
    long,
    on="participant_id",
    validate="one_to_one",
    suffixes=("_short", "_long")
)

if (
    len(paired) != len(short)
    or len(paired) != len(long)
):
    raise ValueError("Unmatched participants")

if not (
    paired.group_short == paired.group_long
).all():
    raise ValueError("Paired diagnosis mismatch")

paired = (
    paired.rename(columns={"group_short": "group"})
    .drop(columns="group_long")
)

paired["delta_180_minus_60"] = (
    paired.fixed_end_180_tv
    - paired.fixed_end_60_tv
)

if len(paired) != 88:
    raise ValueError(
        f"Expected 88 participants, found {len(paired)}"
    )

rng = np.random.default_rng(20261107)


def summarize(name, subset):
    values = subset.delta_180_minus_60.to_numpy(
        dtype=float
    )

    bootstrap = rng.choice(
        values,
        size=(5000, len(values)),
        replace=True
    ).mean(axis=1)

    lo, hi = np.quantile(
        bootstrap, [0.025, 0.975]
    )

    return {
        "group": name,
        "n_participants": len(values),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "fraction_improving": float(
            np.mean(values < 0)
        ),
        "bootstrap_mean_lo": float(lo),
        "bootstrap_mean_hi": float(hi)
    }


rows = [summarize("ALL", paired)]

for group, subset in paired.groupby("group"):
    rows.append(summarize(group, subset))

summary = pd.DataFrame(rows)

overall = float(
    summary.loc[
        summary.group.eq("ALL"), "mean_delta"
    ].iloc[0]
)

print("Previously reported AHEPA mean: -0.003275")
print("Recomputed AHEPA mean:", round(overall, 6))

if abs(overall - (-0.003275)) > 0.0005:
    raise ValueError(
        "AHEPA overall effect differs from prior result. "
        "Check methods before exporting."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUTPUT, index=False, float_format="%.12g"
)

paired.to_csv(
    PAIRS, index=False, float_format="%.12g"
)

print()
print("===== AHEPA DIAGNOSIS-STRATIFIED RESULTS =====")

print(
    summary.round(6).to_string(index=False)
)

print()
print("===== AHEPA AD VERSUS CN =====")

if {"AD", "CN"}.issubset(
    set(paired.group)
):
    ad = paired.loc[
        paired.group.eq("AD"),
        "delta_180_minus_60"
    ].to_numpy()

    cn = paired.loc[
        paired.group.eq("CN"),
        "delta_180_minus_60"
    ].to_numpy()

    print(
        "AD mean minus CN mean:",
        round(float(ad.mean() - cn.mean()), 6)
    )
else:
    print("AD/CN label pair not found.")

print()
print("Saved summary:", OUTPUT)
print("Saved paired measurements:", PAIRS)
print("No EEG arrays loaded.")
print("No results from earlier studies modified.")
print("===== AHEPA AUDIT COMPLETE =====")
