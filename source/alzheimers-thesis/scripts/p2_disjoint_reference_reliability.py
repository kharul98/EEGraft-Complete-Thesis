#!/usr/bin/env python3
"""Disjoint within-recording EEG reliability experiment."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne

from scipy.signal import butter, sosfilt, welch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

INVENTORY = ROOT / "results/p1_raw_recording_inventory.csv"
PREVIOUS = ROOT / "results/p2_prefix_robust_features.csv"

OUT = ROOT / "results/p2_disjoint_reference_reliability.csv"
OUT_REF = ROOT / "results/p2_disjoint_reference_features.csv"
OUT_SUMMARY = ROOT / "docs/audit/p2_disjoint_reference_summary.json"

FS = 500
TIMES = (30, 60, 120, 180)
EPOCH_SAMPLES = 10 * FS

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

SOS = butter(
    4, (1, 40),
    btype="bandpass",
    fs=FS,
    output="sos"
)


def normalized(vector):
    vector = np.asarray(vector, dtype=float)

    if (
        not np.isfinite(vector).all()
        or vector.min() < 0
        or vector.sum() <= 0
    ):
        raise ValueError("Invalid spectral vector")

    return vector / vector.sum()


def tv(a, b):
    return float(
        0.5 * np.abs(
            normalized(a) - normalized(b)
        ).sum()
    )


def epoch_vector(signal):
    frequency, psd = welch(
        signal,
        fs=FS,
        nperseg=4 * FS,
        noverlap=2 * FS,
        axis=-1
    )

    total_mask = (
        (frequency >= 1)
        & (frequency <= 40)
    )

    total = np.trapezoid(
        psd[:, total_mask],
        frequency[total_mask],
        axis=-1
    )

    values = []

    for low, high in BANDS.values():
        mask = (
            (frequency >= low)
            & (frequency <= high)
        )

        band = np.trapezoid(
            psd[:, mask],
            frequency[mask],
            axis=-1
        )

        values.append(
            band / np.maximum(total, 1e-20)
        )

    return np.stack(values, axis=-1)


def mean_interval(values, bootstrap=4000, seed=20261107):
    a = np.asarray(values, dtype=float)

    if not np.isfinite(a).all():
        raise ValueError("Nonfinite comparison values")

    rng = np.random.default_rng(seed)

    samples = rng.integers(
        0,
        len(a),
        (bootstrap, len(a))
    )

    resampled = a[samples].mean(axis=1)

    bounds = np.quantile(
        resampled,
        [0.025, 0.975]
    )

    return [
        float(a.mean()),
        float(bounds[0]),
        float(bounds[1])
    ]


def main():
    inventory = pd.read_csv(INVENTORY)
    previous = pd.read_csv(PREVIOUS)

    if (
        len(inventory) != 88
        or inventory.participant_id.nunique() != 88
    ):
        raise ValueError("Expected 88 raw recordings")

    if (
        len(previous) != 440
        or previous.participant_id.nunique() != 88
    ):
        raise ValueError("Expected 440 prefix records")

    if previous.duplicated(
        ["participant_id", "seconds"]
    ).any():
        raise ValueError("Duplicate prefix records")

    previous = previous.set_index(
        ["participant_id", "seconds"]
    )

    rows = []
    reference_rows = []
    max_reproduction_error = 0.0

    print(
        "===== DISJOINT EEG REFERENCE STUDY =====",
        flush=True
    )

    print(
        "Reference: 240-300s; early prefixes: 30-180s",
        flush=True
    )

    for i, item in enumerate(
        inventory.itertuples(index=False), 1
    ):
        subject = str(item.participant_id)
        group = str(item.group)

        raw = mne.io.read_raw_eeglab(
            str(ROOT / item.path),
            preload=False,
            verbose="ERROR"
        )

        if (
            raw.info["sfreq"] != FS
            or len(raw.ch_names) != 19
            or raw.n_times < 300 * FS
        ):
            raise ValueError(
                f"Unexpected recording: {subject}"
            )

        raw_data = raw.get_data(
            picks="eeg",
            start=0,
            stop=300 * FS
        )

        if (
            raw_data.shape != (19, 300 * FS)
            or not np.isfinite(raw_data).all()
        ):
            raise ValueError(
                f"Invalid EEG data: {subject}"
            )

        filter_state = np.zeros(
            (len(SOS), 19, 2),
            dtype=float
        )

        epochs = []

        for epoch in range(30):
            begin = epoch * EPOCH_SAMPLES

            filtered, filter_state = sosfilt(
                SOS,
                raw_data[
                    :, begin:begin + EPOCH_SAMPLES
                ],
                axis=-1,
                zi=filter_state
            )

            epochs.append(
                epoch_vector(filtered)
            )

        epochs = np.stack(epochs)

        # Independent late temporal reference.
        late = np.median(
            epochs[24:30],
            axis=(0, 1)
        )

        late_first = np.median(
            epochs[24:27],
            axis=(0, 1)
        )

        late_second = np.median(
            epochs[27:30],
            axis=(0, 1)
        )

        # Equal-duration, later-starting control.
        delayed_60 = np.median(
            epochs[12:18],
            axis=(0, 1)
        )

        early_60 = np.median(
            epochs[:6],
            axis=(0, 1)
        )

        ref_row = {
            "participant_id": subject,
            "group": group,
            "reference_start_s": 240,
            "reference_stop_s": 300,
            "late_halves_tv":
                tv(late_first, late_second),
            "delayed_60_tv_to_late":
                tv(delayed_60, late),
            "early_60_tv_to_late":
                tv(early_60, late)
        }

        for j, band in enumerate(BANDS):
            ref_row[
                f"late_median_relative_{band}"
            ] = float(late[j])

        reference_rows.append(ref_row)

        previous_prefix = None

        for seconds in (30, 60, 120, 180, 300):
            vec = np.median(
                epochs[:seconds // 10],
                axis=(0, 1)
            )

            reference = previous.loc[
                (subject, seconds)
            ]

            if str(reference["group"]) != group:
                raise ValueError(
                    f"Group mismatch: {subject}"
                )

            expected = np.array([
                reference[
                    f"median_relative_{band}"
                ]
                for band in BANDS
            ], dtype=float)

            error = float(
                np.max(np.abs(vec - expected))
            )

            max_reproduction_error = max(
                max_reproduction_error,
                error
            )

            if seconds in TIMES:
                rows.append({
                    "participant_id": subject,
                    "group": group,
                    "seconds": seconds,
                    "tv_to_late_240_300":
                        tv(vec, late),
                    "online_spectral_drift": (
                        np.nan
                        if previous_prefix is None
                        else tv(vec, previous_prefix)
                    )
                })

            previous_prefix = vec

        if i % 22 == 0:
            print(
                f"Processed {i}/88 recordings",
                flush=True
            )

    df = pd.DataFrame(rows).sort_values(
        ["participant_id", "seconds"]
    ).reset_index(drop=True)

    refs = pd.DataFrame(reference_rows).sort_values(
        "participant_id"
    ).reset_index(drop=True)

    if len(df) != 352 or len(refs) != 88:
        raise ValueError("Unexpected output count")

    if not df.groupby(
        "participant_id"
    ).seconds.nunique().eq(4).all():
        raise ValueError("Missing duration conditions")

    if max_reproduction_error > 1e-8:
        raise ValueError(
            "Previous causal features did not reproduce: "
            f"{max_reproduction_error}"
        )

    if not df.tv_to_late_240_300.between(0, 1).all():
        raise ValueError("Invalid TV distances")

    if not df.loc[
        df.seconds == 30,
        "online_spectral_drift"
    ].isna().all():
        raise ValueError("Unexpected 30s drift values")

    if not np.isfinite(
        df.loc[
            df.seconds != 30,
            "online_spectral_drift"
        ]
    ).all():
        raise ValueError("Invalid online drift values")

    # Negative control: a different participant
    # from the same diagnostic group.
    reference_by_subject = {
        row.participant_id: np.array([
            getattr(
                row,
                f"late_median_relative_{band}"
            )
            for band in BANDS
        ], dtype=float)
        for row in refs.itertuples(index=False)
    }

    other_map = {}

    for group in sorted(refs.group.unique()):
        ids = sorted(
            refs.loc[
                refs.group == group,
                "participant_id"
            ].tolist()
        )

        if len(ids) < 2:
            raise ValueError(
                "Not enough group-matched participants"
            )

        for index, subject in enumerate(ids):
            other_map[subject] = ids[
                (index + 1) % len(ids)
            ]

    df["tv_to_other_subject_late"] = [
        tv(
            [
                previous.loc[
                    (row.participant_id, int(row.seconds))
                ][f"median_relative_{band}"]
                for band in BANDS
            ],
            reference_by_subject[
                other_map[row.participant_id]
            ]
        )
        for row in df.itertuples(index=False)
    ]

    df["reference_other_participant"] = (
        df.participant_id.map(other_map)
    )

    wide = df.pivot(
        index="participant_id",
        columns="seconds",
        values="tv_to_late_240_300"
    )

    comparisons = {}

    for end in (120, 180):
        delta = (
            wide[end].to_numpy()
            - wide[60].to_numpy()
        )

        comparisons[f"{end}_minus_60"] = {
            "mean_delta_and_bootstrap_95pct_ci":
                mean_interval(
                    delta,
                    seed=20261107 + end
                ),
            "unit": "participant",
            "lower_is_better": True
        }

    s60 = df[df.seconds == 60].copy()

    comparisons[
        "other_subject_minus_same_subject_at_60"
    ] = {
        "mean_delta_and_bootstrap_95pct_ci":
            mean_interval(
                s60.tv_to_other_subject_late
                - s60.tv_to_late_240_300,
                seed=20261267
            ),
        "unit": "participant",
        "positive_means_same_subject_is_closer":
            True
    }

    correlations = {}

    for seconds in (60, 120, 180):
        sub = df[df.seconds == seconds]

        rho = spearmanr(
            sub.online_spectral_drift.to_numpy(),
            sub.tv_to_late_240_300.to_numpy()
        ).statistic

        correlations[str(seconds)] = (
            float(rho) if np.isfinite(rho)
            else None
        )

    report = {
        "study_status":
            "EXPLORATORY within-recording reliability",
        "participants": int(
            refs.participant_id.nunique()
        ),
        "rows": len(df),
        "early_checkpoints_s": list(TIMES),
        "reference_interval_s": [240, 300],
        "reference_is_ground_truth": False,
        "baseline_reproduction_error":
            max_reproduction_error,
        "median_tv_by_checkpoint": {
            str(int(k)): float(v)
            for k, v in df.groupby(
                "seconds"
            ).tv_to_late_240_300.median().items()
        },
        "online_drift_future_error_spearman":
            correlations,
        "paired_descriptive_comparisons":
            comparisons,
        "median_late_halves_tv":
            float(refs.late_halves_tv.median()),
        "early60_median_tv":
            float(
                refs.early_60_tv_to_late.median()
            ),
        "delayed60_median_tv":
            float(
                refs.delayed_60_tv_to_late.median()
            ),
        "limitations": [
            "The reference is a later within-session interval, not an independent cohort.",
            "The later interval is not ground truth.",
            "Both early and late measurements may contain artifacts.",
            "Later prefixes are closer in time to the reference.",
            "The 10-second summaries are available at epoch completion.",
            "All analyses are exploratory on a previously studied cohort."
        ]
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUT,
        index=False,
        float_format="%.12g"
    )

    refs.to_csv(
        OUT_REF,
        index=False,
        float_format="%.12g"
    )

    OUT_SUMMARY.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("===== DISJOINT REFERENCE RESULTS =====")
    print("Participants:", len(refs))
    print("Evaluation rows:", len(df))

    print(
        "Previous feature reproduction error:",
        max_reproduction_error
    )

    print()
    print("Median spectral TV distance:")
    print(
        df.groupby("seconds")
        .tv_to_late_240_300
        .median()
        .round(6)
        .to_string()
    )

    print()
    print("===== PAIRED REPRODUCIBILITY RESULTS =====")

    for name, stats in comparisons.items():
        print(
            name,
            [
                round(x, 6)
                for x in stats[
                    "mean_delta_and_bootstrap_95pct_ci"
                ]
            ]
        )

    print()
    print("===== ONLINE STABILITY PREDICTIVENESS =====")
    print(
        "Spearman correlations:",
        correlations
    )

    print()
    print(
        "Late 30s halves median distance:",
        round(report["median_late_halves_tv"], 6)
    )

    print(
        "Early 60s distance:",
        round(report["early60_median_tv"], 6)
    )

    print(
        "Delayed 60s distance:",
        round(report["delayed60_median_tv"], 6)
    )

    print()
    print(
        "Not an independent-cohort or clinical validation."
    )
    print(
        "P2 DISJOINT REFERENCE ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
