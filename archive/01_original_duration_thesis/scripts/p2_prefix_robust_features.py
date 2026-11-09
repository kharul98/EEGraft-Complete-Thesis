#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import mne

from scipy.signal import butter, sosfilt, welch

ROOT = Path(__file__).resolve().parents[1]

INVENTORY = (
    ROOT / "results/p1_raw_recording_inventory.csv"
)

OUT = (
    ROOT / "results/p2_prefix_robust_features.csv"
)

DURATIONS = (30, 60, 120, 180, 300)

FS = 500
EPOCH_SECONDS = 10
EPOCH_SAMPLES = FS * EPOCH_SECONDS

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

SOS = butter(
    4,
    [1, 40],
    btype="bandpass",
    fs=FS,
    output="sos"
)

inventory = pd.read_csv(INVENTORY)

assert len(inventory) == 88
assert inventory.participant_id.nunique() == 88
assert set(inventory.group) == {"A", "C", "F"}

rows = []
errors = []

print(
    "===== P2 CAUSAL ROBUST FEATURE DEVELOPMENT =====",
    flush=True
)

for index, item in enumerate(
    inventory.itertuples(index=False),
    1
):
    try:
        raw = mne.io.read_raw_eeglab(
            str(ROOT / item.path),
            preload=False,
            verbose="ERROR"
        )

        assert float(raw.info["sfreq"]) == FS
        assert len(raw.ch_names) == 19
        assert raw.n_times >= 300 * FS

        state = np.zeros(
            (len(SOS), 19, 2),
            dtype=np.float64
        )

        epoch_bands = []
        epoch_extreme = []
        epoch_rms = []
        epoch_ptp = []

        for epoch in range(
            300 // EPOCH_SECONDS
        ):
            start = epoch * EPOCH_SAMPLES

            x = raw.get_data(
                picks="eeg",
                start=start,
                stop=start + EPOCH_SAMPLES
            )

            assert x.shape == (
                19,
                EPOCH_SAMPLES
            )

            assert np.isfinite(x).all()

            # Stateful causal filtering.
            # Future samples are never used.
            y, state = sosfilt(
                SOS,
                x,
                axis=-1,
                zi=state
            )

            frequencies, psd = welch(
                y,
                fs=FS,
                nperseg=4 * FS,
                noverlap=2 * FS,
                axis=-1
            )

            total_mask = (
                (frequencies >= 1)
                & (frequencies <= 40)
            )

            total = np.trapezoid(
                psd[:, total_mask],
                frequencies[total_mask],
                axis=-1
            )

            assert np.isfinite(total).all()
            assert (total > 0).all()

            powers = []

            for low, high in BANDS.values():
                mask = (
                    (frequencies >= low)
                    & (frequencies <= high)
                )

                numerator = np.trapezoid(
                    psd[:, mask],
                    frequencies[mask],
                    axis=-1
                )

                powers.append(
                    numerator / total
                )

            epoch_bands.append(
                np.stack(powers, axis=-1)
            )

            epoch_extreme.append(
                float(
                    np.mean(
                        np.abs(x) > 200e-6
                    )
                )
            )

            epoch_rms.append(
                float(
                    np.median(
                        np.sqrt(
                            np.mean(
                                (x * 1e6) ** 2,
                                axis=-1
                            )
                        )
                    )
                )
            )

            epoch_ptp.append(
                float(
                    np.median(
                        np.ptp(
                            x * 1e6,
                            axis=-1
                        )
                    )
                )
            )

        arr = np.stack(epoch_bands)

        previous = None

        for duration in DURATIONS:
            n = duration // EPOCH_SECONDS

            prefix = arr[:n]

            median = np.median(
                prefix,
                axis=(0, 1)
            )

            mean = np.mean(
                prefix,
                axis=(0, 1)
            )

            within_epoch = np.median(
                prefix,
                axis=1
            )

            result = {
                "participant_id":
                    item.participant_id,

                "group":
                    item.group,

                "seconds":
                    duration,

                "raw_fraction_above_200uv":
                    float(
                        np.mean(
                            epoch_extreme[:n]
                        )
                    ),

                "median_epoch_rms_uv":
                    float(
                        np.median(
                            epoch_rms[:n]
                        )
                    ),

                "median_epoch_ptp_uv":
                    float(
                        np.median(
                            epoch_ptp[:n]
                        )
                    ),

                "theta_epoch_iqr":
                    float(
                        np.diff(
                            np.percentile(
                                within_epoch[:, 1],
                                [25, 75]
                            )
                        )[0]
                    ),

                "alpha_epoch_iqr":
                    float(
                        np.diff(
                            np.percentile(
                                within_epoch[:, 2],
                                [25, 75]
                            )
                        )[0]
                    ),
            }

            for j, band in enumerate(BANDS):
                result[
                    f"median_relative_{band}"
                ] = float(median[j])

                result[
                    f"mean_relative_{band}"
                ] = float(mean[j])

            if previous is None:
                result[
                    "spectral_drift_from_prior_prefix"
                ] = np.nan

            else:
                a = median / max(
                    median.sum(),
                    1e-20
                )

                b = previous / max(
                    previous.sum(),
                    1e-20
                )

                result[
                    "spectral_drift_from_prior_prefix"
                ] = float(
                    0.5 * np.abs(a - b).sum()
                )

            previous = median

            rows.append(result)

    except Exception as exc:
        errors.append(
            (
                str(item.participant_id),
                str(exc)
            )
        )

    if index % 22 == 0:
        print(
            f"Processed {index}/88 participants",
            flush=True
        )

if errors:
    for subject, error in errors[:10]:
        print("ERROR", subject, error)

    raise SystemExit(
        f"{len(errors)} recordings require review."
    )

df = pd.DataFrame(rows)

df = df.sort_values(
    ["participant_id", "seconds"]
).reset_index(drop=True)

assert len(df) == 440

assert (
    df.groupby("participant_id")["seconds"]
    .nunique()
    .eq(5)
    .all()
)

excluded = {
    "participant_id",
    "group",
    "spectral_drift_from_prior_prefix"
}

required = [
    c for c in df.columns
    if c not in excluded
]

assert np.isfinite(
    df[required].to_numpy(dtype=float)
).all()

assert df.loc[
    df.seconds == 30,
    "spectral_drift_from_prior_prefix"
].isna().all()

assert np.isfinite(
    df.loc[
        df.seconds != 30,
        "spectral_drift_from_prior_prefix"
    ]
).all()

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT,
    index=False,
    float_format="%.12g"
)

print()
print("===== P2 PREFIX FEATURE SUMMARY =====")
print("Participants:", df.participant_id.nunique())
print("Prefix rows:", len(df))
print(
    "Durations:",
    sorted(df.seconds.unique().tolist())
)

print()
print("Median fraction of samples above 200 uV:")
print(
    df[df.seconds == 300]
    .groupby("group")[
        "raw_fraction_above_200uv"
    ]
    .median()
    .round(4)
    .to_string()
)

print()
print("Spectral drift median by duration:")
print(
    df.groupby("seconds")[
        "spectral_drift_from_prior_prefix"
    ]
    .median()
    .round(4)
    .to_string()
)

print()
print("LIMITATIONS:")
print("- This is NOT validated artifact removal.")
print("- Amplitude flags are descriptive.")
print("- Prefixes are overlapping observations.")
print("- No participants were excluded.")
print()
print("P2 PREFIX-VALID ROBUST FEATURES COMPLETE")
