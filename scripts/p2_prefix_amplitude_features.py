#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import mne

from scipy.signal import butter, sosfilt, welch


ROOT = Path(__file__).resolve().parents[1]

INVENTORY = ROOT / "results/p1_raw_recording_inventory.csv"

BASELINE = ROOT / "results/p2_prefix_robust_features.csv"

OUTPUT = ROOT / "results/p2_amplitude_limited_features.csv"

FS = 500
EPOCH_SECONDS = 10
EPOCH_SAMPLES = EPOCH_SECONDS * FS

DURATIONS = (30, 60, 120, 180, 300)

CLIP_MAD_MULTIPLIER = 8.0

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

SOS = butter(
    4,
    [1, 40],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def spectral_features(filtered):
    frequencies, psd = welch(
        filtered,
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

    bands = []

    for low, high in BANDS.values():
        mask = (
            (frequencies >= low)
            & (frequencies <= high)
        )

        power = np.trapezoid(
            psd[:, mask],
            frequencies[mask],
            axis=-1
        )

        bands.append(
            power / np.maximum(total, 1e-20)
        )

    return np.stack(bands, axis=-1)


inventory = pd.read_csv(INVENTORY)

assert len(inventory) == 88
assert inventory.participant_id.nunique() == 88

rows = []

print(
    "===== P2 PREFIX AMPLITUDE-LIMITING STUDY =====",
    flush=True
)

for index, item in enumerate(
    inventory.itertuples(index=False), 1
):
    raw = mne.io.read_raw_eeglab(
        str(ROOT / item.path),
        preload=False,
        verbose="ERROR"
    )

    assert raw.info["sfreq"] == FS

    raw_state = np.zeros(
        (len(SOS), 19, 2),
        dtype=np.float64
    )

    limited_state = np.zeros_like(raw_state)

    raw_epoch_spectra = []
    limited_epoch_spectra = []
    clipped_fractions = []

    for epoch in range(30):
        start = epoch * EPOCH_SAMPLES

        signal = raw.get_data(
            picks="eeg",
            start=start,
            stop=start + EPOCH_SAMPLES
        )

        assert signal.shape == (19, EPOCH_SAMPLES)
        assert np.isfinite(signal).all()

        # The current complete 10-second epoch is
        # the latest available observation.
        center = np.median(
            signal,
            axis=1,
            keepdims=True
        )

        mad = np.median(
            np.abs(signal - center),
            axis=1,
            keepdims=True
        )

        robust_sigma = np.maximum(
            1.4826 * mad,
            1e-6
        )

        limit = (
            CLIP_MAD_MULTIPLIER
            * robust_sigma
        )

        limited = np.clip(
            signal,
            center - limit,
            center + limit
        )

        clipped_fractions.append(
            float(np.mean(signal != limited))
        )

        # Separate causal filter states prevent
        # cross-contamination between pipelines.
        raw_filtered, raw_state = sosfilt(
            SOS,
            signal,
            axis=-1,
            zi=raw_state
        )

        limited_filtered, limited_state = sosfilt(
            SOS,
            limited,
            axis=-1,
            zi=limited_state
        )

        raw_epoch_spectra.append(
            spectral_features(raw_filtered)
        )

        limited_epoch_spectra.append(
            spectral_features(limited_filtered)
        )

    raw_spectra = np.stack(raw_epoch_spectra)
    limited_spectra = np.stack(limited_epoch_spectra)

    for seconds in DURATIONS:
        count = seconds // EPOCH_SECONDS

        raw_median = np.median(
            raw_spectra[:count],
            axis=(0, 1)
        )

        limited_median = np.median(
            limited_spectra[:count],
            axis=(0, 1)
        )

        row = {
            "participant_id": item.participant_id,
            "group": item.group,
            "seconds": seconds,
            "fraction_amplitude_limited":
                float(
                    np.mean(
                        clipped_fractions[:count]
                    )
                )
        }

        for band_index, name in enumerate(BANDS):
            row[
                f"raw_median_{name}"
            ] = float(raw_median[band_index])

            row[
                f"limited_median_{name}"
            ] = float(limited_median[band_index])

        rows.append(row)

    if index % 22 == 0:
        print(
            f"Processed {index}/88 participants",
            flush=True
        )


df = pd.DataFrame(rows).sort_values(
    ["participant_id", "seconds"]
)

assert len(df) == 440
assert df.participant_id.nunique() == 88

assert np.isfinite(
    df.select_dtypes(include="number").to_numpy()
).all()

# Reproduce our previously implemented
# unmodified epoch-median baseline.
reference = pd.read_csv(BASELINE)

joined = df.merge(
    reference,
    on=["participant_id", "seconds", "group"],
    validate="one_to_one"
)

assert len(joined) == 440

maximum_baseline_error = max(
    float(
        np.max(
            np.abs(
                joined[f"raw_median_{band}"]
                -
                joined[f"median_relative_{band}"]
            )
        )
    )
    for band in BANDS
)

assert maximum_baseline_error < 1e-8, (
    "Unmodified baseline did not reproduce"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False,
    float_format="%.12g"
)

print()
print("===== AMPLITUDE-LIMITING FEATURE RESULTS =====")

print(
    "Baseline reproduction maximum error:",
    maximum_baseline_error
)

print()
print("Median amplitude-limited fraction by duration:")

print(
    df.groupby("seconds")[
        "fraction_amplitude_limited"
    ]
    .median()
    .round(6)
    .to_string()
)

for name in ("theta", "alpha"):
    change = np.abs(
        df[f"limited_median_{name}"]
        -
        df[f"raw_median_{name}"]
    )

    print(
        f"Median absolute {name} feature change:",
        round(float(np.median(change)), 6)
    )

print()
print("LIMITATIONS:")
print("- Fixed exploratory 8-MAD threshold.")
print("- Not validated artifact detection or removal.")
print("- May alter genuine neurophysiological signal.")
print("- Decisions are defined at completed 10-second epochs.")
print("- No model outcomes were used to choose the threshold.")

print()
print("P2 AMPLITUDE-LIMITING FEATURES COMPLETE")
