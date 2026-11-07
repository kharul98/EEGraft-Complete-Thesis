from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfilt, welch

ROOT = Path(__file__).resolve().parents[1]

INVENTORY = ROOT / "results/p1_raw_recording_inventory.csv"
OUT = ROOT / "results/p1_incremental_eeg_features.csv"

DURATIONS = (30, 60, 120, 180, 300)
FS = 500
CHUNK = 10 * FS

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

sos = butter(
    4,
    (1, 40),
    btype="bandpass",
    fs=FS,
    output="sos"
)

inventory = pd.read_csv(INVENTORY)
assert len(inventory) == 88
assert inventory.participant_id.nunique() == 88

rows = []
errors = []

print("===== INCREMENTAL EEG PIPELINE =====", flush=True)

for index, item in enumerate(
    inventory.itertuples(index=False), 1
):
    try:
        raw = mne.io.read_raw_eeglab(
            str(ROOT / item.path),
            preload=False,
            verbose="ERROR"
        )

        assert raw.info["sfreq"] == FS

        x = raw.get_data(
            picks="eeg",
            start=0,
            stop=300 * FS
        )

        assert x.shape == (19, 300 * FS)
        assert np.isfinite(x).all()

        # Full-length causal filtering is used ONLY to
        # verify the independent incremental implementation.
        full = sosfilt(sos, x, axis=-1)

        # Each 10-second chunk carries its causal filter
        # state into the next chunk.
        state = np.zeros(
            (len(sos), x.shape[0], 2),
            dtype=np.float64
        )

        filtered_chunks = []

        for start in range(0, x.shape[1], CHUNK):
            stop = min(start + CHUNK, x.shape[1])

            output, state = sosfilt(
                sos,
                x[:, start:stop],
                axis=-1,
                zi=state
            )

            filtered_chunks.append(output)

        streamed = np.concatenate(
            filtered_chunks,
            axis=-1
        )

        max_error = float(
            np.max(np.abs(streamed - full))
        )

        if max_error > 1e-12:
            raise RuntimeError(
                f"Incremental mismatch: {max_error}"
            )

        for seconds in DURATIONS:
            stop = seconds * FS

            prefix = streamed[:, :stop]

            frequencies, psd = welch(
                prefix,
                fs=FS,
                nperseg=4 * FS,
                noverlap=2 * FS,
                axis=-1
            )

            full_band = (
                (frequencies >= 1)
                & (frequencies <= 40)
            )

            total_power = np.trapezoid(
                psd[:, full_band],
                frequencies[full_band],
                axis=-1
            )

            result = {
                "participant_id": item.participant_id,
                "group": item.group,
                "seconds": seconds,
                "prefix_max_error_v": max_error,
                "raw_fraction_above_200uv":
                    float(
                        np.mean(
                            np.abs(x[:, :stop]) > 200e-6
                        )
                    )
            }

            for name, (low, high) in BANDS.items():
                band = (
                    (frequencies >= low)
                    & (frequencies <= high)
                )

                power = np.trapezoid(
                    psd[:, band],
                    frequencies[band],
                    axis=-1
                )

                relative = power / np.maximum(
                    total_power,
                    1e-20
                )

                result[
                    f"median_relative_{name}"
                ] = float(np.median(relative))

            rows.append(result)

    except Exception as exc:
        errors.append(
            (str(item.participant_id), str(exc))
        )

    if index % 22 == 0:
        print(
            f"Processed {index}/88 participants",
            flush=True
        )

if errors:
    print("FAILED SUBJECTS:", errors[:10])
    raise SystemExit("INCREMENTAL PIPELINE REQUIRES REVIEW")

df = pd.DataFrame(rows).sort_values(
    ["participant_id", "seconds"]
)

assert len(df) == 88 * len(DURATIONS)
assert df.participant_id.nunique() == 88
assert np.isfinite(
    df.select_dtypes(include="number").to_numpy()
).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print()
print("===== INCREMENTAL EEG RESULTS =====")
print("Participants:", df.participant_id.nunique())
print("Feature rows:", len(df))
print("Durations:", sorted(df.seconds.unique()))
print(
    "Maximum streaming difference (V):",
    df.prefix_max_error_v.max()
)

print()
print("Median raw amplitude flag fraction by group:")
print(
    df[df.seconds == 300]
    .groupby("group")["raw_fraction_above_200uv"]
    .median()
    .round(4)
    .to_string()
)

print()
print("Median spectral features by duration:")
print(
    df.groupby("seconds")[
        ["median_relative_theta",
         "median_relative_alpha"]
    ].median().round(4).to_string()
)

print()
print("Important limitations:")
print("- Causal filter only; artifact handling is not complete.")
print("- Raw amplitude flags are not artifact diagnoses.")
print("- Nested 30-300s prefixes are not independent.")
print("- This is NOT a classification or significance test.")
print("- No participants have been excluded.")

print()
print("ALL 88 INCREMENTAL PREFIX TESTS PASSED")
print("INCREMENTAL EEG PIPELINE COMPLETE")
