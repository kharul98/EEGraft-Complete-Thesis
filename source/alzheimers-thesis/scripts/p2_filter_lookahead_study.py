from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfilt, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p2_filter_lookahead_results.csv"
FS = 500
DURATIONS = (30, 60, 120, 180, 300)
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))

sos = butter(
    4, (1, 40), btype="bandpass", fs=FS, output="sos"
)

def spectrum(x):
    f, p = welch(
        x, fs=FS, nperseg=2000,
        noverlap=1000, axis=-1
    )
    reference = (f >= 1) & (f <= 40)
    total = np.trapezoid(
        p[:, reference], f[reference], axis=-1
    )
    values = []
    for low, high in BANDS:
        mask = (f >= low) & (f <= high)
        power = np.trapezoid(
            p[:, mask], f[mask], axis=-1
        )
        values.append(
            np.median(power / np.maximum(total, 1e-20))
        )
    z = np.asarray(values, dtype=float)
    return z / max(z.sum(), 1e-20)

inventory = pd.read_csv(
    ROOT / "results/p1_raw_recording_inventory.csv"
)
assert len(inventory) == 88
rows = []

print("===== P2 FILTER LOOKAHEAD STUDY =====", flush=True)

for i, item in enumerate(inventory.itertuples(index=False), 1):
    raw = mne.io.read_raw_eeglab(
        str(ROOT / item.path),
        preload=False,
        verbose="ERROR"
    )

    assert float(raw.info["sfreq"]) == FS

    x = raw.get_data(
        picks="eeg", start=0, stop=300 * FS
    )
    assert x.shape == (19, 300 * FS)
    assert np.isfinite(x).all()

    causal_full = sosfilt(sos, x, axis=-1)
    zero_full = sosfiltfilt(sos, x, axis=-1)

    for seconds in DURATIONS:
        n = seconds * FS
        prefix = x[:, :n]

        causal_short = sosfilt(sos, prefix, axis=-1)
        causal_error = float(np.max(
            np.abs(causal_short - causal_full[:, :n])
        ))
        assert causal_error <= 1e-12

        zero_short = sosfiltfilt(sos, prefix, axis=-1)
        difference = zero_short - zero_full[:, :n]

        boundary_rms = float(np.sqrt(
            np.mean((difference[:, -FS:] * 1e6) ** 2)
        ))
        full_rms = float(np.sqrt(
            np.mean((difference * 1e6) ** 2)
        ))
        spectral_difference = float(
            0.5 * np.abs(
                spectrum(zero_short)
                - spectrum(zero_full[:, :n])
            ).sum()
        )

        rows.append({
            "participant_id": item.participant_id,
            "group": item.group,
            "seconds": seconds,
            "causal_max_difference_v": causal_error,
            "future_boundary_rms_uv": boundary_rms,
            "future_full_prefix_rms_uv": full_rms,
            "future_spectral_tv": spectral_difference
        })

    if i % 22 == 0:
        print(f"Processed {i}/88", flush=True)

df = pd.DataFrame(rows)
assert len(df) == 440
assert df.participant_id.nunique() == 88
assert df.causal_max_difference_v.le(1e-12).all()
assert np.isfinite(
    df.select_dtypes(include="number").to_numpy()
).all()

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, float_format="%.12g")

print("===== LOOKAHEAD RESULTS =====")
print(df.groupby("seconds")[[
    "future_boundary_rms_uv",
    "future_full_prefix_rms_uv",
    "future_spectral_tv"
]].median().round(8).to_string())
print("Maximum causal error:", df.causal_max_difference_v.max())
print("P2 LOOKAHEAD STUDY COMPLETE")
