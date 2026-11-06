from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

path = Path(
    "data/eeg/on004504_v1.0.0/"
    "sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
)

raw = mne.io.read_raw_eeglab(
    str(path),
    preload=False,
    verbose="ERROR",
)

sfreq = raw.info["sfreq"]

assert sfreq == 500.0

cut = int(60 * sfreq)
stop = int(120 * sfreq)

data = raw.get_data(
    start=0,
    stop=stop,
    picks="eeg",
)

assert data.shape == (19, stop)
assert np.isfinite(data).all()

sos = butter(
    4,
    [1.0, 40.0],
    btype="bandpass",
    fs=sfreq,
    output="sos",
)

prefix = data[:, :cut]

causal_prefix = sosfilt(
    sos,
    prefix,
    axis=-1,
)

causal_long = sosfilt(
    sos,
    data,
    axis=-1,
)[:, :cut]

zero_prefix = sosfiltfilt(
    sos,
    prefix,
    axis=-1,
)

zero_long = sosfiltfilt(
    sos,
    data,
    axis=-1,
)[:, :cut]

causal_error = np.max(
    np.abs(causal_prefix - causal_long)
)

edge = int(sfreq)

zero_edge_rms_uv = (
    np.sqrt(
        np.mean(
            (
                zero_prefix[:, -edge:]
                - zero_long[:, -edge:]
            ) ** 2
        )
    )
    * 1e6
)

zero_edge_max_uv = (
    np.max(
        np.abs(
            zero_prefix[:, -edge:]
            - zero_long[:, -edge:]
        )
    )
    * 1e6
)

print("===== PREFIX PREPROCESSING GATE =====")
print("Input recording:", path.name)
print("Channels:", data.shape[0])
print("Sampling rate:", sfreq)
print("Prefix duration:", 60)
print("Comparison duration:", 120)

print()
print("Causal maximum difference (V):", causal_error)

print(
    "Causal prefix invariant:",
    np.allclose(
        causal_prefix,
        causal_long,
        rtol=1e-10,
        atol=1e-13,
    ),
)

print(
    "Zero-phase boundary RMS difference (uV):",
    round(float(zero_edge_rms_uv), 8),
)

print(
    "Zero-phase boundary maximum difference (uV):",
    round(float(zero_edge_max_uv), 8),
)

assert np.allclose(
    causal_prefix,
    causal_long,
    rtol=1e-10,
    atol=1e-13,
)

print()
print("CAUSAL PREFIX INVARIANCE: PASS")
print("PREFIX PREPROCESSING GATE COMPLETE")
