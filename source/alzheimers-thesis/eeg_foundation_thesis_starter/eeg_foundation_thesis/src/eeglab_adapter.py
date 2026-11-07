"""EEGLAB adapter: 500 Hz EEG -> 200 Hz CBraMod input."""

from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.data import CHANNELS, _norm_name


def read_raw_set(row, *, low_hz=0.3, high_hz=40.0, apply_car=False):
    path = Path(str(row["path"]))

    if path.suffix.lower() != ".set":
        raise ValueError(f"Not an EEGLAB file: {path}")

    if str(row["unit"]).strip().lower() != "v":
        raise ValueError(
            "EEGLAB data read through MNE must use manifest unit V."
        )

    raw = mne.io.read_raw_eeglab(
        str(path),
        preload=False,
        verbose="ERROR"
    )

    try:
        fs = float(raw.info["sfreq"])

        if not np.isclose(fs, float(row["fs_hz"]), atol=0.01):
            raise ValueError(
                f"Sampling-rate mismatch for {path}: {fs}"
            )

        channel_idx = {}

        for i, name in enumerate(raw.ch_names):
            key = _norm_name(name)

            if key in channel_idx:
                raise ValueError(f"Duplicate channel: {key}")

            channel_idx[key] = i

        missing = [
            name for name in CHANNELS
            if _norm_name(name) not in channel_idx
        ]

        if missing:
            raise ValueError(
                f"Missing required channels: {missing}"
            )

        indices = [
            channel_idx[_norm_name(name)]
            for name in CHANNELS
        ]

        # MNE returns EEG signals in volts.
        volts = raw.get_data(
            picks=indices
        ).astype(np.float64, copy=False)

    finally:
        raw.close()

    if volts.shape[1] < 4 * fs:
        raise ValueError("Recording shorter than four seconds")

    if not np.isfinite(volts).all():
        raise ValueError("Recording contains NaN or infinite values")

    # Convert volts to microvolts.
    microvolts = volts * 1e6

    # Consistent bandpass preprocessing.
    hi = min(float(high_hz), fs * 0.45)

    sos = butter(
        4,
        [float(low_hz), hi],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        microvolts,
        axis=-1
    )

    if apply_car:
        filtered -= filtered.mean(
            axis=0,
            keepdims=True
        )

    # Resample 500 Hz -> 200 Hz.
    ratio = Fraction(200 / fs).limit_denominator(10000)

    if abs(ratio.numerator / ratio.denominator - 200 / fs) > 1e-6:
        raise ValueError("Unsupported sampling-rate ratio")

    signal = resample_poly(
        filtered,
        ratio.numerator,
        ratio.denominator,
        axis=-1
    )

    # Fixed amplitude scaling, not fitted using test data.
    return np.ascontiguousarray(
        signal / 100.0,
        dtype=np.float32
    )
