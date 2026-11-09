from pathlib import Path
from collections import Counter
import csv
import inspect
import json
import os
import re
import sys

HOME = Path.home()

WORKSPACE = HOME / "research" / "eegraft-caueeg"
DATA_ROOT = HOME / "research" / "caueeg"

MODEL_ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

sys.path.insert(0, str(MODEL_ROOT))

import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt

from src.data import CHANNELS, read_raw_csv

CONFIG = WORKSPACE / "protocol" / "dataset_layout.json"
PROTOCOL = MODEL_ROOT / "results" / "final_protocol_2026.json"
AUDIT = WORKSPACE / "results" / "08_edf_header_summary.json"

MANIFEST = DATA_ROOT / "manifests" / "07_recording_audit.csv"
OUTPUT = WORKSPACE / "results" / "09_waveform_smoke_summary.json"

ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

ALLOWED_REFERENCES = {
    "", "AVG", "REF", "LE", "A1", "A2",
    "M1", "M2", "CAR"
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_channel(label):
    text = str(label).upper().strip()

    text = re.sub(r"^EEG[\s:._-]*", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "-")

    base, separator, reference = text.partition("-")

    if reference not in ALLOWED_REFERENCES:
        return None

    base = ALIASES.get(base, base)

    mapping = {
        name.upper(): name
        for name in CHANNELS
    }

    return mapping.get(base)


def load_eeg(path):
    raw = mne.io.read_raw_edf(
        str(path),
        preload=False,
        infer_types=False,
        stim_channel=None,
        verbose="ERROR"
    )

    try:
        positions = {}

        for index, name in enumerate(raw.ch_names):
            canonical = normalize_channel(name)

            if canonical is None:
                continue

            if canonical in positions:
                raise ValueError(
                    "Duplicate EEG electrode: " + canonical
                )

            positions[canonical] = index

        missing = [
            channel for channel in CHANNELS
            if channel not in positions
        ]

        if missing:
            raise ValueError(
                "Missing required EEG channels: " + str(missing)
            )

        picks = [
            positions[channel]
            for channel in CHANNELS
        ]

        types = raw.get_channel_types(picks=picks)

        if any(value != "eeg" for value in types):
            raise ValueError(
                "Selected channels are not all EEG in MNE"
            )

        from mne.io.constants import FIFF

        if any(
            raw.info["chs"][index]["unit"] != FIFF.FIFF_UNIT_V
            for index in picks
        ):
            raise ValueError(
                "MNE EEG channel unit is not volts; "
                "scaling needs manual investigation"
            )

        sampling_rate = float(raw.info["sfreq"])

        if abs(sampling_rate - 200.0) > 0.001:
            raise ValueError(
                f"Unexpected sampling rate: {sampling_rate}"
            )

        # MNE supplies calibrated EEG signals in volts.
        eeg_volts = raw.get_data(picks=picks)

        if eeg_volts.shape[0] != 19:
            raise ValueError("Incorrect EEG channel count")

        if not np.isfinite(eeg_volts).all():
            raise ValueError("Nonfinite raw EEG samples")

        # Convert volts to microvolts, matching src.data.
        eeg_uv = eeg_volts.astype(np.float64) * 1e6

        return eeg_uv, sampling_rate, list(raw.ch_names)

    finally:
        raw.close()


def preprocess(eeg_uv, sampling_rate, settings):
    if settings.get("apply_car") is not False:
        raise ValueError(
            "Frozen protocol must explicitly disable CAR"
        )

    seconds = int(settings["seconds"])
    max_windows = int(settings["max_per_subject"])
    target_fs = float(settings["sampling_hz"])

    if target_fs != 200 or seconds != 4 or max_windows != 24:
        raise ValueError(
            "Unexpected original model window settings"
        )

    low_hz = 0.3
    high_hz = float(settings["high_hz"])

    if high_hz != 40.0:
        raise ValueError(
            "Unexpected original high-pass/low-pass settings"
        )

    if settings.get("artifact_limit_uv") is not None:
        raise ValueError(
            "Unexpected artifact exclusion policy"
        )

    # Same fourth-order Butterworth bandpass as
    # the original src.data.read_raw_csv implementation.
    sos = butter(
        4,
        (low_hz, high_hz),
        btype="bandpass",
        fs=sampling_rate,
        output="sos"
    )

    filtered_uv = sosfiltfilt(
        sos,
        eeg_uv,
        axis=-1
    )

    # All CAUEEG recordings are already sampled at 200 Hz.
    # No resampling is required.
    if abs(sampling_rate - target_fs) > 0.001:
        raise ValueError("Resampling policy requires review")

    # Original CBraMod convention: microvolts / 100.
    eeg_scaled = (
        filtered_uv / 100.0
    ).astype(np.float32)

    if not np.isfinite(eeg_scaled).all():
        raise ValueError("Nonfinite preprocessed EEG samples")

    samples_per_window = seconds * int(target_fs)

    complete_windows = (
        eeg_scaled.shape[1] // samples_per_window
    )

    if complete_windows < 1:
        raise ValueError("No complete four-second windows")

    indices = np.unique(
        np.linspace(
            0,
            complete_windows - 1,
            num=min(complete_windows, max_windows),
            dtype=int
        )
    )

    windows = np.stack([
        eeg_scaled[
            :,
            index * samples_per_window:
            (index + 1) * samples_per_window
        ].reshape(19, seconds, int(target_fs))
        for index in indices
    ])

    if windows.ndim != 4:
        raise ValueError("Incorrect tensor dimensions")

    if windows.shape[1:] != (19, 4, 200):
        raise ValueError(
            f"Incorrect model input shape: {windows.shape}"
        )

    if not np.isfinite(windows).all():
        raise ValueError("Nonfinite EEG window values")

    return windows, filtered_uv, indices


def save_without_overwrite(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing report matches")
            return True

        print("[PRESERVED] Existing report is different")
        print("No output overwritten:", path)
        return False

    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)

    os.chmod(path, 0o600)

    print("[SAVED]", path)
    return True


def run():
    print("\n1. Checking previous audit")

    if not all(
        path.is_file()
        for path in (CONFIG, PROTOCOL, AUDIT, MANIFEST)
    ):
        print("[BLOCKED] Previous audit or protocol missing")
        return

    audit = read_json(AUDIT)

    if audit.get("audit_status") != "READY_FOR_WAVEFORM_SMOKE_TEST":
        print("[BLOCKED] EDF audit has unresolved problems")
        return

    config = read_json(CONFIG)
    protocol = read_json(PROTOCOL)

    print("[OK] Previous EDF audit passed")

    print("\n2. Checking frozen preprocessing protocol")

    settings = protocol["windows"]

    signature = inspect.signature(read_raw_csv)

    for parameter, expected in [
        ("low_hz", 0.3),
        ("high_hz", 40.0),
        ("apply_car", False),
    ]:
        actual = signature.parameters[parameter].default

        if actual != expected:
            raise ValueError(
                f"Source preprocessing differs: {parameter}"
            )

    print("Source preprocessing: verified")
    print("Channels:", len(CHANNELS))
    print("Bandpass: 0.3–40 Hz")
    print("Filtering: Butterworth order 4")
    print("Target sampling rate: 200 Hz")
    print("Window length: 4 seconds")
    print("Maximum windows: 24")
    print("Additional CAR: disabled")
    print("Scale: microvolts / 100")

    print("\n3. Choosing training recordings only")

    with MANIFEST.open(
        encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))

    selected = {}

    for record in records:
        if (
            record["split"] == "train"
            and record["class_name"] in (
                "normal", "mci", "dementia"
            )
            and record["edf_exists"] == "1"
        ):
            selected.setdefault(
                record["class_name"], record
            )

    expected_classes = {"normal", "mci", "dementia"}

    if set(selected) != expected_classes:
        raise ValueError(
            "Unable to select one training recording "
            "for each official class"
        )

    print("Selected: 1 Normal + 1 MCI + 1 Dementia")
    print("Validation recordings accessed: 0")
    print("Test recordings accessed: 0")

    edf_dir = Path(config["edf_directory"])
    results = []
    total_windows = 0

    print("\n4. Reading and preprocessing real EEG")

    for label in ("normal", "mci", "dementia"):
        record = selected[label]

        path = edf_dir / (
            record["serial"] + ".edf"
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Selected training EDF is missing"
            )

        print(f"\n[{label.upper()}] Reading waveform")

        eeg_uv, fs, recorded_names = load_eeg(path)

        windows, filtered_uv, selected_indices = preprocess(
            eeg_uv, fs, settings
        )

        duration = eeg_uv.shape[1] / fs

        raw_p99 = float(
            np.percentile(np.abs(eeg_uv), 99)
        )

        filtered_p99 = float(
            np.percentile(np.abs(filtered_uv), 99)
        )

        scaled_p99 = float(
            np.percentile(np.abs(windows), 99)
        )

        print("Original shape:", tuple(eeg_uv.shape))
        print("Duration:", round(duration, 2), "seconds")
        print("Sampling rate:", fs, "Hz")
        print("Raw |EEG| 99th percentile:", round(raw_p99, 3), "uV")
        print("Filtered |EEG| 99th percentile:", round(filtered_p99, 3), "uV")
        print("Model tensor shape:", tuple(windows.shape))
        print("Scaled |EEG| 99th percentile:", round(scaled_p99, 5))
        print("All tensor values finite:", bool(np.isfinite(windows).all()))

        if raw_p99 > 1000:
            print(
                "[WARNING] Unusually large EEG amplitude; "
                "inspect recording before full inference"
            )

        total_windows += len(windows)

        results.append({
            "official_class": label,
            "sampling_rate_hz": fs,
            "duration_seconds": round(duration, 3),
            "raw_shape": list(eeg_uv.shape),
            "model_tensor_shape": list(windows.shape),
            "raw_abs_p99_uv": round(raw_p99, 4),
            "filtered_abs_p99_uv": round(filtered_p99, 4),
            "scaled_abs_p99": round(scaled_p99, 6),
            "selected_window_count": len(selected_indices),
            "all_finite": True
        })

    print("\n5. Final smoke-test verification")

    expected_shape = [24, 19, 4, 200]

    ready = (
        len(results) == 3
        and all(
            result["model_tensor_shape"] == expected_shape
            for result in results
        )
        and all(result["all_finite"] for result in results)
    )

    report = {
        "test_name": "CAUEEG training-only waveform smoke test",
        "model_protocol": "frozen_original_eegraft",
        "recordings_read": 3,
        "validation_waveforms_read": 0,
        "test_waveforms_read": 0,
        "total_windows": total_windows,
        "window_shape": [19, 4, 200],
        "source_preprocessing_match_checked": True,
        "recording_results": results,
        "smoke_status": (
            "READY_FOR_FROZEN_MODEL_INFERENCE_SETUP"
            if ready else "REVIEW_REQUIRED"
        ),
        "note": (
            "Dataset uses AVG-referenced EEG. "
            "Source/target reference differences remain "
            "a domain-shift limitation."
        )
    }

    content = json.dumps(
        report,
        indent=2
    ) + "\n"

    saved = save_without_overwrite(OUTPUT, content)

    if saved:
        print("\nTotal real EEG windows:", total_windows)
        print("Input tensor shape per window: (19, 4, 200)")
        print("WAVEFORM_STATUS:", report["smoke_status"])
    else:
        print("WAVEFORM_STATUS: OUTPUT_CONFLICT")

    print("Original model weights were not modified.")
    print("No model predictions were generated.")


try:
    run()

except Exception as exc:
    print("[ERROR]", type(exc).__name__, str(exc))
    print("WAVEFORM_STATUS: FAILED")

finally:
    print("\n========== CODE 09 FINISHED ==========")
