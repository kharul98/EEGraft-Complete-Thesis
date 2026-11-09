from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import re
import sys
import uuid

HOME = Path.home()

ROOT = (
    HOME / "alzheimers-thesis"
    / "eeg_foundation_thesis_starter"
    / "eeg_foundation_thesis"
)

WORK = HOME / "research" / "eegraft-caueeg"
DATA = HOME / "research" / "caueeg"

sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import mne
import joblib

from scipy.signal import butter, sosfiltfilt
from mne.io.constants import FIFF

from src.data import CHANNELS
from src.model import FoundationClassifier
from scripts.run_classical import features

PROTOCOL_PATH = ROOT / "results/final_protocol_2026.json"

PRETRAINED = ROOT / "checkpoints/cbramod_official.pth"

DSRF_PATH = (
    ROOT / "results/final_development_models_2026"
    / "dsrf/final.pt"
)

XGB_PATH = (
    ROOT / "results/final_development_models_2026"
    / "xgboost/final.joblib"
)

DEV_MANIFEST = ROOT / "manifests/final_dev52_only.csv"

SNAPSHOT = WORK / "protocol/original_model_snapshot.json"
LAYOUT = WORK / "protocol/dataset_layout.json"

AUDIT_07 = WORK / "results/07_metadata_audit_summary.json"
AUDIT_08 = WORK / "results/08_edf_header_summary.json"
AUDIT_09 = WORK / "results/09_waveform_smoke_summary.json"
AUDIT_10 = WORK / "results/10_model_smoke_summary.json"

MANIFEST = DATA / "manifests/07_recording_audit.csv"

TASK_LOCK = WORK / "protocol/caueeg_transfer_task_v1.json"

PRIVATE_ROOT = DATA / "derived/caueeg_transfer_v1"
SUMMARY_PATH = WORK / "results/11_train_val_inference_summary.json"

ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6"
}

REFERENCES = {
    "", "AVG", "REF", "LE", "A1", "A2",
    "M1", "M2", "CAR"
}

EXPECTED_COUNTS = {
    "train": {
        "normal": 367,
        "mci": 334,
        "dementia": 249
    },
    "validation": {
        "normal": 34,
        "mci": 28,
        "dementia": 20
    }
}

VERSION = "caueeg_transfer_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def safe_json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        allow_nan=False
    ) + "\n"

    if path.exists():
        current = path.read_text(encoding="utf-8")

        if current == content:
            return "EXISTS_AND_MATCHES"

        raise RuntimeError(
            f"Different existing result preserved: {path}"
        )

    temporary = path.parent / (
        "." + path.name + "." + uuid.uuid4().hex + ".tmp"
    )

    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600
        )

        with os.fdopen(
            descriptor, "w", encoding="utf-8"
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    f"Concurrent output conflict: {path}"
                )

    finally:
        temporary.unlink(missing_ok=True)

    return "SAVED"


def verify_preflight():
    print("\n1. VERIFYING ALL PREVIOUS AUDITS", flush=True)

    required = [
        PROTOCOL_PATH,
        PRETRAINED,
        DSRF_PATH,
        XGB_PATH,
        DEV_MANIFEST,
        SNAPSHOT,
        LAYOUT,
        AUDIT_07,
        AUDIT_08,
        AUDIT_09,
        AUDIT_10,
        MANIFEST
    ]

    for path in required:
        require(
            path.is_file(),
            f"Missing required file: {path}"
        )

    require(
        read_json(AUDIT_07)["audit_status"] == "READY_FOR_EDF_QC",
        "Code 07 audit did not pass"
    )

    require(
        read_json(AUDIT_08)["audit_status"]
        == "READY_FOR_WAVEFORM_SMOKE_TEST",
        "Code 08 audit did not pass"
    )

    require(
        read_json(AUDIT_09)["smoke_status"]
        == "READY_FOR_FROZEN_MODEL_INFERENCE_SETUP",
        "Code 09 did not pass"
    )

    require(
        read_json(AUDIT_10)["status"]
        == "FROZEN_MODELS_VERIFIED",
        "Code 10 did not pass"
    )

    snapshot = read_json(SNAPSHOT)

    require(
        Path(snapshot["thesis_root"]).resolve()
        == ROOT.resolve(),
        "Original model path mismatch"
    )

    for relative, metadata in snapshot["files"].items():
        path = ROOT / relative

        require(path.is_file(), f"Missing: {relative}")

        require(
            sha256(path) == metadata["sha256"],
            f"Original thesis file changed: {relative}"
        )

    protocol = read_json(PROTOCOL_PATH)

    require(
        protocol["classes"] == ["AD", "CN"],
        "Original classifier class order changed"
    )

    require(
        protocol["primary_architecture"] == "fusion",
        "Original architecture changed"
    )

    require(
        protocol["development_subjects"] == 52
        and protocol["heldout_subjects"] == 13,
        "Original subject counts changed"
    )

    require(
        protocol["windows"] == {
            "seconds": 4,
            "max_per_subject": 24,
            "sampling_hz": 200,
            "apply_car": False,
            "high_hz": 40.0,
            "artifact_limit_uv": None
        },
        "Original preprocessing specification changed"
    )

    require(
        sha256(PRETRAINED)
        == protocol["pretrained_weights_sha256"],
        "Pretrained encoder checksum mismatch"
    )

    require(
        sha256(DEV_MANIFEST)
        == protocol["final_development_manifest_sha256"],
        "Development manifest checksum mismatch"
    )

    require(
        len(CHANNELS) == 19,
        "Unexpected original EEG channel configuration"
    )

    protocol_hash = sha256(PROTOCOL_PATH)

    print("[OK] Original models and preprocessing locked")
    print("[OK] All prior audit gates passed")

    return protocol, protocol_hash


def lock_external_task(protocol_hash):
    print("\n2. LOCKING EXTERNAL ANALYSIS TASK", flush=True)

    task = {
        "version": VERSION,
        "dataset": "CAUEEG",
        "benchmark": "dementia-no-overlap",
        "model_origin": "Original 52-subject AD/CN development models",
        "source_model_classes": ["AD", "CN"],
        "source_positive_class": "AD",
        "primary_target_comparison": {
            "negative": "normal",
            "positive": "dementia",
            "excluded_from_binary_metrics": "mci",
            "level": "recording",
            "decision_threshold": 0.5,
            "threshold_selected_from_external_test": False,
            "primary_metric": "balanced_accuracy"
        },
        "secondary_analysis": [
            "normal_vs_dementia_AUROC",
            "MCI_source_AD_score_distribution",
            "window_probability_stability",
            "AD_symptom_tagged_subgroup_exploratory"
        ],
        "official_splits": [
            "train",
            "validation",
            "test"
        ],
        "patient_identity_independently_verified": False,
        "patient_level_performance_claims_allowed": False,
        "dementia_equals_confirmed_AD": False,
        "source_probability_calibration_validated_on_CAUEEG": False,
        "original_protocol_sha256": protocol_hash,
        "locked_external_test_accessed_by_this_code": False,
        "interpretation": (
            "Exploratory recording-level transfer of source-trained "
            "AD versus CN models to CAUEEG's broader "
            "Normal/MCI/Dementia categories."
        )
    }

    result = safe_json_write(TASK_LOCK, task)

    print("[OK] External protocol:", result)
    print("Primary: Dementia versus Normal")
    print("MCI: separate descriptive analysis")
    print("Unit of evaluation: recording, not verified patient")
    print("Decision threshold: fixed at 0.5")
    print("External test remains protected")

    return sha256(TASK_LOCK)


def read_rows():
    print("\n3. VERIFYING TRAIN/VALIDATION RECORDINGS", flush=True)

    with MANIFEST.open(
        encoding="utf-8",
        newline=""
    ) as handle:
        all_rows = list(csv.DictReader(handle))

    selected = []

    for row in all_rows:
        split = row["split"]
        label = row["class_name"]

        if split not in ("train", "validation", "test"):
            raise RuntimeError("Unexpected dataset split")

        if label not in ("normal", "mci", "dementia"):
            raise RuntimeError("Unexpected CAUEEG class label")

        require(
            re.fullmatch(r"[0-9]{5,8}", row["serial"]) is not None,
            "Unexpected recording serial format"
        )

        if split == "test":
            continue

        require(
            row["edf_exists"] == "1",
            "Missing training/validation EDF"
        )

        selected.append(row)

    seen = set()

    for row in selected:
        key = row["serial"]

        require(
            key not in seen,
            "Duplicate recording in train/validation"
        )

        seen.add(key)

    counts = {
        split: dict(
            Counter(
                row["class_name"]
                for row in selected
                if row["split"] == split
            )
        )
        for split in ("train", "validation")
    }

    require(
        counts == EXPECTED_COUNTS,
        "Official split counts differ from Code 07 audit"
    )

    require(
        len(selected) == 1032,
        "Unexpected number of non-test recordings"
    )

    print("Training recordings:", 950)
    print("Validation recordings:", 82)
    print("Locked test recordings processed:", 0)

    return sorted(
        selected,
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            row["serial"]
        )
    )


def load_models(protocol, protocol_hash):
    print("\n4. LOADING ORIGINAL FROZEN MODELS", flush=True)

    fitted = ROOT / "results/final_development_models_2026"

    for folder, name in (
        ("dsrf", "DSRF"),
        ("xgboost", "XGBoost")
    ):
        metadata = read_json(
            fitted / folder / "training_metadata.json"
        )

        require(
            metadata["model"] == name
            and metadata["protocol_sha256"] == protocol_hash
            and metadata["manifest_sha256"] == sha256(DEV_MANIFEST)
            and metadata["n_development_subjects"] == 52
            and metadata["final_test_evaluated"] is False,
            f"{name} provenance mismatch"
        )

    checkpoint = torch.load(
        DSRF_PATH,
        map_location="cpu",
        weights_only=True
    )

    require(
        checkpoint["model_name"] == "fusion"
        and checkpoint["train_mode"] == "frozen"
        and checkpoint["classes"] == ["AD", "CN"],
        "DSRF checkpoint metadata mismatch"
    )

    require(
        checkpoint["checkpoint_sha256"]
        == protocol["pretrained_weights_sha256"],
        "DSRF pretrained encoder mismatch"
    )

    require(
        checkpoint["selection_protocol_sha256"]
        == protocol_hash,
        "DSRF protocol provenance mismatch"
    )

    require(
        Path(checkpoint["manifest"]).resolve()
        == DEV_MANIFEST.resolve(),
        "DSRF development manifest mismatch"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FoundationClassifier(
        num_classes=2,
        checkpoint=str(PRETRAINED),
        decoder="fusion",
        freeze_encoder=True
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    require(
        model.freeze_encoder,
        "Encoder is not frozen"
    )

    # Only load the already-verified original trusted checkpoint.
    package = joblib.load(XGB_PATH)

    require(
        package["classes"] == ["AD", "CN"]
        and package["protocol_sha256"] == protocol_hash
        and package["manifest_sha256"] == sha256(DEV_MANIFEST),
        "XGBoost package provenance mismatch"
    )

    xgb = package["model"]

    require(
        np.array_equal(
            np.asarray(xgb.classes_),
            [0, 1]
        ),
        "XGBoost class index mismatch"
    )

    feature_count = features(
        np.zeros((1, 19, 4, 200), dtype=np.float32)
    ).shape[1]

    require(
        int(xgb.n_features_in_) == feature_count == 247,
        "XGBoost feature count mismatch"
    )

    print("[OK] CBraMod + DSRF loaded strictly")
    print("[OK] XGBoost loaded with 247 EEG features")
    print("Inference device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return model, xgb, device


def normalize_channel(name):
    value = str(name).upper().strip()

    value = re.sub(r"^EEG[\s:._-]*", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace("_", "-")

    base, _, reference = value.partition("-")

    if reference not in REFERENCES:
        return None

    base = ALIASES.get(base, base)

    lookup = {
        channel.upper(): channel
        for channel in CHANNELS
    }

    return lookup.get(base)


def prepare_windows(path):
    raw = mne.io.read_raw_edf(
        str(path),
        preload=False,
        infer_types=False,
        stim_channel=None,
        verbose="ERROR"
    )

    try:
        positions = {}

        for i, label in enumerate(raw.ch_names):
            channel = normalize_channel(label)

            if channel is None:
                continue

            require(
                channel not in positions,
                "Duplicate EEG electrode"
            )

            positions[channel] = i

        require(
            set(CHANNELS).issubset(positions),
            "Required EEG electrode missing"
        )

        picks = [
            positions[channel]
            for channel in CHANNELS
        ]

        require(
            all(
                kind == "eeg"
                for kind in raw.get_channel_types(picks=picks)
            ),
            "Incorrect EEG channel type"
        )

        require(
            all(
                raw.info["chs"][index]["unit"]
                == FIFF.FIFF_UNIT_V
                for index in picks
            ),
            "Unexpected EEG physical unit"
        )

        fs = float(raw.info["sfreq"])

        require(
            abs(fs - 200.0) < 0.001,
            "Sampling frequency is not 200 Hz"
        )

        eeg_uv = (
            raw.get_data(picks=picks).astype(np.float64)
            * 1e6
        )

    finally:
        raw.close()

    require(
        eeg_uv.shape[0] == 19
        and np.isfinite(eeg_uv).all(),
        "Invalid EEG waveform"
    )

    sos = butter(
        4,
        (0.3, 40.0),
        btype="bandpass",
        fs=200.0,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        eeg_uv,
        axis=-1
    )

    scaled = (filtered / 100.0).astype(np.float32)

    require(
        np.isfinite(scaled).all(),
        "Nonfinite filtered EEG"
    )

    samples_per_window = 800
    total_windows = scaled.shape[1] // samples_per_window

    require(
        total_windows >= 24,
        "Unexpected shortage of complete EEG windows"
    )

    indices = np.unique(
        np.linspace(
            0,
            total_windows - 1,
            num=24,
            dtype=int
        )
    )

    require(
        len(indices) == 24,
        "Expected 24 distinct windows"
    )

    windows = np.stack([
        scaled[
            :,
            index * samples_per_window:
            (index + 1) * samples_per_window
        ].reshape(19, 4, 200)
        for index in indices
    ])

    require(
        windows.shape == (24, 19, 4, 200)
        and np.isfinite(windows).all(),
        "Unexpected model input tensor"
    )

    start_seconds = [
        int(index * 4)
        for index in indices
    ]

    duration_seconds = (
        scaled.shape[1] / 200.0
    )

    return windows, start_seconds, duration_seconds


def predict_models(dsrf, xgb, device, windows):
    dsrf_predictions = []

    with torch.inference_mode():
        for start in range(0, len(windows), 4):
            tensor = torch.from_numpy(
                windows[start:start + 4].copy()
            ).to(device)

            logits = dsrf(tensor)

            require(
                logits.ndim == 2
                and logits.shape[1] == 2,
                "Invalid DSRF output"
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            dsrf_predictions.append(
                probabilities.cpu().numpy()
            )

    dsrf_prob = np.concatenate(
        dsrf_predictions, axis=0
    )

    input_features = features(windows)

    require(
        input_features.shape == (24, 247)
        and np.isfinite(input_features).all(),
        "Invalid XGBoost EEG features"
    )

    xgb_prob = xgb.predict_proba(input_features)

    for name, array in (
        ("DSRF", dsrf_prob),
        ("XGBoost", xgb_prob)
    ):
        require(
            array.shape == (24, 2),
            f"{name} prediction shape mismatch"
        )

        require(
            np.isfinite(array).all(),
            f"{name} has nonfinite probabilities"
        )

        require(
            np.all(array >= 0)
            and np.all(array <= 1),
            f"{name} probability is outside [0,1]"
        )

        require(
            np.allclose(
                array.sum(axis=1),
                1.0,
                atol=1e-5
            ),
            f"{name} probabilities do not sum to one"
        )

    return dsrf_prob, xgb_prob


def main():
    protocol, protocol_hash = verify_preflight()

    task_hash = lock_external_task(protocol_hash)

    rows = read_rows()

    layout = read_json(LAYOUT)
    edf_dir = Path(layout["edf_directory"])

    require(
        edf_dir.is_dir(),
        "Verified EDF directory missing"
    )

    dsrf, xgb, device = load_models(
        protocol, protocol_hash
    )

    print("\n5. RUNNING FULL FROZEN INFERENCE", flush=True)
    print("All output files remain local and restricted.")
    print("Existing matching results will be reused.")
    print("The locked test partition will not be processed.")

    hashes = {
        "protocol_sha256": protocol_hash,
        "task_sha256": task_hash,
        "pretrained_sha256": sha256(PRETRAINED),
        "dsrf_sha256": sha256(DSRF_PATH),
        "xgboost_sha256": sha256(XGB_PATH),
        "inference_script_sha256": sha256(Path(__file__))
    }

    completed = Counter()
    reused = 0
    created = 0
    total_windows = 0

    for position, row in enumerate(rows, start=1):
        split = row["split"]
        label = row["class_name"]
        serial = row["serial"]

        require(
            split in ("train", "validation"),
            "Attempted to process protected test split"
        )

        edf_path = edf_dir / (serial + ".edf")

        require(
            edf_path.is_file(),
            "A required training/validation EDF is missing"
        )

        destination = PRIVATE_ROOT / split / (
            serial + ".json"
        )

        file_hash = sha256(edf_path)

        if destination.exists():
            cached = read_json(destination)

            require(
                cached.get("version") == VERSION
                and cached.get("split") == split
                and cached.get("class_name") == label
                and cached.get("serial") == serial
                and cached.get("edf_sha256") == file_hash
                and cached.get("model_hashes") == hashes
                and len(cached.get("dsrf_window_p_ad", [])) == 24
                and len(cached.get("xgb_window_p_ad", [])) == 24,
                "Cached prediction is inconsistent; "
                "existing data preserved"
            )

            reused += 1

        else:
            windows, window_starts, duration = (
                prepare_windows(edf_path)
            )

            dsrf_prob, xgb_prob = predict_models(
                dsrf, xgb, device, windows
            )

            dsrf_ad = [
                float(value)
                for value in dsrf_prob[:, 0]
            ]

            xgb_ad = [
                float(value)
                for value in xgb_prob[:, 0]
            ]

            result = {
                "version": VERSION,
                "dataset": "CAUEEG",
                "benchmark": "dementia-no-overlap",
                "serial": serial,
                "split": split,
                "class_name": label,
                "edf_sha256": file_hash,
                "model_hashes": hashes,
                "source_model_classes": ["AD", "CN"],
                "window_count": 24,
                "window_start_seconds": window_starts,
                "recording_duration_seconds": float(duration),
                "dsrf_window_p_ad": dsrf_ad,
                "xgb_window_p_ad": xgb_ad,
                "dsrf_mean_p_ad": float(
                    np.mean(dsrf_ad)
                ),
                "xgb_mean_p_ad": float(
                    np.mean(xgb_ad)
                ),
                "preprocessing": {
                    "channels": CHANNELS,
                    "sampling_hz": 200,
                    "bandpass_hz": [0.3, 40.0],
                    "butterworth_order": 4,
                    "additional_car": False,
                    "amplitude_scale": "microvolts / 100",
                    "windows_per_recording": 24,
                    "seconds_per_window": 4
                },
                "patient_id_verified": False,
                "interpretation": (
                    "Source AD probability applied to a "
                    "different external diagnostic benchmark; "
                    "not a clinical diagnosis."
                )
            }

            safe_json_write(destination, result)

            created += 1

        completed[(split, label)] += 1
        total_windows += 24

        if (
            position == 1
            or position % 20 == 0
            or position == len(rows)
        ):
            print(
                f"[PROGRESS] {position}/{len(rows)} "
                f"recordings | new={created} | reused={reused}",
                flush=True
            )

    print("\n6. VERIFYING COMPLETION", flush=True)

    for split in ("train", "validation"):
        for label in ("normal", "mci", "dementia"):
            require(
                completed[(split, label)]
                == EXPECTED_COUNTS[split][label],
                f"Output count mismatch for {split}/{label}"
            )

    require(
        sum(completed.values()) == 1032,
        "Incomplete development inference"
    )

    require(
        total_windows == 1032 * 24,
        "Unexpected total EEG window count"
    )

    summary = {
        "status": "TRAIN_VALIDATION_INFERENCE_COMPLETED",
        "version": VERSION,
        "benchmark": "dementia-no-overlap",
        "source_classifier_classes": ["AD", "CN"],
        "model_hashes": hashes,
        "completed_counts": {
            split: {
                label: completed[(split, label)]
                for label in ("normal", "mci", "dementia")
            }
            for split in ("train", "validation")
        },
        "total_recordings": 1032,
        "total_windows": total_windows,
        "test_recordings_processed": 0,
        "original_model_retraining": False,
        "patient_level_metrics_authorized": False,
        "private_prediction_directory": str(PRIVATE_ROOT),
        "note": (
            "External recording-level inference with original "
            "frozen AD/CN models; no performance metrics or "
            "clinical validity inferred."
        )
    }

    write_status = safe_json_write(
        SUMMARY_PATH, summary
    )

    print("\nSummary:", write_status)
    print("Training recordings completed:", 950)
    print("Validation recordings completed:", 82)
    print("Total EEG windows:", total_windows)
    print("Locked test recordings processed:", 0)

    print("\nINFERENCE_STATUS: TRAIN_VALIDATION_INFERENCE_COMPLETED")


try:
    main()

except Exception as exc:
    print(
        "\n[ERROR]",
        type(exc).__name__,
        str(exc),
        flush=True
    )
    print(
        "Completed individual recordings remain preserved."
    )
    print(
        "INFERENCE_STATUS: INCOMPLETE_OR_FAILED"
    )

finally:
    print(
        "\n========== CODE 11 FINISHED ==========",
        flush=True
    )
