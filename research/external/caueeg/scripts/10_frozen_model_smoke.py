from pathlib import Path
import csv
import hashlib
import json
import os
import re
import sys

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


PROTOCOL = ROOT / "results/final_protocol_2026.json"
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

PREVIOUS = WORK / "results/09_waveform_smoke_summary.json"

MANIFEST = DATA / "manifests/07_recording_audit.csv"

SUMMARY_OUTPUT = WORK / "results/10_model_smoke_summary.json"

PRIVATE_OUTPUT = (
    DATA / "derived/10_training_smoke_predictions.json"
)

ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

REFERENCES = {
    "", "AVG", "REF", "LE", "A1", "A2",
    "M1", "M2", "CAR"
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preserve_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        obj, indent=2, sort_keys=True
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            print("[OK] Existing result matches:", path.name)
            return True

        print("[PRESERVED] Different existing result:", path)
        return False

    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)

    os.chmod(path, 0o600)

    print("[SAVED]", path)
    return True


def verify_original_files():
    print("\n1. Verifying frozen original thesis")

    require(SNAPSHOT.is_file(), "Code 03 snapshot missing")
    require(PREVIOUS.is_file(), "Code 09 result missing")

    previous = read_json(PREVIOUS)

    require(
        previous.get("smoke_status")
        == "READY_FOR_FROZEN_MODEL_INFERENCE_SETUP",
        "Code 09 did not pass"
    )

    snapshot = read_json(SNAPSHOT)

    require(
        Path(snapshot["thesis_root"]).resolve() == ROOT.resolve(),
        "Original thesis root changed"
    )

    for relative, metadata in snapshot["files"].items():
        path = ROOT / relative

        require(path.is_file(), f"Original file missing: {relative}")

        require(
            sha256(path) == metadata["sha256"],
            f"Original file changed after Code 03: {relative}"
        )

        print("[VERIFIED]", relative)

    protocol = read_json(PROTOCOL)

    require(
        protocol["classes"] == ["AD", "CN"],
        "Unexpected original class order"
    )

    require(
        protocol["primary_architecture"] == "fusion",
        "Unexpected original architecture"
    )

    require(
        protocol["development_subjects"] == 52
        and protocol["heldout_subjects"] == 13,
        "Original cohort protocol changed"
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
        "Original preprocessing protocol changed"
    )

    require(
        sha256(PRETRAINED)
        == protocol["pretrained_weights_sha256"],
        "Pretrained CBraMod hash mismatch"
    )

    require(
        sha256(DEV_MANIFEST)
        == protocol["final_development_manifest_sha256"],
        "Original development manifest hash mismatch"
    )

    protocol_hash = sha256(PROTOCOL)

    print("[OK] Original model and protocol hashes verified")

    return protocol, protocol_hash


def load_frozen_models(protocol, protocol_hash):
    print("\n2. Loading DSRF + XGBoost")

    for name, expected in (
        ("dsrf", "DSRF"),
        ("xgboost", "XGBoost")
    ):
        metadata_path = (
            ROOT / "results/final_development_models_2026"
            / name / "training_metadata.json"
        )

        metadata = read_json(metadata_path)

        require(
            metadata["model"] == expected
            and metadata["protocol_sha256"] == protocol_hash
            and metadata["manifest_sha256"] == sha256(DEV_MANIFEST)
            and metadata["n_development_subjects"] == 52,
            f"{expected} training provenance mismatch"
        )

    checkpoint = torch.load(
        DSRF_PATH,
        map_location="cpu",
        weights_only=True
    )

    require(
        checkpoint["classes"] == ["AD", "CN"]
        and checkpoint["model_name"] == "fusion"
        and checkpoint["train_mode"] == "frozen",
        "DSRF checkpoint is incompatible"
    )

    require(
        checkpoint["checkpoint_sha256"]
        == protocol["pretrained_weights_sha256"],
        "DSRF encoder provenance mismatch"
    )

    require(
        checkpoint["selection_protocol_sha256"]
        == protocol_hash,
        "DSRF selection protocol mismatch"
    )

    require(
        Path(checkpoint["manifest"]).resolve()
        == DEV_MANIFEST.resolve(),
        "DSRF training manifest mismatch"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading pretrained CBraMod encoder...")

    dsrf = FoundationClassifier(
        num_classes=2,
        checkpoint=str(PRETRAINED),
        decoder="fusion",
        freeze_encoder=True
    )

    dsrf.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    dsrf = dsrf.to(device)
    dsrf.eval()

    require(
        dsrf.freeze_encoder,
        "DSRF encoder is unexpectedly unfrozen"
    )

    print("[OK] CBraMod + DSRF strictly loaded")

    # Load only our own original trusted local joblib file.
    xgb_package = joblib.load(XGB_PATH)

    require(
        xgb_package["classes"] == ["AD", "CN"],
        "XGBoost class order mismatch"
    )

    require(
        xgb_package["protocol_sha256"] == protocol_hash
        and xgb_package["manifest_sha256"]
        == sha256(DEV_MANIFEST),
        "XGBoost checkpoint provenance mismatch"
    )

    xgb = xgb_package["model"]

    require(
        np.array_equal(np.asarray(xgb.classes_), [0, 1]),
        "Unexpected XGBoost probability column order"
    )

    dummy_features = features(
        np.zeros((1, 19, 4, 200), dtype=np.float32)
    )

    require(
        xgb.n_features_in_ == dummy_features.shape[1],
        "XGBoost feature dimension mismatch"
    )

    print("[OK] XGBoost strictly verified")
    print("Class order: 0 = AD, 1 = CN")
    print("Feature count:", xgb.n_features_in_)
    print("Inference device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return dsrf, xgb, device


def normalize_channel(label):
    name = str(label).upper().strip()

    name = re.sub(r"^EEG[\s:._-]*", "", name)
    name = re.sub(r"\s+", "", name)
    name = name.replace("_", "-")

    base, _, reference = name.partition("-")

    if reference not in REFERENCES:
        return None

    base = ALIASES.get(base, base)

    mapping = {
        channel.upper(): channel
        for channel in CHANNELS
    }

    return mapping.get(base)


def create_windows(path):
    raw = mne.io.read_raw_edf(
        str(path),
        preload=False,
        infer_types=False,
        stim_channel=None,
        verbose="ERROR"
    )

    try:
        positions = {}

        for index, label in enumerate(raw.ch_names):
            name = normalize_channel(label)

            if name is None:
                continue

            require(
                name not in positions,
                f"Duplicate EEG channel: {name}"
            )

            positions[name] = index

        require(
            set(CHANNELS).issubset(positions),
            "Missing original 19-channel montage"
        )

        picks = [
            positions[name]
            for name in CHANNELS
        ]

        require(
            all(
                raw.info["chs"][index]["unit"]
                == FIFF.FIFF_UNIT_V
                for index in picks
            ),
            "Unexpected MNE EEG units"
        )

        require(
            all(
                value == "eeg"
                for value in raw.get_channel_types(picks=picks)
            ),
            "Some selected channels are not EEG"
        )

        fs = float(raw.info["sfreq"])

        require(
            abs(fs - 200.0) < 0.001,
            "Unexpected sampling rate"
        )

        # MNE returns EEG in volts.
        # Original thesis preprocessing expects microvolts.
        eeg_uv = raw.get_data(picks=picks) * 1e6

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
        fs=200,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos, eeg_uv, axis=-1
    )

    scaled = (filtered / 100.0).astype(np.float32)

    chunk = 4 * 200
    count = scaled.shape[1] // chunk

    require(count >= 1, "Recording too short")

    selected = np.unique(
        np.linspace(
            0, count - 1,
            num=min(count, 24),
            dtype=int
        )
    )

    windows = np.stack([
        scaled[:, j * chunk:(j + 1) * chunk]
        .reshape(19, 4, 200)
        for j in selected
    ])

    require(
        windows.shape == (24, 19, 4, 200),
        "Unexpected window shape"
    )

    require(
        np.isfinite(windows).all(),
        "Nonfinite model input"
    )

    return windows


def infer_dsrf(model, windows, device):
    batches = []

    with torch.inference_mode():
        for start in range(0, len(windows), 4):
            batch = torch.from_numpy(
                windows[start:start + 4].copy()
            ).to(device)

            logits = model(batch)

            require(
                logits.ndim == 2
                and logits.shape[1] == 2,
                "Incorrect DSRF output shape"
            )

            probabilities = torch.softmax(
                logits, dim=1
            )

            batches.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(batches, axis=0)


def infer_xgboost(model, windows):
    x = features(windows)

    require(
        np.isfinite(x).all(),
        "Nonfinite EEG features"
    )

    return model.predict_proba(x)


def check_probabilities(probabilities):
    require(
        probabilities.shape == (24, 2),
        "Incorrect probability array shape"
    )

    require(
        np.isfinite(probabilities).all(),
        "Nonfinite prediction probabilities"
    )

    require(
        np.all(probabilities >= 0)
        and np.all(probabilities <= 1),
        "Predictions are outside the probability range"
    )

    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5
        ),
        "Prediction probabilities do not sum to one"
    )


def run():
    protocol, protocol_hash = verify_original_files()

    dsrf, xgb, device = load_frozen_models(
        protocol, protocol_hash
    )

    print("\n3. Selecting training-only EEG recordings")

    require(MANIFEST.is_file(), "Recording manifest missing")
    require(LAYOUT.is_file(), "Dataset layout missing")

    layout = read_json(LAYOUT)
    edf_dir = Path(layout["edf_directory"])

    with MANIFEST.open(
        encoding="utf-8", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))

    chosen = {}

    for record in records:
        if record["split"] != "train":
            continue

        label = record["class_name"]

        if (
            label in ("normal", "mci", "dementia")
            and record["edf_exists"] == "1"
        ):
            chosen.setdefault(label, record)

    require(
        set(chosen) == {"normal", "mci", "dementia"},
        "Training example selection failed"
    )

    print("[OK] 3 training recordings selected")
    print("Validation recordings selected: 0")
    print("Test recordings selected: 0")

    print("\n4. Running actual frozen model predictions")

    results = []

    for label in ("normal", "mci", "dementia"):
        record = chosen[label]

        path = edf_dir / (record["serial"] + ".edf")

        require(path.is_file(), "Selected EDF file missing")

        windows = create_windows(path)

        dsrf_prob = infer_dsrf(
            dsrf, windows, device
        )

        xgb_prob = infer_xgboost(
            xgb, windows
        )

        check_probabilities(dsrf_prob)
        check_probabilities(xgb_prob)

        dsrf_ad = float(
            dsrf_prob[:, 0].mean()
        )

        xgb_ad = float(
            xgb_prob[:, 0].mean()
        )

        print(
            f"[{label.upper()}] "
            f"windows={len(windows)}, "
            f"DSRF AD-score={dsrf_ad:.4f}, "
            f"XGBoost AD-score={xgb_ad:.4f}"
        )

        results.append({
            "caueeg_recording_class": label,
            "window_count": len(windows),
            "dsrf_mean_source_ad_probability": round(
                dsrf_ad, 6
            ),
            "xgboost_mean_source_ad_probability": round(
                xgb_ad, 6
            )
        })

    print("\n5. Saving compatibility results")

    summary = {
        "status": "FROZEN_MODELS_VERIFIED",
        "protocol_sha256": protocol_hash,
        "model_classes": ["AD", "CN"],
        "dsrf": "CBraMod + frozen fusion decoder checkpoint",
        "comparator": "frozen XGBoost checkpoint",
        "recordings_processed": 3,
        "windows_processed": 72,
        "recording_split": "train_only",
        "validation_waveforms_accessed": False,
        "test_waveforms_accessed": False,
        "retraining_performed": False,
        "interpretation": (
            "Compatibility smoke test only. "
            "CAUEEG diagnostic categories differ from "
            "the source AD/CN training categories."
        )
    }

    private_report = {
        "note": "Restricted local training-only predictions",
        "records": results,
        "patient_identifiers_verified": False,
        "not_a_clinical_diagnosis": True
    }

    saved1 = preserve_json(
        SUMMARY_OUTPUT, summary
    )

    saved2 = preserve_json(
        PRIVATE_OUTPUT, private_report
    )

    if saved1 and saved2:
        print("\nMODEL_STATUS: FROZEN_MODELS_VERIFIED")
    else:
        print("\nMODEL_STATUS: OUTPUT_CONFLICT")

    print("Original checkpoints unchanged.")
    print("No model was trained.")
    print("Locked external test waveforms were not accessed.")


try:
    run()

except Exception as exc:
    print("\n[ERROR]", type(exc).__name__, str(exc))
    print("MODEL_STATUS: FAILED")

finally:
    print("\n========== CODE 10 FINISHED ==========")
