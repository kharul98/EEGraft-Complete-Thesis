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

WORK = HOME / "research/eegraft-caueeg"
DATA = HOME / "research/caueeg"

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


PLAN_FILE = WORK / "protocol/caueeg_final_test_plan_v1.json"

PREFLIGHT_FILE = WORK / "results/17_final_test_preflight.json"

LAYOUT_FILE = WORK / "protocol/dataset_layout.json"

MANIFEST_FILE = DATA / "manifests/07_recording_audit.csv"

PROTOCOL_FILE = ROOT / "results/final_protocol_2026.json"

PRETRAINED = ROOT / "checkpoints/cbramod_official.pth"

DSRF_FILE = (
    ROOT / "results/final_development_models_2026"
    / "dsrf/final.pt"
)

XGB_FILE = (
    ROOT / "results/final_development_models_2026"
    / "xgboost/final.joblib"
)

DEV_MANIFEST = ROOT / "manifests/final_dev52_only.csv"

TEST_DIRECTORY = (
    DATA / "derived/caueeg_transfer_v1/test"
)

SUMMARY_FILE = (
    WORK / "results/18_locked_test_inference_summary.json"
)

VERSION = "caueeg_transfer_v1"

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

EXPECTED_TEST_COUNTS = {
    "normal": 35,
    "mci": 33,
    "dementia": 22
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data,
        indent=2,
        sort_keys=True,
        allow_nan=False
    ) + "\n"

    if path.exists():
        require(
            path.read_text(encoding="utf-8") == content,
            "Existing output differs: " + str(path)
        )
        return "REUSED"

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
            descriptor,
            "w",
            encoding="utf-8"
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(temporary, path)

        except FileExistsError:
            require(
                path.read_text(encoding="utf-8") == content,
                "Concurrent output conflict"
            )

    finally:
        temporary.unlink(missing_ok=True)

    return "SAVED"


def verify_final_lock():
    print("\n1. VERIFYING FINAL TEST FREEZE", flush=True)

    require(
        PLAN_FILE.is_file(),
        "Code 17 final test plan missing. Run Code 17 first."
    )

    require(
        PREFLIGHT_FILE.is_file(),
        "Code 17 preflight missing. Run Code 17 first."
    )

    plan = read_json(PLAN_FILE)
    preflight = read_json(PREFLIGHT_FILE)

    plan_hash = sha256(PLAN_FILE)

    require(
        preflight.get("status")
        == "READY_FOR_LOCKED_TEST_INFERENCE",
        "Code 17 did not authorize locked test inference"
    )

    require(
        preflight.get("plan_sha256") == plan_hash,
        "Frozen test plan checksum mismatch"
    )

    require(
        plan["version"] == "caueeg_final_test_plan_v1",
        "Unexpected experiment version"
    )

    require(
        plan["official_task"] == "dementia-no-overlap",
        "Unexpected official CAUEEG benchmark"
    )

    require(
        plan["test_recordings"] == 90
        and plan["primary_binary_test_recordings"] == 57
        and plan["secondary_mci_recordings"] == 33,
        "Unexpected test population"
    )

    require(
        plan["test_class_counts"] == EXPECTED_TEST_COUNTS,
        "Test class distribution differs from frozen plan"
    )

    require(
        plan["analysis_unit"] == "recording"
        and plan["patient_level_performance_claims_allowed"] is False,
        "Invalid evaluation unit"
    )

    primary = plan["primary_comparison"]

    require(
        primary["models"] == ["DSRF", "XGBoost"]
        and primary["source_classes"] == ["AD", "CN"]
        and primary["target_positive"] == "dementia"
        and primary["target_negative"] == "normal",
        "Primary evaluation definition changed"
    )

    require(
        float(primary["DSRF_threshold"]) == 0.5
        and float(primary["XGBoost_threshold"]) == 0.5,
        "Original frozen decision thresholds changed"
    )

    require(
        plan["safety"]["no_retraining_original_models"] is True
        and plan["safety"]["no_threshold_selection_using_test"] is True
        and plan["safety"]["no_variant_selection_using_test"] is True,
        "Final experiment safety rules changed"
    )

    print("[OK] Final test plan verified")
    print("[OK] Primary threshold: 0.5 for both models")
    print("[OK] Original models remain frozen")

    return plan, plan_hash


def verify_artifact_hashes(plan):
    print("\n2. VERIFYING ALL FROZEN ARTIFACT HASHES", flush=True)

    layout = read_json(LAYOUT_FILE)

    expected = plan["artifact_hashes"]

    artifact_files = {
        "official_split_json_sha256":
            Path(layout["split_file"]),

        "annotation_json_sha256":
            Path(layout["annotation"]),

        "recording_manifest_sha256":
            MANIFEST_FILE,

        "edf_header_audit_sha256":
            WORK / "results/08_edf_header_summary.json",

        "training_inference_summary_sha256":
            WORK / "results/11_train_val_inference_summary.json",

        "threshold_lock_sha256":
            WORK / "protocol/caueeg_adapted_thresholds_v1.json",

        "calibration_lock_sha256":
            WORK / "protocol/caueeg_platt_calibration_v1.json",

        "validation_comparison_sha256":
            WORK / "results/16_validation_variant_comparison.json",
    }

    for name, path in artifact_files.items():
        require(
            path.is_file(),
            f"Required experiment artifact missing: {path}"
        )

        require(
            sha256(path) == expected[name],
            "Frozen artifact checksum mismatch: " + name
        )

        print("[VERIFIED]", name, flush=True)

    model_paths = {
        "protocol_sha256": PROTOCOL_FILE,

        "task_sha256":
            WORK / "protocol/caueeg_transfer_task_v1.json",

        "pretrained_sha256": PRETRAINED,

        "dsrf_sha256": DSRF_FILE,

        "xgboost_sha256": XGB_FILE,

        "inference_script_sha256":
            WORK / "scripts/11_train_val_inference.py",
    }

    for name, path in model_paths.items():
        require(path.is_file(), f"Missing model artifact: {path}")

        require(
            sha256(path) == plan["model_hashes"][name],
            "Original model checksum mismatch: " + name
        )

    require(
        sha256(DEV_MANIFEST)
        == read_json(PROTOCOL_FILE)[
            "final_development_manifest_sha256"
        ],
        "Original training manifest changed"
    )

    print("[OK] Dataset annotations and official splits unchanged")
    print("[OK] All original checkpoints unchanged")

    return layout


def load_test_manifest(plan):
    print("\n3. VERIFYING LOCKED TEST COHORT", flush=True)

    with MANIFEST_FILE.open(
        newline="",
        encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    require(
        len(rows) == 1122,
        "Unexpected official benchmark manifest size"
    )

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    require(
        len(test_rows) == 90,
        "Expected exactly 90 locked test recordings"
    )

    require(
        len({row["serial"] for row in test_rows}) == 90,
        "Duplicate recording identifiers in locked test"
    )

    counts = Counter(
        row["class_name"] for row in test_rows
    )

    require(
        dict(counts) == EXPECTED_TEST_COUNTS,
        "Locked test class counts differ"
    )

    for row in test_rows:
        require(
            re.fullmatch(r"[0-9]{5,8}", row["serial"]) is not None,
            "Invalid recording identifier"
        )

        require(
            row["edf_exists"] == "1"
            and row["annotation_exists"] == "1",
            "Locked test recording is missing"
        )

    print("[OK] Exactly 90 official test recordings")
    print("[OK] No duplicate test serials")
    print("[OK] Original official split preserved")
    print("No test performance metrics will be calculated.")

    return sorted(test_rows, key=lambda r: r["serial"])


def load_models(plan):
    print("\n4. LOADING ORIGINAL FROZEN MODELS", flush=True)

    protocol = read_json(PROTOCOL_FILE)
    protocol_hash = sha256(PROTOCOL_FILE)

    require(
        protocol["classes"] == ["AD", "CN"]
        and protocol["primary_architecture"] == "fusion",
        "Original classifier configuration mismatch"
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

    for folder, model_name in (
        ("dsrf", "DSRF"),
        ("xgboost", "XGBoost")
    ):
        metadata_path = (
            ROOT / "results/final_development_models_2026"
            / folder / "training_metadata.json"
        )

        metadata = read_json(metadata_path)

        require(
            metadata["model"] == model_name
            and metadata["protocol_sha256"] == protocol_hash
            and metadata["manifest_sha256"] == sha256(DEV_MANIFEST)
            and metadata["n_development_subjects"] == 52
            and metadata["final_test_evaluated"] is False,
            model_name + " training provenance mismatch"
        )

    checkpoint = torch.load(
        DSRF_FILE,
        map_location="cpu",
        weights_only=True
    )

    require(
        checkpoint["model_name"] == "fusion"
        and checkpoint["train_mode"] == "frozen"
        and checkpoint["classes"] == ["AD", "CN"],
        "DSRF architecture or class mismatch"
    )

    require(
        checkpoint["checkpoint_sha256"]
        == protocol["pretrained_weights_sha256"]
        and checkpoint["selection_protocol_sha256"]
        == protocol_hash,
        "DSRF checkpoint provenance mismatch"
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

    # Trusted original XGBoost artifact only.
    xgb_package = joblib.load(XGB_FILE)

    require(
        xgb_package["classes"] == ["AD", "CN"]
        and xgb_package["protocol_sha256"] == protocol_hash
        and xgb_package["manifest_sha256"]
        == sha256(DEV_MANIFEST),
        "XGBoost source provenance mismatch"
    )

    xgb = xgb_package["model"]

    require(
        np.array_equal(
            np.asarray(xgb.classes_), [0, 1]
        ),
        "XGBoost class ordering mismatch"
    )

    feature_count = features(
        np.zeros(
            (1, 19, 4, 200),
            dtype=np.float32
        )
    ).shape[1]

    require(
        feature_count == 247
        and int(xgb.n_features_in_) == 247,
        "XGBoost feature configuration mismatch"
    )

    print("[OK] CBraMod encoder strictly loaded")
    print("[OK] DSRF decoder strictly loaded")
    print("[OK] XGBoost 247-feature model verified")
    print("Inference device:", device, flush=True)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return model, xgb, device


def normalize_channel(label):
    value = str(label).upper().strip()

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

        for index, label in enumerate(raw.ch_names):
            channel = normalize_channel(label)

            if channel is None:
                continue

            require(
                channel not in positions,
                "Duplicate EEG electrode"
            )

            positions[channel] = index

        require(
            set(CHANNELS).issubset(positions),
            "Required 19-channel montage is incomplete"
        )

        picks = [
            positions[channel]
            for channel in CHANNELS
        ]

        require(
            all(
                value == "eeg"
                for value in raw.get_channel_types(picks=picks)
            ),
            "Selected channel type is not EEG"
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
            "Unexpected sampling frequency"
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
        "Invalid EEG waveform values"
    )

    # Identical preprocessing specification to Code 11.
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

    window_samples = 800
    available = scaled.shape[1] // window_samples

    require(
        available >= 24,
        "Recording has fewer than 24 full EEG windows"
    )

    indices = np.unique(
        np.linspace(
            0,
            available - 1,
            num=24,
            dtype=int
        )
    )

    require(
        len(indices) == 24,
        "Incorrect sampled window count"
    )

    windows = np.stack([
        scaled[
            :,
            index * window_samples:
            (index + 1) * window_samples
        ].reshape(19, 4, 200)
        for index in indices
    ])

    require(
        windows.shape == (24, 19, 4, 200)
        and np.isfinite(windows).all(),
        "Invalid EEG model input tensor"
    )

    start_seconds = [
        int(index * 4)
        for index in indices
    ]

    duration_seconds = float(
        scaled.shape[1] / 200.0
    )

    return windows, start_seconds, duration_seconds


def validate_probability_matrix(array, model_name):
    require(
        array.shape == (24, 2),
        model_name + " output shape mismatch"
    )

    require(
        np.isfinite(array).all(),
        model_name + " produced nonfinite probabilities"
    )

    require(
        np.all((array >= 0) & (array <= 1)),
        model_name + " produced invalid probabilities"
    )

    require(
        np.allclose(
            array.sum(axis=1),
            1.0,
            atol=1e-5
        ),
        model_name + " probabilities do not sum to 1"
    )


def predict_windows(model, xgb, device, windows):
    neural_batches = []

    with torch.inference_mode():
        for start in range(0, len(windows), 4):
            tensor = torch.from_numpy(
                windows[start:start + 4].copy()
            ).to(device)

            logits = model(tensor)

            require(
                logits.shape[1] == 2,
                "DSRF output class count mismatch"
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            neural_batches.append(
                probabilities.cpu().numpy()
            )

    dsrf_prob = np.concatenate(
        neural_batches,
        axis=0
    )

    input_features = features(windows)

    require(
        input_features.shape == (24, 247)
        and np.isfinite(input_features).all(),
        "Invalid XGBoost features"
    )

    xgb_prob = xgb.predict_proba(input_features)

    validate_probability_matrix(
        dsrf_prob, "DSRF"
    )

    validate_probability_matrix(
        xgb_prob, "XGBoost"
    )

    return dsrf_prob, xgb_prob


def verify_cached_result(
    cached,
    serial,
    label,
    plan_hash,
    model_hashes,
    code_hash,
    edf_hash
):
    require(
        cached.get("version") == VERSION
        and cached.get("split") == "test"
        and cached.get("serial") == serial
        and cached.get("class_name") == label,
        "Cached test recording metadata mismatch"
    )

    require(
        cached.get("final_test_plan_sha256") == plan_hash
        and cached.get("model_hashes") == model_hashes
        and cached.get("inference_code_sha256") == code_hash
        and cached.get("edf_sha256") == edf_hash,
        "Cached result provenance mismatch"
    )

    require(
        cached.get("source_model_classes") == ["AD", "CN"]
        and cached.get("window_count") == 24
        and cached.get("patient_id_verified") is False,
        "Cached model configuration mismatch"
    )

    starts = cached.get("window_start_seconds", [])

    require(
        len(starts) == 24
        and all(
            starts[i] < starts[i + 1]
            for i in range(23)
        ),
        "Cached EEG window coordinates invalid"
    )

    for prefix in ("dsrf", "xgb"):
        values = np.asarray(
            cached.get(prefix + "_window_p_ad", []),
            dtype=float
        )

        require(
            values.shape == (24,)
            and np.isfinite(values).all()
            and np.all((values >= 0) & (values <= 1)),
            "Cached probability values invalid"
        )

        require(
            np.isclose(
                float(values.mean()),
                float(cached[prefix + "_mean_p_ad"]),
                rtol=0,
                atol=1e-7
            ),
            "Cached mean score mismatch"
        )


def main():
    # All lock checks complete BEFORE any test waveform is read.
    plan, plan_hash = verify_final_lock()

    layout = verify_artifact_hashes(plan)

    rows = load_test_manifest(plan)

    edf_dir = Path(layout["edf_directory"])

    require(
        edf_dir.is_dir(),
        "Verified EDF directory missing"
    )

    require(
        not TEST_DIRECTORY.resolve().is_relative_to(
            ROOT.resolve()
        ),
        "Private test outputs must not be in the original repository"
    )

    model, xgb, device = load_models(plan)

    print("\n5. BEGINNING LOCKED TEST INFERENCE", flush=True)
    print("Models: frozen DSRF and XGBoost")
    print("Recording count:", len(rows))
    print("EEG windows per recording: 24")
    print("Total expected windows: 2160")
    print("No hyperparameter tuning will occur.")
    print("No evaluation metric will be calculated.")
    print("Predictions will remain in the restricted local folder.")

    hashes = plan["model_hashes"]
    code_hash = sha256(Path(__file__))

    completed = Counter()
    newly_generated = 0
    reused = 0

    for index, row in enumerate(rows, start=1):
        serial = row["serial"]
        label = row["class_name"]

        edf_path = edf_dir / (serial + ".edf")

        require(
            edf_path.is_file(),
            "A locked test EDF file is missing"
        )

        destination = TEST_DIRECTORY / (
            serial + ".json"
        )

        edf_hash = sha256(edf_path)

        if destination.exists():
            cached = read_json(destination)

            verify_cached_result(
                cached,
                serial,
                label,
                plan_hash,
                hashes,
                code_hash,
                edf_hash
            )

            reused += 1

        else:
            windows, starts, duration = prepare_windows(
                edf_path
            )

            dsrf_prob, xgb_prob = predict_windows(
                model,
                xgb,
                device,
                windows
            )

            dsrf_scores = [
                float(value)
                for value in dsrf_prob[:, 0]
            ]

            xgb_scores = [
                float(value)
                for value in xgb_prob[:, 0]
            ]

            result = {
                "version": VERSION,
                "dataset": "CAUEEG",
                "benchmark": "dementia-no-overlap",
                "split": "test",
                "serial": serial,
                "class_name": label,
                "edf_sha256": edf_hash,
                "final_test_plan_sha256": plan_hash,
                "model_hashes": hashes,
                "inference_code_sha256": code_hash,
                "source_model_classes": ["AD", "CN"],
                "window_count": 24,
                "window_start_seconds": starts,
                "recording_duration_seconds": duration,
                "dsrf_window_p_ad": dsrf_scores,
                "xgb_window_p_ad": xgb_scores,
                "dsrf_mean_p_ad": float(
                    np.mean(dsrf_scores)
                ),
                "xgb_mean_p_ad": float(
                    np.mean(xgb_scores)
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
                "clinical_diagnosis_claim": False,
                "interpretation": (
                    "Frozen source AD/CN model scores transferred "
                    "to a broader external dementia benchmark."
                )
            }

            safe_write_json(destination, result)

            newly_generated += 1

        completed[label] += 1

        if (
            index == 1
            or index % 10 == 0
            or index == len(rows)
        ):
            print(
                f"[PROGRESS] {index}/90 "
                f"| generated={newly_generated} "
                f"| reused={reused}",
                flush=True
            )

    print("\n6. VERIFYING COMPLETE TEST INFERENCE", flush=True)

    require(
        dict(completed) == EXPECTED_TEST_COUNTS,
        "Test output class-count mismatch"
    )

    require(
        sum(completed.values()) == 90,
        "Incomplete test inference"
    )

    actual_files = list(
        TEST_DIRECTORY.glob("*.json")
    )

    require(
        len(actual_files) == 90,
        "Unexpected number of stored test prediction files"
    )

    expected_serials = {
        row["serial"]
        for row in rows
    }

    require(
        {path.stem for path in actual_files}
        == expected_serials,
        "Stored test recording IDs do not match the official split"
    )

    print("[OK] All 90 test recordings completed")
    print("[OK] Exactly 2160 EEG windows processed")
    print("[OK] Both frozen model outputs stored")
    print("[OK] No model parameters updated")

    print("\n7. SAVING COMPLETION RECORD", flush=True)

    summary = {
        "status": "LOCKED_TEST_INFERENCE_COMPLETED",
        "experiment": "caueeg_final_test_plan_v1",
        "final_test_plan_sha256": plan_hash,
        "inference_code_sha256": code_hash,
        "model_hashes": hashes,
        "test_recordings": 90,
        "test_eeg_windows": 2160,
        "model_names": ["DSRF", "XGBoost"],
        "source_model_classes": ["AD", "CN"],
        "test_metrics_calculated": False,
        "original_models_retrained": False,
        "original_thresholds_changed": False,
        "patient_level_claims_permitted": False,
        "private_prediction_directory": str(TEST_DIRECTORY),
        "next_stage": (
            "Evaluate all predeclared variants once, "
            "using the frozen final analysis plan."
        )
    }

    status = safe_write_json(
        SUMMARY_FILE,
        summary
    )

    print("Completion record:", status)

    print("\nTEST_INFERENCE_STATUS: COMPLETED")

    print(
        "Individual test predictions were not printed "
        "or uploaded."
    )

    print(
        "Original model checkpoints and thresholds "
        "remain unchanged."
    )


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
        "TEST_INFERENCE_STATUS: INCOMPLETE_OR_FAILED"
    )

    print(
        "Previously completed prediction files "
        "were preserved."
    )

finally:
    print(
        "\n========== CODE 18 FINISHED ==========",
        flush=True
    )
