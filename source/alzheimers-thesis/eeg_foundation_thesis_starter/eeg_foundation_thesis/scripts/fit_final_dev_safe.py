"""Fit final thesis models on development subjects only."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    TensorWindows,
    build_windows,
    load_manifest
)

from src.model import (
    FoundationClassifier,
    sha256_file
)

from scripts.run_classical import features

from scripts.train import (
    seed_everything,
    train_epoch
)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def main():

    protocol_path = (
        ROOT / "results/final_protocol_2026.json"
    )

    if not protocol_path.is_file():
        raise FileNotFoundError(
            "Run the protocol-freezing block first."
        )

    protocol = json.loads(
        protocol_path.read_text()
    )

    assert protocol["heldout_status"] == "NOT_EVALUATED"
    assert protocol["development_subjects"] == 52
    assert protocol["heldout_subjects"] == 13
    assert protocol["classes"] == ["AD", "CN"]
    assert protocol["primary_architecture"] == "fusion"

    manifest = Path(
        protocol["final_development_manifest"]
    ).resolve()

    expected_manifest = (
        ROOT / "manifests/final_dev52_only.csv"
    ).resolve()

    if manifest != expected_manifest:
        raise RuntimeError(
            "Unexpected development manifest."
        )

    weights_path = (
        ROOT / "checkpoints/cbramod_official.pth"
    )

    if sha256_file(weights_path) != protocol[
        "pretrained_weights_sha256"
    ]:
        raise RuntimeError(
            "Pretrained weights have changed."
        )

    if sha(manifest) != protocol[
        "final_development_manifest_sha256"
    ]:
        raise RuntimeError(
            "Development manifest has changed."
        )

    df = load_manifest(manifest)

    if (
        len(df) != 52
        or df.subject_id.nunique() != 52
        or set(df.split) != {"train"}
        or set(df.label) != {"AD", "CN"}
    ):
        raise RuntimeError(
            "Final manifest must contain exactly "
            "52 development training subjects."
        )

    # The canonical split is inspected as CSV metadata.
    # No original final-test EEG signals are loaded.

    canonical = pd.read_csv(
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )

    reserved = set(
        canonical.loc[
            canonical.split == "test",
            "subject_id"
        ]
    )

    development = set(
        canonical.loc[
            canonical.split.isin(["train", "val"]),
            "subject_id"
        ]
    )

    if (
        len(reserved) != 13
        or len(development) != 52
        or set(df.subject_id) != development
    ):
        raise RuntimeError(
            "Final-test protection failed."
        )

    settings = protocol["windows"]

    assert settings["seconds"] == 4
    assert settings["max_per_subject"] == 24
    assert settings["sampling_hz"] == 200

    print(
        "Loading EEG from 52 DEVELOPMENT patients only.",
        flush=True
    )

    store = build_windows(
        df,
        "train",
        classes=["AD", "CN"],
        seconds=settings["seconds"],
        max_per_subject=settings["max_per_subject"],
        apply_car=settings["apply_car"],
        high_hz=settings["high_hz"],
        artifact_limit_uv=settings["artifact_limit_uv"]
    )

    assert set(store.subject_ids) == development

    print("Patients:", len(set(store.subject_ids)))
    print("EEG windows:", len(store))

    subject_counts = Counter(
        store.subject_ids.tolist()
    )

    subject_class = dict(
        zip(store.subject_ids, store.y)
    )

    class_counts = Counter(
        subject_class.values()
    )

    sample_weights = np.asarray([
        1.0 / (
            subject_counts[sid]
            * class_counts[subject_class[sid]]
        )
        for sid in store.subject_ids
    ], dtype=np.float64)

    sample_weights /= sample_weights.mean()

    output_root = (
        ROOT / "results/final_development_models_2026"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    protocol_sha = sha(protocol_path)

    def already_completed(folder, expected_files, name):

        if not folder.exists():
            return False

        if not all(
            (folder / filename).is_file()
            for filename in expected_files
        ):
            raise RuntimeError(
                f"Incomplete existing folder preserved: {folder}"
            )

        metadata = json.loads(
            (folder / "training_metadata.json").read_text()
        )

        if (
            metadata.get("protocol_sha256") != protocol_sha
            or metadata.get("manifest_sha256") != sha(manifest)
            or metadata.get("model") != name
            or metadata.get("n_development_subjects") != 52
            or metadata.get("final_test_evaluated") is not False
        ):
            raise RuntimeError(
                f"Existing model verification failed: {folder}"
            )

        print("Already completed:", name)

        return True

    # ===========================================================
    # DSRF: CBraMod frozen + custom decoder.
    # ===========================================================

    dsrf_folder = output_root / "dsrf"

    if not already_completed(
        dsrf_folder,
        ["final.pt", "training_metadata.json", "loss_history.csv"],
        "DSRF"
    ):

        seed_everything(
            int(protocol["dsrf"]["seed"])
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = FoundationClassifier(
            num_classes=2,
            checkpoint=str(weights_path),
            decoder="fusion",
            freeze_encoder=True
        ).to(device)

        sampler = WeightedRandomSampler(
            torch.tensor(
                sample_weights,
                dtype=torch.double
            ),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(42)
        )

        loader = DataLoader(
            TensorWindows(store),
            batch_size=8,
            sampler=sampler,
            num_workers=0
        )

        optimizer = torch.optim.AdamW(
            [
                p for p in model.parameters()
                if p.requires_grad
            ],
            lr=float(
                protocol["dsrf"]["learning_rate"]
            ),
            weight_decay=float(
                protocol["dsrf"]["weight_decay"]
            )
        )

        epochs = int(
            protocol["dsrf"]["epochs"]
        )

        assert 1 <= epochs <= 15

        history = []

        print(
            f"\nTRAINING FINAL DSRF: {epochs} FIXED EPOCHS",
            flush=True
        )

        for epoch in range(1, epochs + 1):

            loss = train_epoch(
                model,
                loader,
                optimizer,
                device
            )

            history.append({
                "epoch": epoch,
                "train_loss": loss
            })

            print(
                f"DSRF epoch {epoch}/{epochs}: "
                f"loss={loss:.4f}",
                flush=True
            )

        dsrf_folder.mkdir(
            parents=True,
            exist_ok=False
        )

        torch.save({
            "model_state": model.state_dict(),
            "classes": ["AD", "CN"],
            "model_name": "fusion",
            "train_mode": "frozen",
            "checkpoint_sha256":
                protocol["pretrained_weights_sha256"],
            "manifest": str(manifest),
            "selection_protocol_sha256": protocol_sha,
            "epochs": epochs,
            "seed": 42
        }, dsrf_folder / "final.pt")

        pd.DataFrame(history).to_csv(
            dsrf_folder / "loss_history.csv",
            index=False
        )

        metadata = {
            "model": "DSRF",
            "protocol_sha256": protocol_sha,
            "manifest_sha256": sha(manifest),
            "pretrained_sha256":
                protocol["pretrained_weights_sha256"],
            "n_development_subjects": 52,
            "n_training_windows": len(store),
            "epochs": epochs,
            "final_test_evaluated": False
        }

        (
            dsrf_folder / "training_metadata.json"
        ).write_text(
            json.dumps(metadata, indent=2)
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            "Final DSRF saved:",
            dsrf_folder,
            flush=True
        )

    # ===========================================================
    # XGBoost: selected classical baseline.
    # ===========================================================

    xgb_folder = output_root / "xgboost"

    if not already_completed(
        xgb_folder,
        ["final.joblib", "training_metadata.json"],
        "XGBoost"
    ):

        seed_everything(
            int(protocol["xgboost"]["seed"])
        )

        x = features(store.x)

        if not np.isfinite(x).all():
            raise RuntimeError(
                "Nonfinite classical features."
            )

        config = protocol["xgboost"]

        model = XGBClassifier(
            n_estimators=int(
                config["n_estimators"]
            ),
            max_depth=int(
                config["max_depth"]
            ),
            learning_rate=float(
                config["learning_rate"]
            ),
            subsample=float(
                config["subsample"]
            ),
            colsample_bytree=float(
                config["colsample_bytree"]
            ),
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=2,
            random_state=int(
                config["seed"]
            )
        )

        print(
            "\nTRAINING FINAL XGBOOST",
            flush=True
        )

        model.fit(
            x,
            store.y,
            sample_weight=sample_weights
        )

        xgb_folder.mkdir(
            parents=True,
            exist_ok=False
        )

        joblib.dump({
            "model": model,
            "classes": ["AD", "CN"],
            "protocol_sha256": protocol_sha,
            "manifest_sha256": sha(manifest)
        }, xgb_folder / "final.joblib")

        metadata = {
            "model": "XGBoost",
            "protocol_sha256": protocol_sha,
            "manifest_sha256": sha(manifest),
            "n_development_subjects": 52,
            "n_training_windows": len(store),
            "n_features": int(x.shape[1]),
            "max_depth": int(
                config["max_depth"]
            ),
            "final_test_evaluated": False
        }

        (
            xgb_folder / "training_metadata.json"
        ).write_text(
            json.dumps(metadata, indent=2)
        )

        print(
            "Final XGBoost saved:",
            xgb_folder,
            flush=True
        )

    print("\n========== FINAL TRAINING COMPLETE ==========")
    print("DSRF and XGBoost trained on 52 development patients.")
    print("No validation or test evaluation during final fitting.")
    print("Original 13 held-out patients untouched.")
    print("Models saved:", output_root)


if __name__ == "__main__":
    main()
