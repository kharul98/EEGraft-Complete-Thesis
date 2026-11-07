"""Evaluate outer DEVELOPMENT folds without touching final locked test."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_windows, load_manifest
from src.model import FoundationClassifier, sha256_file
from scripts.train import get_metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    canonical_path = (
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv"
    )

    canonical = pd.read_csv(
        canonical_path,
        dtype={"subject_id": str}
    )

    if (
        len(canonical) != 65
        or canonical.subject_id.nunique() != 65
    ):
        raise RuntimeError(
            "Canonical 65-subject manifest changed."
        )

    locked = set(
        canonical.loc[
            canonical.split == "test", "subject_id"
        ]
    )

    development = set(
        canonical.loc[
            canonical.split.isin(["train", "val"]),
            "subject_id"
        ]
    )

    if (
        len(locked) != 13
        or len(development) != 52
        or locked.intersection(development)
    ):
        raise RuntimeError(
            "Final-test/development partition is inconsistent."
        )

    manifest_path = Path(args.manifest).resolve()

    fold_folder = (
        ROOT / "manifests/nested_cv5_dev2026"
    ).resolve()

    allowed_names = {
        f"fold_{i}.csv" for i in range(5)
    }

    if (
        manifest_path.parent != fold_folder
        or manifest_path.name not in allowed_names
    ):
        raise RuntimeError(
            "Only approved development fold manifests are allowed."
        )

    df = load_manifest(manifest_path)

    if (
        len(df) != 52
        or set(df.subject_id) != development
    ):
        raise RuntimeError(
            "Unexpected subjects in development fold."
        )

    if set(df.split) != {"train", "val", "test"}:
        raise RuntimeError(
            "Expected train, validation and outer-development splits."
        )

    if df.groupby("subject_id").split.nunique().max() != 1:
        raise RuntimeError("Patient split leakage detected.")

    outer_ids = set(
        df.loc[df.split == "test", "subject_id"]
    )

    if (
        not 10 <= len(outer_ids) <= 11
        or outer_ids.intersection(locked)
    ):
        raise RuntimeError(
            "Invalid outer development fold."
        )

    if set(df.label) != {"AD", "CN"}:
        raise RuntimeError("Unexpected class labels.")

    for split in ("train", "val", "test"):
        if set(
            df.loc[df.split == split, "label"]
        ) != {"AD", "CN"}:
            raise RuntimeError(
                f"Missing class in {split}."
            )

    workdir = ROOT / "results" / args.run_name

    report_path = workdir / "metrics.json"
    checkpoint_path = workdir / "best.pt"

    output_path = workdir / "outer_dev_metrics.json"

    predictions_path = (
        workdir / "outer_dev_subject_predictions.csv"
    )

    if output_path.exists() or predictions_path.exists():
        raise FileExistsError(
            "Existing outer-fold results preserved. Refusing overwrite."
        )

    if (
        not report_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise FileNotFoundError(
            "Training must finish before outer evaluation."
        )

    report = json.loads(report_path.read_text())

    if (
        report.get("test_not_evaluated") is not True
        or report.get("test") is not None
    ):
        raise RuntimeError(
            "Training was not confirmed as validation-only."
        )

    config = report.get("config", {})

    if config.get("validation_only") is not True:
        raise RuntimeError(
            "Missing validation-only training flag."
        )

    if Path(
        config.get("manifest", "")
    ).resolve() != manifest_path:
        raise RuntimeError(
            "Training used a different manifest."
        )

    if config.get("run_name") != args.run_name:
        raise RuntimeError("Run name mismatch.")

    if (
        config.get("train_mode") != "frozen"
        or config.get("model") not in (
            "fusion", "no_skip", "high_only", "low_only"
        )
    ):
        raise RuntimeError(
            "Unsupported architecture or training mode."
        )

    if (
        int(config.get("seconds", 0)) != 4
        or int(config.get("max_windows_per_subject", 0)) != 24
    ):
        raise RuntimeError(
            "Inconsistent EEG window configuration."
        )

    expected_counts = {
        s: int((df.split == s).sum())
        for s in ("train", "val")
    }

    if report.get("subjects_per_split") != expected_counts:
        raise RuntimeError(
            "Training subjects do not match the manifest."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    weights_path = (
        ROOT / "checkpoints/cbramod_official.pth"
    )

    actual_sha = sha256_file(weights_path)

    if (
        checkpoint.get("checkpoint_sha256") != actual_sha
        or report.get("backbone_weights_sha256") != actual_sha
    ):
        raise RuntimeError(
            "Pretrained checkpoint fingerprint mismatch."
        )

    if Path(
        checkpoint.get("manifest", "")
    ).resolve() != manifest_path:
        raise RuntimeError(
            "Checkpoint was trained on another manifest."
        )

    if (
        checkpoint.get("model_name") != config["model"]
        or checkpoint.get("train_mode") != "frozen"
    ):
        raise RuntimeError(
            "Checkpoint architecture mismatch."
        )

    classes = list(checkpoint["classes"])

    if classes != ["AD", "CN"]:
        raise RuntimeError(
            "Unexpected class encoding."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FoundationClassifier(
        num_classes=2,
        checkpoint=str(weights_path),
        decoder=config["model"],
        freeze_encoder=True
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True
    )

    model = model.to(device).eval()

    print(
        "Evaluating outer DEVELOPMENT fold:",
        len(outer_ids),
        "patients"
    )

    store = build_windows(
        df,
        "test",
        classes=classes,
        seconds=4,
        max_per_subject=24,
        apply_car=bool(config.get("car", False)),
        high_hz=float(config.get("high_hz", 40.0)),
        artifact_limit_uv=config.get("artifact_limit_uv")
    )

    metrics, (
        ids,
        ys,
        probs
    ) = get_metrics(
        model,
        store,
        device,
        classes,
        args.batch_size
    )

    if (
        set(ids) != outer_ids
        or len(ids) != len(outer_ids)
    ):
        raise RuntimeError(
            "Predicted patient IDs differ from outer fold."
        )

    labels = np.asarray(classes)

    predictions = pd.DataFrame({
        "subject_id": ids,
        "true_label": labels[ys],
        "predicted_label": labels[probs.argmax(axis=1)],
        "p_AD": probs[:, 0],
        "p_CN": probs[:, 1]
    })

    predictions.to_csv(
        predictions_path,
        index=False
    )

    output_path.write_text(
        json.dumps({
            "run_name": args.run_name,
            "manifest": str(manifest_path),
            "n_outer_subjects": len(ids),
            "outer_development_metrics": metrics,
            "original_final_test_evaluated": False
        }, indent=2)
    )

    print("\n========== OUTER DEVELOPMENT RESULTS ==========")

    print(
        "Balanced accuracy:",
        round(metrics["balanced_accuracy"], 4)
    )

    print(
        "Macro-F1:",
        round(metrics["macro_f1"], 4)
    )

    print("Saved:", output_path)
    print("Original final-test patients were excluded.")


if __name__ == "__main__":
    main()
