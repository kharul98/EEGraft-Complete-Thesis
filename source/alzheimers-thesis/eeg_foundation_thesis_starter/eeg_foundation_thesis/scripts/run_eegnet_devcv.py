"""Five-fold development-only EEGNet-inspired baseline."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CHANNELS, build_windows, load_manifest
from src.model import EEGNetLike
from scripts.train import get_metrics

CLASSES = ["AD", "CN"]


def main():

    canonical = pd.read_csv(
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )

    assert len(canonical) == 65
    assert canonical.subject_id.nunique() == 65

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

    assert len(locked) == 13
    assert len(development) == 52

    for fold in range(5):

        manifest = (
            ROOT /
            f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
        ).resolve()

        df = load_manifest(manifest)

        assert len(df) == 52
        assert set(df.subject_id) == development
        assert not locked.intersection(df.subject_id)

        outer_ids = set(
            df.loc[
                df.split == "test", "subject_id"
            ]
        )

        assert 10 <= len(outer_ids) <= 11

        for split in ("train", "val", "test"):
            assert set(
                df.loc[df.split == split, "label"]
            ) == set(CLASSES)

        run_name = (
            f"nestedcv5_f{fold}_eegnet_like_s42"
        )

        run = ROOT / "results" / run_name

        report_path = run / "metrics.json"
        best_path = run / "best.pt"

        output_path = (
            run / "outer_dev_metrics.json"
        )

        prediction_path = (
            run / "outer_dev_subject_predictions.csv"
        )

        print(
            f"\n========== EEGNET FOLD {fold} ==========",
            flush=True
        )

        if (
            output_path.is_file()
            and prediction_path.is_file()
        ):

            old = json.loads(
                output_path.read_text()
            )

            old_ids = set(
                pd.read_csv(
                    prediction_path,
                    dtype={"subject_id": str}
                ).subject_id
            )

            if (
                old.get(
                    "original_final_test_evaluated"
                ) is not False
                or old_ids != outer_ids
            ):
                raise RuntimeError(
                    "Existing result mismatch"
                )

            print("Already completed. Skipping.")
            continue

        if (
            output_path.exists()
            or prediction_path.exists()
        ):
            raise RuntimeError(
                f"Partial output preserved: {run}"
            )

        if not run.exists():

            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/train.py"),
                "--manifest", str(manifest),
                "--run-name", run_name,
                "--model", "eegnet_like",
                "--train-mode", "frozen",
                "--validation-only",
                "--seconds", "4",
                "--max-windows-per-subject", "24",
                "--batch-size", "8",
                "--epochs", "15",
                "--patience", "5",
                "--seed", "42"
            ]

            print("TRAINING:", run_name, flush=True)

            status = subprocess.call(
                command,
                cwd=ROOT
            )

            if status != 0:
                raise RuntimeError(
                    f"Training failed with status {status}. "
                    f"Files preserved in {run}"
                )

        elif not (
            report_path.is_file()
            and best_path.is_file()
        ):
            raise RuntimeError(
                f"Incomplete training folder: {run}"
            )

        report = json.loads(
            report_path.read_text()
        )

        config = report.get("config", {})

        if (
            report.get("test_not_evaluated") is not True
            or report.get("test") is not None
            or report.get(
                "trained_with_pretrained_weights"
            ) is not False
            or config.get("validation_only") is not True
            or config.get("model") != "eegnet_like"
            or config.get("run_name") != run_name
            or config.get("train_mode") != "frozen"
            or int(config.get("seconds", 0)) != 4
            or int(config.get(
                "max_windows_per_subject", 0
            )) != 24
            or int(config.get("seed", -1)) != 42
            or Path(
                config.get("manifest", "")
            ).resolve() != manifest
        ):
            raise RuntimeError(
                "EEGNet training metadata mismatch"
            )

        checkpoint = torch.load(
            best_path,
            map_location="cpu",
            weights_only=True
        )

        if (
            checkpoint.get("model_name") != "eegnet_like"
            or checkpoint.get(
                "checkpoint_sha256"
            ) is not None
            or Path(
                checkpoint.get("manifest", "")
            ).resolve() != manifest
            or checkpoint.get("classes") != CLASSES
        ):
            raise RuntimeError(
                "Incorrect EEGNet checkpoint"
            )

        model = EEGNetLike(
            len(CHANNELS), 2
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model.to(device).eval()

        store = build_windows(
            df,
            "test",
            classes=CLASSES,
            seconds=4,
            max_per_subject=24,
            high_hz=40.0,
            apply_car=False,
            artifact_limit_uv=None
        )

        metrics, (
            ids,
            targets,
            probabilities
        ) = get_metrics(
            model,
            store,
            device,
            CLASSES,
            8
        )

        if (
            len(ids) != len(outer_ids)
            or set(ids) != outer_ids
        ):
            raise RuntimeError(
                "Outer patient mismatch"
            )

        labels = np.asarray(CLASSES)

        predictions = pd.DataFrame({
            "subject_id": ids,
            "true_label": labels[targets],
            "predicted_label": labels[
                probabilities.argmax(axis=1)
            ],
            "p_AD": probabilities[:, 0],
            "p_CN": probabilities[:, 1]
        })

        result = {
            "model": "eegnet_like",
            "fold": fold,
            "outer_development_metrics": metrics,
            "n_outer_subjects": len(ids),
            "original_final_test_evaluated": False,
            "from_scratch": True
        }

        predictions.to_csv(
            prediction_path,
            index=False
        )

        output_path.write_text(
            json.dumps(result, indent=2)
        )

        print(
            "OUTER DEVELOPMENT BAcc:",
            round(metrics["balanced_accuracy"], 4)
        )

        print(
            "Macro-F1:",
            round(metrics["macro_f1"], 4)
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nEEGNET DEVELOPMENT EVALUATION COMPLETE")
    print("Original 13 final-test subjects excluded.")


if __name__ == "__main__":
    main()
