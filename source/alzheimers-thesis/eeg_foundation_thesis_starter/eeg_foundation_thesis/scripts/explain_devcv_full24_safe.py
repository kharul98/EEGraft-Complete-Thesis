"""Interpret out-of-fold DSRF predictions on development patients only."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CHANNELS, build_windows, load_manifest
from src.model import FoundationClassifier, sha256_file

CLASSES = ["AD", "CN"]

SAVE = ROOT / "results/devcv_dsr_fusion_interpret_full24_2026"
WEIGHTS = ROOT / "checkpoints/cbramod_official.pth"


def main():

    canonical = pd.read_csv(
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )

    if len(canonical) != 65 or canonical.subject_id.nunique() != 65:
        raise RuntimeError("Canonical manifest changed")

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
            "Original final-test split invalid"
        )

    if SAVE.exists():

        outputs = [
            SAVE / "per_subject_channel.csv",
            SAVE / "interpretation_summary.json",
            SAVE / "gate_ad.png",
            SAVE / "gate_cn.png"
        ]

        if all(p.is_file() for p in outputs):
            print("Interpretation already completed:", SAVE)
            return

        raise RuntimeError(
            f"Incomplete output preserved: {SAVE}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    weights_sha = sha256_file(WEIGHTS)

    print("Device:", device)

    records = []
    gate_maps = {label: [] for label in CLASSES}
    covered = set()

    for fold in range(5):

        manifest_path = (
            ROOT /
            f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
        ).resolve()

        df = load_manifest(manifest_path)

        if (
            len(df) != 52
            or set(df.subject_id) != development
            or set(df.subject_id).intersection(locked)
        ):
            raise RuntimeError(
                "Unsafe development manifest"
            )

        outer_ids = set(
            df.loc[df.split == "test", "subject_id"]
        )

        if covered.intersection(outer_ids):
            raise RuntimeError(
                "Repeated outer-fold patients"
            )

        covered.update(outer_ids)

        checkpoint_path = (
            ROOT /
            f"results/nestedcv5_f{fold}_fusion_s42/best.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if (
            checkpoint.get("model_name") != "fusion"
            or checkpoint.get("train_mode") != "frozen"
            or checkpoint.get("classes") != CLASSES
            or checkpoint.get("checkpoint_sha256") != weights_sha
            or Path(
                checkpoint.get("manifest", "")
            ).resolve() != manifest_path
        ):
            raise RuntimeError(
                f"Checkpoint mismatch in fold {fold}"
            )

        model = FoundationClassifier(
            num_classes=2,
            checkpoint=str(WEIGHTS),
            decoder="fusion",
            freeze_encoder=True
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True
        )

        model.to(device).eval()

        store = build_windows(
            df,
            "test",
            classes=CLASSES,
            seconds=4,
            max_per_subject=24,
            apply_car=False,
            high_hz=40.0,
            artifact_limit_uv=None
        )

        if set(store.subject_ids) != outer_ids:
            raise RuntimeError(
                "Outer-fold patient mismatch"
            )

        for subject in sorted(outer_ids):

            indices = np.flatnonzero(
                store.subject_ids == subject
            )

            selected = indices[
                np.unique(
                    np.linspace(
                        0,
                        len(indices) - 1,
                        num=min(24, len(indices)),
                        dtype=int
                    )
                )
            ]

            x = torch.from_numpy(
                store.x[selected].copy()
            ).to(device)

            true_index = int(
                store.y[selected][0]
            )

            if not np.all(
                store.y[selected] == true_index
            ):
                raise RuntimeError(
                    "Conflicting window labels"
                )

            true_label = CLASSES[true_index]

            with torch.no_grad():

                logits, extras = model(
                    x, return_extras=True
                )

                clean_probs = (
                    logits.softmax(dim=1).mean(dim=0)
                )

                predicted_label = CLASSES[
                    int(clean_probs.argmax())
                ]

                gate_map = (
                    extras["gate_map"]
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )

                attention = (
                    extras["class_attention"][:, true_index]
                    .mean(dim=0)
                    .sum(dim=-1)
                    .cpu()
                    .numpy()
                )

            gate_maps[true_label].append(gate_map)

            # Gradient x signal for true-class probability.
            grad_input = (
                x.detach()
                .clone()
                .requires_grad_(True)
            )

            score = (
                model(grad_input)
                .softmax(dim=1)[:, true_index]
                .mean()
            )

            gradient = torch.autograd.grad(
                score, grad_input
            )[0]

            gradient_importance = (
                (gradient * grad_input)
                .abs()
                .mean(dim=(0, 2, 3))
                .detach()
                .cpu()
                .numpy()
            )

            del grad_input, gradient

            # Mask each electrode with its within-window mean.
            # This is sensitivity analysis, not a causal intervention.
            occlusion = np.zeros(
                len(CHANNELS),
                dtype=float
            )

            with torch.no_grad():

                for start in range(0, len(CHANNELS), 2):

                    electrodes = list(
                        range(
                            start,
                            min(start + 2, len(CHANNELS))
                        )
                    )

                    modified_list = []

                    for ch in electrodes:

                        modified = x.clone()

                        modified[:, ch] = (
                            modified[:, ch]
                            .mean(
                                dim=(1, 2),
                                keepdim=True
                            )
                        )

                        modified_list.append(modified)

                    masked = torch.cat(
                        modified_list, dim=0
                    )

                    changed = (
                        model(masked)
                        .softmax(dim=1)
                        .reshape(
                            len(electrodes),
                            len(x),
                            2
                        )
                        .mean(dim=1)
                    )

                    for offset, ch in enumerate(electrodes):

                        occlusion[ch] = float(
                            clean_probs[true_index]
                            - changed[offset, true_index]
                        )

            for ch, channel in enumerate(CHANNELS):

                records.append({
                    "fold": fold,
                    "subject_id": subject,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == true_label,
                    "channel": channel,
                    "clean_p_true": float(
                        clean_probs[true_index]
                    ),
                    "occlusion_delta_p_true": float(
                        occlusion[ch]
                    ),
                    "gradient_x_input_abs": float(
                        gradient_importance[ch]
                    ),
                    "true_class_attention_mass": float(
                        attention[ch]
                    ),
                    "mean_residual_gate": float(
                        gate_map[ch].mean()
                    ),
                    "n_explained_windows": len(selected)
                })

            print(
                f"Fold {fold}: {subject} "
                f"{true_label} -> {predicted_label}",
                flush=True
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if (
        len(covered) != 52
        or len(records) != 52 * len(CHANNELS)
    ):
        raise RuntimeError(
            "Incomplete development-patient coverage"
        )

    result = pd.DataFrame(records)

    SAVE.mkdir(
        parents=True,
        exist_ok=False
    )

    result.to_csv(
        SAVE / "per_subject_channel.csv",
        index=False
    )

    for label in CLASSES:

        mean_map = np.mean(
            np.stack(gate_maps[label]),
            axis=0
        )

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        image = ax.imshow(
            mean_map,
            aspect="auto",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )

        ax.set_yticks(
            range(len(CHANNELS)),
            CHANNELS
        )

        ax.set_xticks(
            range(4),
            ["0-1s", "1-2s", "2-3s", "3-4s"]
        )

        ax.set_title(
            f"DSRF residual gate: {label} (descriptive)"
        )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        fig.savefig(
            SAVE / f"gate_{label.lower()}.png",
            dpi=160
        )

        plt.close(fig)

    metadata = {
        "n_development_patients": 52,
        "n_outer_folds": 5,
        "n_windows_per_patient": 24,
        "target": "True-class probability",
        "original_final_test_evaluated": False,
        "limitation": (
            "Electrode masking and gradients measure model "
            "sensitivity, not causal or clinically validated "
            "brain localization. Gate values are descriptive."
        )
    }

    (
        SAVE / "interpretation_summary.json"
    ).write_text(
        json.dumps(metadata, indent=2)
    )

    print("\n========== CHANNEL INTERPRETABILITY ==========")

    for label in CLASSES:

        means = (
            result[result.true_label == label]
            .groupby("channel", as_index=False)[[
                "occlusion_delta_p_true",
                "mean_residual_gate",
                "true_class_attention_mass"
            ]]
            .mean()
        )

        print("\nCLASS:", label)

        print(
            means.nlargest(
                5, "occlusion_delta_p_true"
            ).to_string(index=False)
        )

    print("\nSAVED:", SAVE)
    print("52 development patients interpreted.")
    print("Original final-test patients excluded.")


if __name__ == "__main__":
    main()
