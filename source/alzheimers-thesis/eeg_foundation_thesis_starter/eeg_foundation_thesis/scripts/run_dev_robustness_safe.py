"""Development-only robustness with frozen CBraMod checkpoints."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_windows, load_manifest
from src.metrics import aggregate_subjects
from src.model import FoundationClassifier, sha256_file
from scripts.train import get_metrics, infer_probabilities

CLASSES = ["AD", "CN"]

MODELS = {
    "DSRF": "fusion",
    "No skip": "no_skip",
    "High only": "high_only"
}

CONDITIONS = {
    "clean": {
        "noise_ratio": 0.0,
        "drop_channels": 0
    },
    "noise_10pct": {
        "noise_ratio": 0.10,
        "drop_channels": 0
    },
    "noise_20pct": {
        "noise_ratio": 0.20,
        "drop_channels": 0
    },
    "one_channel_zeroed": {
        "noise_ratio": 0.0,
        "drop_channels": 1
    }
}

WEIGHTS = ROOT / "checkpoints/cbramod_official.pth"
OUT = ROOT / "results/devcv_dsr_robustness_v1"


def main():

    if OUT.exists():

        result = OUT / "robustness_summary.csv"
        samples = OUT / "robustness_patients.csv"

        if result.is_file() and samples.is_file():
            print("Previous results preserved:", OUT)
            print(
                pd.read_csv(result)
                .round(4)
                .to_string(index=False)
            )
            return

        raise RuntimeError(
            f"Incomplete results folder preserved: {OUT}"
        )

    canonical = pd.read_csv(
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )

    assert len(canonical) == 65
    assert canonical.subject_id.nunique() == 65

    final_ids = set(
        canonical.loc[
            canonical.split == "test", "subject_id"
        ]
    )

    dev = canonical.loc[
        canonical.split.isin(["train", "val"])
    ]

    expected = dict(
        zip(dev.subject_id, dev.label)
    )

    assert len(final_ids) == 13
    assert len(expected) == 52

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    fingerprint = sha256_file(WEIGHTS)

    records = []
    visited = set()

    for fold in range(5):

        manifest = (
            ROOT /
            f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
        ).resolve()

        df = load_manifest(manifest)

        assert len(df) == 52
        assert set(df.subject_id) == set(expected)

        outer = set(
            df.loc[
                df.split == "test", "subject_id"
            ]
        )

        assert not outer.intersection(
            visited | final_ids
        )

        visited.update(outer)

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

        assert set(store.subject_ids) == outer

        for model_label, mode in MODELS.items():

            run = (
                ROOT /
                f"results/nestedcv5_f{fold}_{mode}_s42"
            )

            ckpt = torch.load(
                run / "best.pt",
                map_location="cpu",
                weights_only=True
            )

            if (
                ckpt.get("model_name") != mode
                or ckpt.get("train_mode") != "frozen"
                or ckpt.get("classes") != CLASSES
                or ckpt.get("checkpoint_sha256") != fingerprint
                or Path(
                    ckpt.get("manifest", "")
                ).resolve() != manifest
            ):
                raise RuntimeError(
                    f"Checkpoint mismatch: {run}"
                )

            model = FoundationClassifier(
                2,
                str(WEIGHTS),
                decoder=mode,
                freeze_encoder=True
            )

            model.load_state_dict(
                ckpt["model_state"],
                strict=True
            )

            model.to(device).eval()

            original_predictions = pd.read_csv(
                run / "outer_dev_subject_predictions.csv",
                dtype={"subject_id": str}
            ).set_index("subject_id")

            assert set(original_predictions.index) == outer

            clean_pred = None

            for condition, kwargs in CONDITIONS.items():

                if condition == "clean":

                    _, (
                        ids,
                        ys,
                        probabilities
                    ) = get_metrics(
                        model,
                        store,
                        device,
                        CLASSES,
                        batch_size=8
                    )

                    clean_pred = dict(
                        zip(ids, probabilities.argmax(axis=1))
                    )

                    for i, sid in enumerate(ids):

                        expected_p = original_predictions.loc[
                            sid,
                            ["p_AD", "p_CN"]
                        ].to_numpy(dtype=float)

                        if not np.allclose(
                            probabilities[i],
                            expected_p,
                            atol=2e-4,
                            rtol=0
                        ):
                            raise RuntimeError(
                                f"Clean prediction mismatch: {run}, {sid}"
                            )

                else:

                    win_y, win_p, win_ids = infer_probabilities(
                        model,
                        store,
                        device,
                        batch_size=8,
                        **kwargs
                    )

                    ids, ys, probabilities = aggregate_subjects(
                        win_y,
                        win_p,
                        win_ids
                    )

                assert set(ids) == outer

                for i, sid in enumerate(ids):

                    if expected[sid] != CLASSES[int(ys[i])]:
                        raise RuntimeError(
                            f"Label mismatch: {sid}"
                        )

                    records.append({
                        "fold": fold,
                        "model": model_label,
                        "condition": condition,
                        "subject_id": sid,
                        "true_label": CLASSES[int(ys[i])],
                        "predicted_label": CLASSES[
                            int(probabilities[i].argmax())
                        ],
                        "flip_vs_clean": int(
                            probabilities[i].argmax()
                            != clean_pred[sid]
                        ),
                        "p_AD": float(probabilities[i, 0]),
                        "p_CN": float(probabilities[i, 1])
                    })

                print(
                    f"Fold {fold} | {model_label} | "
                    f"{condition} complete",
                    flush=True
                )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    assert visited == set(expected)

    data = pd.DataFrame(records)
    summary = []

    for (model_label, condition), sub in data.groupby(
        ["model", "condition"],
        sort=False
    ):

        assert len(sub) == 52
        assert sub.subject_id.nunique() == 52

        y = sub.true_label
        pred = sub.predicted_label

        onehot = np.eye(2)[
            (y == "CN").astype(int)
        ]

        prob = sub[
            ["p_AD", "p_CN"]
        ].to_numpy(dtype=float)

        summary.append({
            "model": model_label,
            "condition": condition,
            "balanced_accuracy": float(
                balanced_accuracy_score(y, pred)
            ),
            "accuracy": float(
                accuracy_score(y, pred)
            ),
            "macro_f1": float(
                f1_score(y, pred, average="macro")
            ),
            "flip_vs_clean": float(
                sub.flip_vs_clean.mean()
            ),
            "multiclass_brier": float(
                np.mean(
                    np.sum(
                        (prob - onehot) ** 2,
                        axis=1
                    )
                )
            ),
            "correct_patients": int(
                (y == pred).sum()
            )
        })

    summary = pd.DataFrame(summary)

    OUT.mkdir(
        parents=True,
        exist_ok=False
    )

    data.to_csv(
        OUT / "robustness_patients.csv",
        index=False
    )

    summary.to_csv(
        OUT / "robustness_summary.csv",
        index=False
    )

    (
        OUT / "metadata.json"
    ).write_text(
        json.dumps({
            "patients": 52,
            "original_final_test_evaluated": False,
            "noise_perturbation": (
                "Window SD-scaled Gaussian noise "
                "as implemented in scripts/train.py"
            ),
            "dropout_perturbation": (
                "One randomly zeroed electrode per window"
            ),
            "pretrained_checkpoint_sha256": fingerprint,
            "limitations": (
                "Artificial stress tests; not clinically "
                "realistic acquisition perturbations."
            )
        }, indent=2)
    )

    print("\n========== ROBUSTNESS SUMMARY ==========")

    print(
        summary.round(4).to_string(index=False)
    )

    print(
        "\nConstant-label control: "
        "flip rate = 0; binary balanced accuracy = 0.50."
    )

    print("Original 13 final-test patients excluded.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
