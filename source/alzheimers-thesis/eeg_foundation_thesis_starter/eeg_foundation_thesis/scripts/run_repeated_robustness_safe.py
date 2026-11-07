"""Matched, repeated synthetic EEG robustness evaluation."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_windows, load_manifest
from src.model import FoundationClassifier, sha256_file
from scripts.train import get_metrics

CLASSES = ["AD", "CN"]

MODELS = {
    "DSRF": "fusion",
    "No skip": "no_skip",
    "High only": "high_only"
}

CONDITIONS = [
    "noise_10pct",
    "noise_20pct",
    "one_channel_zeroed"
]

REPEATS = 5

WEIGHTS = ROOT / "checkpoints/cbramod_official.pth"

OUTPUT = ROOT / "results/devcv_robustness_repeated_v1"


def file_hash(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def perturb(store, fold, repeat, condition):

    x = np.array(
        store.x,
        dtype=np.float32,
        copy=True
    )

    seed = 20261108 + 1000 * fold + 17 * repeat

    if condition in ("noise_10pct", "noise_20pct"):

        rng = np.random.default_rng(seed)

        ratio = (
            0.10
            if condition == "noise_10pct"
            else 0.20
        )

        std = x.std(
            axis=(1, 2, 3),
            keepdims=True
        )

        noise = rng.normal(
            size=x.shape
        ).astype(np.float32)

        x += noise * std * ratio

    elif condition == "one_channel_zeroed":

        rng = np.random.default_rng(
            seed + 900000
        )

        choices = rng.integers(
            0,
            x.shape[1],
            size=len(x)
        )

        x[
            np.arange(len(x)),
            choices,
            :,
            :
        ] = 0.0

    else:
        raise ValueError(condition)

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Nonfinite EEG after perturbation"
        )

    return replace(
        store,
        x=np.ascontiguousarray(x)
    )


def verify_rows(df, outer, fold):

    expected = (
        len(outer)
        * REPEATS
        * len(MODELS)
        * len(CONDITIONS)
    )

    if len(df) != expected:
        raise RuntimeError(
            f"Wrong row count for fold {fold}"
        )

    if set(df.subject_id) != outer:
        raise RuntimeError(
            "Wrong outer-development patients"
        )

    keys = [
        "repeat",
        "model",
        "condition",
        "subject_id"
    ]

    if df.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate repeated predictions"
        )

    for model in MODELS:
        for condition in CONDITIONS:
            for repeat in range(REPEATS):

                part = df[
                    (df.model == model)
                    & (df.condition == condition)
                    & (df.repeat == repeat)
                ]

                if set(part.subject_id) != outer:
                    raise RuntimeError(
                        "Incomplete repeated experiment"
                    )


def run_fold(fold, development, locked, weight_hash, device):

    manifest = (
        ROOT /
        f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
    ).resolve()

    df = load_manifest(manifest)

    if (
        len(df) != 52
        or df.subject_id.nunique() != 52
        or set(df.subject_id) != development
        or set(df.subject_id).intersection(locked)
    ):
        raise RuntimeError(
            "Development manifest safety check failed"
        )

    outer = set(
        df.loc[
            df.split == "test", "subject_id"
        ]
    )

    assert len(outer) in (10, 11)

    result_file = OUTPUT / f"fold_{fold}.csv"
    metadata_file = OUTPUT / f"fold_{fold}.json"

    metadata = {
        "fold": fold,
        "manifest_sha256": file_hash(manifest),
        "backbone_sha256": weight_hash,
        "n_repeats": REPEATS,
        "conditions": CONDITIONS,
        "original_final_test_evaluated": False
    }

    if result_file.exists() or metadata_file.exists():

        if not (
            result_file.is_file()
            and metadata_file.is_file()
        ):
            raise RuntimeError(
                "Incomplete existing fold preserved"
            )

        old_meta = json.loads(
            metadata_file.read_text()
        )

        if old_meta != metadata:
            raise RuntimeError(
                "Saved experiment configuration mismatch"
            )

        old = pd.read_csv(
            result_file,
            dtype={"subject_id": str}
        )

        verify_rows(old, outer, fold)

        print(
            f"Fold {fold}: completed results verified",
            flush=True
        )

        return

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

    if set(store.subject_ids) != outer:
        raise RuntimeError(
            "Window subject IDs mismatch"
        )

    rows = []

    for model_name, mode in MODELS.items():

        folder = (
            ROOT /
            f"results/nestedcv5_f{fold}_{mode}_s42"
        )

        checkpoint = torch.load(
            folder / "best.pt",
            map_location="cpu",
            weights_only=True
        )

        if (
            checkpoint.get("model_name") != mode
            or checkpoint.get("train_mode") != "frozen"
            or checkpoint.get("classes") != CLASSES
            or checkpoint.get("checkpoint_sha256")
               != weight_hash
            or Path(
                checkpoint.get("manifest", "")
            ).resolve() != manifest
        ):
            raise RuntimeError(
                f"Checkpoint mismatch: {folder}"
            )

        reference = pd.read_csv(
            folder / "outer_dev_subject_predictions.csv",
            dtype={"subject_id": str}
        ).set_index("subject_id")

        if (
            len(reference) != len(outer)
            or not reference.index.is_unique
            or set(reference.index) != outer
        ):
            raise RuntimeError(
                "Original predictions mismatch"
            )

        model = FoundationClassifier(
            num_classes=2,
            checkpoint=str(WEIGHTS),
            decoder=mode,
            freeze_encoder=True
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True
        )

        model.to(device).eval()

        # Verify clean outputs before generating any stress results.
        _, (ids, ys, probabilities) = get_metrics(
            model,
            store,
            device,
            CLASSES,
            8
        )

        if set(ids) != outer:
            raise RuntimeError(
                "Clean evaluation patient mismatch"
            )

        for sid, yy, pp in zip(
            ids, ys, probabilities
        ):

            existing = reference.loc[
                sid,
                ["p_AD", "p_CN"]
            ].to_numpy(dtype=float)

            if not np.allclose(
                pp, existing,
                atol=2e-4,
                rtol=0
            ):
                raise RuntimeError(
                    f"Clean probability mismatch: {sid}"
                )

            if CLASSES[int(yy)] != str(
                reference.loc[sid, "true_label"]
            ):
                raise RuntimeError(
                    "Clean class label mismatch"
                )

        for repeat in range(REPEATS):

            for condition in CONDITIONS:

                altered = perturb(
                    store,
                    fold,
                    repeat,
                    condition
                )

                _, (
                    ids,
                    targets,
                    probabilities
                ) = get_metrics(
                    model,
                    altered,
                    device,
                    CLASSES,
                    8
                )

                if set(ids) != outer:
                    raise RuntimeError(
                        "Perturbation patient mismatch"
                    )

                for sid, target, prob in zip(
                    ids, targets, probabilities
                ):

                    true_label = CLASSES[int(target)]

                    if true_label != str(
                        reference.loc[sid, "true_label"]
                    ):
                        raise RuntimeError(
                            "Patient label mismatch"
                        )

                    predicted = CLASSES[
                        int(prob.argmax())
                    ]

                    previous = str(
                        reference.loc[
                            sid, "predicted_label"
                        ]
                    )

                    rows.append({
                        "fold": fold,
                        "repeat": repeat,
                        "model": model_name,
                        "condition": condition,
                        "subject_id": sid,
                        "true_label": true_label,
                        "predicted_label": predicted,
                        "flip_vs_clean": int(
                            predicted != previous
                        ),
                        "p_AD": float(prob[0]),
                        "p_CN": float(prob[1])
                    })

                print(
                    f"Fold {fold} | {model_name} | "
                    f"repeat {repeat + 1}/{REPEATS} | "
                    f"{condition}",
                    flush=True
                )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows)

    verify_rows(result, outer, fold)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = OUTPUT / f"fold_{fold}.partial.csv"

    if temporary.exists():
        raise RuntimeError(
            "Unexpected temporary file preserved"
        )

    result.to_csv(
        temporary,
        index=False
    )

    temporary.rename(result_file)

    metadata_file.write_text(
        json.dumps(metadata, indent=2)
    )

    print(
        f"Saved fold {fold}: {result_file}",
        flush=True
    )


def main():

    canonical = pd.read_csv(
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )

    if (
        len(canonical) != 65
        or canonical.subject_id.nunique() != 65
    ):
        raise RuntimeError(
            "Canonical manifest changed"
        )

    locked = set(
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
        len(locked) != 13
        or len(development) != 52
        or locked.intersection(development)
    ):
        raise RuntimeError(
            "Original subject partition invalid"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    weight_hash = sha256_file(WEIGHTS)

    print("Device:", device)
    print("Development subjects:", len(development))
    print("Reserved final-test subjects:", len(locked))

    for fold in range(5):

        run_fold(
            fold,
            development,
            locked,
            weight_hash,
            device
        )

    combined = pd.concat(
        [
            pd.read_csv(
                OUTPUT / f"fold_{fold}.csv",
                dtype={"subject_id": str}
            )
            for fold in range(5)
        ],
        ignore_index=True
    )

    if set(combined.subject_id) != development:
        raise RuntimeError(
            "Cross-fold patient coverage mismatch"
        )

    summary = []

    for model_name in MODELS:

        for condition in CONDITIONS:

            subset = combined[
                (combined.model == model_name)
                & (combined.condition == condition)
            ]

            bacc_scores = []
            flip_rates = []
            brier_scores = []

            for repeat in range(REPEATS):

                part = subset[
                    subset.repeat == repeat
                ]

                if (
                    len(part) != 52
                    or part.subject_id.nunique() != 52
                ):
                    raise RuntimeError(
                        "Incomplete repeat"
                    )

                y = part.true_label.to_numpy()
                predicted = (
                    part.predicted_label.to_numpy()
                )

                probabilities = part[
                    ["p_AD", "p_CN"]
                ].to_numpy(dtype=float)

                target_onehot = np.eye(2)[
                    (y == "CN").astype(int)
                ]

                bacc_scores.append(
                    balanced_accuracy_score(
                        y, predicted
                    )
                )

                flip_rates.append(
                    float(part.flip_vs_clean.mean())
                )

                brier_scores.append(
                    float(np.mean(
                        np.sum(
                            (probabilities - target_onehot) ** 2,
                            axis=1
                        )
                    ))
                )

            summary.append({
                "model": model_name,
                "condition": condition,
                "mean_BAcc_pct": round(
                    100 * np.mean(bacc_scores), 2
                ),
                "BAcc_min_pct": round(
                    100 * np.min(bacc_scores), 2
                ),
                "BAcc_max_pct": round(
                    100 * np.max(bacc_scores), 2
                ),
                "mean_flip_pct": round(
                    100 * np.mean(flip_rates), 2
                ),
                "mean_Brier": round(
                    np.mean(brier_scores), 4
                )
            })

    summary = pd.DataFrame(summary)

    summary_file = OUTPUT / "summary_repeated.csv"

    if summary_file.exists():

        old = pd.read_csv(summary_file)

        pd.testing.assert_frame_equal(
            old, summary, check_dtype=False
        )

        print("Existing summary verified.")

    else:

        summary.to_csv(
            summary_file,
            index=False
        )

    print("\n========== REPEATED ROBUSTNESS RESULTS ==========")

    print(
        summary.to_string(index=False)
    )

    print("\nFive corruption realizations per condition.")
    print("Repetitions are not independent patients.")
    print("Original 13 final-test patients excluded.")
    print("REPEATED ROBUSTNESS COMPLETE")


if __name__ == "__main__":
    main()
