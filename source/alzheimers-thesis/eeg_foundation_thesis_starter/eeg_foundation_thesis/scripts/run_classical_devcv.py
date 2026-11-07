"""Patient-isolated development-only classical EEG baselines."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_windows, load_manifest
from src.metrics import aggregate_subjects, summarize_subject_metrics
from scripts.run_classical import features

CLASSES = ["AD", "CN"]
METHODS = ("svm", "random_forest", "xgboost")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_fold(fold):
    source = ROOT / "manifests/ds004504_ad_cn_preprocessed.csv"

    canonical = pd.read_csv(
        source, dtype={"subject_id": str}
    )

    assert len(canonical) == 65
    assert canonical.subject_id.nunique() == 65

    reserved = set(
        canonical.loc[
            canonical.split == "test", "subject_id"
        ]
    )

    dev = canonical.loc[
        canonical.split.isin(["train", "val"])
    ].copy()

    assert len(reserved) == 13
    assert len(dev) == 52

    manifest = (
        ROOT / f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
    )

    df = load_manifest(manifest)

    assert len(df) == 52
    assert df.subject_id.nunique() == 52
    assert set(df.subject_id) == set(dev.subject_id)
    assert not reserved.intersection(df.subject_id)

    canonical_map = dev.set_index("subject_id")[
        ["label", "path"]
    ].sort_index()

    fold_map = df.set_index("subject_id")[
        ["label", "path"]
    ].sort_index()

    pd.testing.assert_frame_equal(
        canonical_map, fold_map
    )

    for split in ("train", "val", "test"):
        assert set(
            df.loc[df.split == split, "label"]
        ) == set(CLASSES)

    assert 10 <= (df.split == "test").sum() <= 11

    outer_ids = set(
        df.loc[df.split == "test", "subject_id"]
    )

    return df, manifest, outer_ids


def make_model(method, parameter, seed):

    if method == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=parameter,
                probability=True,
                random_state=seed
            )
        )

    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=parameter,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=seed
        )

    if method == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=180,
            max_depth=parameter,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=2,
            random_state=seed
        )

    raise ValueError(method)


def fit_model(model, method, x, y, weights):

    if method == "svm":
        return model.fit(
            x, y, svc__sample_weight=weights
        )

    return model.fit(x, y, sample_weight=weights)


def evaluate(model, x, store):

    probabilities = np.asarray(
        model.predict_proba(x),
        dtype=np.float64
    )

    if (
        probabilities.shape != (len(x), 2)
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("Invalid probabilities")

    ids, targets, prob = aggregate_subjects(
        store.y,
        probabilities,
        store.subject_ids
    )

    metrics = summarize_subject_metrics(
        targets, prob, CLASSES
    )

    return metrics, (ids, targets, prob)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds", default="0,1,2,3,4"
    )

    parser.add_argument(
        "--models",
        default="svm,random_forest,xgboost"
    )

    args = parser.parse_args()

    folds = [
        int(x) for x in args.folds.split(",")
    ]

    methods = [
        x.strip() for x in args.models.split(",")
    ]

    if (
        not folds
        or len(set(folds)) != len(folds)
        or not set(folds).issubset(range(5))
    ):
        raise ValueError("Invalid folds")

    if (
        not methods
        or len(set(methods)) != len(methods)
        or not set(methods).issubset(METHODS)
    ):
        raise ValueError("Invalid models")

    if "xgboost" in methods:
        import xgboost
        print("XGBoost:", xgboost.__version__)

    for fold in folds:

        df, manifest, outer_ids = check_fold(fold)

        pending = []

        for method in methods:

            dest = (
                ROOT /
                "results/devcv_classical_2026" /
                f"fold_{fold}" /
                method
            )

            metrics_file = (
                dest / "outer_dev_metrics.json"
            )

            predictions_file = (
                dest / "outer_dev_subject_predictions.csv"
            )

            if (
                metrics_file.is_file()
                and predictions_file.is_file()
            ):

                existing = json.loads(
                    metrics_file.read_text()
                )

                if (
                    existing.get("manifest_sha256")
                    != checksum(manifest)
                    or existing.get("model") != method
                    or existing.get(
                        "original_final_test_evaluated"
                    ) is not False
                ):
                    raise RuntimeError(
                        f"Existing result mismatch: {dest}"
                    )

                saved_ids = set(
                    pd.read_csv(
                        predictions_file,
                        dtype={"subject_id": str}
                    ).subject_id
                )

                if saved_ids != outer_ids:
                    raise RuntimeError(
                        f"Existing subject mismatch: {dest}"
                    )

                print("Already completed:", dest)

            elif dest.exists():
                raise RuntimeError(
                    f"Incomplete folder preserved: {dest}"
                )

            else:
                pending.append((method, dest))

        if not pending:
            continue

        print(
            f"\n========== FOLD {fold} ==========",
            flush=True
        )

        stores = {
            split: build_windows(
                df,
                split,
                classes=CLASSES,
                seconds=4,
                max_per_subject=24,
                apply_car=False,
                high_hz=40.0,
                artifact_limit_uv=None
            )
            for split in ("train", "val", "test")
        }

        x = {
            split: features(stores[split].x)
            for split in stores
        }

        for split in x:

            if not np.isfinite(x[split]).all():
                raise RuntimeError(
                    f"Nonfinite features in {split}"
                )

            print(
                split,
                len(np.unique(stores[split].subject_ids)),
                "subjects,",
                len(x[split]),
                "windows"
            )

        # Patient-balanced and class-balanced training weights.

        counts = Counter(
            stores["train"].subject_ids
        )

        labels_by_subject = dict(
            zip(
                stores["train"].subject_ids,
                stores["train"].y
            )
        )

        class_counts = Counter(
            labels_by_subject.values()
        )

        weights = np.asarray([
            1.0 / (
                counts[sid] *
                class_counts[labels_by_subject[sid]]
            )
            for sid in stores["train"].subject_ids
        ], dtype=float)

        weights /= weights.mean()

        for method, dest in pending:

            configurations = {
                "svm": [0.1, 1.0, 10.0],
                "random_forest": [4, 8],
                "xgboost": [2, 3]
            }[method]

            best = None

            print(
                f"\nMODEL: {method} | FOLD: {fold}",
                flush=True
            )

            for parameter in configurations:

                model = make_model(
                    method,
                    parameter,
                    seed=42 + fold
                )

                fit_model(
                    model,
                    method,
                    x["train"],
                    stores["train"].y,
                    weights
                )

                val_metrics, _ = evaluate(
                    model,
                    x["val"],
                    stores["val"]
                )

                rank = (
                    val_metrics["balanced_accuracy"],
                    val_metrics["macro_f1"]
                )

                print(
                    "Parameter:", parameter,
                    "Inner BAcc:", round(rank[0], 4),
                    "Macro-F1:", round(rank[1], 4),
                    flush=True
                )

                if best is None or rank > best[0]:
                    best = (
                        rank,
                        parameter,
                        model,
                        val_metrics
                    )

            _, selected, trained, val_metrics = best

            outer_metrics, (
                ids, targets, probabilities
            ) = evaluate(
                trained,
                x["test"],
                stores["test"]
            )

            if set(ids) != outer_ids:
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

            report = {
                "model": method,
                "fold": fold,
                "selected_hyperparameter": selected,
                "selection_rule":
                    "Inner-validation patient BAcc, then macro-F1",
                "train_n_subjects": int(
                    len(np.unique(
                        stores["train"].subject_ids
                    ))
                ),
                "inner_val_n_subjects": int(
                    len(np.unique(
                        stores["val"].subject_ids
                    ))
                ),
                "outer_dev_n_subjects": len(ids),
                "manifest_sha256": checksum(manifest),
                "features":
                    "Band power and Hjorth EEG features",
                "inner_validation_metrics": val_metrics,
                "outer_development_metrics": outer_metrics,
                "original_final_test_evaluated": False
            }

            dest.mkdir(
                parents=True,
                exist_ok=False
            )

            predictions.to_csv(
                dest / "outer_dev_subject_predictions.csv",
                index=False
            )

            (
                dest / "outer_dev_metrics.json"
            ).write_text(
                json.dumps(report, indent=2)
            )

            print(
                "OUTER DEVELOPMENT BAcc:",
                round(
                    outer_metrics["balanced_accuracy"], 4
                ),
                "Macro-F1:",
                round(
                    outer_metrics["macro_f1"], 4
                ),
                flush=True
            )

    print("\nCLASSICAL DEVELOPMENT BASELINES COMPLETE")
    print("Original 13 final-test patients untouched.")


if __name__ == "__main__":
    main()
