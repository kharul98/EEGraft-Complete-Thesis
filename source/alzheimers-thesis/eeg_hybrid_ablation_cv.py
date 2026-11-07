#!/usr/bin/env python3

"""Spectral + frozen CBraMod feature-fusion ablation."""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

import numpy as np
from scipy.stats import binomtest

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
R = ROOT / "reports"
SEED = 42
N_BOOT = 2000

SPEC = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def index_rows(rows, label):
    result = {r["subject"]: r for r in rows}

    if len(result) != len(rows):
        raise ValueError(
            f"Duplicate subject in {label}"
        )

    return result


def classifier():
    return LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=SEED
    )


def calculate_metrics(y, probability):
    prediction = (
        probability >= 0.5
    ).astype(int)

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            y,
            prediction,
            labels=[0, 1]
        ).ravel()
    )

    return {
        "accuracy": accuracy_score(
            y, prediction
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y, prediction
            ),
        "roc_auc": roc_auc_score(
            y, probability
        ),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def save_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


def main():

    # =====================================
    # LOAD SAME 46 SUBJECTS
    # =====================================

    manifest = read_csv(
        R / "openneuro_5min_manifest_v2.csv"
    )

    rows = [
        r for r in manifest
        if r["status"] == "CANDIDATE"
        and r["group"] in ("A", "C")
    ]

    groups = Counter(
        r["group"] for r in rows
    )

    if (
        len(rows) != 46
        or groups != {"A": 23, "C": 23}
    ):
        raise ValueError(
            "Expected 23 AD and 23 Healthy subjects."
        )

    ids = [r["subject"] for r in rows]

    if len(set(ids)) != 46:
        raise ValueError("Duplicate subjects")

    y = np.array([
        int(r["group"] == "A")
        for r in rows
    ])

    spectral = index_rows(
        read_csv(R / "baseline_features_v2.csv"),
        "spectral"
    )

    baseline = index_rows(
        read_csv(R / "baseline_predictions_v2.csv"),
        "baseline"
    )

    if (
        set(spectral) != set(ids)
        or set(baseline) != set(ids)
    ):
        raise ValueError(
            "Subject mismatch in baseline files."
        )

    for row in rows:
        sid = row["subject"]

        if (
            spectral[sid]["group"] != row["group"]
            or baseline[sid]["group"] != row["group"]
        ):
            raise ValueError(
                f"Label mismatch: {sid}"
            )

    # =====================================
    # SPECTRAL FEATURE MATRIX
    # =====================================

    X_spectral = np.array([
        [
            float(spectral[sid][key])
            for key in SPEC
        ]
        for sid in ids
    ])

    if (
        X_spectral.shape != (46, 6)
        or not np.isfinite(X_spectral).all()
    ):
        raise ValueError(
            "Invalid spectral features."
        )

    # =====================================
    # FROZEN CBRAMOD FEATURE MATRIX
    # =====================================

    checkpoint = (
        ROOT /
        "pretrained_weights/cbramod_official.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    checkpoint_hash = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()

    embeddings = []

    for row in rows:

        sid = row["subject"]

        cache = (
            R /
            "cbramod_subject_cache_v1" /
            f"{sid}.npz"
        )

        with np.load(
            cache,
            allow_pickle=False
        ) as saved:

            provenance = json.loads(
                saved["provenance"].item()
            )

            feature = np.asarray(
                saved["feature"],
                dtype=np.float64
            )

        source = Path(
            row["file"]
        ).resolve()

        stat = source.stat()

        # Protect against stale cached features.
        valid = (
            provenance["checkpoint"]
                == checkpoint_hash
            and provenance["subject"]
                == sid
            and provenance["source"]
                == str(source)
            and provenance["start_sec"]
                == row["start_sec"]
            and provenance["file_size"]
                == stat.st_size
            and provenance["file_mtime"]
                == stat.st_mtime_ns
        )

        if not valid:
            raise ValueError(
                f"Stale CBraMod cache: {sid}"
            )

        if (
            feature.shape != (400,)
            or not np.isfinite(feature).all()
        ):
            raise ValueError(
                f"Invalid embedding: {sid}"
            )

        embeddings.append(feature)

    X_cbramod = np.asarray(embeddings)

    X_fusion = np.column_stack([
        X_spectral,
        X_cbramod
    ])

    print(
        "===== SPECTRAL + CBRAMOD "
        "FUSION EXPERIMENT ====="
    )

    print("Subjects:", len(y))
    print("AD:", int(y.sum()))
    print("Healthy:", int((1-y).sum()))
    print("Spectral dimensions:", 6)
    print("CBraMod dimensions:", 400)
    print("Fusion dimensions:", 406)
    print("Pretrained network remains frozen.")

    # =====================================
    # THREE ABLATION PIPELINES
    # =====================================

    spectral_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier())
    ])

    cbramod_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                whiten=True,
                svd_solver="full"
            )
        ),
        ("classifier", classifier())
    ])

    fusion_pipeline = Pipeline([

        (
            "features",
            ColumnTransformer(
                transformers=[
                    (
                        "spectral",
                        StandardScaler(),
                        list(range(6))
                    ),
                    (
                        "foundation",
                        Pipeline([
                            (
                                "scaler",
                                StandardScaler()
                            ),
                            (
                                "pca",
                                PCA(
                                    whiten=True,
                                    svd_solver="full"
                                )
                            )
                        ]),
                        list(range(6, 406))
                    )
                ],
                transformer_weights={
                    "spectral": 1.0,
                    "foundation": 1.0
                }
            )
        ),

        ("classifier", classifier())
    ])

    experiments = {
        "Spectral": (
            X_spectral,
            spectral_pipeline,
            {
                "classifier__C": [
                    0.1, 1.0, 10.0
                ]
            }
        ),

        "CBraMod": (
            X_cbramod,
            cbramod_pipeline,
            {
                "pca__n_components": [5, 10],
                "classifier__C": [
                    0.1, 1.0, 10.0
                ]
            }
        ),

        "Fusion": (
            X_fusion,
            fusion_pipeline,
            {
                "features__foundation__pca__n_components":
                    [5, 10],

                "features__transformer_weights": [
                    {
                        "spectral": 1.0,
                        "foundation": weight
                    }
                    for weight in (0.5, 1.0, 2.0)
                ],

                "classifier__C": [
                    0.1, 1.0, 10.0
                ]
            }
        )
    }

    # =====================================
    # NESTED FIVE-FOLD CV
    # =====================================

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    oof = {
        name: np.full(
            len(y),
            np.nan
        )
        for name in experiments
    }

    fold_ids = np.zeros(
        len(y),
        dtype=int
    )

    tuning = []

    print(
        "\n===== FIVE-FOLD NESTED ABLATION =====",
        flush=True
    )

    for fold, (train, test) in enumerate(
        outer.split(X_spectral, y),
        1
    ):

        # Verify original fold membership.
        for j in test:
            sid = ids[int(j)]

            if int(baseline[sid]["fold"]) != fold:
                raise ValueError(
                    f"Fold mismatch: {sid}"
                )

        fold_ids[test] = fold

        inner = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=100 + fold
        )

        print(
            f"\nFold {fold}",
            flush=True
        )

        for name, (
            X,
            estimator,
            grid
        ) in experiments.items():

            search = GridSearchCV(
                estimator,
                grid,
                cv=inner,
                scoring="balanced_accuracy",
                n_jobs=1,
                refit=True,
                error_score="raise"
            )

            search.fit(
                X[train],
                y[train]
            )

            probability = (
                search.predict_proba(
                    X[test]
                )[:, 1]
            )

            oof[name][test] = probability

            pred = (
                probability >= 0.5
            ).astype(int)

            bacc = balanced_accuracy_score(
                y[test],
                pred
            )

            print(
                f"  {name}: "
                f"BACC={bacc:.3f}, "
                f"inner score={search.best_score_:.3f}",
                flush=True
            )

            print(
                "  Best parameters:",
                search.best_params_,
                flush=True
            )

            tuning.append({
                "fold": fold,
                "model": name,
                "best_inner_balanced_accuracy":
                    round(
                        float(search.best_score_),
                        5
                    ),
                "best_params_json":
                    json.dumps(
                        search.best_params_,
                        sort_keys=True
                    )
            })

    # =====================================
    # OUT-OF-FOLD RESULTS
    # =====================================

    if (
        any(
            not np.isfinite(p).all()
            for p in oof.values()
        )
        or not np.all(fold_ids > 0)
    ):
        raise RuntimeError(
            "Missing subject predictions."
        )

    summaries = []

    print(
        "\n===== FUSION ABLATION RESULTS ====="
    )

    for name, probability in oof.items():

        result = calculate_metrics(
            y,
            probability
        )

        summaries.append({
            "model": name,
            "subjects": len(y),
            **{
                k: round(float(v), 5)
                for k, v in result.items()
            }
        })

        print(
            f"\nMODEL: {name}"
        )

        for key, value in result.items():
            print(
                key + ":",
                round(float(value), 4)
            )

        print(
            "Confusion matrix:",
            [
                [
                    result["tn"],
                    result["fp"]
                ],
                [
                    result["fn"],
                    result["tp"]
                ]
            ]
        )

    save_csv(
        R / "fusion_ablation_cv_summary.csv",
        summaries
    )

    save_csv(
        R / "fusion_ablation_cv_tuning.csv",
        tuning
    )

    prediction_rows = []

    for i, row in enumerate(rows):

        record = {
            "subject": row["subject"],
            "group": row["group"],
            "fold": int(fold_ids[i]),
            "true_label": int(y[i])
        }

        for name, probability in oof.items():

            key = name.lower()

            record[key + "_prob"] = round(
                float(probability[i]),
                7
            )

            record[key + "_prediction"] = int(
                probability[i] >= 0.5
            )

        prediction_rows.append(record)

    save_csv(
        R / "fusion_ablation_cv_predictions.csv",
        prediction_rows
    )

    # =====================================
    # PAIRED COMPARISONS
    # =====================================

    svm_pred = np.array([
        int(baseline[s]["SVM_prediction"])
        for s in ids
    ])

    svm_prob = np.array([
        float(
            baseline[s]["SVM_ad_probability"]
        )
        for s in ids
    ])

    print(
        "\n===== REFERENCE SVM ====="
    )

    print(
        "Accuracy:",
        round(accuracy_score(y, svm_pred), 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc_score(y, svm_prob), 4)
    )

    comparisons = {
        "Fusion minus SVM": (
            oof["Fusion"],
            svm_pred,
            svm_prob
        ),
        "Fusion minus Spectral": (
            oof["Fusion"],
            None,
            oof["Spectral"]
        ),
        "Fusion minus CBraMod": (
            oof["Fusion"],
            None,
            oof["CBraMod"]
        )
    }

    rng = np.random.default_rng(2026)

    healthy = np.where(y == 0)[0]
    alzheimers = np.where(y == 1)[0]

    paired = []

    print(
        "\n===== PAIRED FUSION COMPARISONS ====="
    )

    for title, (
        candidate_prob,
        reference_pred,
        reference_prob
    ) in comparisons.items():

        candidate_pred = (
            candidate_prob >= 0.5
        ).astype(int)

        if reference_pred is None:
            reference_pred = (
                reference_prob >= 0.5
            ).astype(int)

        accuracy_difference = float(
            np.mean(candidate_pred == y)
            - np.mean(reference_pred == y)
        )

        auc_difference = float(
            roc_auc_score(y, candidate_prob)
            - roc_auc_score(y, reference_prob)
        )

        candidate_only = int(np.sum(
            (candidate_pred == y)
            & (reference_pred != y)
        ))

        reference_only = int(np.sum(
            (candidate_pred != y)
            & (reference_pred == y)
        ))

        discordant = (
            candidate_only + reference_only
        )

        if discordant:
            p_value = float(
                binomtest(
                    min(
                        candidate_only,
                        reference_only
                    ),
                    n=discordant,
                    p=0.5
                ).pvalue
            )
        else:
            p_value = 1.0

        accuracy_boot = []
        auc_boot = []

        for _ in range(N_BOOT):

            sample = np.concatenate([
                rng.choice(
                    healthy,
                    len(healthy),
                    replace=True
                ),
                rng.choice(
                    alzheimers,
                    len(alzheimers),
                    replace=True
                )
            ])

            accuracy_boot.append(
                np.mean(
                    candidate_pred[sample]
                    == y[sample]
                )
                -
                np.mean(
                    reference_pred[sample]
                    == y[sample]
                )
            )

            auc_boot.append(
                roc_auc_score(
                    y[sample],
                    candidate_prob[sample]
                )
                -
                roc_auc_score(
                    y[sample],
                    reference_prob[sample]
                )
            )

        acc_low, acc_high = np.percentile(
            accuracy_boot,
            [2.5, 97.5]
        )

        auc_low, auc_high = np.percentile(
            auc_boot,
            [2.5, 97.5]
        )

        print("\n", title)

        print(
            "Accuracy difference:",
            round(accuracy_difference, 4)
        )

        print(
            "95% paired accuracy CI:",
            round(float(acc_low), 4),
            round(float(acc_high), 4)
        )

        print(
            "AUC difference:",
            round(auc_difference, 4)
        )

        print(
            "95% paired AUC CI:",
            round(float(auc_low), 4),
            round(float(auc_high), 4)
        )

        print(
            "Exact McNemar p (uncorrected):",
            round(p_value, 4)
        )

        paired.append({
            "comparison": title,
            "accuracy_difference":
                round(accuracy_difference, 5),
            "accuracy_ci_low":
                round(float(acc_low), 5),
            "accuracy_ci_high":
                round(float(acc_high), 5),
            "auc_difference":
                round(auc_difference, 5),
            "auc_ci_low":
                round(float(auc_low), 5),
            "auc_ci_high":
                round(float(auc_high), 5),
            "candidate_only_correct":
                candidate_only,
            "reference_only_correct":
                reference_only,
            "mcnemar_p_uncorrected":
                round(p_value, 5)
        })

    save_csv(
        R / "fusion_paired_comparisons.csv",
        paired
    )

    print(
        "\n===== EXPERIMENT COMPLETE ====="
    )

    print(
        "Summary:",
        "reports/fusion_ablation_cv_summary.csv"
    )

    print(
        "Predictions:",
        "reports/fusion_ablation_cv_predictions.csv"
    )

    print(
        "Tuning:",
        "reports/fusion_ablation_cv_tuning.csv"
    )

    print(
        "Paired statistics:",
        "reports/fusion_paired_comparisons.csv"
    )

    print("Original EEG data unchanged.")
    print("No pretrained weights modified.")
    print("All model fitting uses training folds.")
    print("Results remain exploratory.")
    print(
        "Independent validation is still required."
    )


if __name__ == "__main__":
    main()

