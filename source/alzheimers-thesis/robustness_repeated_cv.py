#!/usr/bin/env python3

"""
Repeated nested subject-level cross-validation.

Models:
    Logistic Regression
    RBF-SVM

Dataset:
    OpenNeuro AD vs Healthy
    46 independent participants

Purpose:
    Measure performance variability
    across different subject-level splits.

Original EEG is never modified.
"""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json

import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)


# =====================================
# CONFIGURATION
# =====================================

REPORTS = Path("reports")

REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 3
BASE_SEED = 42

FEATURES = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
]


# =====================================
# FILE UTILITIES
# =====================================

def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):

    with path.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


# =====================================
# MODEL DEFINITIONS
# =====================================

def make_estimators():

    return {

        "LogisticRegression": (

            Pipeline([
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),

            {
                "model__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0
                ]
            }
        ),

        "RBF_SVM": (

            Pipeline([
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=False
                    )
                )
            ]),

            {
                "model__C": [
                    0.1,
                    1.0,
                    10.0
                ],

                "model__gamma": [
                    "scale",
                    0.01,
                    0.1
                ]
            }
        )
    }


# =====================================
# MAIN EXPERIMENT
# =====================================

def main():

    REPORTS.mkdir(
        exist_ok=True
    )

    manifest = [
        r for r in read_csv(
            REPORTS /
            "openneuro_5min_manifest_v2.csv"
        )
        if r["status"] == "CANDIDATE"
        and r["group"] in ("A", "C")
    ]

    feature_rows = read_csv(
        REPORTS / "baseline_features_v2.csv"
    )

    groups = Counter(
        r["group"] for r in manifest
    )

    if (
        len(manifest) != 46
        or groups != {"A": 23, "C": 23}
    ):
        raise ValueError(
            "Expected 23 AD and 23 Healthy subjects."
        )

    feature_by_id = {
        r["subject"]: r
        for r in feature_rows
    }

    ids = [
        r["subject"] for r in manifest
    ]

    if (
        len(set(ids)) != 46
        or len(feature_by_id) != len(feature_rows)
        or set(ids) != set(feature_by_id)
    ):
        raise ValueError(
            "Duplicate or mismatched participants."
        )

    for row in manifest:

        sid = row["subject"]

        if (
            feature_by_id[sid]["group"]
            != row["group"]
        ):
            raise ValueError(
                f"Diagnosis mismatch: {sid}"
            )

    X = np.array([
        [
            float(feature_by_id[sid][feature])
            for feature in FEATURES
        ]
        for sid in ids
    ], dtype=float)

    y = np.array([
        int(r["group"] == "A")
        for r in manifest
    ], dtype=int)

    if (
        X.shape != (46, 6)
        or not np.isfinite(X).all()
    ):
        raise ValueError(
            "Missing or invalid spectral features."
        )

    models = make_estimators()

    run_rows = []
    fold_rows = []

    person_correct = {
        name: defaultdict(list)
        for name in models
    }

    print(
        "===== REPEATED NESTED CV: SPECTRAL EEG =====",
        flush=True
    )

    print("Subjects:", len(ids))
    print("AD:", int(y.sum()))
    print("Healthy:", int((1 - y).sum()))

    print(
        "Repeated outer CV:",
        REPEATS,
        "x",
        OUTER_FOLDS
    )

    print(
        "Inner CV:",
        INNER_FOLDS,
        "folds"
    )

    print(
        "Hyperparameter selection occurs "
        "inside training folds only."
    )

    print(
        "AUC uses model decision scores; "
        "accuracy uses model predictions."
    )

    # =====================================
    # REPEATED NESTED CROSS-VALIDATION
    # =====================================

    for repeat in range(REPEATS):

        outer_seed = (
            BASE_SEED + 100 * repeat
        )

        outer = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=outer_seed
        )

        splits = list(
            outer.split(X, y)
        )

        for name, (
            estimator,
            grid
        ) in models.items():

            prediction = np.full(
                len(y),
                -1,
                dtype=int
            )

            decision = np.full(
                len(y),
                np.nan,
                dtype=float
            )

            fold_auc_values = []

            for fold, (
                train,
                test
            ) in enumerate(splits, start=1):

                inner = StratifiedKFold(
                    n_splits=INNER_FOLDS,
                    shuffle=True,
                    random_state=(
                        1000
                        + repeat * 100
                        + fold
                    )
                )

                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring="balanced_accuracy",
                    cv=inner,
                    n_jobs=1,
                    refit=True,
                    error_score="raise"
                )

                search.fit(
                    X[train],
                    y[train]
                )

                model = search.best_estimator_

                prediction[test] = model.predict(
                    X[test]
                )

                decision[test] = (
                    model.decision_function(
                        X[test]
                    )
                )

                fold_auc = roc_auc_score(
                    y[test],
                    decision[test]
                )

                fold_auc_values.append(
                    fold_auc
                )

                fold_rows.append({
                    "repeat": repeat + 1,
                    "seed": outer_seed,
                    "model": name,
                    "fold": fold,
                    "inner_best_bacc": round(
                        float(search.best_score_),
                        6
                    ),
                    "best_params": json.dumps(
                        search.best_params_,
                        sort_keys=True
                    ),
                    "fold_balanced_accuracy": round(
                        float(
                            balanced_accuracy_score(
                                y[test],
                                prediction[test]
                            )
                        ),
                        6
                    ),
                    "fold_auc": round(
                        float(fold_auc),
                        6
                    )
                })

            # =================================
            # REPEAT-LEVEL EVALUATION
            # =================================

            if (
                np.any(prediction < 0)
                or not np.isfinite(decision).all()
            ):
                raise RuntimeError(
                    f"Incomplete predictions: {name}"
                )

            tn, fp, fn, tp = [
                int(v)
                for v in confusion_matrix(
                    y,
                    prediction,
                    labels=[0, 1]
                ).ravel()
            ]

            run_rows.append({
                "repeat": repeat + 1,
                "seed": outer_seed,
                "model": name,

                "accuracy": round(
                    float(
                        accuracy_score(
                            y,
                            prediction
                        )
                    ),
                    6
                ),

                "balanced_accuracy": round(
                    float(
                        balanced_accuracy_score(
                            y,
                            prediction
                        )
                    ),
                    6
                ),

                "pooled_oof_auc": round(
                    float(
                        roc_auc_score(
                            y,
                            decision
                        )
                    ),
                    6
                ),

                "mean_fold_auc": round(
                    float(
                        np.mean(fold_auc_values)
                    ),
                    6
                ),

                "sensitivity": round(
                    tp / (tp + fn),
                    6
                ),

                "specificity": round(
                    tn / (tn + fp),
                    6
                ),

                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            })

            for i, sid in enumerate(ids):

                person_correct[name][sid].append(
                    int(
                        prediction[i] == y[i]
                    )
                )

        print(
            f"Repeat {repeat + 1:02d}/"
            f"{REPEATS} completed",
            flush=True
        )

    # =====================================
    # SAVE REPORTS
    # =====================================

    save_csv(
        REPORTS /
        "robustness_repeated_nested_cv_runs.csv",
        run_rows
    )

    save_csv(
        REPORTS /
        "robustness_repeated_nested_cv_folds.csv",
        fold_rows
    )

    subject_rows = []

    for i, sid in enumerate(ids):

        row = {
            "subject": sid,
            "group": (
                "A" if y[i] else "C"
            )
        }

        for name in models:

            row[
                name + "_correct_fraction"
            ] = round(
                float(
                    np.mean(
                        person_correct[name][sid]
                    )
                ),
                3
            )

        subject_rows.append(row)

    save_csv(
        REPORTS /
        "robustness_subject_stability.csv",
        subject_rows
    )

    # =====================================
    # SUMMARY
    # =====================================

    print(
        "\n===== STABILITY ACROSS RANDOM SPLITS ====="
    )

    for name in models:

        selected = [
            r for r in run_rows
            if r["model"] == name
        ]

        for metric in (
            "balanced_accuracy",
            "mean_fold_auc",
            "pooled_oof_auc"
        ):

            values = np.array([
                r[metric]
                for r in selected
            ])

            print(
                f"{name} {metric}: "
                f"mean={values.mean():.4f} "
                f"sd={values.std(ddof=1):.4f} "
                f"min={values.min():.4f} "
                f"max={values.max():.4f}"
            )

    print("\n===== COMPLETE =====")

    print(
        "Run summaries:",
        "reports/robustness_repeated_nested_cv_runs.csv"
    )

    print(
        "Fold details:",
        "reports/robustness_repeated_nested_cv_folds.csv"
    )

    print(
        "Per-subject stability:",
        "reports/robustness_subject_stability.csv"
    )

    print(
        "Original EEG, existing reports and "
        "checkpoints unchanged."
    )

    print(
        "These are descriptive stability results, "
        "not independent validation or "
        "confidence intervals."
    )


if __name__ == "__main__":
    main()

