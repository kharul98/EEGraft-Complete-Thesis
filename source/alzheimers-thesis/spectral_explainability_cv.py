#!/usr/bin/env python3

"""
Cross-validated spectral EEG explainability.

Uses:
- 46 AD/Healthy subjects
- 10 x 5 nested subject-level CV
- Logistic Regression
- Exact additive log-odds contributions
- Held-out permutation importance

No original EEG files are modified.
"""

from pathlib import Path
import csv

import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score
)


# =====================================
# CONFIGURATION
# =====================================

R = Path("reports")

SEED = 42
REPEATS = 10
N_PERM = 5

FEATURES = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
]

GROUPS = {
    name: [i]
    for i, name in enumerate(FEATURES)
}

# Assess correlated theta/alpha features together.
GROUPS["theta_alpha_family"] = [1, 2, 5]


# =====================================
# UTILITIES
# =====================================

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


# =====================================
# MAIN
# =====================================

def main():

    manifest = [
        r for r in read(
            R / "openneuro_5min_manifest_v2.csv"
        )
        if r["status"] == "CANDIDATE"
        and r["group"] in ("A", "C")
    ]

    by_id = {
        r["subject"]: r
        for r in read(
            R / "baseline_features_v2.csv"
        )
    }

    ids = [
        r["subject"]
        for r in manifest
    ]

    assert len(ids) == 46
    assert len(set(ids)) == 46
    assert set(ids) == set(by_id)

    assert sum(
        r["group"] == "A"
        for r in manifest
    ) == 23

    for r in manifest:
        assert (
            r["group"]
            == by_id[r["subject"]]["group"]
        )

    X = np.array([
        [
            float(by_id[s][c])
            for c in FEATURES
        ]
        for s in ids
    ])

    y = np.array([
        int(r["group"] == "A")
        for r in manifest
    ])

    assert np.isfinite(X).all()

    # Each subject receives one held-out
    # explanation in each repetition.
    attributions = np.full(
        (REPEATS, len(y), len(FEATURES)),
        np.nan
    )

    coefficients = []
    subject_preds = []
    perm_runs = []

    rng = np.random.default_rng(20261107)

    print(
        "===== FOLD-AWARE SPECTRAL "
        "EXPLAINABILITY ====="
    )

    print("Subjects:", len(y))
    print("AD:", int(y.sum()))
    print("Healthy:", int(len(y) - y.sum()))

    # =====================================
    # REPEATED NESTED CV
    # =====================================

    for repeat in range(REPEATS):

        oof = np.full(
            len(y),
            np.nan
        )

        perturbed = {
            name: np.full(
                (N_PERM, len(y)),
                np.nan
            )
            for name in GROUPS
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42 + repeat * 100
        )

        for fold, (train, test) in enumerate(
            cv.split(X, y),
            1
        ):

            inner = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=(
                    1000 + repeat * 100 + fold
                )
            )

            pipeline = Pipeline([
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
            ])

            search = GridSearchCV(
                pipeline,
                {
                    "model__C": [
                        0.01,
                        0.1,
                        1.0,
                        10.0
                    ]
                },
                cv=inner,
                scoring="balanced_accuracy",
                refit=True,
                n_jobs=1,
                error_score="raise"
            )

            search.fit(
                X[train],
                y[train]
            )

            fitted = search.best_estimator_

            prob = fitted.predict_proba(
                X[test]
            )[:, 1]

            oof[test] = prob

            scaler = fitted.named_steps["scale"]
            linear = fitted.named_steps["model"]

            # Exact additive contribution
            # to model log-odds, relative to
            # the training-fold feature means.
            attrs = (
                (X[test] - scaler.mean_)
                / scaler.scale_
            ) * linear.coef_[0]

            np.testing.assert_allclose(
                linear.intercept_[0]
                + attrs.sum(axis=1),
                fitted.decision_function(
                    X[test]
                ),
                atol=1e-7,
                rtol=1e-7
            )

            attributions[
                repeat, test, :
            ] = attrs

            for idx in test:

                subject_preds.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "subject": ids[int(idx)],
                    "group": (
                        "A" if y[idx] else "C"
                    ),
                    "true_label": int(y[idx]),
                    "ad_probability": float(
                        oof[idx]
                    ),
                    "predicted_label": int(
                        oof[idx] >= 0.5
                    ),
                    "chosen_C": float(
                        search.best_params_[
                            "model__C"
                        ]
                    )
                })

            for j, name in enumerate(FEATURES):

                coefficients.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "feature": name,
                    "standardized_coef": float(
                        linear.coef_[0, j]
                    ),
                    "chosen_C": float(
                        search.best_params_[
                            "model__C"
                        ]
                    )
                })

            # =================================
            # HELD-OUT PERMUTATION IMPORTANCE
            # =================================

            for name, cols in GROUPS.items():

                for k in range(N_PERM):

                    order = rng.permutation(
                        len(test)
                    )

                    variant = X[test].copy()

                    variant[:, cols] = (
                        X[test][order][:, cols]
                    )

                    perturbed[name][
                        k, test
                    ] = fitted.predict_proba(
                        variant
                    )[:, 1]

        assert np.isfinite(oof).all()

        assert np.isfinite(
            attributions[repeat]
        ).all()

        base_bacc = balanced_accuracy_score(
            y,
            oof >= 0.5
        )

        base_auc = roc_auc_score(
            y,
            oof
        )

        for name, variants in perturbed.items():

            assert np.isfinite(variants).all()

            for k in range(N_PERM):

                perm_runs.append({
                    "repeat": repeat + 1,
                    "feature_or_group": name,
                    "permutation": k + 1,

                    "delta_balanced_accuracy":
                        float(
                            base_bacc
                            - balanced_accuracy_score(
                                y,
                                variants[k] >= 0.5
                            )
                        ),

                    "delta_auc":
                        float(
                            base_auc
                            - roc_auc_score(
                                y,
                                variants[k]
                            )
                        )
                })

        print(
            f"Repeat {repeat+1:02d}/{REPEATS} "
            f"BACC={base_bacc:.3f} "
            f"AUC={base_auc:.3f}",
            flush=True
        )

    # =====================================
    # AGGREGATE FEATURE CONTRIBUTIONS
    # =====================================

    averaged = attributions.mean(
        axis=0
    )

    coefs_by_feature = {
        name: np.array([
            c["standardized_coef"]
            for c in coefficients
            if c["feature"] == name
        ])
        for name in FEATURES
    }

    summary = []

    for j, name in enumerate(FEATURES):

        arr = averaged[:, j]

        summary.append({

            "feature": name,

            "mean_absolute_logodds_attribution":
                float(
                    np.abs(arr).mean()
                ),

            "mean_signed_AD":
                float(
                    arr[y == 1].mean()
                ),

            "mean_signed_Healthy":
                float(
                    arr[y == 0].mean()
                ),

            "positive_coefficient_fraction":
                float(
                    (
                        coefs_by_feature[name] > 0
                    ).mean()
                )
        })

    summary.sort(
        key=lambda r: r[
            "mean_absolute_logodds_attribution"
        ],
        reverse=True
    )

    # =====================================
    # AGGREGATE PERMUTATION IMPORTANCE
    # =====================================

    perm_summary = []

    for name in GROUPS:

        runs = [
            r for r in perm_runs
            if r["feature_or_group"] == name
        ]

        bacc = np.array([
            np.mean([
                r["delta_balanced_accuracy"]
                for r in runs
                if r["repeat"] == rep
            ])
            for rep in range(1, REPEATS + 1)
        ])

        auc = np.array([
            np.mean([
                r["delta_auc"]
                for r in runs
                if r["repeat"] == rep
            ])
            for rep in range(1, REPEATS + 1)
        ])

        perm_summary.append({

            "feature_or_group": name,

            "mean_bacc_drop":
                float(bacc.mean()),

            "sd_bacc_drop_across_repeats":
                float(
                    bacc.std(ddof=1)
                ),

            "mean_auc_drop":
                float(auc.mean()),

            "sd_auc_drop_across_repeats":
                float(
                    auc.std(ddof=1)
                )
        })

    perm_summary.sort(
        key=lambda r: r["mean_bacc_drop"],
        reverse=True
    )

    # =====================================
    # SUBJECT-LEVEL EXPLANATIONS
    # =====================================

    person_rows = []

    for i, subject in enumerate(ids):

        r = {
            "subject": subject,
            "group": (
                "A" if y[i] else "C"
            )
        }

        for j, name in enumerate(FEATURES):

            r[
                name + "_mean_signed_logodds"
            ] = float(
                averaged[i, j]
            )

            r[
                name + "_mean_abs_logodds"
            ] = float(
                np.abs(
                    attributions[:, i, j]
                ).mean()
            )

        person_rows.append(r)

    # =====================================
    # SAVE REPORTS
    # =====================================

    save(
        R / "spectral_explain_attributions.csv",
        summary
    )

    save(
        R / "spectral_explain_permutation.csv",
        perm_summary
    )

    save(
        R / "spectral_explain_permutation_runs.csv",
        perm_runs
    )

    save(
        R / "spectral_explain_coefficients.csv",
        coefficients
    )

    save(
        R / "spectral_explain_subjects.csv",
        person_rows
    )

    save(
        R / "spectral_explain_cv_predictions.csv",
        subject_preds
    )

    # =====================================
    # PRINT RESULTS
    # =====================================

    print(
        "\n===== LOG-ODDS CONTRIBUTION RANKING ====="
    )

    for rank, r in enumerate(summary, 1):

        print(
            f"{rank}. {r['feature']}: "
            f"mean |contribution|="
            f"{r['mean_absolute_logodds_attribution']:.3f}, "
            f"AD signed="
            f"{r['mean_signed_AD']:+.3f}, "
            f"Healthy signed="
            f"{r['mean_signed_Healthy']:+.3f}, "
            f"positive coefficient in "
            f"{100*r['positive_coefficient_fraction']:.0f}% "
            f"of fits"
        )

    print(
        "\n===== HELD-OUT PERMUTATION IMPORTANCE ====="
    )

    for r in perm_summary:

        print(
            f"{r['feature_or_group']}: "
            f"BACC drop="
            f"{r['mean_bacc_drop']:+.3f} "
            f"(SD="
            f"{r['sd_bacc_drop_across_repeats']:.3f}), "
            f"AUC drop="
            f"{r['mean_auc_drop']:+.3f}"
        )

    print("\n===== COMPLETE =====")

    print(
        "Reports: reports/spectral_explain_*.csv"
    )

    print(
        "No original EEG modified; "
        "no participants removed."
    )

    print(
        "Explanations are model-level associations, "
        "not causal biomarkers."
    )

    print(
        "Same 46 subjects reused; "
        "independent generalization not tested."
    )


if __name__ == "__main__":
    main()

