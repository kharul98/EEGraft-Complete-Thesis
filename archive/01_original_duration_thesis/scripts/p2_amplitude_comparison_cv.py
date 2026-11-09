#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score
)


ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    ROOT / "results/p2_amplitude_limited_features.csv"
)

BASELINE_FILE = (
    ROOT / "results/p2_feature_representation_oof.csv"
)

OUT_PRED = (
    ROOT / "results/p2_amplitude_comparison_oof.csv"
)

OUT_METRICS = (
    ROOT / "results/p2_amplitude_comparison_metrics.csv"
)

BANDS = ("delta", "theta", "alpha", "beta", "gamma")
DURATIONS = (30, 60, 120, 180, 300)

FEATURES = [
    f"limited_median_{band}"
    for band in BANDS
]

data = pd.read_csv(FEATURES_FILE)

assert len(data) == 440
assert data.participant_id.nunique() == 88

predictions = []

print(
    "===== P2 MATCHED AMPLITUDE FEATURE CV =====",
    flush=True
)

for task, negative in (
    ("AD_vs_CN", "C"),
    ("AD_vs_FTD", "F")
):
    participants = (
        data.loc[
            data.group.isin(["A", negative]),
            ["participant_id", "group"]
        ]
        .drop_duplicates()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    subjects = participants.participant_id.to_numpy()

    y = (
        participants.group.to_numpy() == "A"
    ).astype(int)

    expected = (
        65 if task == "AD_vs_CN"
        else 59
    )

    assert len(subjects) == expected

    panels = {}

    for seconds in DURATIONS:
        panel = (
            data.loc[
                data.group.isin(["A", negative])
                & data.seconds.eq(seconds)
            ]
            .set_index("participant_id")
            .loc[subjects]
        )

        assert np.array_equal(
            panel.group.to_numpy(),
            participants.group.to_numpy()
        )

        x = panel[FEATURES].to_numpy(dtype=float)

        assert x.shape == (expected, 5)
        assert np.isfinite(x).all()

        panels[seconds] = x

    outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=20261107
    )

    for split_index, (train, test) in enumerate(
        outer.split(subjects, y)
    ):
        repeat = split_index // 5 + 1
        fold = split_index % 5 + 1

        inner = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=3049 + split_index
        )

        for seconds in DURATIONS:
            x = panels[seconds]

            model = Pipeline([
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2500
                    )
                )
            ])

            search = GridSearchCV(
                model,
                {
                    "clf__C": [
                        0.01, 0.1, 1.0, 10.0
                    ]
                },
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise"
            )

            search.fit(
                x[train],
                y[train]
            )

            probability = (
                search.predict_proba(x[test])[:, 1]
            )

            for j, subject_index in enumerate(test):
                p = float(probability[j])

                predictions.append({
                    "task": task,
                    "repeat": repeat,
                    "fold": fold,
                    "participant_id":
                        str(subjects[subject_index]),
                    "seconds": seconds,
                    "y_true": int(y[subject_index]),
                    "probability_ad": p,
                    "y_pred": int(p >= 0.5),
                    "best_C":
                        float(
                            search.best_params_["clf__C"]
                        )
                })

    print(
        f"Completed {task}: {expected} participants",
        flush=True
    )


pred = pd.DataFrame(predictions)

assert len(pred) == (65 + 59) * 3 * 5

assert not pred.duplicated([
    "task",
    "repeat",
    "participant_id",
    "seconds"
]).any()


# Compare with the saved baseline from the same
# subject-level split design.
baseline = pd.read_csv(BASELINE_FILE)

baseline = baseline.loc[
    baseline.representation == "epoch_median"
].copy()

keys = [
    "task",
    "repeat",
    "fold",
    "participant_id",
    "seconds"
]

comparison = pred.merge(
    baseline[
        keys + [
            "y_true",
            "probability_ad",
            "y_pred"
        ]
    ],
    on=keys,
    validate="one_to_one",
    suffixes=("_limited", "_baseline")
)

assert len(comparison) == len(pred)

assert (
    comparison.y_true_limited
    ==
    comparison.y_true_baseline
).all()


metrics = []

for keys_group, frame in comparison.groupby(
    ["task", "repeat", "seconds"]
):
    task, repeat, seconds = keys_group

    true = frame.y_true_limited.to_numpy(dtype=int)

    limited_pred = (
        frame.y_pred_limited.to_numpy(dtype=int)
    )

    baseline_pred = (
        frame.y_pred_baseline.to_numpy(dtype=int)
    )

    limited_probability = (
        frame.probability_ad_limited.to_numpy(dtype=float)
    )

    baseline_probability = (
        frame.probability_ad_baseline.to_numpy(dtype=float)
    )

    limited_ba = balanced_accuracy_score(
        true,
        limited_pred
    )

    baseline_ba = balanced_accuracy_score(
        true,
        baseline_pred
    )

    metrics.append({
        "task": task,
        "repeat": int(repeat),
        "seconds": int(seconds),
        "participants": len(frame),
        "limited_ba": float(limited_ba),
        "baseline_ba": float(baseline_ba),
        "delta_ba": float(
            limited_ba - baseline_ba
        ),
        "limited_sensitivity": float(
            recall_score(
                true,
                limited_pred,
                pos_label=1
            )
        ),
        "baseline_sensitivity": float(
            recall_score(
                true,
                baseline_pred,
                pos_label=1
            )
        ),
        "limited_auc": float(
            roc_auc_score(
                true,
                limited_probability
            )
        ),
        "baseline_auc": float(
            roc_auc_score(
                true,
                baseline_probability
            )
        )
    })


results = pd.DataFrame(metrics)

assert len(results) == 30

OUT_PRED.parent.mkdir(
    parents=True,
    exist_ok=True
)

pred.to_csv(
    OUT_PRED,
    index=False,
    float_format="%.12g"
)

results.to_csv(
    OUT_METRICS,
    index=False,
    float_format="%.12g"
)

print()
print("===== P2 AMPLITUDE METHOD COMPARISON =====")

print(
    results.groupby(
        ["task", "seconds"]
    )[
        [
            "limited_ba",
            "baseline_ba",
            "delta_ba",
            "limited_sensitivity",
            "baseline_sensitivity"
        ]
    ]
    .mean()
    .round(4)
    .to_string()
)

print()
print("IMPORTANT:")
print("- Same participants and matched outer folds.")
print("- Training-only standardization and C selection.")
print("- The threshold was not selected by CV.")
print("- Results are exploratory on repeatedly used data.")
print("- No statistically significant improvement is claimed.")
print("- This does not establish clinical artifact removal.")

print()
print("P2 AMPLITUDE COMPARISON COMPLETE")
