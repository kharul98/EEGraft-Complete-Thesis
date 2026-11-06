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

SOURCE = (
    ROOT / "results/p2_prefix_robust_features.csv"
)

OUT_PRED = (
    ROOT / "results/p2_quality_control_oof.csv"
)

OUT_METRICS = (
    ROOT / "results/p2_quality_control_metrics.csv"
)

FEATURES = [
    "raw_fraction_above_200uv",
    "median_epoch_rms_uv",
    "median_epoch_ptp_uv"
]

DURATIONS = (30, 60, 300)

data = pd.read_csv(SOURCE)

assert len(data) == 440
assert data.participant_id.nunique() == 88

assert not data.duplicated([
    "participant_id",
    "seconds"
]).any()

assert np.isfinite(
    data[FEATURES].to_numpy(dtype=float)
).all()

predictions = []
metrics = []

print(
    "===== P2 EEG-QUALITY NEGATIVE CONTROL =====",
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

    expected = 65 if task == "AD_vs_CN" else 59

    assert len(subjects) == expected

    outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=20261107
    )

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

        matrix = panel[FEATURES].to_numpy(
            dtype=float
        )

        assert matrix.shape == (expected, 3)
        assert np.isfinite(matrix).all()

        panels[seconds] = matrix

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

            pipeline = Pipeline([
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=2500
                    )
                )
            ])

            search = GridSearchCV(
                pipeline,
                {
                    "clf__C": [
                        0.01,
                        0.1,
                        1.0,
                        10.0
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
                    "seconds": seconds,
                    "participant_id":
                        str(subjects[subject_index]),
                    "y_true":
                        int(y[subject_index]),
                    "probability_ad": p,
                    "y_pred":
                        int(p >= 0.5),
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

assert len(pred) == (
    (65 + 59) * 3 * len(DURATIONS)
)

assert not pred.duplicated([
    "task",
    "repeat",
    "seconds",
    "participant_id"
]).any()


for keys, frame in pred.groupby(
    ["task", "repeat", "seconds"]
):

    task, repeat, seconds = keys

    true = frame.y_true.to_numpy(dtype=int)
    predicted = frame.y_pred.to_numpy(dtype=int)
    probability = frame.probability_ad.to_numpy(dtype=float)

    expected = 65 if task == "AD_vs_CN" else 59

    assert len(frame) == expected

    metrics.append({
        "task": task,
        "repeat": int(repeat),
        "seconds": int(seconds),
        "participants": expected,
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    true,
                    predicted
                )
            ),
        "sensitivity_ad":
            float(
                recall_score(
                    true,
                    predicted,
                    pos_label=1
                )
            ),
        "roc_auc":
            float(
                roc_auc_score(
                    true,
                    probability
                )
            )
    })


results = pd.DataFrame(metrics)

assert len(results) == 18

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
print("===== QUALITY-ONLY CLASSIFICATION =====")

print(
    results.groupby(
        ["task", "seconds"]
    )[
        [
            "balanced_accuracy",
            "sensitivity_ad",
            "roc_auc"
        ]
    ]
    .mean()
    .round(3)
    .to_string()
)

print()
print("===== QUALITY FEATURE GROUP DISTRIBUTIONS =====")

print(
    data.loc[data.seconds == 300]
    .groupby("group")[FEATURES]
    .median()
    .round(4)
    .to_string()
)

print()
print("INTERPRETATION:")
print("- Quality-only prediction is a confounding probe.")
print("- Predictive QC features do not prove a specific cause.")
print("- Failure to predict does not exclude confounding.")
print("- Classification uses independent participant folds.")
print("- Results remain exploratory.")
print()
print("P2 QUALITY NEGATIVE CONTROL COMPLETE")
