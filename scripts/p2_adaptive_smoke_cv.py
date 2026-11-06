#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
)

from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "results/p1_incremental_eeg_features.csv"
)

OUT_PRED = (
    ROOT / "results/p2_adaptive_pilot_participant_oof.csv"
)

OUT_METRICS = (
    ROOT / "results/p2_adaptive_pilot_repeat_metrics.csv"
)

DURATIONS = (30, 60, 120, 180, 300)

FEATURES = [
    f"median_relative_{band}"
    for band in (
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma"
    )
]

QUALITY = "raw_fraction_above_200uv"

POLICIES = (
    "fixed_30",
    "fixed_60",
    "fixed_120",
    "fixed_180",
    "fixed_300",
    "confidence_only",
    "stability_only",
    "confidence_stability",
    "confidence_quality",
    "combined_quality_confidence_stability",
)


def tv_distance(first, second):
    a = np.asarray(
        first,
        dtype=float
    ).copy()

    b = np.asarray(
        second,
        dtype=float
    ).copy()

    a /= max(a.sum(), 1e-20)
    b /= max(b.sum(), 1e-20)

    return float(
        np.abs(a - b).sum() / 2
    )


def selected_time(
    policy,
    probabilities,
    features,
    quality,
    training_thresholds
):
    if policy.startswith("fixed_"):
        return int(
            policy.split("_")[1]
        )

    quality_cap, stability_cap = (
        training_thresholds
    )

    previous_t = None

    for t in DURATIONS[:-1]:
        p = probabilities[t]

        confident = (
            p <= 0.20
            or p >= 0.80
        )

        highly_confident = (
            p <= 0.15
            or p >= 0.85
        )

        quality_ok = (
            quality[t] <= quality_cap
        )

        stable = (
            previous_t is not None
            and tv_distance(
                features[t],
                features[previous_t]
            ) <= stability_cap
        )

        if (
            policy == "confidence_only"
            and highly_confident
        ):
            return t

        if (
            policy == "stability_only"
            and stable
        ):
            return t

        if (
            policy == "confidence_stability"
            and confident
            and stable
        ):
            return t

        if (
            policy == "confidence_quality"
            and confident
            and quality_ok
        ):
            return t

        if (
            policy
            == "combined_quality_confidence_stability"
            and confident
            and stable
            and quality_ok
        ):
            return t

        previous_t = t

    return DURATIONS[-1]


raw = pd.read_csv(SOURCE)

assert len(raw) == 440
assert raw.participant_id.nunique() == 88

assert (
    raw.groupby("participant_id")["seconds"]
    .nunique()
    .eq(5)
    .all()
)

assert set(raw.group) == {"A", "C", "F"}

assert not raw.duplicated(
    ["participant_id", "seconds"]
).any()

assert np.isfinite(
    raw[FEATURES + [QUALITY]].to_numpy()
).all()

all_predictions = []

print(
    "===== P2 SUBJECT-LEVEL ADAPTIVE SMOKE TEST =====",
    flush=True
)

print("Source: existing 440 raw causal-prefix rows")
print("Evaluation: repeated 5-fold CV, 3 repeats")
print("Classifier probabilities: NOT calibrated")
print("Stopping rules: EXPLORATORY")

for task, negative in (
    ("AD_vs_CN", "C"),
    ("AD_vs_FTD", "F"),
):
    subset = raw.loc[
        raw.group.isin(["A", negative])
    ].copy()

    participants = (
        subset[
            ["participant_id", "group"]
        ]
        .drop_duplicates()
        .sort_values("participant_id")
    )

    subjects = participants.participant_id.to_numpy()

    y = (
        participants.group.to_numpy()
        == "A"
    ).astype(int)

    panels = {}

    for duration in DURATIONS:
        panel = (
            subset.loc[
                subset.seconds == duration
            ]
            .set_index("participant_id")
            .loc[subjects]
        )

        assert (
            panel.group.to_numpy().tolist()
            ==
            participants.group.to_numpy().tolist()
        )

        panels[duration] = panel

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=20261107
    )

    for fold_num, (train, test) in enumerate(
        cv.split(subjects, y)
    ):
        repeat = fold_num // 5 + 1
        fold = fold_num % 5 + 1

        predictions = {}

        for duration in DURATIONS:
            x = (
                panels[duration][FEATURES]
                .to_numpy(dtype=float)
            )

            pipeline = Pipeline([
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=2500
                    )
                )
            ])

            inner_cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=3049 + fold_num
            )

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
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=1,
                refit=True,
                error_score="raise"
            )

            search.fit(
                x[train],
                y[train]
            )

            predictions[duration] = (
                search.predict_proba(
                    x[test]
                )[:, 1]
            )

        # Thresholds are derived only from
        # outer-training participants.
        train_quality = (
            panels[30][QUALITY]
            .to_numpy(dtype=float)[train]
        )

        quality_cap = float(
            np.quantile(
                train_quality,
                0.75
            )
        )

        earlier = (
            panels[30][FEATURES]
            .to_numpy(dtype=float)[train]
        )

        later = (
            panels[60][FEATURES]
            .to_numpy(dtype=float)[train]
        )

        stability_cap = float(
            np.quantile(
                [
                    tv_distance(a, b)
                    for a, b in zip(
                        earlier,
                        later
                    )
                ],
                0.75
            )
        )

        for idx, test_idx in enumerate(test):
            probability = {
                t: float(predictions[t][idx])
                for t in DURATIONS
            }

            bands = {
                t: (
                    panels[t][FEATURES]
                    .iloc[test_idx]
                    .to_numpy(dtype=float)
                )
                for t in DURATIONS
            }

            quality = {
                t: float(
                    panels[t][QUALITY]
                    .iloc[test_idx]
                )
                for t in DURATIONS
            }

            for policy in POLICIES:
                duration = selected_time(
                    policy,
                    probability,
                    bands,
                    quality,
                    (
                        quality_cap,
                        stability_cap
                    )
                )

                all_predictions.append({
                    "task":
                        task,

                    "repeat":
                        repeat,

                    "fold":
                        fold,

                    "participant_id":
                        str(subjects[test_idx]),

                    "y_true":
                        int(y[test_idx]),

                    "policy":
                        policy,

                    "stop_seconds":
                        duration,

                    "probability_ad":
                        probability[duration],

                    "y_pred":
                        int(
                            probability[duration]
                            >= 0.5
                        ),

                    "train_quality_cap":
                        quality_cap,

                    "train_stability_cap":
                        stability_cap,
                })

    print(
        f"Completed {task}: "
        f"{len(subjects)} participants, "
        f"3 repeats, {len(POLICIES)} policies",
        flush=True
    )

pred = pd.DataFrame(all_predictions)

assert not pred.duplicated(
    [
        "task",
        "repeat",
        "policy",
        "participant_id"
    ]
).any()

metrics = []

for (
    task,
    repeat,
    policy
), group in pred.groupby(
    ["task", "repeat", "policy"],
    sort=True
):
    expected = (
        65 if task == "AD_vs_CN"
        else 59
    )

    assert len(group) == expected

    assert (
        group.participant_id.nunique()
        == expected
    )

    y_true = group.y_true.to_numpy(dtype=int)
    y_pred = group.y_pred.to_numpy(dtype=int)

    prob = group.probability_ad.to_numpy(
        dtype=float
    )

    assert np.unique(y_true).tolist() == [0, 1]

    metrics.append({
        "task":
            task,

        "repeat":
            int(repeat),

        "policy":
            policy,

        "participants":
            len(group),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "sensitivity_ad":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1
                )
            ),

        "specificity_nonad":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=0
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    prob
                )
            ),

        "mean_seconds":
            float(
                group.stop_seconds.mean()
            ),

        "median_seconds":
            float(
                group.stop_seconds.median()
            ),

        "early_stop_fraction":
            float(
                (
                    group.stop_seconds < 300
                ).mean()
            ),

        "false_negative_ad_count":
            int(
                (
                    (y_true == 1)
                    & (y_pred == 0)
                ).sum()
            ),
    })

results = (
    pd.DataFrame(metrics)
    .sort_values(
        ["task", "repeat", "policy"]
    )
)

assert len(results) == (
    2 * 3 * len(POLICIES)
)

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
print(
    "===== P2 POLICY COMPARISON ====="
)

print(
    "Descriptive means across three CV repeats"
)

for task in (
    "AD_vs_CN",
    "AD_vs_FTD"
):
    print()
    print("TASK:", task)

    summary = (
        results[
            results.task == task
        ]
        .groupby("policy")[
            [
                "balanced_accuracy",
                "sensitivity_ad",
                "mean_seconds",
                "early_stop_fraction"
            ]
        ]
        .mean()
        .round(3)
    )

    print(summary.to_string())

print()
print("IMPORTANT LIMITATIONS:")
print("- Repeated CV folds are not independent cohorts.")
print("- No p-values or equivalence claims are made.")
print("- Probabilities are not calibrated.")
print("- Raw features remain artifact-sensitive.")
print("- This is exploratory method development.")

print()
print(
    "P2 PARTICIPANT-LEVEL ADAPTIVE PILOT COMPLETE"
)
