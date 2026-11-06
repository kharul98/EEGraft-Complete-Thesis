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
    roc_auc_score,
    brier_score_loss
)


ROOT = Path(__file__).resolve().parents[1]

OLD = ROOT / "results/p1_incremental_eeg_features.csv"
ROBUST = ROOT / "results/p2_prefix_robust_features.csv"

REFERENCE = (
    ROOT / "results/p2_adaptive_pilot_participant_oof.csv"
)

OUT_PRED = (
    ROOT / "results/p2_feature_representation_oof.csv"
)

OUT_METRICS = (
    ROOT / "results/p2_feature_representation_metrics.csv"
)

DURATIONS = (30, 60, 120, 180, 300)

BANDS = ("delta", "theta", "alpha", "beta", "gamma")

old = pd.read_csv(OLD)
robust = pd.read_csv(ROBUST)

assert len(old) == len(robust) == 440

for df in (old, robust):
    assert not df.duplicated(
        ["participant_id", "seconds"]
    ).any()

alignment = old[
    ["participant_id", "seconds", "group"]
].merge(
    robust[
        ["participant_id", "seconds", "group"]
    ],
    on=["participant_id", "seconds"],
    validate="one_to_one",
    suffixes=("_old", "_robust")
)

assert len(alignment) == 440
assert (
    alignment.group_old == alignment.group_robust
).all()

representations = {
    "pilot_global_psd": (
        old,
        [f"median_relative_{b}" for b in BANDS]
    ),
    "epoch_median": (
        robust,
        [f"median_relative_{b}" for b in BANDS]
    ),
    "epoch_mean": (
        robust,
        [f"mean_relative_{b}" for b in BANDS]
    )
}

predictions = []

print("===== P2 FAIR FEATURE REPRESENTATION CV =====", flush=True)

for task, negative in (
    ("AD_vs_CN", "C"),
    ("AD_vs_FTD", "F")
):

    participants = (
        old.loc[
            old.group.isin(["A", negative]),
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
    assert len(np.unique(subjects)) == expected

    panels = {}

    for name, (data, features) in representations.items():

        for seconds in DURATIONS:

            panel = (
                data[
                    (data.group.isin(["A", negative]))
                    & (data.seconds == seconds)
                ]
                .set_index("participant_id")
                .loc[subjects]
            )

            assert np.array_equal(
                panel.group.to_numpy(),
                participants.group.to_numpy()
            )

            matrix = panel[features].to_numpy(dtype=float)

            assert matrix.shape == (expected, 5)
            assert np.isfinite(matrix).all()

            panels[(name, seconds)] = matrix

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

            for name in representations:

                x = panels[(name, seconds)]

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
                    error_score="raise",
                    refit=True
                )

                search.fit(x[train], y[train])

                probabilities = (
                    search.predict_proba(x[test])[:, 1]
                )

                for j, subject_index in enumerate(test):

                    probability = float(probabilities[j])

                    predictions.append({
                        "task": task,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id":
                            str(subjects[subject_index]),
                        "y_true": int(y[subject_index]),
                        "representation": name,
                        "seconds": seconds,
                        "probability_ad": probability,
                        "y_pred": int(probability >= 0.5),
                        "best_C":
                            float(search.best_params_["clf__C"])
                    })

    print(
        f"Completed {task}: {expected} participants",
        flush=True
    )

pred = pd.DataFrame(predictions)

assert len(pred) == (65 + 59) * 3 * 5 * 3

assert not pred.duplicated([
    "task",
    "repeat",
    "participant_id",
    "representation",
    "seconds"
]).any()

# Independent reproducibility check against the previously
# saved fixed-duration results.
previous = pd.read_csv(REFERENCE)

previous = previous[
    previous.policy.str.startswith("fixed_")
][[
    "task",
    "repeat",
    "participant_id",
    "stop_seconds",
    "probability_ad"
]].rename(columns={
    "stop_seconds": "seconds",
    "probability_ad": "previous_probability"
})

rerun = pred[
    pred.representation == "pilot_global_psd"
]

check = rerun.merge(
    previous,
    on=["task", "repeat", "participant_id", "seconds"],
    validate="one_to_one"
)

assert len(check) == len(rerun)

maximum_difference = float(
    np.max(
        np.abs(
            check.probability_ad
            - check.previous_probability
        )
    )
)

print(
    "Original prediction replication max difference:",
    maximum_difference,
    flush=True
)

assert maximum_difference < 1e-8, (
    "Previous pilot could not be reproduced"
)

metrics = []

for keys, frame in pred.groupby(
    ["task", "repeat", "representation", "seconds"]
):

    task, repeat, representation, seconds = keys

    y_true = frame.y_true.to_numpy(dtype=int)
    y_pred = frame.y_pred.to_numpy(dtype=int)
    prob = frame.probability_ad.to_numpy(dtype=float)

    metrics.append({
        "task": task,
        "repeat": int(repeat),
        "representation": representation,
        "seconds": int(seconds),
        "participants": len(frame),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity_ad": float(
            recall_score(y_true, y_pred, pos_label=1)
        ),
        "specificity_nonad": float(
            recall_score(y_true, y_pred, pos_label=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, prob)
        ),
        "brier_score": float(
            brier_score_loss(y_true, prob)
        )
    })

results = pd.DataFrame(metrics)

assert len(results) == 90

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
print("===== FEATURE REPRESENTATION COMPARISON =====")
print("OOF prediction rows:", len(pred))
print("Metric rows:", len(results))

summary = (
    results.groupby(
        ["task", "representation", "seconds"]
    )[
        ["balanced_accuracy", "sensitivity_ad", "roc_auc"]
    ]
    .mean()
    .round(3)
)

print(summary.to_string())

print()
print("INTERPRETATION: EXPLORATORY")
print("No final feature method selected.")
print("Repeated CV estimates are correlated.")
print("No p-values or clinical claims.")
print("P2 FEATURE REPRESENTATION CV COMPLETE")
