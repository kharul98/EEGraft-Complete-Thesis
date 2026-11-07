from __future__ import annotations

from pathlib import Path
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

CONFIG = json.loads(
    (
        ROOT
        / "configs"
        / "eeg_duration_protocol.json"
    ).read_text()
)

RESULT_DIR = (
    ROOT
    / "results"
    / "duration_experiment"
)

FEATURE_FILE = (
    ROOT
    / "results"
    / "duration_features"
    / "eeg_duration_features.csv"
)

PRED_FILE = (
    RESULT_DIR
    / "oof_predictions.csv"
)

SEED = int(CONFIG["random_seed"])
BOOTSTRAPS = int(CONFIG["bootstrap_samples"])
PERMUTATIONS = int(CONFIG["permutation_samples"])
C_GRID = list(CONFIG["logistic_C"])

DURATIONS = list(
    CONFIG["durations_seconds"]
)

ANCHORS = list(
    CONFIG["anchors"]
)

TASKS = {
    "AD_vs_CN": {
        "positive": "A",
        "negative": "C",
    },
    "AD_vs_FTD": {
        "positive": "A",
        "negative": "F",
    },
}


def balanced_accuracy(
    y: np.ndarray,
    probability: np.ndarray,
) -> float:

    pred = (
        probability >= 0.5
    ).astype(int)

    return float(
        balanced_accuracy_score(
            y,
            pred,
        )
    )


def validate_predictions(
    predictions: pd.DataFrame,
) -> None:

    expected_tasks = {
        "AD_vs_CN": 65,
        "AD_vs_FTD": 59,
    }

    required = {
        "task",
        "duration_seconds",
        "anchor",
        "repeat",
        "participant_id",
        "label",
        "probability",
    }

    missing = (
        required
        - set(predictions.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing prediction columns: {missing}"
        )

    if predictions[
        "probability"
    ].isna().any():
        raise RuntimeError(
            "NaNs found in probabilities."
        )

    for task, n_subjects in (
        expected_tasks.items()
    ):

        part = predictions[
            predictions["task"] == task
        ]

        expected_rows = (
            n_subjects
            * len(DURATIONS)
            * len(ANCHORS)
            * 10
        )

        if len(part) != expected_rows:
            raise RuntimeError(
                f"{task}: expected "
                f"{expected_rows} prediction rows, "
                f"got {len(part)}"
            )

        counts = (
            part
            .groupby(
                [
                    "duration_seconds",
                    "anchor",
                    "participant_id",
                ]
            )
            .size()
        )

        if not (
            counts == 10
        ).all():
            raise RuntimeError(
                f"{task}: not every participant "
                "has exactly 10 OOF predictions "
                "per condition."
            )

    print(
        "OOF prediction completeness: PASS"
    )


def bootstrap_delta(
    y: np.ndarray,
    short_probability: np.ndarray,
    reference_probability: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:

    observed = (
        balanced_accuracy(
            y,
            short_probability,
        )
        -
        balanced_accuracy(
            y,
            reference_probability,
        )
    )

    idx0 = np.flatnonzero(
        y == 0
    )

    idx1 = np.flatnonzero(
        y == 1
    )

    values = np.empty(
        BOOTSTRAPS,
        dtype=float,
    )

    for b in range(
        BOOTSTRAPS
    ):

        sample0 = rng.choice(
            idx0,
            size=len(idx0),
            replace=True,
        )

        sample1 = rng.choice(
            idx1,
            size=len(idx1),
            replace=True,
        )

        sample = np.concatenate(
            [
                sample0,
                sample1,
            ]
        )

        values[b] = (
            balanced_accuracy(
                y[sample],
                short_probability[sample],
            )
            -
            balanced_accuracy(
                y[sample],
                reference_probability[sample],
            )
        )

    low, high = np.percentile(
        values,
        [2.5, 97.5],
    )

    return (
        float(observed),
        float(low),
        float(high),
    )


def make_pipeline() -> GridSearchCV:

    pipeline = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    inner = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED,
    )

    return GridSearchCV(
        pipeline,
        {
            "model__C": C_GRID
        },
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=1,
        refit=True,
    )


def nested_5fold_score(
    X: np.ndarray,
    y: np.ndarray,
    split_seed: int,
) -> float:

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=split_seed,
    )

    probabilities = np.zeros(
        len(y),
        dtype=float,
    )

    for train_idx, test_idx in (
        splitter.split(
            X,
            y,
        )
    ):

        grid = make_pipeline()

        grid.fit(
            X[train_idx],
            y[train_idx],
        )

        probabilities[
            test_idx
        ] = (
            grid
            .predict_proba(
                X[test_idx]
            )[:, 1]
        )

    return balanced_accuracy(
        y,
        probabilities,
    )


def one_permutation(
    permutation_index: int,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:

    rng = np.random.default_rng(
        SEED
        + 10000
        + permutation_index
    )

    shuffled_y = rng.permutation(
        y
    )

    score = nested_5fold_score(
        X,
        shuffled_y,
        SEED
        + 1000
        + permutation_index,
    )

    return {
        "permutation":
            permutation_index,
        "balanced_accuracy":
            float(score),
    }


def main() -> None:

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PRED_FILE
    )

    validate_predictions(
        predictions
    )

    averaged = (
        predictions
        .groupby(
            [
                "task",
                "duration_seconds",
                "anchor",
                "participant_id",
                "label",
            ],
            as_index=False,
        )["probability"]
        .mean()
    )

    rng = np.random.default_rng(
        SEED
    )

    bootstrap_rows = []

    for task in TASKS:

        for anchor in ANCHORS:

            reference = (
                averaged[
                    (
                        averaged[
                            "task"
                        ]
                        == task
                    )
                    & (
                        averaged[
                            "anchor"
                        ]
                        == anchor
                    )
                    & (
                        averaged[
                            "duration_seconds"
                        ]
                        == 300
                    )
                ]
                .sort_values(
                    "participant_id"
                )
                .reset_index(
                    drop=True
                )
            )

            for duration in (
                30,
                60,
                120,
                180,
            ):

                short = (
                    averaged[
                        (
                            averaged[
                                "task"
                            ]
                            == task
                        )
                        & (
                            averaged[
                                "anchor"
                            ]
                            == anchor
                        )
                        & (
                            averaged[
                                "duration_seconds"
                            ]
                            == duration
                        )
                    ]
                    .sort_values(
                        "participant_id"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                if not np.array_equal(
                    short[
                        "participant_id"
                    ].to_numpy(),
                    reference[
                        "participant_id"
                    ].to_numpy(),
                ):
                    raise RuntimeError(
                        "Participant pairing failure: "
                        f"{task} {anchor} {duration}"
                    )

                if not np.array_equal(
                    short[
                        "label"
                    ].to_numpy(),
                    reference[
                        "label"
                    ].to_numpy(),
                ):
                    raise RuntimeError(
                        "Label pairing failure."
                    )

                delta, low, high = (
                    bootstrap_delta(
                        short[
                            "label"
                        ].to_numpy(),
                        short[
                            "probability"
                        ].to_numpy(),
                        reference[
                            "probability"
                        ].to_numpy(),
                        rng,
                    )
                )

                bootstrap_rows.append(
                    {
                        "task":
                            task,
                        "anchor":
                            anchor,
                        "duration_seconds":
                            duration,
                        "reference_seconds":
                            300,
                        "balanced_accuracy_delta":
                            delta,
                        "ci95_low":
                            low,
                        "ci95_high":
                            high,
                    }
                )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap.to_csv(
        RESULT_DIR
        / "paired_bootstrap_vs_300.csv",
        index=False,
    )

    print()
    print(
        "PAIRED BOOTSTRAP COMPLETE"
    )
    print(
        "Rows:",
        len(bootstrap),
    )

    print()
    print(
        bootstrap
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------
    # Negative control
    # -----------------------------------------

    features = pd.read_csv(
        FEATURE_FILE
    )

    metadata = {
        "participant_id",
        "Group",
        "Age",
        "Gender",
        "MMSE",
        "duration_seconds",
        "anchor",
        "window_start_seconds",
        "window_stop_seconds",
        "recording_seconds",
    }

    feature_columns = [
        c
        for c in features.columns
        if c not in metadata
    ]

    subset = (
        features[
            (
                features[
                    "Group"
                ]
                .isin(
                    ["A", "C"]
                )
            )
            & (
                features[
                    "duration_seconds"
                ]
                == 300
            )
            & (
                features[
                    "anchor"
                ]
                == "center"
            )
        ]
        .copy()
        .sort_values(
            "participant_id"
        )
        .reset_index(
            drop=True
        )
    )

    X = (
        subset[
            feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        subset["Group"]
        == "A"
    ).astype(
        int
    ).to_numpy()

    if len(y) != 65:
        raise RuntimeError(
            f"Expected 65 AD/CN participants; "
            f"got {len(y)}"
        )

    print()
    print(
        "Running observed 5-fold "
        "nested negative-control reference..."
    )

    observed = nested_5fold_score(
        X,
        y,
        SEED,
    )

    print(
        "Observed balanced accuracy:",
        round(
            observed,
            6,
        ),
    )

    print()
    print(
        f"Running {PERMUTATIONS} "
        "prespecified label permutations..."
    )

    permutation_rows = Parallel(
        n_jobs=6,
        verbose=10,
    )(
        delayed(
            one_permutation
        )(
            i,
            X,
            y,
        )
        for i in range(
            PERMUTATIONS
        )
    )

    permutation_df = (
        pd.DataFrame(
            permutation_rows
        )
        .sort_values(
            "permutation"
        )
        .reset_index(
            drop=True
        )
    )

    permutation_df.to_csv(
        RESULT_DIR
        / "permutation_null.csv",
        index=False,
    )

    empirical_p = (
        1
        + int(
            (
                permutation_df[
                    "balanced_accuracy"
                ]
                >= observed
            )
            .sum()
        )
    ) / (
        PERMUTATIONS + 1
    )

    summary = {
        "condition":
            "AD_vs_CN_300s_center",
        "observed_balanced_accuracy":
            float(observed),
        "null_mean":
            float(
                permutation_df[
                    "balanced_accuracy"
                ].mean()
            ),
        "null_std":
            float(
                permutation_df[
                    "balanced_accuracy"
                ].std(
                    ddof=1
                )
            ),
        "null_min":
            float(
                permutation_df[
                    "balanced_accuracy"
                ].min()
            ),
        "null_max":
            float(
                permutation_df[
                    "balanced_accuracy"
                ].max()
            ),
        "empirical_p":
            float(empirical_p),
        "permutations":
            int(PERMUTATIONS),
    }

    (
        RESULT_DIR
        / "permutation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "PERMUTATION CONTROL COMPLETE"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # -----------------------------------------
    # Final structural verification
    # -----------------------------------------

    if len(bootstrap) != 24:
        raise RuntimeError(
            "Expected 24 bootstrap rows."
        )

    if len(
        permutation_df
    ) != 100:
        raise RuntimeError(
            "Expected 100 permutations."
        )

    if not (
        0.0
        <= empirical_p
        <= 1.0
    ):
        raise RuntimeError(
            "Invalid empirical p-value."
        )

    print()
    print(
        "ALL RECOVERY CHECKS PASSED"
    )


if __name__ == "__main__":
    main()
