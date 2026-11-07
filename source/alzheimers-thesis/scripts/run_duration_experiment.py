from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
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

FEATURE_FILE = (
    ROOT
    / "results"
    / "duration_features"
    / "eeg_duration_features.csv"
)

OUT = (
    ROOT
    / "results"
    / "duration_experiment"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = int(
    CONFIG["random_seed"]
)

DURATIONS = list(
    CONFIG["durations_seconds"]
)

ANCHORS = list(
    CONFIG["anchors"]
)

C_GRID = list(
    CONFIG["logistic_C"]
)

N_SPLITS = int(
    CONFIG["outer_folds"]
)

N_REPEATS = int(
    CONFIG["outer_repeats"]
)

INNER_SPLITS = int(
    CONFIG["inner_folds"]
)

BOOTSTRAPS = int(
    CONFIG["bootstrap_samples"]
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


def metrics(
    y: np.ndarray,
    prob: np.ndarray,
) -> dict:

    pred = (
        prob >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1],
        )
        .ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    return {
        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),
        "roc_auc":
            roc_auc_score(
                y,
                prob,
            ),
        "sensitivity":
            sensitivity,
        "specificity":
            specificity,
        "f1":
            f1_score(
                y,
                pred,
                zero_division=0,
            ),
        "mcc":
            matthews_corrcoef(
                y,
                pred,
            ),
        "brier":
            brier_score_loss(
                y,
                prob,
            ),
    }


def model_grid() -> GridSearchCV:

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
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__C": C_GRID
        },
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )


def stratified_bootstrap_delta(
    y: np.ndarray,
    p_short: np.ndarray,
    p_ref: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
    float,
]:

    observed = (
        metrics(
            y,
            p_short,
        )[
            "balanced_accuracy"
        ]
        -
        metrics(
            y,
            p_ref,
        )[
            "balanced_accuracy"
        ]
    )

    idx0 = np.where(
        y == 0
    )[0]

    idx1 = np.where(
        y == 1
    )[0]

    deltas = np.empty(
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

        idx = np.concatenate(
            [
                sample0,
                sample1,
            ]
        )

        deltas[b] = (
            metrics(
                y[idx],
                p_short[idx],
            )[
                "balanced_accuracy"
            ]
            -
            metrics(
                y[idx],
                p_ref[idx],
            )[
                "balanced_accuracy"
            ]
        )

    low, high = np.percentile(
        deltas,
        [2.5, 97.5],
    )

    return (
        float(observed),
        float(low),
        float(high),
    )


def make_dataset(
    df: pd.DataFrame,
    task: str,
    duration: int,
    anchor: str,
    feature_columns: list[str],
):

    spec = TASKS[task]

    subset = (
        df[
            (
                df["Group"]
                .isin(
                    [
                        spec["positive"],
                        spec["negative"],
                    ]
                )
            )
            & (
                df[
                    "duration_seconds"
                ]
                == duration
            )
            & (
                df["anchor"]
                == anchor
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

    y = (
        subset["Group"]
        == spec["positive"]
    ).astype(int).to_numpy()

    X = (
        subset[
            feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    ids = (
        subset[
            "participant_id"
        ]
        .to_numpy()
    )

    return (
        subset,
        X,
        y,
        ids,
    )


def make_splits(
    y: np.ndarray,
):

    splitter = (
        RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=SEED,
        )
    )

    dummy = np.zeros(
        (
            len(y),
            1,
        )
    )

    return list(
        splitter.split(
            dummy,
            y,
        )
    )


def evaluate_condition(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    splits,
    task: str,
    duration: int,
    anchor: str,
):

    prediction_rows = []
    repeat_metrics = []

    for outer_index, (
        train_idx,
        test_idx,
    ) in enumerate(splits):

        repeat = (
            outer_index
            // N_SPLITS
        )

        fold = (
            outer_index
            % N_SPLITS
        )

        grid = model_grid()

        grid.fit(
            X[train_idx],
            y[train_idx],
        )

        prob = (
            grid
            .predict_proba(
                X[test_idx]
            )[:, 1]
        )

        pred = (
            prob >= 0.5
        ).astype(int)

        for local_i, idx in enumerate(
            test_idx
        ):

            prediction_rows.append(
                {
                    "task": task,
                    "duration_seconds":
                        duration,
                    "anchor": anchor,
                    "repeat": repeat,
                    "fold": fold,
                    "participant_id":
                        ids[idx],
                    "label":
                        int(y[idx]),
                    "probability":
                        float(
                            prob[local_i]
                        ),
                    "prediction":
                        int(
                            pred[local_i]
                        ),
                    "best_C":
                        float(
                            grid.best_params_[
                                "model__C"
                            ]
                        ),
                }
            )

    pred_df = pd.DataFrame(
        prediction_rows
    )

    for repeat, part in (
        pred_df.groupby(
            "repeat"
        )
    ):

        m = metrics(
            part[
                "label"
            ].to_numpy(),
            part[
                "probability"
            ].to_numpy(),
        )

        m.update(
            {
                "task": task,
                "duration_seconds":
                    duration,
                "anchor":
                    anchor,
                "repeat":
                    int(repeat),
            }
        )

        repeat_metrics.append(
            m
        )

    return (
        pred_df,
        pd.DataFrame(
            repeat_metrics
        ),
    )


def main():

    df = pd.read_csv(
        FEATURE_FILE
    )

    metadata_cols = {
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
        for c in df.columns
        if c not in metadata_cols
    ]

    print(
        "EEG feature count:",
        len(feature_columns),
    )

    all_predictions = []
    all_repeat_metrics = []
    all_summary = []
    bootstrap_rows = []
    demographic_rows = []

    for task in TASKS:

        print()
        print(
            "TASK:",
            task,
        )

        (
            base_subset,
            _,
            base_y,
            base_ids,
        ) = make_dataset(
            df,
            task,
            DURATIONS[0],
            ANCHORS[0],
            feature_columns,
        )

        splits = make_splits(
            base_y
        )

        expected_ids = list(
            base_ids
        )

        for anchor in ANCHORS:

            for duration in DURATIONS:

                (
                    subset,
                    X,
                    y,
                    ids,
                ) = make_dataset(
                    df,
                    task,
                    duration,
                    anchor,
                    feature_columns,
                )

                if list(
                    ids
                ) != expected_ids:

                    raise RuntimeError(
                        "Participant order changed "
                        f"for {task} "
                        f"{duration} {anchor}"
                    )

                print(
                    f"  {anchor:6s} "
                    f"{duration:3d}s "
                    f"N={len(y)}"
                )

                (
                    pred_df,
                    repeat_df,
                ) = evaluate_condition(
                    X=X,
                    y=y,
                    ids=ids,
                    splits=splits,
                    task=task,
                    duration=duration,
                    anchor=anchor,
                )

                all_predictions.append(
                    pred_df
                )

                all_repeat_metrics.append(
                    repeat_df
                )

                averaged = (
                    pred_df
                    .groupby(
                        [
                            "participant_id",
                            "label",
                        ],
                        as_index=False,
                    )[
                        "probability"
                    ]
                    .mean()
                    .sort_values(
                        "participant_id"
                    )
                )

                m = metrics(
                    averaged[
                        "label"
                    ].to_numpy(),
                    averaged[
                        "probability"
                    ].to_numpy(),
                )

                m.update(
                    {
                        "task": task,
                        "duration_seconds":
                            duration,
                        "anchor":
                            anchor,
                        "n_participants":
                            len(averaged),
                    }
                )

                all_summary.append(
                    m
                )

        # Demographic-only baseline
        demo_subset = (
            base_subset
            .sort_values(
                "participant_id"
            )
            .reset_index(
                drop=True
            )
        )

        demo_X = np.column_stack(
            [
                demo_subset[
                    "Age"
                ].to_numpy(
                    dtype=float
                ),
                (
                    demo_subset[
                        "Gender"
                    ]
                    == "F"
                ).astype(
                    float
                ).to_numpy(),
            ]
        )

        (
            demo_pred,
            _,
        ) = evaluate_condition(
            X=demo_X,
            y=base_y,
            ids=base_ids,
            splits=splits,
            task=task,
            duration=-1,
            anchor="demographic",
        )

        demo_avg = (
            demo_pred
            .groupby(
                [
                    "participant_id",
                    "label",
                ],
                as_index=False,
            )[
                "probability"
            ]
            .mean()
        )

        demo_m = metrics(
            demo_avg[
                "label"
            ].to_numpy(),
            demo_avg[
                "probability"
            ].to_numpy(),
        )

        demo_m[
            "task"
        ] = task

        demographic_rows.append(
            demo_m
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    repeat_metrics = pd.concat(
        all_repeat_metrics,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        all_summary
    )

    demographics = pd.DataFrame(
        demographic_rows
    )

    predictions.to_csv(
        OUT
        / "oof_predictions.csv",
        index=False,
    )

    repeat_metrics.to_csv(
        OUT
        / "repeat_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "condition_summary.csv",
        index=False,
    )

    demographics.to_csv(
        OUT
        / "demographic_baseline.csv",
        index=False,
    )

    # ---------------------------------------------
    # Paired bootstrap vs 300 s
    # ---------------------------------------------

    averaged_predictions = (
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
        )[
            "probability"
        ]
        .mean()
    )

    rng = np.random.default_rng(
        SEED
    )

    for task in TASKS:

        for anchor in ANCHORS:

            reference = (
                averaged_predictions[
                    (
                        averaged_predictions[
                            "task"
                        ]
                        == task
                    )
                    & (
                        averaged_predictions[
                            "anchor"
                        ]
                        == anchor
                    )
                    & (
                        averaged_predictions[
                            "duration_seconds"
                        ]
                        == 300
                    )
                ]
                .sort_values(
                    "participant_id"
                )
            )

            for duration in DURATIONS:

                if duration == 300:
                    continue

                short = (
                    averaged_predictions[
                        (
                            averaged_predictions[
                                "task"
                            ]
                            == task
                        )
                        & (
                            averaged_predictions[
                                "anchor"
                            ]
                            == anchor
                        )
                        & (
                            averaged_predictions[
                                "duration_seconds"
                            ]
                            == duration
                        )
                    ]
                    .sort_values(
                        "participant_id"
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
                        "Bootstrap pairing failure"
                    )

                y = short[
                    "label"
                ].to_numpy()

                delta, low, high = (
                    stratified_bootstrap_delta(
                        y=y,
                        p_short=short[
                            "probability"
                        ].to_numpy(),
                        p_ref=reference[
                            "probability"
                        ].to_numpy(),
                        rng=rng,
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

    pd.DataFrame(
        bootstrap_rows
    ).to_csv(
        OUT
        / "paired_bootstrap_vs_300.csv",
        index=False,
    )

    # ---------------------------------------------
    # Permutation negative control
    # AD vs CN, 300 s center
    # ---------------------------------------------

    task = "AD_vs_CN"

    (
        perm_subset,
        X,
        y,
        ids,
    ) = make_dataset(
        df,
        task,
        300,
        "center",
        feature_columns,
    )

    one_repeat_splitter = (
        StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        )
    )

    observed_pipeline = (
        model_grid()
    )

    observed_probs = np.zeros(
        len(y),
        dtype=float,
    )

    for train_idx, test_idx in (
        one_repeat_splitter.split(
            X,
            y,
        )
    ):

        observed_pipeline = (
            model_grid()
        )

        observed_pipeline.fit(
            X[train_idx],
            y[train_idx],
        )

        observed_probs[
            test_idx
        ] = (
            observed_pipeline
            .predict_proba(
                X[test_idx]
            )[:, 1]
        )

    observed_ba = metrics(
        y,
        observed_probs,
    )[
        "balanced_accuracy"
    ]

    permutation_rows = []

    perm_rng = (
        np.random.default_rng(
            SEED + 99
        )
    )

    n_perm = int(
        CONFIG[
            "permutation_samples"
        ]
    )

    for permutation in range(
        n_perm
    ):

        y_perm = (
            perm_rng
            .permutation(y)
        )

        splitter = (
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=(
                    SEED
                    + permutation
                    + 1000
                ),
            )
        )

        probs = np.zeros(
            len(y_perm),
            dtype=float,
        )

        for train_idx, test_idx in (
            splitter.split(
                X,
                y_perm,
            )
        ):

            grid = model_grid()

            grid.fit(
                X[train_idx],
                y_perm[train_idx],
            )

            probs[
                test_idx
            ] = (
                grid
                .predict_proba(
                    X[test_idx]
                )[:, 1]
            )

        ba = metrics(
            y_perm,
            probs,
        )[
            "balanced_accuracy"
        ]

        permutation_rows.append(
            {
                "permutation":
                    permutation,
                "balanced_accuracy":
                    ba,
            }
        )

        if (
            permutation + 1
        ) % 10 == 0:
            print(
                "Permutation",
                permutation + 1,
                "/",
                n_perm,
            )

    perm_df = pd.DataFrame(
        permutation_rows
    )

    empirical_p = (
        1
        + int(
            (
                perm_df[
                    "balanced_accuracy"
                ]
                >= observed_ba
            ).sum()
        )
    ) / (
        n_perm + 1
    )

    perm_df.to_csv(
        OUT
        / "permutation_null.csv",
        index=False,
    )

    (
        OUT
        / "permutation_summary.json"
    ).write_text(
        json.dumps(
            {
                "observed_balanced_accuracy":
                    float(
                        observed_ba
                    ),
                "null_mean":
                    float(
                        perm_df[
                            "balanced_accuracy"
                        ].mean()
                    ),
                "null_std":
                    float(
                        perm_df[
                            "balanced_accuracy"
                        ].std()
                    ),
                "empirical_p":
                    float(
                        empirical_p
                    ),
                "permutations":
                    n_perm,
            },
            indent=2,
        )
    )

    print()
    print(
        "======================================"
    )
    print(
        "PRIMARY CONDITION SUMMARY"
    )
    print(
        "======================================"
    )

    primary = (
        summary[
            summary["task"]
            == "AD_vs_CN"
        ]
        .sort_values(
            [
                "anchor",
                "duration_seconds",
            ]
        )
    )

    print(
        primary[
            [
                "duration_seconds",
                "anchor",
                "balanced_accuracy",
                "roc_auc",
                "sensitivity",
                "specificity",
                "mcc",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "DEMOGRAPHIC BASELINE"
    )

    print(
        demographics
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "PERMUTATION SUMMARY"
    )

    print(
        (
            OUT
            / "permutation_summary.json"
        ).read_text()
    )


if __name__ == "__main__":
    main()
