from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
OUT = RESULTS / "final_robustness"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

CONFIG = json.loads(
    (
        ROOT
        / "configs"
        / "eeg_duration_protocol.json"
    ).read_text()
)

SEED = int(CONFIG["random_seed"])
BOOTSTRAPS = 5000

predictions = pd.read_csv(
    RESULTS
    / "duration_experiment"
    / "oof_predictions.csv"
)

condition_summary = pd.read_csv(
    RESULTS
    / "duration_experiment"
    / "condition_summary.csv"
)

demographic_baseline = pd.read_csv(
    RESULTS
    / "duration_experiment"
    / "demographic_baseline.csv"
)

features = pd.read_csv(
    RESULTS
    / "duration_features"
    / "eeg_duration_features.csv"
)


def balanced_accuracy(y, probability):
    return balanced_accuracy_score(
        y,
        (probability >= 0.5).astype(int),
    )


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    return float(
        (
            (x[:, None] > y[None, :]).sum()
            - (x[:, None] < y[None, :]).sum()
        )
        / (len(x) * len(y))
    )


def bh_adjust(p):
    p = np.asarray(p, dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    n = len(p)

    adjusted = (
        ranked
        * n
        / np.arange(1, n + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    result = np.empty(n)
    result[order] = adjusted

    return result


# ==================================================
# Average repeated OOF predictions per participant
# ==================================================

avg = (
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


# ==================================================
# 1. Absolute stratified participant bootstrap CIs
# ==================================================

absolute_rows = []

rng = np.random.default_rng(
    SEED + 50000
)

for keys, part in avg.groupby(
    [
        "task",
        "duration_seconds",
        "anchor",
    ]
):

    task, duration, anchor = keys

    part = (
        part
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    y = part["label"].to_numpy()
    p = part["probability"].to_numpy()

    observed_ba = balanced_accuracy(y, p)
    observed_auc = roc_auc_score(y, p)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    ba_boot = np.empty(
        BOOTSTRAPS,
        dtype=float,
    )

    auc_boot = np.empty(
        BOOTSTRAPS,
        dtype=float,
    )

    for b in range(BOOTSTRAPS):

        s0 = rng.choice(
            idx0,
            size=len(idx0),
            replace=True,
        )

        s1 = rng.choice(
            idx1,
            size=len(idx1),
            replace=True,
        )

        idx = np.concatenate(
            [s0, s1]
        )

        ba_boot[b] = balanced_accuracy(
            y[idx],
            p[idx],
        )

        auc_boot[b] = roc_auc_score(
            y[idx],
            p[idx],
        )

    ba_low, ba_high = np.percentile(
        ba_boot,
        [2.5, 97.5],
    )

    auc_low, auc_high = np.percentile(
        auc_boot,
        [2.5, 97.5],
    )

    absolute_rows.append(
        {
            "task": task,
            "duration_seconds": int(duration),
            "anchor": anchor,
            "balanced_accuracy":
                float(observed_ba),
            "ba_ci95_low":
                float(ba_low),
            "ba_ci95_high":
                float(ba_high),
            "roc_auc":
                float(observed_auc),
            "auc_ci95_low":
                float(auc_low),
            "auc_ci95_high":
                float(auc_high),
        }
    )

absolute = pd.DataFrame(
    absolute_rows
).sort_values(
    [
        "task",
        "anchor",
        "duration_seconds",
    ]
)

absolute.to_csv(
    TABLES
    / "table_absolute_performance_ci.csv",
    index=False,
)


# ==================================================
# 2. Temporal-anchor variability per participant
# ==================================================

anchor_subject_rows = []
anchor_summary_rows = []

for (task, duration), part in avg.groupby(
    [
        "task",
        "duration_seconds",
    ]
):

    pivot = part.pivot(
        index=[
            "participant_id",
            "label",
        ],
        columns="anchor",
        values="probability",
    )

    expected = {
        "start",
        "center",
        "end",
    }

    if set(pivot.columns) != expected:
        raise RuntimeError(
            f"Missing anchors for {task} {duration}"
        )

    probability_range = (
        pivot.max(axis=1)
        - pivot.min(axis=1)
    )

    probability_sd = pivot.std(
        axis=1,
        ddof=0,
    )

    hard = (
        pivot >= 0.5
    ).astype(int)

    all_same = (
        hard.nunique(axis=1)
        == 1
    )

    for idx, row in pivot.iterrows():

        participant_id, label = idx

        anchor_subject_rows.append(
            {
                "task": task,
                "duration_seconds":
                    int(duration),
                "participant_id":
                    participant_id,
                "label":
                    int(label),
                "probability_range":
                    float(
                        probability_range.loc[idx]
                    ),
                "probability_sd":
                    float(
                        probability_sd.loc[idx]
                    ),
                "same_classification_all_anchors":
                    bool(
                        all_same.loc[idx]
                    ),
            }
        )

    anchor_summary_rows.append(
        {
            "task":
                task,
            "duration_seconds":
                int(duration),
            "mean_probability_range":
                float(
                    probability_range.mean()
                ),
            "median_probability_range":
                float(
                    probability_range.median()
                ),
            "mean_probability_sd":
                float(
                    probability_sd.mean()
                ),
            "all_anchor_classification_agreement_rate":
                float(
                    all_same.mean()
                ),
        }
    )

anchor_subject = pd.DataFrame(
    anchor_subject_rows
)

anchor_summary = pd.DataFrame(
    anchor_summary_rows
).sort_values(
    [
        "task",
        "duration_seconds",
    ]
)

anchor_subject.to_csv(
    OUT
    / "participant_anchor_variability.csv",
    index=False,
)

anchor_summary.to_csv(
    TABLES
    / "table_anchor_probability_variability.csv",
    index=False,
)


# ==================================================
# 3. Regional relative-power effects
# ==================================================

regions = [
    "frontal",
    "central",
    "temporal",
    "parietal",
    "occipital",
]

bands = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]

regional_features = [
    f"{region}_{band}"
    for region in regions
    for band in bands
]

tasks = {
    "AD_vs_CN": ("A", "C"),
    "AD_vs_FTD": ("A", "F"),
}

regional_rows = []

for task, (
    positive_group,
    negative_group,
) in tasks.items():

    for duration in [180, 300]:

        condition = features[
            (
                features["duration_seconds"]
                == duration
            )
            & (
                features["anchor"]
                == "center"
            )
            & (
                features["Group"].isin(
                    [
                        positive_group,
                        negative_group,
                    ]
                )
            )
        ]

        temp_rows = []

        for feature in regional_features:

            x = (
                condition[
                    condition["Group"]
                    == positive_group
                ][feature]
                .to_numpy(dtype=float)
            )

            y = (
                condition[
                    condition["Group"]
                    == negative_group
                ][feature]
                .to_numpy(dtype=float)
            )

            test = mannwhitneyu(
                x,
                y,
                alternative="two-sided",
            )

            region, band = feature.split(
                "_",
                1,
            )

            temp_rows.append(
                {
                    "task": task,
                    "duration_seconds":
                        duration,
                    "anchor": "center",
                    "region": region,
                    "band": band,
                    "feature": feature,
                    "positive_mean":
                        float(np.mean(x)),
                    "negative_mean":
                        float(np.mean(y)),
                    "cliffs_delta":
                        cliffs_delta(x, y),
                    "mannwhitney_p":
                        float(test.pvalue),
                }
            )

        pvals = [
            row["mannwhitney_p"]
            for row in temp_rows
        ]

        qvals = bh_adjust(pvals)

        for row, q in zip(
            temp_rows,
            qvals,
        ):
            row[
                "bh_q_within_task_duration"
            ] = float(q)

        regional_rows.extend(
            temp_rows
        )

regional = pd.DataFrame(
    regional_rows
)

regional.to_csv(
    TABLES
    / "table_exploratory_regional_effects.csv",
    index=False,
)


# ==================================================
# Regional effect heatmaps
# ==================================================

for task in tasks:

    for duration in [180, 300]:

        part = regional[
            (
                regional["task"]
                == task
            )
            & (
                regional[
                    "duration_seconds"
                ]
                == duration
            )
        ]

        matrix = (
            part
            .pivot(
                index="region",
                columns="band",
                values="cliffs_delta",
            )
            .reindex(
                index=regions,
                columns=bands,
            )
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            vmin=-1,
            vmax=1,
        )

        ax.set_xticks(
            np.arange(len(bands))
        )

        ax.set_xticklabels(
            bands
        )

        ax.set_yticks(
            np.arange(len(regions))
        )

        ax.set_yticklabels(
            regions
        )

        ax.set_title(
            f"{task}: regional spectral effects, "
            f"{duration}s center"
        )

        for i in range(
            len(regions)
        ):
            for j in range(
                len(bands)
            ):
                ax.text(
                    j,
                    i,
                    f"{matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                )

        fig.colorbar(
            image,
            ax=ax,
            label="Cliff's delta",
        )

        fig.tight_layout()

        fig.savefig(
            FIGURES
            / (
                "fig07_"
                f"{task.lower()}_"
                f"regional_{duration}s.png"
            ),
            dpi=300,
        )

        plt.close(fig)


# ==================================================
# 4. EEG + age/sex sensitivity
# ==================================================

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

eeg_columns = [
    c
    for c in features.columns
    if c not in metadata_cols
]

C_GRID = list(
    CONFIG["logistic_C"]
)

outer_folds = int(
    CONFIG["outer_folds"]
)

outer_repeats = int(
    CONFIG["outer_repeats"]
)

inner_folds = int(
    CONFIG["inner_folds"]
)


def make_grid():
    pipe = Pipeline(
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
        n_splits=inner_folds,
        shuffle=True,
        random_state=SEED,
    )

    return GridSearchCV(
        pipe,
        {
            "model__C": C_GRID
        },
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )


def evaluate_combined(
    subset,
):

    subset = (
        subset
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    y = (
        subset["Group"]
        == "A"
    ).astype(int).to_numpy()

    sex = (
        subset["Gender"]
        == "F"
    ).astype(float).to_numpy()

    X = np.column_stack(
        [
            subset[eeg_columns]
            .to_numpy(dtype=float),
            subset["Age"]
            .to_numpy(dtype=float),
            sex,
        ]
    )

    ids = (
        subset["participant_id"]
        .to_numpy()
    )

    outer = RepeatedStratifiedKFold(
        n_splits=outer_folds,
        n_repeats=outer_repeats,
        random_state=SEED,
    )

    rows = []

    for outer_index, (
        train,
        test,
    ) in enumerate(
        outer.split(X, y)
    ):

        repeat = (
            outer_index
            // outer_folds
        )

        grid = make_grid()

        grid.fit(
            X[train],
            y[train],
        )

        probability = (
            grid
            .predict_proba(
                X[test]
            )[:, 1]
        )

        for local_i, index in enumerate(
            test
        ):
            rows.append(
                {
                    "repeat":
                        repeat,
                    "participant_id":
                        ids[index],
                    "label":
                        int(y[index]),
                    "probability":
                        float(
                            probability[
                                local_i
                            ]
                        ),
                }
            )

    predictions = pd.DataFrame(
        rows
    )

    averaged = (
        predictions
        .groupby(
            [
                "participant_id",
                "label",
            ],
            as_index=False,
        )["probability"]
        .mean()
    )

    y_final = (
        averaged["label"]
        .to_numpy()
    )

    p_final = (
        averaged["probability"]
        .to_numpy()
    )

    return {
        "balanced_accuracy":
            float(
                balanced_accuracy(
                    y_final,
                    p_final,
                )
            ),
        "roc_auc":
            float(
                roc_auc_score(
                    y_final,
                    p_final,
                )
            ),
    }


sensitivity_rows = []

demo_primary = (
    demographic_baseline[
        demographic_baseline["task"]
        == "AD_vs_CN"
    ]
    .iloc[0]
)

for anchor in [
    "start",
    "center",
    "end",
]:

    for duration in [
        180,
        300,
    ]:

        subset = features[
            (
                features["Group"]
                .isin(["A", "C"])
            )
            & (
                features[
                    "duration_seconds"
                ]
                == duration
            )
            & (
                features["anchor"]
                == anchor
            )
        ]

        combined = evaluate_combined(
            subset
        )

        eeg_only = (
            condition_summary[
                (
                    condition_summary[
                        "task"
                    ]
                    == "AD_vs_CN"
                )
                & (
                    condition_summary[
                        "duration_seconds"
                    ]
                    == duration
                )
                & (
                    condition_summary[
                        "anchor"
                    ]
                    == anchor
                )
            ]
            .iloc[0]
        )

        sensitivity_rows.extend(
            [
                {
                    "duration_seconds":
                        duration,
                    "anchor":
                        anchor,
                    "model":
                        "EEG_only",
                    "balanced_accuracy":
                        float(
                            eeg_only[
                                "balanced_accuracy"
                            ]
                        ),
                    "roc_auc":
                        float(
                            eeg_only[
                                "roc_auc"
                            ]
                        ),
                },
                {
                    "duration_seconds":
                        duration,
                    "anchor":
                        anchor,
                    "model":
                        "EEG_plus_age_sex",
                    **combined,
                },
                {
                    "duration_seconds":
                        duration,
                    "anchor":
                        anchor,
                    "model":
                        "age_sex_only",
                    "balanced_accuracy":
                        float(
                            demo_primary[
                                "balanced_accuracy"
                            ]
                        ),
                    "roc_auc":
                        float(
                            demo_primary[
                                "roc_auc"
                            ]
                        ),
                },
            ]
        )

sensitivity = pd.DataFrame(
    sensitivity_rows
)

sensitivity.to_csv(
    TABLES
    / "table_age_sex_sensitivity.csv",
    index=False,
)


# ==================================================
# 5. Duration-level worst/mean/best anchor robustness
# ==================================================

robust_rows = []

primary_summary = condition_summary[
    condition_summary["task"]
    == "AD_vs_CN"
]

for duration, part in primary_summary.groupby(
    "duration_seconds"
):

    robust_rows.append(
        {
            "duration_seconds":
                int(duration),
            "balanced_accuracy_worst_anchor":
                float(
                    part[
                        "balanced_accuracy"
                    ].min()
                ),
            "balanced_accuracy_mean_anchor":
                float(
                    part[
                        "balanced_accuracy"
                    ].mean()
                ),
            "balanced_accuracy_best_anchor":
                float(
                    part[
                        "balanced_accuracy"
                    ].max()
                ),
            "balanced_accuracy_anchor_range":
                float(
                    part[
                        "balanced_accuracy"
                    ].max()
                    -
                    part[
                        "balanced_accuracy"
                    ].min()
                ),
            "roc_auc_worst_anchor":
                float(
                    part[
                        "roc_auc"
                    ].min()
                ),
            "roc_auc_mean_anchor":
                float(
                    part[
                        "roc_auc"
                    ].mean()
                ),
            "roc_auc_best_anchor":
                float(
                    part[
                        "roc_auc"
                    ].max()
                ),
        }
    )

duration_robustness = pd.DataFrame(
    robust_rows
).sort_values(
    "duration_seconds"
)

duration_robustness.to_csv(
    TABLES
    / "table_duration_robustness_summary.csv",
    index=False,
)


# ==================================================
# Print concise results
# ==================================================

print()
print(
    "========================================"
)
print(
    "ABSOLUTE AD VS CN CONFIDENCE INTERVALS"
)
print(
    "========================================"
)

print(
    absolute[
        absolute["task"]
        == "AD_vs_CN"
    ]
    .round(4)
    .to_string(index=False)
)

print()
print(
    "========================================"
)
print(
    "DURATION ROBUSTNESS SUMMARY"
)
print(
    "========================================"
)

print(
    duration_robustness
    .round(4)
    .to_string(index=False)
)

print()
print(
    "========================================"
)
print(
    "TEMPORAL-ANCHOR PROBABILITY VARIABILITY"
)
print(
    "========================================"
)

print(
    anchor_summary[
        anchor_summary["task"]
        == "AD_vs_CN"
    ]
    .round(4)
    .to_string(index=False)
)

print()
print(
    "========================================"
)
print(
    "AGE / SEX SENSITIVITY"
)
print(
    "========================================"
)

print(
    sensitivity
    .round(4)
    .to_string(index=False)
)

print()
print(
    "========================================"
)
print(
    "STRONGEST REGIONAL AD VS CN EFFECTS"
)
print(
    "========================================"
)

display = (
    regional[
        regional["task"]
        == "AD_vs_CN"
    ]
    .copy()
)

display[
    "abs_delta"
] = (
    display[
        "cliffs_delta"
    ].abs()
)

print(
    display
    .sort_values(
        [
            "duration_seconds",
            "abs_delta",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .groupby(
        "duration_seconds"
    )
    .head(10)[
        [
            "duration_seconds",
            "region",
            "band",
            "cliffs_delta",
            "mannwhitney_p",
            "bh_q_within_task_duration",
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print()
print(
    "========================================"
)
print(
    "GLOBAL BIOMARKERS WITH FDR"
)
print(
    "========================================"
)

global_bio = pd.read_csv(
    ROOT
    / "tables"
    / "table_biomarkers_300s_center.csv"
)

print(
    global_bio[
        [
            "task",
            "feature",
            "cliffs_delta",
            "mannwhitney_p",
            "bh_q_within_task",
        ]
    ]
    .round(6)
    .to_string(index=False)
)

print()
print(
    "FINAL ROBUSTNESS ANALYSIS COMPLETE"
)
