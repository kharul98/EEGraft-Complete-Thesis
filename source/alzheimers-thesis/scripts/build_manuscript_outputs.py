from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]

RES = ROOT / "results"
OUT = RES / "manuscript_summary"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


summary = pd.read_csv(
    RES / "duration_experiment" / "condition_summary.csv"
)

bootstrap = pd.read_csv(
    RES / "duration_experiment" / "paired_bootstrap_vs_300.csv"
)

pred = pd.read_csv(
    RES / "duration_experiment" / "oof_predictions.csv"
)

demo = pd.read_csv(
    RES / "duration_experiment" / "demographic_baseline.csv"
)

perm = pd.read_csv(
    RES / "duration_experiment" / "permutation_null.csv"
)

perm_summary = json.loads(
    (
        RES
        / "duration_experiment"
        / "permutation_summary.json"
    ).read_text()
)

stability = pd.read_csv(
    RES
    / "spectral_stability"
    / "key_feature_stability_summary.csv"
)

effects = pd.read_csv(
    RES
    / "spectral_stability"
    / "global_biomarker_effects.csv"
)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)

    n = len(p)

    order = np.argsort(p)

    ranked = p[order]

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

    out = np.empty(
        n,
        dtype=float,
    )

    out[order] = adjusted

    return out


# ==================================================
# Main tables
# ==================================================

primary = (
    summary[
        summary["task"] == "AD_vs_CN"
    ]
    .sort_values(
        ["anchor", "duration_seconds"]
    )
    .reset_index(drop=True)
)

secondary = (
    summary[
        summary["task"] == "AD_vs_FTD"
    ]
    .sort_values(
        ["anchor", "duration_seconds"]
    )
    .reset_index(drop=True)
)

primary.to_csv(
    TABLES / "table_primary_ad_vs_cn_performance.csv",
    index=False,
)

secondary.to_csv(
    TABLES / "table_secondary_ad_vs_ftd_performance.csv",
    index=False,
)

demo.to_csv(
    TABLES / "table_demographic_baseline.csv",
    index=False,
)


primary_bootstrap = (
    bootstrap[
        bootstrap["task"] == "AD_vs_CN"
    ]
    .sort_values(
        ["anchor", "duration_seconds"]
    )
    .reset_index(drop=True)
)

secondary_bootstrap = (
    bootstrap[
        bootstrap["task"] == "AD_vs_FTD"
    ]
    .sort_values(
        ["anchor", "duration_seconds"]
    )
    .reset_index(drop=True)
)

primary_bootstrap.to_csv(
    TABLES / "table_primary_bootstrap_vs_300.csv",
    index=False,
)

secondary_bootstrap.to_csv(
    TABLES / "table_secondary_bootstrap_vs_300.csv",
    index=False,
)


# ==================================================
# Anchor sensitivity
# ==================================================

anchor_rows = []

for task, task_df in summary.groupby("task"):

    for duration, part in task_df.groupby("duration_seconds"):

        anchor_rows.append(
            {
                "task": task,
                "duration_seconds": int(duration),

                "balanced_accuracy_min":
                    float(part["balanced_accuracy"].min()),

                "balanced_accuracy_max":
                    float(part["balanced_accuracy"].max()),

                "balanced_accuracy_range":
                    float(
                        part["balanced_accuracy"].max()
                        - part["balanced_accuracy"].min()
                    ),

                "roc_auc_min":
                    float(part["roc_auc"].min()),

                "roc_auc_max":
                    float(part["roc_auc"].max()),

                "roc_auc_range":
                    float(
                        part["roc_auc"].max()
                        - part["roc_auc"].min()
                    ),
            }
        )

anchor_sensitivity = pd.DataFrame(anchor_rows)

anchor_sensitivity.to_csv(
    TABLES / "table_anchor_sensitivity.csv",
    index=False,
)


# ==================================================
# Prediction stability versus 300 s
# ==================================================

avg_pred = (
    pred
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

prediction_rows = []

for task in avg_pred["task"].unique():

    for anchor in avg_pred["anchor"].unique():

        reference = (
            avg_pred[
                (avg_pred["task"] == task)
                & (avg_pred["anchor"] == anchor)
                & (avg_pred["duration_seconds"] == 300)
            ]
            .sort_values("participant_id")
            .reset_index(drop=True)
        )

        for duration in [30, 60, 120, 180, 300]:

            current = (
                avg_pred[
                    (avg_pred["task"] == task)
                    & (avg_pred["anchor"] == anchor)
                    & (avg_pred["duration_seconds"] == duration)
                ]
                .sort_values("participant_id")
                .reset_index(drop=True)
            )

            if not np.array_equal(
                current["participant_id"].to_numpy(),
                reference["participant_id"].to_numpy(),
            ):
                raise RuntimeError(
                    "Participant mismatch in probability stability."
                )

            rho = spearmanr(
                current["probability"].to_numpy(),
                reference["probability"].to_numpy(),
            ).statistic

            mae = np.mean(
                np.abs(
                    current["probability"].to_numpy()
                    - reference["probability"].to_numpy()
                )
            )

            agreement = np.mean(
                (
                    current["probability"].to_numpy() >= 0.5
                )
                ==
                (
                    reference["probability"].to_numpy() >= 0.5
                )
            )

            prediction_rows.append(
                {
                    "task": task,
                    "anchor": anchor,
                    "duration_seconds": duration,
                    "probability_spearman_vs_300":
                        float(rho),
                    "probability_mae_vs_300":
                        float(mae),
                    "classification_agreement_vs_300":
                        float(agreement),
                }
            )

prediction_stability = pd.DataFrame(
    prediction_rows
)

prediction_stability.to_csv(
    TABLES / "table_prediction_stability_vs_300.csv",
    index=False,
)


# ==================================================
# Prespecified biomarkers at 300 s center
# ==================================================

bio = (
    effects[
        (effects["duration_seconds"] == 300)
        & (effects["anchor"] == "center")
    ]
    .copy()
    .reset_index(drop=True)
)

bio["bh_q_within_task"] = np.nan

for task in bio["task"].unique():

    idx = bio["task"] == task

    bio.loc[
        idx,
        "bh_q_within_task",
    ] = bh_adjust(
        bio.loc[
            idx,
            "mannwhitney_p",
        ].to_numpy()
    )

bio.to_csv(
    TABLES / "table_biomarkers_300s_center.csv",
    index=False,
)


stability.to_csv(
    TABLES / "table_key_spectral_stability.csv",
    index=False,
)


# ==================================================
# Figures
# ==================================================

durations = [30, 60, 120, 180, 300]


# Figure 1: primary balanced accuracy
plt.figure(figsize=(7, 5))

for anchor in ["start", "center", "end"]:

    part = (
        primary[
            primary["anchor"] == anchor
        ]
        .sort_values("duration_seconds")
    )

    plt.plot(
        part["duration_seconds"],
        part["balanced_accuracy"],
        marker="o",
        label=anchor,
    )

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Available EEG duration (seconds)")
plt.ylabel("Balanced accuracy")
plt.title("AD vs CN classification across EEG duration")
plt.xticks(durations)
plt.legend(title="Segment position")
plt.tight_layout()

plt.savefig(
    FIGURES / "fig01_primary_balanced_accuracy.png",
    dpi=300,
)

plt.close()


# Figure 2: spectral stability
plt.figure(figsize=(7, 5))

for anchor in ["start", "center", "end"]:

    part = (
        stability[
            stability["anchor"] == anchor
        ]
        .sort_values("duration_seconds")
    )

    plt.plot(
        part["duration_seconds"],
        part["median"],
        marker="o",
        label=anchor,
    )

plt.xlabel("Available EEG duration (seconds)")
plt.ylabel("Median Spearman correlation vs 300 s")
plt.title("Stability of prespecified spectral biomarkers")
plt.xticks(durations)
plt.ylim(0, 1.05)
plt.legend(title="Segment position")
plt.tight_layout()

plt.savefig(
    FIGURES / "fig02_spectral_stability.png",
    dpi=300,
)

plt.close()


# Figure 3: bootstrap forest plot
plot_boot = (
    primary_bootstrap
    .sort_values(
        ["anchor", "duration_seconds"]
    )
    .reset_index(drop=True)
)

labels = [
    f"{row.anchor} {int(row.duration_seconds)} s"
    for row in plot_boot.itertuples()
]

ypos = np.arange(len(plot_boot))

estimate = (
    plot_boot["balanced_accuracy_delta"]
    .to_numpy()
)

lower = (
    plot_boot["ci95_low"]
    .to_numpy()
)

upper = (
    plot_boot["ci95_high"]
    .to_numpy()
)

xerr = np.vstack(
    [
        estimate - lower,
        upper - estimate,
    ]
)

plt.figure(figsize=(8, 7))

plt.errorbar(
    estimate,
    ypos,
    xerr=xerr,
    fmt="o",
    capsize=3,
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.yticks(
    ypos,
    labels,
)

plt.xlabel(
    "Balanced accuracy difference: shorter duration - 300 s"
)

plt.title(
    "Paired participant bootstrap: AD vs CN"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "fig03_primary_bootstrap_vs_300.png",
    dpi=300,
)

plt.close()


# Figure 4: permutation control
plt.figure(figsize=(7, 5))

plt.hist(
    perm["balanced_accuracy"],
    bins=15,
)

plt.axvline(
    perm_summary["observed_balanced_accuracy"],
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Balanced accuracy")
plt.ylabel("Permutation count")
plt.title(
    "Shuffled-label negative control: AD vs CN, 300 s center"
)
plt.tight_layout()

plt.savefig(
    FIGURES / "fig04_permutation_control.png",
    dpi=300,
)

plt.close()


# Figure 5A: AD vs CN biomarker effects
for task, filename, title in [
    (
        "AD_vs_CN",
        "fig05a_biomarker_effects_ad_vs_cn.png",
        "AD vs CN: 300 s center spectral effects",
    ),
    (
        "AD_vs_FTD",
        "fig05b_biomarker_effects_ad_vs_ftd.png",
        "AD vs FTD: 300 s center spectral effects",
    ),
]:

    part = (
        bio[bio["task"] == task]
        .sort_values("cliffs_delta")
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        part["feature"],
        part["cliffs_delta"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel("Cliff's delta")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300,
    )

    plt.close()


# Figure 6: probability stability
primary_prediction = (
    prediction_stability[
        prediction_stability["task"]
        == "AD_vs_CN"
    ]
)

plt.figure(figsize=(7, 5))

for anchor in ["start", "center", "end"]:

    part = (
        primary_prediction[
            primary_prediction["anchor"]
            == anchor
        ]
        .sort_values("duration_seconds")
    )

    plt.plot(
        part["duration_seconds"],
        part["probability_spearman_vs_300"],
        marker="o",
        label=anchor,
    )

plt.xlabel("Available EEG duration (seconds)")
plt.ylabel("Prediction correlation vs 300 s")
plt.title("Participant prediction stability: AD vs CN")
plt.xticks(durations)
plt.ylim(0, 1.05)
plt.legend(title="Segment position")
plt.tight_layout()

plt.savefig(
    FIGURES / "fig06_prediction_stability.png",
    dpi=300,
)

plt.close()


# ==================================================
# Numeric snapshot
# ==================================================

best_primary_idx = (
    primary["balanced_accuracy"]
    .idxmax()
)

best_primary = primary.loc[
    best_primary_idx
]

snapshot = {
    "best_primary_condition": {
        "duration_seconds":
            int(best_primary["duration_seconds"]),
        "anchor":
            str(best_primary["anchor"]),
        "balanced_accuracy":
            float(best_primary["balanced_accuracy"]),
        "roc_auc":
            float(best_primary["roc_auc"]),
    },

    "permutation_control":
        perm_summary,

    "demographic_ad_vs_cn":
        (
            demo[
                demo["task"] == "AD_vs_CN"
            ]
            .iloc[0]
            .to_dict()
        ),
}

(
    OUT / "numeric_snapshot.json"
).write_text(
    json.dumps(
        snapshot,
        indent=2,
    )
)

print(
    "MANUSCRIPT OUTPUT BUILD COMPLETE"
)

print()
print(
    "Best primary point-estimate condition:"
)

print(
    json.dumps(
        snapshot["best_primary_condition"],
        indent=2,
    )
)

print()
print(
    "Primary bootstrap rows:"
)

print(
    primary_bootstrap
    .round(4)
    .to_string(index=False)
)

print()
print(
    "Anchor sensitivity:"
)

print(
    anchor_sensitivity[
        anchor_sensitivity["task"]
        == "AD_vs_CN"
    ]
    .round(4)
    .to_string(index=False)
)

print()
print(
    "Prediction stability:"
)

print(
    primary_prediction
    .round(4)
    .to_string(index=False)
)

print()
print(
    "300 s center biomarkers with BH q-values:"
)

print(
    bio[
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
    "Figures:"
)

for path in sorted(
    FIGURES.glob("fig*.png")
):
    print(path)
