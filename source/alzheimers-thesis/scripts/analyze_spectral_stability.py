from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    spearmanr,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "results"
    / "duration_features"
    / "eeg_duration_features.csv"
)

OUT = (
    ROOT
    / "results"
    / "spectral_stability"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


KEY_FEATURES = [
    "global_delta",
    "global_theta",
    "global_alpha",
    "global_beta",
    "global_gamma",
    "global_slowing_ratio",
    "global_alpha_theta_ratio",
    "global_paf_hz",
]


TASKS = {
    "AD_vs_CN": (
        "A",
        "C",
    ),
    "AD_vs_FTD": (
        "A",
        "F",
    ),
}


def cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    x = np.asarray(x)
    y = np.asarray(y)

    greater = 0
    lower = 0

    for value in x:
        greater += int(
            np.sum(
                value > y
            )
        )

        lower += int(
            np.sum(
                value < y
            )
        )

    return (
        greater - lower
    ) / (
        len(x)
        * len(y)
    )


def main():

    df = pd.read_csv(
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
        for c in df.columns
        if c not in metadata
    ]

    stability_rows = []

    for anchor in sorted(
        df["anchor"].unique()
    ):

        reference = (
            df[
                (
                    df["anchor"]
                    == anchor
                )
                & (
                    df[
                        "duration_seconds"
                    ]
                    == 300
                )
            ]
            .sort_values(
                "participant_id"
            )
        )

        for duration in sorted(
            df[
                "duration_seconds"
            ].unique()
        ):

            current = (
                df[
                    (
                        df["anchor"]
                        == anchor
                    )
                    & (
                        df[
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
                reference[
                    "participant_id"
                ].to_numpy(),
                current[
                    "participant_id"
                ].to_numpy(),
            ):

                raise RuntimeError(
                    "Feature-stability pairing failure"
                )

            for feature in (
                feature_columns
            ):

                x = current[
                    feature
                ].to_numpy(
                    dtype=float
                )

                ref = reference[
                    feature
                ].to_numpy(
                    dtype=float
                )

                rho = spearmanr(
                    x,
                    ref,
                ).statistic

                mae = float(
                    np.mean(
                        np.abs(
                            x - ref
                        )
                    )
                )

                stability_rows.append(
                    {
                        "anchor":
                            anchor,
                        "duration_seconds":
                            duration,
                        "feature":
                            feature,
                        "spearman_rho":
                            float(rho),
                        "mae":
                            mae,
                    }
                )

    stability = pd.DataFrame(
        stability_rows
    )

    stability.to_csv(
        OUT
        / "feature_stability_vs_300.csv",
        index=False,
    )

    effect_rows = []

    for task, (
        group1,
        group0,
    ) in TASKS.items():

        for anchor in sorted(
            df[
                "anchor"
            ].unique()
        ):

            for duration in sorted(
                df[
                    "duration_seconds"
                ].unique()
            ):

                condition = df[
                    (
                        df[
                            "anchor"
                        ]
                        == anchor
                    )
                    & (
                        df[
                            "duration_seconds"
                        ]
                        == duration
                    )
                ]

                for feature in KEY_FEATURES:

                    x = (
                        condition[
                            condition[
                                "Group"
                            ]
                            == group1
                        ][feature]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    y = (
                        condition[
                            condition[
                                "Group"
                            ]
                            == group0
                        ][feature]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    test = mannwhitneyu(
                        x,
                        y,
                        alternative="two-sided",
                    )

                    effect_rows.append(
                        {
                            "task":
                                task,
                            "anchor":
                                anchor,
                            "duration_seconds":
                                duration,
                            "feature":
                                feature,
                            "group_positive":
                                group1,
                            "group_negative":
                                group0,
                            "positive_mean":
                                float(
                                    np.mean(x)
                                ),
                            "negative_mean":
                                float(
                                    np.mean(y)
                                ),
                            "positive_median":
                                float(
                                    np.median(x)
                                ),
                            "negative_median":
                                float(
                                    np.median(y)
                                ),
                            "cliffs_delta":
                                float(
                                    cliffs_delta(
                                        x,
                                        y,
                                    )
                                ),
                            "mannwhitney_p":
                                float(
                                    test.pvalue
                                ),
                        }
                    )

    effects = pd.DataFrame(
        effect_rows
    )

    effects.to_csv(
        OUT
        / "global_biomarker_effects.csv",
        index=False,
    )

    summary = (
        stability[
            stability[
                "feature"
            ].isin(
                KEY_FEATURES
            )
        ]
        .groupby(
            [
                "anchor",
                "duration_seconds",
            ]
        )[
            "spearman_rho"
        ]
        .agg(
            [
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    summary.to_csv(
        OUT
        / "key_feature_stability_summary.csv",
        index=False,
    )

    print()
    print(
        "======================================"
    )
    print(
        "KEY FEATURE STABILITY VS 300s"
    )
    print(
        "======================================"
    )

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "======================================"
    )
    print(
        "300s CENTER BIOMARKER EFFECTS"
    )
    print(
        "======================================"
    )

    display = effects[
        (
            effects[
                "duration_seconds"
            ]
            == 300
        )
        & (
            effects[
                "anchor"
            ]
            == "center"
        )
    ]

    print(
        display[
            [
                "task",
                "feature",
                "positive_mean",
                "negative_mean",
                "cliffs_delta",
                "mannwhitney_p",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
