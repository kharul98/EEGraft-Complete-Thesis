#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "results/p2_feature_representation_oof.csv"
)

POLICY_FILE = (
    ROOT / "results/p2_adaptive_pilot_participant_oof.csv"
)

OUTPUT = (
    ROOT / "results/p2_paired_uncertainty.csv"
)

features = pd.read_csv(FEATURE_FILE)
policies = pd.read_csv(POLICY_FILE)

BOOTSTRAPS = 5000
SEED = 20261107

# These are post-hoc exploratory comparisons.
# They were selected after examining pilot results.

COMPARISONS = [
    (
        "AD_vs_CN",
        "epoch_median@300",
        "pilot_global_psd@300"
    ),
    (
        "AD_vs_CN",
        "epoch_median@300",
        "pilot_global_psd@60"
    ),
    (
        "AD_vs_CN",
        "stability_only",
        "fixed_60"
    ),
    (
        "AD_vs_FTD",
        "epoch_median@300",
        "pilot_global_psd@300"
    ),
]


def extract(task, method):

    if "@" in method:

        representation, duration = method.split("@")
        duration = int(duration)

        frame = features.loc[
            (features.task == task)
            & (features.representation == representation)
            & (features.seconds == duration)
        ].copy()

        frame["stop_seconds"] = duration

    else:

        frame = policies.loc[
            (policies.task == task)
            & (policies.policy == method)
        ].copy()

    columns = [
        "repeat",
        "fold",
        "participant_id",
        "y_true",
        "y_pred",
        "stop_seconds"
    ]

    frame = frame[columns].copy()

    frame["participant_id"] = (
        frame.participant_id.astype(str)
    )

    assert not frame.duplicated([
        "repeat",
        "participant_id"
    ]).any()

    assert set(frame.repeat) == {1, 2, 3}

    return frame


def analyze(task, candidate, reference, seed):

    a = extract(task, candidate)
    b = extract(task, reference)

    joined = a.merge(
        b,
        on=["repeat", "participant_id"],
        suffixes=("_candidate", "_reference"),
        validate="one_to_one"
    )

    assert len(joined) == len(a) == len(b)

    assert (
        joined.y_true_candidate
        == joined.y_true_reference
    ).all()

    assert (
        joined.fold_candidate
        == joined.fold_reference
    ).all()

    ids = sorted(
        joined.participant_id.unique()
    )

    expected = 65 if task == "AD_vs_CN" else 59
    assert len(ids) == expected

    reference_repeat = (
        joined.loc[joined.repeat == 1]
        .set_index("participant_id")
        .loc[ids]
    )

    y = reference_repeat.y_true_candidate.to_numpy(
        dtype=int
    )

    assert set(y) == {0, 1}

    candidate_predictions = []
    reference_predictions = []

    candidate_times = []
    reference_times = []

    for repeat in (1, 2, 3):

        frame = (
            joined.loc[joined.repeat == repeat]
            .set_index("participant_id")
            .loc[ids]
        )

        assert len(frame) == expected

        assert np.array_equal(
            frame.y_true_candidate.to_numpy(dtype=int),
            y
        )

        candidate_predictions.append(
            frame.y_pred_candidate.to_numpy(dtype=int)
        )

        reference_predictions.append(
            frame.y_pred_reference.to_numpy(dtype=int)
        )

        candidate_times.append(
            frame.stop_seconds_candidate.to_numpy(dtype=float)
        )

        reference_times.append(
            frame.stop_seconds_reference.to_numpy(dtype=float)
        )

    a_pred = np.stack(candidate_predictions)
    b_pred = np.stack(reference_predictions)

    a_time = np.stack(candidate_times)
    b_time = np.stack(reference_times)

    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)

    a_correct = (a_pred == y[None, :]).astype(float)
    b_correct = (b_pred == y[None, :]).astype(float)

    difference = a_correct - b_correct

    def balanced_accuracy(correct):
        return float(
            (
                correct[:, positive].mean()
                + correct[:, negative].mean()
            ) / 2
        )

    candidate_ba = balanced_accuracy(a_correct)
    reference_ba = balanced_accuracy(b_correct)

    rng = np.random.default_rng(seed)

    # Resample unique participants within diagnostic
    # strata. The same sampled participant IDs are
    # reused in all three CV repeats.
    sampled_positive = rng.choice(
        positive,
        size=(BOOTSTRAPS, len(positive)),
        replace=True
    )

    sampled_negative = rng.choice(
        negative,
        size=(BOOTSTRAPS, len(negative)),
        replace=True
    )

    positive_deltas = (
        difference[:, sampled_positive]
        .mean(axis=(0, 2))
    )

    negative_deltas = (
        difference[:, sampled_negative]
        .mean(axis=(0, 2))
    )

    bootstrap_ba = (
        positive_deltas + negative_deltas
    ) / 2

    delta_seconds = a_time - b_time

    bootstrap_seconds = (
        (
            delta_seconds[:, sampled_positive]
            .mean(axis=(0, 2))
            * len(positive)
        )
        +
        (
            delta_seconds[:, sampled_negative]
            .mean(axis=(0, 2))
            * len(negative)
        )
    ) / len(y)

    ba_low, ba_high = np.quantile(
        bootstrap_ba,
        [0.025, 0.975]
    )

    sec_low, sec_high = np.quantile(
        bootstrap_seconds,
        [0.025, 0.975]
    )

    sensitivity_difference = float(
        difference[:, positive].mean()
    )

    record = {
        "task": task,
        "candidate": candidate,
        "reference": reference,
        "unique_participants": len(y),
        "cv_repeats": 3,
        "candidate_ba": candidate_ba,
        "reference_ba": reference_ba,
        "delta_ba": candidate_ba - reference_ba,
        "delta_ba_ci_low": float(ba_low),
        "delta_ba_ci_high": float(ba_high),
        "delta_sensitivity_ad":
            sensitivity_difference,
        "candidate_mean_seconds":
            float(a_time.mean()),
        "reference_mean_seconds":
            float(b_time.mean()),
        "delta_seconds":
            float(delta_seconds.mean()),
        "delta_seconds_ci_low":
            float(sec_low),
        "delta_seconds_ci_high":
            float(sec_high),
        "bootstrap_resamples": BOOTSTRAPS
    }

    return record


print("===== P2 PAIRED UNCERTAINTY =====")
print("Exploratory comparisons, selected post hoc")
print("Unique participants resampled within strata")
print("All three CV repeats kept together")

records = []

for index, comparison in enumerate(COMPARISONS):

    row = analyze(
        *comparison,
        seed=SEED + index
    )

    records.append(row)

    print()
    print(
        row["task"],
        row["candidate"],
        "vs",
        row["reference"]
    )

    print(
        "Balanced accuracy:",
        round(row["candidate_ba"], 4),
        "vs",
        round(row["reference_ba"], 4)
    )

    print(
        "Paired BA difference:",
        round(row["delta_ba"], 4)
    )

    print(
        "Exploratory 95% bootstrap interval:",
        (
            round(row["delta_ba_ci_low"], 4),
            round(row["delta_ba_ci_high"], 4)
        )
    )

    print(
        "Mean recording-time difference (s):",
        round(row["delta_seconds"], 2)
    )

result = pd.DataFrame(records)

assert len(result) == 4
assert result.delta_ba.between(-1, 1).all()
assert np.isfinite(
    result.select_dtypes(include="number").to_numpy()
).all()

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False,
    float_format="%.12g"
)

print()
print("IMPORTANT LIMITATIONS:")
print("- Comparisons selected after inspecting pilot results.")
print("- Intervals are descriptive, not confirmatory.")
print("- Bootstrap conditions on saved OOF predictions.")
print("- Model retraining variability is not fully captured.")
print("- No multiple-comparison adjustment was performed.")
print("- No non-inferiority or clinical claim is justified.")

print()
print("P2 PAIRED UNCERTAINTY COMPLETE")
