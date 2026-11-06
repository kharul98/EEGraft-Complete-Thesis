#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRED = ROOT / "results/p2_adaptive_pilot_participant_oof.csv"
METRICS = ROOT / "results/p2_adaptive_pilot_repeat_metrics.csv"
OUT = ROOT / "docs/audit/p2_policy_diagnostics.json"

pred = pd.read_csv(PRED)
metrics = pd.read_csv(METRICS)

assert len(pred) == 3720
assert len(metrics) == 60

assert not pred.duplicated(
    ["task", "repeat", "policy", "participant_id"]
).any()

report = {
    "status": "EXPLORATORY",
    "objective": "Descriptive policy comparison, not significance testing",
    "tasks": {}
}

print("===== P2 POLICY REALITY CHECK =====")

for task in ("AD_vs_CN", "AD_vs_FTD"):

    subset = metrics[metrics.task == task]

    summary = (
        subset.groupby("policy")[
            [
                "balanced_accuracy",
                "sensitivity_ad",
                "mean_seconds",
                "early_stop_fraction"
            ]
        ]
        .mean()
    )

    task_results = {
        "policies": {},
        "paired_fixed60_comparisons": []
    }

    print()
    print("TASK:", task)

    for policy, row in summary.iterrows():

        dominators = []

        for other, comparison in summary.iterrows():

            if policy == other:
                continue

            better_accuracy = (
                comparison.balanced_accuracy
                >= row.balanced_accuracy - 1e-12
            )

            no_slower = (
                comparison.mean_seconds
                <= row.mean_seconds + 1e-12
            )

            strictly_better = (
                comparison.balanced_accuracy
                > row.balanced_accuracy + 1e-12
                or comparison.mean_seconds
                < row.mean_seconds - 1e-12
            )

            if better_accuracy and no_slower and strictly_better:
                dominators.append(other)

        selected = pred[
            (pred.task == task)
            & (pred.policy == policy)
        ]

        stops = (
            selected.stop_seconds
            .value_counts()
            .sort_index()
            .to_dict()
        )

        task_results["policies"][policy] = {
            "mean_balanced_accuracy":
                float(row.balanced_accuracy),

            "mean_sensitivity":
                float(row.sensitivity_ad),

            "mean_seconds":
                float(row.mean_seconds),

            "early_stop_fraction":
                float(row.early_stop_fraction),

            "dominated_in_accuracy_time_plane":
                bool(dominators),

            "dominated_by":
                dominators,

            "stop_counts_across_repeats":
                {str(k): int(v) for k, v in stops.items()}
        }

        print(
            f"{policy:40s}",
            f"BA={row.balanced_accuracy:.3f}",
            f"time={row.mean_seconds:.1f}s",
            "DOMINATED" if dominators else "FRONTIER"
        )

    for policy in (
        "stability_only",
        "combined_quality_confidence_stability"
    ):

        candidate = pred[
            (pred.task == task)
            & (pred.policy == policy)
        ]

        reference = pred[
            (pred.task == task)
            & (pred.policy == "fixed_60")
        ][[
            "task",
            "repeat",
            "participant_id",
            "y_pred",
            "stop_seconds"
        ]]

        joined = candidate.merge(
            reference,
            on=["task", "repeat", "participant_id"],
            suffixes=("_candidate", "_fixed60"),
            validate="one_to_one"
        )

        for repeat, group in joined.groupby("repeat"):

            candidate_correct = (
                group.y_pred_candidate == group.y_true
            )

            reference_correct = (
                group.y_pred_fixed60 == group.y_true
            )

            gains = int((
                candidate_correct & ~reference_correct
            ).sum())

            losses = int((
                ~candidate_correct & reference_correct
            ).sum())

            ad = group.y_true.eq(1)

            new_ad_misses = int((
                ad & ~candidate_correct & reference_correct
            ).sum())

            recovered_ad = int((
                ad & candidate_correct & ~reference_correct
            ).sum())

            entry = {
                "policy": policy,
                "repeat": int(repeat),
                "participants": len(group),
                "candidate_correct_fixed60_wrong": gains,
                "candidate_wrong_fixed60_correct": losses,
                "additional_ad_misses": new_ad_misses,
                "recovered_ad_cases": recovered_ad,
                "candidate_mean_seconds":
                    float(group.stop_seconds_candidate.mean()),
                "fixed60_mean_seconds":
                    float(group.stop_seconds_fixed60.mean())
            }

            task_results[
                "paired_fixed60_comparisons"
            ].append(entry)

            print(
                f"  {policy}, repeat {repeat}:",
                f"corrected={gains},",
                f"new_errors={losses},",
                f"new_AD_misses={new_ad_misses},",
                f"recovered_AD={recovered_ad}"
            )

    report["tasks"][task] = task_results

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")

print()
print("PARETO STATUS: DESCRIPTIVE ONLY")
print("REPEATED CV FOLDS ARE NOT INDEPENDENT SUBJECTS")
print("NO SIGNIFICANCE OR CLINICAL CLAIM")
print("P2 POLICY DIAGNOSTICS COMPLETE")
