#!/usr/bin/env python3

"""Local research-evidence audit. No training."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
R = ROOT / "reports"
F = R / "thesis_final"

OUT = F / "publication_evidence_v1"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "manifest":
        R / "openneuro_5min_manifest_v2.csv",

    "legacy_features":
        R / "baseline_features_v2.csv",

    "verified_models":
        F / "audit/verified_model_metrics.csv",

    "duration_features":
        F / "duration_locked_v1/duration_features.csv",

    "duration_runs":
        F / "duration_locked_v1/nested_cv_runs.csv",

    "duration_predictions":
        F / "duration_locked_v1/held_out_predictions.csv",

    "first_minute_blocks":
        F / "temporal_repeated_v1/"
        "all_repeats_heldout_block_predictions.csv",

    "repeat_level_results":
        F / "temporal_repeated_v1/repeat_level_results.csv",

    "position_transfer":
        F / "temporal_transfer_matrix_v1/"
        "cross_block_held_out_predictions.csv",

    "augmented_blocks":
        F / "block_augmented_svm_v1/"
        "augmented_heldout_block_predictions.csv",

    "specificity_contrasts":
        F / "augmentation_specificity_audit_v1/"
        "matched_subject_contrasts.csv"
}

FEATURES = (
    "delta", "theta", "alpha",
    "beta", "gamma", "theta_alpha_ratio"
)

REPEATS = range(1, 6)
BLOCKS = range(1, 6)
DURATIONS = (60, 180, 300)


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as handle:
        return list(csv.DictReader(handle))


def unique(rows, key, description):
    result = {}

    for row in rows:
        k = key(row)

        if k in result:
            raise RuntimeError(
                f"Duplicate {description}: {k}"
            )

        result[k] = row

    return result


def save(name, rows):
    if not rows:
        raise RuntimeError(
            "Empty output: " + name
        )

    keys = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with (OUT / name).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=keys
        )
        writer.writeheader()
        writer.writerows(rows)


print(
    "===== PUBLICATION EVIDENCE FREEZE =====",
    flush=True
)

# ========================================
# 1. LOCK SOURCE FILE FINGERPRINTS
# ========================================

for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(
            f"{name}: {path}"
        )

hashes = {
    name: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for name, path in FILES.items()
}

lock_path = OUT / "input_fingerprints.json"

serialized = (
    json.dumps(
        hashes,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if lock_path.exists():

    if lock_path.read_text() != serialized:
        raise RuntimeError(
            "SOURCE FILE CHANGED since evidence freeze. "
            "Review differences before a new version."
        )

else:
    lock_path.write_text(serialized)


# ========================================
# 2. VERIFY COHORT AND EXCLUSIONS
# ========================================

manifest = read(FILES["manifest"])

if len(manifest) != 88:
    raise RuntimeError(
        "Manifest is not 88 subjects"
    )

if len({
    r["subject"] for r in manifest
}) != 88:
    raise RuntimeError(
        "Duplicate manifest subject"
    )

cohort = {
    r["subject"]: r["group"]
    for r in manifest
    if r["group"] in ("A", "C")
    and r["status"] == "CANDIDATE"
}

if Counter(cohort.values()) != {
    "A": 23,
    "C": 23
}:
    raise RuntimeError(
        "Primary cohort changed"
    )

review = [
    r for r in manifest
    if r["status"] == "REVIEW"
    and r["group"] in ("A", "C")
]

if len(review) != 3:
    raise RuntimeError(
        "Unexpected number of review cases"
    )

flow = []

for (group, status), n in sorted(
    Counter(
        (r["group"], r["status"])
        for r in manifest
    ).items()
):
    flow.append({
        "group": group,
        "selection_status": status,
        "participants": n
    })

save("cohort_flow.csv", flow)

save(
    "review_cases_to_justify.csv",
    [
        {
            k: r.get(k, "")
            for k in (
                "subject", "group", "status",
                "start_sec", "end_sec",
                "multi200_pct", "any300_pct",
                "flat_pct", "note"
            )
        }
        for r in review
    ]
)


# ========================================
# 3. VERIFY HISTORICAL SPECTRAL FEATURES
# ========================================

legacy = unique(
    read(FILES["legacy_features"]),
    lambda r: r["subject"],
    "legacy feature"
)

duration = unique(
    read(FILES["duration_features"]),
    lambda r: (
        r["subject"],
        int(r["duration_sec"])
    ),
    "duration feature"
)

if (
    set(legacy) != set(cohort)
    or len(duration) != 138
):
    raise RuntimeError(
        "Feature cohort mismatch"
    )

max_error = 0.0

for sid in cohort:

    for seconds in DURATIONS:

        if (sid, seconds) not in duration:
            raise RuntimeError(
                "Missing duration: " + sid
            )

        if duration[
            (sid, seconds)
        ]["group"] != cohort[sid]:
            raise RuntimeError(
                "Diagnosis mismatch: " + sid
            )

    if int(float(
        legacy[sid]["clean_epochs"]
    )) != int(float(
        duration[(sid, 300)]["clean_epochs"]
    )):
        raise RuntimeError(
            "Clean-epoch mismatch: " + sid
        )

    for feature in FEATURES:

        x = float(
            legacy[sid][feature]
        )

        z = float(
            duration[(sid, 300)][feature]
        )

        if (
            not np.isfinite(x)
            or not np.isfinite(z)
        ):
            raise RuntimeError(
                "Nonfinite feature"
            )

        max_error = max(
            max_error,
            abs(x - z)
        )

if max_error > 1e-6:
    raise RuntimeError(
        "Historical 300-second features "
        f"NOT reproduced: {max_error}"
    )

print(
    "PASS: 46-subject 300s feature reproduction, "
    f"max difference={max_error:.3g}"
)


# ========================================
# 4. EXISTING MODEL COMPARISON
# ========================================

model_rows = read(
    FILES["verified_models"]
)

if (
    len(model_rows) != 9
    or len({
        r["model"] for r in model_rows
    }) != 9
):
    raise RuntimeError(
        "Unexpected model count"
    )

save(
    "model_comparison_from_saved_oof.csv",
    [
        {
            "model": r["model"],
            "status": r["status"],
            "n_participants":
                r.get("subjects", ""),
            "accuracy":
                r.get("accuracy", ""),
            "balanced_accuracy":
                r.get("balanced_accuracy", ""),
            "AUROC":
                r.get("roc_auc", "")
        }
        for r in model_rows
    ]
)


# ========================================
# 5. VERIFY DURATION RESULTS
# ========================================

runs = read(
    FILES["duration_runs"]
)

run_index = unique(
    runs,
    lambda r: (
        int(r["repeat"]),
        r["model"],
        int(r["duration_sec"])
    ),
    "duration CV run"
)

if len(run_index) != 30:
    raise RuntimeError(
        "Expected 30 nested CV run rows"
    )

duration_summary = []

for model in (
    "RBF_SVM",
    "LogisticRegression"
):

    for seconds in DURATIONS:

        values = np.array([
            float(
                run_index[
                    (rep, model, seconds)
                ]["balanced_accuracy"]
            )
            for rep in REPEATS
        ])

        duration_summary.append({
            "model": model,
            "duration_sec": seconds,
            "n_individuals": 46,
            "cv_repeats_correlated": 5,
            "mean_balanced_accuracy":
                float(values.mean()),
            "sd_across_repeats_not_a_CI":
                float(values.std(ddof=1))
        })

save(
    "duration_bacc_summary.csv",
    duration_summary
)


# ========================================
# 6. RECONCILE HELD-OUT PREDICTIONS
# ========================================

hist = unique(
    (
        r for r in read(
            FILES["duration_predictions"]
        )
        if r["model"] == "RBF_SVM"
        and int(r["duration_sec"]) == 60
    ),
    lambda r: (
        int(r["repeat"]),
        r["subject"]
    ),
    "historical first minute"
)

base = unique(
    read(
        FILES["first_minute_blocks"]
    ),
    lambda r: (
        int(r["repeat"]),
        r["subject"],
        int(r["block"])
    ),
    "baseline block"
)

aug = unique(
    read(
        FILES["augmented_blocks"]
    ),
    lambda r: (
        int(r["repeat"]),
        r["subject"],
        int(r["test_block"])
    ),
    "augmented block"
)

transfer = unique(
    read(
        FILES["position_transfer"]
    ),
    lambda r: (
        int(r["repeat"]),
        r["subject"],
        int(r["train_block"]),
        int(r["test_block"])
    ),
    "transfer prediction"
)

if (
    len(hist),
    len(base),
    len(aug),
    len(transfer)
) != (
    230, 1150, 1150, 5750
):
    raise RuntimeError(
        "Prediction-file row count mismatch"
    )

ids = sorted(cohort)

y = np.array([
    int(cohort[sid] == "A")
    for sid in ids
])

ad = np.flatnonzero(y == 1)
hc = np.flatnonzero(y == 0)

arrays = {
    key: np.full(
        (5, 46, 5),
        -1,
        dtype=int
    )
    for key in (
        "first_minute_training",
        "position_matched_training",
        "five_block_training"
    )
}

for rep in REPEATS:

    for i, sid in enumerate(ids):

        original = hist[
            (rep, sid)
        ]

        fold = int(
            original["fold"]
        )

        if int(
            original["true_label"]
        ) != y[i]:
            raise RuntimeError(
                "Historical diagnosis mismatch"
            )

        for block in BLOCKS:

            b = base[
                (rep, sid, block)
            ]

            a = aug[
                (rep, sid, block)
            ]

            s = transfer[
                (rep, sid, block, block)
            ]

            t = transfer[
                (rep, sid, 1, block)
            ]

            for row, col in (
                (b, "fold"),
                (a, "outer_fold"),
                (s, "outer_fold"),
                (t, "outer_fold")
            ):
                if (
                    int(row[col]) != fold
                    or int(row["true_label"]) != y[i]
                ):
                    raise RuntimeError(
                        "Fold/label mismatch: "
                        f"{rep}, {sid}, {block}"
                    )

            if int(
                b["predicted_label"]
            ) != int(
                t["predicted_label"]
            ):
                raise RuntimeError(
                    "Transfer/source block mismatch"
                )

            for key, row in (
                ("first_minute_training", b),
                ("position_matched_training", s),
                ("five_block_training", a)
            ):

                value = int(
                    row["predicted_label"]
                )

                if value not in (0, 1):
                    raise RuntimeError(
                        "Invalid prediction"
                    )

                arrays[key][
                    rep-1, i, block-1
                ] = value

        if arrays[
            "first_minute_training"
        ][rep-1, i, 0] != int(
            original["predicted_label"]
        ):
            raise RuntimeError(
                "Saved 60s SVM prediction mismatch"
            )

for name, arr in arrays.items():

    if np.any(arr < 0):
        raise RuntimeError(
            "Incomplete predictions: " + name
        )

print(
    "PASS: 1,150 predictions for each "
    "of three training strategies, "
    "matched by subject/fold"
)


# ========================================
# 7. RECOMPUTE PRIMARY METRICS
# ========================================

def bacc(arr):

    sensitivity = (
        arr[:, ad, :] == 1
    ).mean()

    specificity = (
        arr[:, hc, :] == 0
    ).mean()

    return float(
        (sensitivity + specificity) / 2
    )


strategy_rows = []

for name, arr in arrays.items():

    for block in (0, *BLOCKS):

        selected = (
            arr if block == 0
            else arr[:, :, block-1:block]
        )

        strategy_rows.append({
            "strategy": name,
            "test_block":
                "all" if block == 0 else block,
            "balanced_accuracy":
                bacc(selected),
            "n_independent_subjects": 46,
            "correlated_cv_repeats": 5,
            "training_seconds_per_subject":
                300 if name == "five_block_training"
                else 60
        })

save(
    "training_strategy_comparison.csv",
    strategy_rows
)


# ========================================
# 8. RECOMPUTE PREDICTION REVERSALS
# ========================================

flip = (
    arrays["first_minute_training"]
    != arrays["first_minute_training"][:, :, :1]
).any(axis=2)

flip_count = int(
    flip.sum()
)

rate = float(
    flip.mean()
)

any_flip = int(
    flip.any(axis=0).sum()
)

repeat_levels = read(
    FILES["repeat_level_results"]
)

if (
    len(repeat_levels) != 5
    or sum(
        int(r["class_reversals_n"])
        for r in repeat_levels
    ) != flip_count
):
    raise RuntimeError(
        "Reversal count disagrees "
        "with saved repeated CV summary"
    )

rng = np.random.default_rng(
    20261107
)

flip_boot = []

for _ in range(3000):

    sampled = np.r_[
        rng.choice(
            ad,
            size=len(ad),
            replace=True
        ),
        rng.choice(
            hc,
            size=len(hc),
            replace=True
        )
    ]

    flip_boot.append(
        float(
            flip[:, sampled].mean()
        )
    )

flip_ci = np.percentile(
    flip_boot,
    [2.5, 97.5]
)


# ========================================
# 9. VERIFY PAIRED COMPARISON RESULTS
# ========================================

contrasts = read(
    FILES["specificity_contrasts"]
)

if len(contrasts) != 6:
    raise RuntimeError(
        "Unexpected number of contrasts"
    )

primary = next(
    (
        r for r in contrasts
        if r["contrast"].startswith("PRIMARY:")
    ),
    None
)

if primary is None:
    raise RuntimeError(
        "Missing primary contrast"
    )

observed = (
    bacc(arrays["five_block_training"])
    -
    bacc(arrays["first_minute_training"])
)

if abs(
    observed
    -
    float(primary["delta_balanced_accuracy"])
) > 1e-9:
    raise RuntimeError(
        "Primary contrast disagrees "
        "with canonical predictions"
    )

save(
    "paired_comparisons_existing_CIs.csv",
    contrasts
)


# ========================================
# 10. CLAIM REGISTER
# ========================================

m1 = bacc(
    arrays["first_minute_training"][:, :, 4:5]
)

m5 = bacc(
    arrays["five_block_training"][:, :, 4:5]
)

sp = bacc(
    arrays["position_matched_training"][:, :, 4:5]
)

best_duration = max(
    duration_summary,
    key=lambda r: r["mean_balanced_accuracy"]
)

claim_rows = [
    {
        "finding":
            "Within-recording class reversal "
            "across five blocks",
        "estimate": rate,
        "ci_low": float(flip_ci[0]),
        "ci_high": float(flip_ci[1]),
        "ci_type":
            "conditional subject bootstrap",
        "source":
            "first_minute_blocks + repeat_level_results",
        "status": "exploratory"
    },
    {
        "finding":
            "Mean 180-second SVM duration BACC",
        "estimate": next(
            r["mean_balanced_accuracy"]
            for r in duration_summary
            if r["model"] == "RBF_SVM"
            and r["duration_sec"] == 180
        ),
        "ci_low": "",
        "ci_high": "",
        "ci_type":
            "mean of correlated CV repeats",
        "source": "duration_runs",
        "status": "exploratory"
    }
]

for row in contrasts:

    claim_rows.append({
        "finding": row["contrast"],
        "estimate":
            row["delta_balanced_accuracy"],
        "ci_low":
            row["conditional_ci_low"],
        "ci_high":
            row["conditional_ci_high"],
        "ci_type":
            "conditional participant bootstrap",
        "source":
            "specificity_contrasts",
        "status": "exploratory"
    })

save(
    "claim_registry.csv",
    claim_rows
)


# ========================================
# 11. PRINT VERIFIED NUMBERS
# ========================================

print(
    "\n===== VERIFIED PUBLICATION NUMBERS ====="
)

print(
    "Retrospective cohort: "
    "46 (23 AD, 23 HC)"
)

print(
    f"Within-recording reversals: "
    f"{flip_count}/230 subject-repeat evaluations "
    f"= {rate:.4f}"
)

print(
    "Conditional reversal-rate 95% CI: "
    f"[{flip_ci[0]:.4f}, {flip_ci[1]:.4f}]"
)

print(
    f"Participants reversing in at least "
    f"one repeat: {any_flip}/46"
)

for r in strategy_rows:

    if r["test_block"] in ("all", 5):

        print(
            f"{r['strategy']}, "
            f"test {r['test_block']}: "
            f"BACC={float(r['balanced_accuracy']):.4f}"
        )

print(
    "Primary overall augmented advantage: "
    f"{observed:+.4f}, "
    "conditional 95% CI "
    f"[{float(primary['conditional_ci_low']):+.4f}, "
    f"{float(primary['conditional_ci_high']):+.4f}]"
)


# ========================================
# 12. GENERATE TWO FINAL EVIDENCE FIGURES
# ========================================

fig, ax = plt.subplots(
    figsize=(7.3, 4.7)
)

for name, arr in arrays.items():

    ax.plot(
        list(BLOCKS),
        [
            bacc(
                arr[:, :, j:j+1]
            )
            for j in range(5)
        ],
        marker="o",
        label=name.replace("_", " ")
    )

ax.set(
    xlabel="Held-out 60-second block",
    ylabel="Participant-level balanced accuracy",
    ylim=(0.45, 0.9),
    xticks=list(BLOCKS),
    title="Recording-position robustness"
)

ax.grid(
    axis="y",
    alpha=0.2
)

ax.legend(fontsize=8)
fig.tight_layout()

fig.savefig(
    OUT / "temporal_training_strategies.png",
    dpi=190
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(7.1, 4.4)
)

for model in (
    "RBF_SVM",
    "LogisticRegression"
):

    items = [
        r for r in duration_summary
        if r["model"] == model
    ]

    ax.errorbar(
        [
            r["duration_sec"]
            for r in items
        ],
        [
            r["mean_balanced_accuracy"]
            for r in items
        ],
        yerr=[
            r["sd_across_repeats_not_a_CI"]
            for r in items
        ],
        marker="o",
        capsize=3,
        label=model
    )

ax.set(
    xlabel="Retained EEG duration (seconds)",
    ylabel="Participant-level balanced accuracy",
    ylim=(0.6, 0.9),
    xticks=list(DURATIONS),
    title="Duration analysis: mean and CV-repeat SD"
)

ax.grid(
    axis="y",
    alpha=0.2
)

ax.legend()
fig.tight_layout()

fig.savefig(
    OUT / "duration_performance_sd.png",
    dpi=190
)

plt.close(fig)


# ========================================
# 13. SAVE EVIDENCE SUMMARY, NOT MANUSCRIPT
# ========================================

notes = f"""# Publication evidence freeze

This is an evidence summary, not a finished thesis
manuscript or a preregistered study.

The same 46 participants were examined repeatedly
during exploratory method development.

## Verified study scope

- OpenNeuro ds004504: 88 total participants.
- Primary cohort: 46 (23 AD, 23 healthy).
- FTD is excluded from the binary endpoint.
- Of 19 excluded AD/healthy participants, 16 lacked
  a selected uninterrupted window and 3 have REVIEW
  status requiring final objective justification.
- Selected-window continuity audit reported no errors.
- Historical 300-second spectral reproduction:
  maximum absolute difference {max_error:.3g}.
- The outcome is classification of retrospective
  dataset diagnostic labels, not early detection.

## Reproducible findings

- First-minute trained, mean all-block BACC:
  {bacc(arrays['first_minute_training']):.4f}.
- Position-matched trained, mean all-block BACC:
  {bacc(arrays['position_matched_training']):.4f}.
- Five-block trained, mean all-block BACC:
  {bacc(arrays['five_block_training']):.4f}.
- Overall augmented minus original:
  {observed:+.4f}, conditional 95% interval
  [{float(primary['conditional_ci_low']):+.4f},
   {float(primary['conditional_ci_high']):+.4f}].
- No established overall advantage of augmentation.
- Test minute five BACC: original {m1:.4f};
  position-matched {sp:.4f}; augmented {m5:.4f}.
- The augmented model is not established as superior
  to the minute-five position-matched model.
- Within-recording reversal: {flip_count}/230
  subject-repeat observations ({100*rate:.1f}%).
  Conditional subject-bootstrap 95% interval:
  [{flip_ci[0]:.4f}, {flip_ci[1]:.4f}].
- {any_flip}/46 participants reversed at least once
  across five correlated CV repetitions.
- Highest observed duration-model BACC:
  {best_duration['mean_balanced_accuracy']:.4f}
  at {best_duration['duration_sec']} seconds,
  using {best_duration['model']}.
- This does not establish an optimal EEG duration.

## What is not established

- No state-of-the-art accuracy result.
- No superiority of five-block augmentation.
- No proven optimal acquisition duration.
- No demonstrated causal biomarker mechanism.
- No independent external validation.
- No guarantee of clinical applicability.
- No claim of 60-second acquisition-to-prediction;
  preprocessing may use the longer original session.
- Bootstrap uncertainty is conditional on fitted
  cross-validation results, and repeated observations
  are correlated.
- CBraMod conclusions apply to the frozen-feature
  configuration actually evaluated.

## Before journal submission

1. Document exclusion reasons for sub-027,
   sub-030 and sub-064.
2. Verify model training code and signal preprocessing
   provenance, including CBraMod input configuration.
3. Preserve participant-level splits and dependencies.
4. Verify all citations and literature comparisons.
5. Do not introduce new models after examining
   these test results.

## Evidence files

- input_fingerprints.json
- model_comparison_from_saved_oof.csv
- cohort_flow.csv
- review_cases_to_justify.csv
- duration_bacc_summary.csv
- training_strategy_comparison.csv
- paired_comparisons_existing_CIs.csv
- claim_registry.csv
- temporal_training_strategies.png
- duration_performance_sd.png
"""

(
    OUT / "EVIDENCE_SUMMARY.md"
).write_text(
    notes,
    encoding="utf-8"
)

print(
    "\n===== PUBLICATION EVIDENCE FREEZE COMPLETE ====="
)

print(
    "Output directory:",
    OUT
)

print(
    "Original data, training code "
    "and previous reports unchanged."
)

