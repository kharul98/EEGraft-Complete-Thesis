#!/usr/bin/env python3
"""Exploratory nested-CV, training-only adaptive acquisition."""

from pathlib import Path
import hashlib
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
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "results/p2_prefix_robust_features.csv"
REFERENCE = ROOT / "results/p2_feature_representation_oof.csv"

OUT_PRED = ROOT / "results/p2_training_only_gate_oof.csv"
OUT_FOLDS = ROOT / "results/p2_training_only_gate_folds.csv"
OUT_METRICS = ROOT / "results/p2_training_only_gate_metrics.csv"

TIMES = np.array([30, 60, 120, 180, 300], dtype=int)
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
FEATURES = [f"median_relative_{b}" for b in BANDS]

POLICIES = (
    "fixed_30",
    "fixed_60",
    "fixed_120",
    "fixed_180",
    "fixed_300",
    "stability_only_75",
    "budget_random_60_120",
    "training_selected_gate"
)


def classifier():
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2500
            )
        )
    ])


def fit_search(x, y, splits, seed):
    cv = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=seed
    )

    search = GridSearchCV(
        classifier(),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=1,
        error_score="raise",
        refit=True
    )

    search.fit(x, y)
    return search


def spectral_tv(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / np.maximum(
        a.sum(axis=-1, keepdims=True), 1e-20
    )
    b = b / np.maximum(
        b.sum(axis=-1, keepdims=True), 1e-20
    )

    return 0.5 * np.abs(a - b).sum(axis=-1)


def simulate(prob, bands, uncertainty, cfg):
    """
    Stop decisions at checkpoint k use only
    current and previous checkpoint information.
    """
    n = len(prob)

    if cfg["kind"] == "fixed":
        return np.full(
            n, cfg["index"], dtype=int
        )

    stops = np.full(n, 4, dtype=int)
    active = np.ones(n, dtype=bool)

    for k in (1, 2, 3):
        drift = spectral_tv(
            bands[:, k],
            bands[:, k - 1]
        )

        ready = drift <= cfg["drift_cap"]

        if cfg.get("margin") is not None:
            ready &= (
                np.abs(prob[:, k] - 0.5)
                >= cfg["margin"]
            )

        if cfg.get("uncertainty_cap") is not None:
            ready &= (
                uncertainty[:, k]
                <= cfg["uncertainty_cap"]
            )

        newly = active & ready
        stops[newly] = k
        active[newly] = False

    return stops


def select_training_policy(prob, bands, unc, y):
    reference = (prob[:, 1] >= 0.5).astype(int)

    ref_ba = balanced_accuracy_score(
        y, reference
    )
    ref_sensitivity = recall_score(
        y, reference, pos_label=1
    )

    drift60 = spectral_tv(
        bands[:, 1],
        bands[:, 0]
    )

    candidates = []

    for drift_q in (0.5, 0.75):
        cap = float(
            np.quantile(drift60, drift_q)
        )

        for kind in (
            "stability",
            "margin",
            "uncertainty",
            "combined"
        ):
            margins = (
                (0.05, 0.15)
                if kind in ("margin", "combined")
                else (None,)
            )

            uncertainty_qs = (
                (0.5, 0.75)
                if kind in ("uncertainty", "combined")
                else (None,)
            )

            for margin in margins:
                for uq in uncertainty_qs:
                    unc_cap = (
                        None if uq is None
                        else float(
                            np.quantile(unc[:, 1], uq)
                        )
                    )

                    candidates.append({
                        "kind": kind,
                        "drift_cap": cap,
                        "margin": margin,
                        "uncertainty_cap": unc_cap,
                        "name": (
                            f"{kind}_dq{drift_q}"
                            f"_m{margin}_uq{uq}"
                        )
                    })

    feasible = []

    for cfg in candidates:
        stops = simulate(
            prob, bands, unc, cfg
        )

        predicted = (
            prob[np.arange(len(stops)), stops]
            >= 0.5
        ).astype(int)

        ba = balanced_accuracy_score(
            y, predicted
        )

        sensitivity = recall_score(
            y, predicted, pos_label=1
        )

        mean_seconds = float(
            TIMES[stops].mean()
        )

        # Empirical training-only guardrails.
        # These are NOT clinical risk guarantees.
        allowed = (
            ba >= ref_ba - 0.03
            and sensitivity >= ref_sensitivity - 0.05
            and mean_seconds <= 120.0
        )

        if allowed:
            score = (
                ba - 0.04 * mean_seconds / 300
            )

            feasible.append((
                score,
                -mean_seconds,
                cfg["name"],
                cfg
            ))

    if not feasible:
        return (
            {
                "kind": "fixed",
                "index": 1,
                "name": "fixed_60_fallback"
            },
            ref_ba,
            ref_sensitivity,
            60.0,
            len(candidates),
            0
        )

    best = sorted(
        feasible,
        key=lambda x: (
            -x[0],
            -x[1],
            x[2]
        )
    )[0][3]

    stops = simulate(
        prob, bands, unc, best
    )

    return (
        best,
        ref_ba,
        ref_sensitivity,
        float(TIMES[stops].mean()),
        len(candidates),
        len(feasible)
    )


def stable_uniform(subject, repeat):
    value = f"{subject}|{repeat}|budget"

    digest = hashlib.sha256(
        value.encode()
    ).digest()

    return (
        int.from_bytes(digest[:8], "big")
        / 2**64
    )


data = pd.read_csv(SOURCE)

assert len(data) == 440
assert data.participant_id.nunique() == 88

assert not data.duplicated([
    "participant_id", "seconds"
]).any()

assert set(data.group) == {"A", "C", "F"}

assert np.isfinite(
    data[
        FEATURES
        + ["theta_epoch_iqr", "alpha_epoch_iqr"]
    ].to_numpy(dtype=float)
).all()

print(
    "===== TRAINING-ONLY RELIABILITY GATE =====",
    flush=True
)

print("Task: AD versus controls")
print("Outer CV: five folds, one exploratory repeat")
print("Stopping parameters: training-only")
print("Classifier probabilities: not calibrated")

participants = (
    data.loc[
        data.group.isin(["A", "C"]),
        ["participant_id", "group"]
    ]
    .drop_duplicates()
    .sort_values("participant_id")
    .reset_index(drop=True)
)

subjects = (
    participants.participant_id
    .astype(str)
    .to_numpy()
)

y = (
    participants.group.to_numpy() == "A"
).astype(int)

assert len(subjects) == 65
assert int(y.sum()) == 36

x_by_time = {}
unc_by_time = {}

for t in TIMES:
    panel = (
        data.loc[
            data.group.isin(["A", "C"])
            & data.seconds.eq(t)
        ]
        .set_index("participant_id")
        .loc[subjects]
    )

    assert np.array_equal(
        panel.group.to_numpy(),
        participants.group.to_numpy()
    )

    x_by_time[t] = panel[
        FEATURES
    ].to_numpy(dtype=float)

    unc_by_time[t] = (
        panel.theta_epoch_iqr.to_numpy(dtype=float)
        + panel.alpha_epoch_iqr.to_numpy(dtype=float)
    ) / np.sqrt(t / 10)

full_x = np.stack(
    [x_by_time[t] for t in TIMES],
    axis=1
)

full_unc = np.stack(
    [unc_by_time[t] for t in TIMES],
    axis=1
)

pred_rows = []
fold_rows = []

outer = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=1,
    random_state=20261107
)

for split_no, (train, test) in enumerate(
    outer.split(subjects, y)
):
    repeat = split_no // 5 + 1
    fold = split_no % 5 + 1

    y_train = y[train]

    crossfit = np.full(
        (len(train), len(TIMES)),
        np.nan
    )

    test_probs = np.full(
        (len(test), len(TIMES)),
        np.nan
    )

    meta = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=6100 + split_no
    )

    meta_splits = list(
        meta.split(train, y_train)
    )

    for time_idx, t in enumerate(TIMES):
        x_train = x_by_time[t][train]

        # Cross-fitting for training-only gate selection.
        # C=1 was fixed before these predictions.
        for meta_idx, (
            meta_train,
            meta_validation
        ) in enumerate(meta_splits):

            nested = classifier()
            nested.set_params(clf__C=1.0)

            nested.fit(
                x_train[meta_train],
                y_train[meta_train]
            )

            crossfit[
                meta_validation, time_idx
            ] = nested.predict_proba(
                x_train[meta_validation]
            )[:, 1]

        # Separately tune the final duration-specific
        # classifier using outer-training subjects.
        final = fit_search(
            x_train,
            y_train,
            3,
            seed=3049 + split_no
        )

        test_probs[:, time_idx] = (
            final.predict_proba(
                x_by_time[t][test]
            )[:, 1]
        )

    assert np.isfinite(crossfit).all()
    assert np.isfinite(test_probs).all()

    (
        learned,
        train_ba,
        train_sensitivity,
        train_mean,
        candidate_count,
        eligible_count
    ) = select_training_policy(
        crossfit,
        full_x[train],
        full_unc[train],
        y_train
    )

    drift75 = float(
        np.quantile(
            spectral_tv(
                full_x[train, 1],
                full_x[train, 0]
            ),
            0.75
        )
    )

    simple_stability = {
        "kind": "stability",
        "drift_cap": drift75
    }

    # Randomized comparator's expected budget
    # is determined from training subjects only.
    budget_probability = float(
        np.clip(
            (train_mean - 60) / 60,
            0,
            1
        )
    )

    fold_rows.append({
        "repeat": repeat,
        "fold": fold,
        "selected_gate": learned["name"],
        "candidate_count": candidate_count,
        "eligible_count": eligible_count,
        "training_reference_ba": train_ba,
        "training_reference_ad_sensitivity":
            train_sensitivity,
        "training_selected_mean_seconds":
            train_mean,
        "budget_random_probability_120":
            budget_probability
    })

    for policy in POLICIES:
        if policy.startswith("fixed_"):
            duration = int(
                policy.split("_")[1]
            )

            stop = simulate(
                test_probs,
                full_x[test],
                full_unc[test],
                {
                    "kind": "fixed",
                    "index": list(TIMES).index(duration)
                }
            )

        elif policy == "stability_only_75":
            stop = simulate(
                test_probs,
                full_x[test],
                full_unc[test],
                simple_stability
            )

        elif policy == "training_selected_gate":
            stop = simulate(
                test_probs,
                full_x[test],
                full_unc[test],
                learned
            )

        elif policy == "budget_random_60_120":
            stop = np.array([
                2 if stable_uniform(
                    subjects[i], repeat
                ) < budget_probability
                else 1
                for i in test
            ], dtype=int)

        else:
            raise ValueError(policy)

        for j, idx in enumerate(test):
            k = int(stop[j])
            p = float(test_probs[j, k])

            pred_rows.append({
                "task": "AD_vs_CN",
                "repeat": repeat,
                "fold": fold,
                "participant_id": subjects[idx],
                "y_true": int(y[idx]),
                "policy": policy,
                "stop_seconds": int(TIMES[k]),
                "probability_ad": p,
                "y_pred": int(p >= 0.5),
                "selected_gate": (
                    learned["name"]
                    if policy == "training_selected_gate"
                    else ""
                )
            })

    print(
        f"Completed fold {split_no + 1}/5; "
        f"gate={learned['name']}",
        flush=True
    )

pred = pd.DataFrame(pred_rows)
folds = pd.DataFrame(fold_rows)

assert len(pred) == 65 * len(POLICIES)
assert len(folds) == 5

assert not pred.duplicated([
    "repeat",
    "participant_id",
    "policy"
]).any()

# Reproduce previously frozen epoch-median
# fixed-duration predictions on the same folds.
if not REFERENCE.is_file():
    raise FileNotFoundError(
        "Frozen reference predictions are missing."
    )

previous = pd.read_csv(REFERENCE)

previous = previous.loc[
    previous.task.eq("AD_vs_CN")
    & previous.representation.eq("epoch_median")
    & previous.repeat.eq(1)
].copy()

previous["participant_id"] = (
    previous.participant_id.astype(str)
)

fixed = pred.loc[
    pred.policy.str.startswith("fixed_")
].copy()

fixed["seconds"] = fixed.stop_seconds

merged = fixed.merge(
    previous[
        [
            "repeat",
            "fold",
            "participant_id",
            "seconds",
            "probability_ad"
        ]
    ],
    on=[
        "repeat",
        "fold",
        "participant_id",
        "seconds"
    ],
    validate="one_to_one",
    suffixes=("_new", "_prior")
)

assert len(merged) == 65 * 5

deviation = float(
    np.max(
        np.abs(
            merged.probability_ad_new
            - merged.probability_ad_prior
        )
    )
)

print(
    "Frozen fixed-policy replication error:",
    deviation
)

assert deviation < 1e-8

metric_rows = []

for (repeat, policy), frame in pred.groupby(
    ["repeat", "policy"]
):
    actual = frame.y_true.to_numpy(dtype=int)
    estimate = frame.y_pred.to_numpy(dtype=int)
    probability = (
        frame.probability_ad.to_numpy(dtype=float)
    )

    assert len(frame) == 65
    assert len(np.unique(actual)) == 2

    metric_rows.append({
        "task": "AD_vs_CN",
        "repeat": int(repeat),
        "policy": policy,
        "participants": len(frame),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, estimate)
        ),
        "ad_sensitivity": float(
            recall_score(
                actual, estimate, pos_label=1
            )
        ),
        "control_specificity": float(
            recall_score(
                actual, estimate, pos_label=0
            )
        ),
        "auc": float(
            roc_auc_score(actual, probability)
        ),
        "mean_seconds": float(
            frame.stop_seconds.mean()
        ),
        "early_stop_fraction": float(
            (frame.stop_seconds < 300).mean()
        )
    })

metrics = pd.DataFrame(metric_rows)

assert len(metrics) == len(POLICIES)

assert np.isfinite(
    metrics.select_dtypes(
        include="number"
    ).to_numpy()
).all()

OUT_PRED.parent.mkdir(
    parents=True,
    exist_ok=True
)

pred.to_csv(
    OUT_PRED,
    index=False,
    float_format="%.12g"
)

folds.to_csv(
    OUT_FOLDS,
    index=False,
    float_format="%.12g"
)

metrics.to_csv(
    OUT_METRICS,
    index=False,
    float_format="%.12g"
)

print()
print("===== TRAINING-ONLY GATE POLICY RESULTS =====")

print(
    metrics.groupby("policy")[
        [
            "balanced_accuracy",
            "ad_sensitivity",
            "mean_seconds"
        ]
    ]
    .mean()
    .round(4)
    .to_string()
)

print()
print("===== TRAINING GATE SELECTION FREQUENCY =====")

print(
    folds.selected_gate
    .value_counts()
    .to_string()
)

print()
print("Unique participants: 65")
print("Outer CV repeats: 1")
print("Results: exploratory")
print("Training thresholds are not clinical guarantees.")
print("No confirmatory significance or novelty claim.")
print("P2 TRAINING-ONLY RELIABILITY GATE COMPLETE")
