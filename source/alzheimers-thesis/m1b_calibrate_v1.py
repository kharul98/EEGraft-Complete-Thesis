#!/usr/bin/env python3
"""Milestone 1B: outer-subject-held-out SVM sigmoid calibration."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE = ROOT / "reports/thesis_final"

OUT = BASE / "milestone_1b_calibration_v1"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "reports/openneuro_5min_manifest_v2.csv"

STATES = (
    BASE / "milestone_1a_reliability_v1"
    / "subject_repeat_states.csv"
)

TASKS = {
    "AD_vs_HC": {
        "positive": "A",
        "n": 46,
        "features":
            BASE / "duration_locked_v1/duration_features.csv",
        "blocks":
            BASE / "duration_locked_v1/five_60s_blocks.csv",
        "reference":
            BASE / "temporal_repeated_v1"
            / "all_repeats_heldout_block_predictions.csv",
    },
    "FTD_vs_HC": {
        "positive": "F",
        "n": 41,
        "features":
            BASE / "ftd_extension_v1/locked_study"
            / "duration_features.csv",
        "blocks":
            BASE / "ftd_extension_v1/locked_study"
            / "five_60s_blocks.csv",
        "reference":
            BASE / "ftd_extension_v1/locked_study"
            / "held_out_60s_block_predictions.csv",
    },
}

FEATS = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
)

REPEATS = 5
NBINS = 5
NBOOT = 3000


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def index(rows, keyfn, desc):
    d = {}

    for row in rows:
        k = keyfn(row)

        if k in d:
            raise RuntimeError(
                f"Duplicate {desc}: {k}"
            )

        d[k] = row

    return d


def save(name, rows):
    if not rows:
        raise RuntimeError(
            "Empty result: " + name
        )

    keys = list(dict.fromkeys(
        k for r in rows for k in r
    ))

    with (OUT / name).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=keys
        )
        w.writeheader()
        w.writerows(rows)


def bacc(y, label):
    sensitivity = (
        label[y == 1] == 1
    ).mean()

    specificity = (
        label[y == 0] == 0
    ).mean()

    return float(
        0.5 * (
            sensitivity + specificity
        )
    )


def ece(y, p, n_bins=NBINS):
    bins = np.minimum(
        (p * n_bins).astype(int),
        n_bins - 1
    )

    return float(
        sum(
            (bins == b).mean()
            * abs(
                p[bins == b].mean()
                - y[bins == b].mean()
            )
            for b in range(n_bins)
            if np.any(bins == b)
        )
    )


def classifier():
    return Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "clf",
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        ),
    ])


def metrics(y, p, decision, prior):
    clipped = np.clip(
        p,
        1e-9,
        1 - 1e-9
    )

    new_label = (
        p >= 0.5
    ).astype(int)

    return {
        "original_svm_bacc":
            bacc(y, decision),

        "calibrated_threshold_bacc":
            bacc(y, new_label),

        "changed_prediction_fraction":
            float(
                (new_label != decision).mean()
            ),

        "brier":
            float(
                np.mean((p - y) ** 2)
            ),

        "training_prior_brier":
            float(
                np.mean((prior - y) ** 2)
            ),

        "nll":
            float(
                -np.mean(
                    y * np.log(clipped)
                    + (1-y) * np.log(1-clipped)
                )
            ),

        "ece_five_equal_width_bins":
            ece(y, p),

        "auc":
            float(
                roc_auc_score(y, p)
            ),

        "average_probability":
            float(p.mean()),
    }


# =========================================
# LOCK PROTOCOL AND SOURCE CHECKSUMS
# =========================================

protocol = {
    "version": "milestone_1b_v1",

    "status":
        "post-hoc exploratory, original diagnostic cohorts",

    "outer_cv":
        "5 repeats x 5 subject folds, "
        "identical to prior SVM study",

    "model_selection":
        "same train-only 3-fold grid, BACC scoring",

    "calibration":
        "fixed C=1 sigmoid logistic regression "
        "on 3-fold out-of-fold first-minute "
        "training margins",

    "test":
        "held-out participants only; "
        "5 nonoverlapping 60-second blocks",

    "primary_calibration_check":
        "first-minute Brier, NLL, ECE(5), "
        "AUC and training-fold prevalence baseline",

    "important":
        "calibration may change the p>=0.5 "
        "decision; raw SVM labels remain unchanged",

    "limitations":
        "same dataset reused; small training folds; "
        "no external test; conditional estimates",
}

lock = OUT / "locked_protocol.json"

encoded = (
    json.dumps(
        protocol,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if lock.exists():
    if lock.read_text() != encoded:
        raise RuntimeError(
            "Locked protocol changed"
        )
else:
    lock.write_text(encoded)

paths = {
    "manifest": MANIFEST,
    "m1a_states": STATES
}

for name, cfg in TASKS.items():
    for label in (
        "features",
        "blocks",
        "reference"
    ):
        paths[
            f"{name}_{label}"
        ] = cfg[label]

checksums = {
    name: hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    for name, path in paths.items()
    if path.is_file()
}

if len(checksums) != len(paths):
    raise RuntimeError(
        "Missing required input file"
    )

hash_path = OUT / "input_sha256.json"

hashtext = (
    json.dumps(
        checksums,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if hash_path.exists():
    if hash_path.read_text() != hashtext:
        raise RuntimeError(
            "Inputs changed since previous execution"
        )
else:
    hash_path.write_text(hashtext)

manifest = read(MANIFEST)

if (
    len(manifest) != 88
    or len({
        r["subject"] for r in manifest
    }) != 88
):
    raise RuntimeError(
        "Manifest invalid"
    )

states = index(
    read(STATES),
    lambda r: (
        r["task"],
        r["subject"],
        int(r["repeat"])
    ),
    "milestone 1A state"
)

print(
    "===== MILESTONE 1B: TRAINING-ONLY SVM CALIBRATION =====",
    flush=True
)

all_probs = []
all_folds = []
all_metrics = []
all_states = []


# =========================================
# EACH DIAGNOSTIC TASK
# =========================================

for task, cfg in TASKS.items():

    cohort = {
        r["subject"]: r["group"]
        for r in manifest
        if (
            r["group"] in (
                cfg["positive"], "C"
            )
            and r["status"] == "CANDIDATE"
        )
    }

    expected = (
        {"A": 23, "C": 23}
        if task == "AD_vs_HC"
        else {"F": 18, "C": 23}
    )

    if Counter(cohort.values()) != expected:
        raise RuntimeError(
            "Cohort changed for " + task
        )

    ids = sorted(cohort)
    n = len(ids)

    y = np.asarray([
        int(
            cohort[s] == cfg["positive"]
        )
        for s in ids
    ], dtype=int)

    duration = index(
        (
            r for r in read(cfg["features"])
            if int(r["duration_sec"]) == 60
        ),
        lambda r: r["subject"],
        "first-minute features"
    )

    block = index(
        read(cfg["blocks"]),
        lambda r: (
            r["subject"],
            int(r["block_number"])
        ),
        "block features"
    )

    reference = index(
        read(cfg["reference"]),
        lambda r: (
            int(r["repeat"]),
            r["subject"],
            int(r["block"])
        ),
        "prior prediction"
    )

    if (
        set(duration) != set(ids)
        or len(block) != n * 5
        or len(reference) != REPEATS * n * 5
    ):
        raise RuntimeError(
            task + ": incorrect file coverage"
        )

    X = np.asarray([
        [
            float(duration[s][f])
            for f in FEATS
        ]
        for s in ids
    ], dtype=float)

    XB = np.asarray([
        [
            [
                float(
                    block[(s, b)][f]
                )
                for f in FEATS
            ]
            for b in range(1, 6)
        ]
        for s in ids
    ], dtype=float)

    if (
        not np.isfinite(X).all()
        or not np.isfinite(XB).all()
        or not np.allclose(
            X,
            XB[:, 0, :],
            atol=1e-12,
            rtol=0
        )
    ):
        raise RuntimeError(
            "Feature consistency failed for " + task
        )

    probs = np.full(
        (REPEATS, n, 5),
        np.nan
    )

    raw = np.full_like(
        probs,
        np.nan
    )

    original = np.full(
        (REPEATS, n, 5),
        -1,
        dtype=int
    )

    priors = np.full(
        (REPEATS, n),
        np.nan
    )

    fold_ids = np.full(
        (REPEATS, n),
        -1,
        dtype=int
    )

    # =====================================
    # MATCH ORIGINAL OUTER CV
    # =====================================

    for rep in range(REPEATS):

        outer = StratifiedKFold(
            5,
            shuffle=True,
            random_state=42 + 100 * rep
        )

        for fold, (tr, te) in enumerate(
            outer.split(X, y),
            1
        ):

            inner = StratifiedKFold(
                3,
                shuffle=True,
                random_state=(
                    1000 + 100 * rep + fold
                )
            )

            # Original SVM selection.
            # Only outer-training subjects.
            search = GridSearchCV(
                classifier(),
                {
                    "clf__C": [
                        0.1, 1.0, 10.0
                    ],
                    "clf__gamma": [
                        "scale", 0.01, 0.1
                    ]
                },
                cv=inner,
                scoring="balanced_accuracy",
                refit=True,
                n_jobs=1,
                error_score="raise"
            )

            search.fit(
                X[tr],
                y[tr]
            )

            model = search.best_estimator_

            xb = XB[te].reshape(
                -1,
                len(FEATS)
            )

            z = model.decision_function(
                xb
            ).reshape(
                len(te), 5
            )

            hard = model.predict(
                xb
            ).reshape(
                len(te), 5
            )

            # Exact held-out reproduction.
            for j, i in enumerate(te):

                sid = ids[i]

                for b in range(5):

                    ref = reference[
                        (rep + 1, sid, b + 1)
                    ]

                    if (
                        int(ref["fold"]) != fold
                        or int(ref["true_label"]) != y[i]
                        or ref["group"] != cohort[sid]
                    ):
                        raise RuntimeError(
                            "Label/fold mismatch: "
                            + task + " " + sid
                        )

                    if (
                        int(ref["predicted_label"])
                        != int(hard[j, b])
                        or not np.isclose(
                            float(ref["decision_score"]),
                            z[j, b],
                            atol=1e-7,
                            rtol=0
                        )
                    ):
                        raise RuntimeError(
                            "Historical SVM reproduction failed: "
                            + task
                            + f" repeat {rep+1} "
                            + sid
                            + f" block {b+1}"
                        )

            # =================================
            # CROSS-FITTED CALIBRATION SCORES
            # =================================

            oof = np.full(
                len(tr),
                np.nan
            )

            for local_train, local_val in inner.split(
                X[tr],
                y[tr]
            ):

                submodel = clone(model)

                submodel.fit(
                    X[tr[local_train]],
                    y[tr[local_train]]
                )

                oof[local_val] = (
                    submodel.decision_function(
                        X[tr[local_val]]
                    )
                )

            if not np.isfinite(oof).all():
                raise RuntimeError(
                    "Incomplete crossfit margins"
                )

            # Fixed calibration method.
            # No held-out subjects involved.
            calibrator = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000
            )

            calibrator.fit(
                oof.reshape(-1, 1),
                y[tr]
            )

            p = calibrator.predict_proba(
                z.reshape(-1, 1)
            )[:, 1].reshape(
                len(te), 5
            )

            if (
                not np.isfinite(p).all()
                or np.any(p <= 0)
                or np.any(p >= 1)
            ):
                raise RuntimeError(
                    "Invalid calibrated probability"
                )

            probs[rep, te] = p
            raw[rep, te] = z
            original[rep, te] = hard

            priors[rep, te] = float(
                y[tr].mean()
            )

            fold_ids[rep, te] = fold

            all_folds.append({
                "task": task,
                "repeat": rep + 1,
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "best_C":
                    search.best_params_["clf__C"],
                "best_gamma":
                    search.best_params_["clf__gamma"],
                "calibrator_coef":
                    float(calibrator.coef_[0, 0]),
                "calibrator_intercept":
                    float(calibrator.intercept_[0]),
                "calibration_train_oof_auc":
                    float(roc_auc_score(
                        y[tr],
                        oof
                    )),
                "negative_slope_warning":
                    int(
                        calibrator.coef_[0, 0] < 0
                    )
            })

        print(
            f"{task}: reproduced and calibrated "
            f"CV repeat {rep+1}/{REPEATS}",
            flush=True
        )

    if (
        not np.isfinite(probs).all()
        or not np.isfinite(raw).all()
        or np.any(original < 0)
        or np.any(fold_ids < 1)
    ):
        raise RuntimeError(
            "Incomplete held-out arrays"
        )

    # =====================================
    # SAVE PARTICIPANT-LEVEL PROBABILITIES
    # =====================================

    for rep in range(REPEATS):

        for i, sid in enumerate(ids):

            for b in range(5):

                p = float(
                    probs[rep, i, b]
                )

                all_probs.append({
                    "task": task,
                    "repeat": rep + 1,
                    "subject": sid,
                    "group": cohort[sid],
                    "true_label": int(y[i]),
                    "fold": int(
                        fold_ids[rep, i]
                    ),
                    "minute": b + 1,
                    "original_svm_label": int(
                        original[rep, i, b]
                    ),
                    "original_svm_decision_score":
                        float(raw[rep, i, b]),
                    "calibrated_positive_probability": p,
                    "calibrated_threshold_label":
                        int(p >= 0.5),
                    "calibrated_confidence":
                        max(p, 1-p),
                    "calibrated_entropy_nats":
                        float(
                            -(
                                p * np.log(p)
                                + (1-p) * np.log(1-p)
                            )
                        ),
                    "train_fold_prevalence":
                        float(priors[rep, i])
                })

            state = states.get(
                (task, sid, rep + 1)
            )

            if state is None:
                raise RuntimeError(
                    "Missing 1A state for " + sid
                )

            q = float(
                probs[rep, i, 0]
            )

            all_states.append({
                "task": task,
                "repeat": rep + 1,
                "subject": sid,
                "group": cohort[sid],
                "state": state["state"],
                "first_minute_calibrated_confidence":
                    max(q, 1-q),
                "first_minute_calibrated_entropy":
                    float(
                        -(
                            q*np.log(q)
                            + (1-q)*np.log(1-q)
                        )
                    ),
                "five_minute_probability_sd":
                    float(
                        probs[rep, i].std(
                            ddof=1
                        )
                    ),
                "correct_first_minute_original":
                    int(
                        original[rep, i, 0]
                        == y[i]
                    )
            })

        for b in range(5):

            result = metrics(
                y,
                probs[rep, :, b],
                original[rep, :, b],
                priors[rep]
            )

            all_metrics.append({
                "task": task,
                "repeat": rep + 1,
                "minute": b + 1,
                "n_subjects": n,
                **result
            })

    # =====================================
    # PRINT CALIBRATION RESULTS
    # =====================================

    print(
        "\n===== "
        + task
        + " CALIBRATION RESULTS ====="
    )

    for b in range(5):

        runs = [
            r for r in all_metrics
            if (
                r["task"] == task
                and r["minute"] == b + 1
            )
        ]

        def avg(field):
            return float(np.mean([
                r[field] for r in runs
            ]))

        print(
            f"minute {b+1}: "
            f"original BACC={avg('original_svm_bacc'):.4f}, "
            f"calibrated-label BACC="
            f"{avg('calibrated_threshold_bacc'):.4f}, "
            f"Brier={avg('brier'):.4f}, "
            f"prior Brier={avg('training_prior_brier'):.4f}, "
            f"ECE5={avg('ece_five_equal_width_bins'):.4f}, "
            f"NLL={avg('nll'):.4f}, "
            f"AUC={avg('auc'):.4f}"
        )

    print(
        "\n===== "
        + task
        + " UNCERTAINTY BY 1A STATE ====="
    )

    for state_name in (
        "stable_correct",
        "stable_wrong",
        "unstable"
    ):

        selected = [
            r for r in all_states
            if (
                r["task"] == task
                and r["state"] == state_name
            )
        ]

        if not selected:
            continue

        print(
            f"{state_name}: "
            f"n_subject_repeats={len(selected)}, "
            "mean first-minute confidence="
            f"{np.mean([r['first_minute_calibrated_confidence'] for r in selected]):.4f}, "
            "mean first-minute entropy="
            f"{np.mean([r['first_minute_calibrated_entropy'] for r in selected]):.4f}, "
            "mean five-minute probability SD="
            f"{np.mean([r['five_minute_probability_sd'] for r in selected]):.4f}"
        )

    # =====================================
    # FIRST-MINUTE RELIABILITY DIAGRAM
    # =====================================

    flat_p = probs[:, :, 0].reshape(-1)

    flat_y = np.broadcast_to(
        y,
        (REPEATS, n)
    ).reshape(-1)

    bins = np.minimum(
        (flat_p * NBINS).astype(int),
        NBINS - 1
    )

    xs = []
    ys = []

    for b in range(NBINS):

        mask = bins == b

        if mask.any():
            xs.append(
                float(
                    flat_p[mask].mean()
                )
            )

            ys.append(
                float(
                    flat_y[mask].mean()
                )
            )

    fig, ax = plt.subplots(
        figsize=(5.7, 4.7)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Ideal calibration"
    )

    ax.plot(
        xs,
        ys,
        "o-",
        label="Held-out first-minute estimates"
    )

    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted positive probability",
        ylabel="Observed positive fraction",
        title=task + " first-minute reliability (5 bins)"
    )

    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(
        OUT / (
            task
            + "_first_minute_reliability.png"
        ),
        dpi=190
    )

    plt.close(fig)

    # =====================================
    # BRIER SCORE PAIRED UNCERTAINTY
    # =====================================

    per_subject = np.mean(
        (probs[:, :, 0] - y[None, :])**2
        -
        (priors - y[None, :])**2,
        axis=0
    )

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    rng = np.random.default_rng(20261107)

    bootstrap = np.empty(NBOOT)

    for z in range(NBOOT):

        sampled = np.r_[
            rng.choice(
                pos,
                len(pos),
                replace=True
            ),
            rng.choice(
                neg,
                len(neg),
                replace=True
            )
        ]

        bootstrap[z] = (
            per_subject[sampled].mean()
        )

    lo, hi = np.percentile(
        bootstrap,
        [2.5, 97.5]
    )

    print(
        f"{task} first-minute Brier minus "
        "training-prior Brier: "
        f"{per_subject.mean():+.4f}; "
        "conditional paired 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )


# =========================================
# SAVE CANONICAL CALIBRATION OUTPUTS
# =========================================

save(
    "calibrated_held_out_block_predictions.csv",
    all_probs
)

save(
    "calibration_fold_audit.csv",
    all_folds
)

save(
    "calibration_metrics_by_repeat_minute.csv",
    all_metrics
)

save(
    "uncertainty_by_subject_repeat.csv",
    all_states
)

print(
    "\n===== MILESTONE 1B COMPLETE ====="
)

print(
    "Saved:",
    OUT
)

print(
    "All historical hard predictions and "
    "SVM margins reproduced; none modified."
)

print(
    "Sigmoid calibration trained without "
    "held-out participant labels."
)

print(
    "Small-cohort conditional internal validation; "
    "no clinical calibration claim."
)

