#!/usr/bin/env python3

"""Separate, reproducible 60/180/300-second spectral EEG study."""

from pathlib import Path
from collections import Counter
import csv
import json

import numpy as np
import mne

from scipy.signal import welch

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
REPORTS = ROOT / "reports"

OUT = REPORTS / "thesis_final" / "duration_locked_v1"
OUT.mkdir(parents=True, exist_ok=True)

REPEATS = 5
SEED = 42

DURATIONS = (60, 180, 300)

FEATURES = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "theta_alpha_ratio"
)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, records):
    if not records:
        raise RuntimeError("No rows to save: " + str(path))

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


# ==========================================
# FEATURE AGGREGATION
# ==========================================

def record_features(
    relative,
    clean,
    a,
    b,
    sid,
    group,
    duration,
    block=None
):

    allowed = clean[a:b]
    n = int(allowed.sum())
    n_total = b - a

    # Predefined minimum of 50% usable epochs
    # and at least ten clean epochs.
    if n < max(10, n_total // 2):

        if block is not None:
            return None

        raise RuntimeError(
            f"{sid} has only {n}/{n_total} "
            f"clean epochs in the first {duration}s. "
            "Stop and review."
        )

    values = {
        name: float(
            np.median(v[a:b][allowed])
        )
        for name, v in relative.items()
    }

    values["theta_alpha_ratio"] = (
        values["theta"]
        / max(values["alpha"], 1e-12)
    )

    ep_theta = np.median(
        relative["theta"][a:b],
        axis=1
    )[allowed]

    ep_alpha = np.median(
        relative["alpha"][a:b],
        axis=1
    )[allowed]

    log_ratio = np.log(
        (ep_theta + 1e-12)
        / (ep_alpha + 1e-12)
    )

    mad = float(
        1.4826
        * np.median(
            np.abs(
                log_ratio - np.median(log_ratio)
            )
        )
    )

    return {
        "subject": sid,
        "group": group,
        "duration_sec": duration,
        "block_number": "" if block is None else block,
        "clean_epochs": n,
        "clean_fraction": n / n_total,
        "theta_alpha_epoch_log_mad": mad,
        **values
    }


# ==========================================
# EXTRACT CONSISTENT DURATION FEATURES
# ==========================================

def extract(manifest, legacy):

    main_rows = []
    block_rows = []
    differences = []

    for idx, row in enumerate(manifest, 1):

        sid = row["subject"]
        group = row["group"]

        source = Path(row["file"])

        if not source.is_absolute():
            source = ROOT / source

        raw = mne.io.read_raw_eeglab(
            str(source),
            preload=False,
            verbose="ERROR"
        )

        sf = float(raw.info["sfreq"])

        picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[]
        )

        if abs(sf - 500) > 0.01 or len(picks) != 19:
            raise RuntimeError(
                f"{sid}: unexpected sampling rate "
                "or channel count"
            )

        start = int(
            round(float(row["start_sec"]) * sf)
        )

        end = start + 300 * 500

        if start < 0 or end > raw.n_times:
            raise RuntimeError(
                f"{sid}: invalid recording interval"
            )

        signal = raw.get_data(
            picks=picks,
            start=start,
            stop=end
        ) * 1e6

        if (
            signal.shape != (19, 150000)
            or not np.isfinite(signal).all()
        ):
            raise RuntimeError(
                f"{sid}: invalid EEG samples"
            )

        # 150 x 19 x 1000:
        # epochs, channels, samples.

        epochs = signal.reshape(
            19, 150, 1000
        ).transpose(1, 0, 2)

        ptp = np.ptp(epochs, axis=-1)
        sd = epochs.std(axis=-1)

        clean = ~(
            ((ptp > 200).sum(axis=1) >= 3)
            | (ptp > 300).any(axis=1)
            | (sd < 0.5).any(axis=1)
        )

        # Verify the artifact screen against
        # the original saved baseline.

        expected = int(
            float(legacy[sid]["clean_epochs"])
        )

        if int(clean.sum()) != expected:
            raise RuntimeError(
                f"{sid}: clean-epoch count "
                f"{int(clean.sum())} != {expected}. "
                "Stop and review."
            )

        freqs, psd = welch(
            epochs,
            fs=500,
            nperseg=1000,
            noverlap=0,
            axis=-1,
            window="hann",
            detrend="constant",
            scaling="density"
        )

        base_mask = (
            (freqs >= 1)
            & (freqs <= 45)
        )

        denominator = psd[
            ..., base_mask
        ].sum(axis=-1)

        if np.any(denominator <= 0):
            raise RuntimeError(
                f"{sid}: invalid spectral power"
            )

        relative = {}

        for name, (lo, hi) in BANDS.items():

            if name == "gamma":
                mask = (
                    (freqs >= lo)
                    & (freqs <= hi)
                )
            else:
                mask = (
                    (freqs >= lo)
                    & (freqs < hi)
                )

            relative[name] = (
                psd[..., mask].sum(axis=-1)
                / denominator
            )

        # Main nested duration conditions.
        for seconds in DURATIONS:

            main_rows.append(
                record_features(
                    relative,
                    clean,
                    0,
                    seconds // 2,
                    sid,
                    group,
                    seconds
                )
            )

        # Five nonoverlapping 60s blocks.
        for b in range(5):

            record = record_features(
                relative,
                clean,
                b * 30,
                (b + 1) * 30,
                sid,
                group,
                60,
                b + 1
            )

            if record is None:

                segment = clean[
                    b * 30:(b + 1) * 30
                ]

                record = {
                    "subject": sid,
                    "group": group,
                    "duration_sec": 60,
                    "block_number": b + 1,
                    "clean_epochs": int(segment.sum()),
                    "clean_fraction": float(segment.mean()),
                    "theta_alpha_epoch_log_mad": "",
                    **{
                        feature: ""
                        for feature in FEATURES
                    }
                }

            block_rows.append(record)

        derived = main_rows[-1]

        diffs = [
            abs(
                derived[name]
                - float(legacy[sid][name])
            )
            for name in FEATURES
        ]

        differences.append(
            (sid, max(diffs))
        )

        if idx % 10 == 0 or idx == len(manifest):
            print(
                f"EEG features extracted: "
                f"{idx}/{len(manifest)}",
                flush=True
            )

    if len(main_rows) != 138 or len(block_rows) != 230:
        raise RuntimeError(
            "Unexpected duration-feature row count"
        )

    save_csv(
        OUT / "duration_features.csv",
        main_rows
    )

    save_csv(
        OUT / "five_60s_blocks.csv",
        block_rows
    )

    print(
        "Maximum 300s difference from historical features:",
        f"{max(v for _, v in differences):.8g}"
    )

    print(
        "NOTE: This is a separate consistent "
        "duration protocol, not an exact "
        "reproduction of the historical baseline."
    )

    return main_rows, block_rows


# ==========================================
# CLASSIFICATION MODELS
# ==========================================

def model_specs():

    return {
        "LogisticRegression": (
            Pipeline([
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),
            {
                "clf__C": [
                    0.01, 0.1, 1.0, 10.0
                ]
            }
        ),

        "RBF_SVM": (
            Pipeline([
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced"
                    )
                )
            ]),
            {
                "clf__C": [
                    0.1, 1.0, 10.0
                ],
                "clf__gamma": [
                    "scale", 0.01, 0.1
                ]
            }
        )
    }


# ==========================================
# MATCHED REPEATED NESTED CV
# ==========================================

def evaluate(main_rows, block_rows, ids, y):

    main = {
        (r["subject"], r["duration_sec"]): r
        for r in main_rows
    }

    blocks = {
        (r["subject"], r["block_number"]): r
        for r in block_rows
    }

    arrays = {
        duration: np.array([
            [
                main[(sid, duration)][f]
                for f in FEATURES
            ]
            for sid in ids
        ], dtype=float)
        for duration in DURATIONS
    }

    models = model_specs()

    results = []
    predictions = []
    heldout_blocks = []

    print(
        "\n===== MATCHED REPEATED NESTED CV =====",
        flush=True
    )

    for repeat in range(REPEATS):

        outer = StratifiedKFold(
            5,
            shuffle=True,
            random_state=SEED + repeat * 100
        )

        splits = list(
            outer.split(
                np.zeros(len(y)),
                y
            )
        )

        for duration in DURATIONS:

            X = arrays[duration]

            for name, (pipeline, params) in models.items():

                pred = np.full(
                    len(y),
                    -1,
                    dtype=int
                )

                score = np.full(
                    len(y),
                    np.nan
                )

                fold_id = np.zeros(
                    len(y),
                    dtype=int
                )

                fold_aucs = []

                for fold, (train, test) in enumerate(
                    splits, 1
                ):

                    if set(train).intersection(test):
                        raise RuntimeError(
                            "Participant leakage detected"
                        )

                    inner = StratifiedKFold(
                        3,
                        shuffle=True,
                        random_state=(
                            1000
                            + repeat * 100
                            + fold
                        )
                    )

                    search = GridSearchCV(
                        pipeline,
                        params,
                        cv=inner,
                        scoring="balanced_accuracy",
                        refit=True,
                        n_jobs=1,
                        error_score="raise"
                    )

                    search.fit(
                        X[train],
                        y[train]
                    )

                    fitted = search.best_estimator_

                    pred[test] = fitted.predict(
                        X[test]
                    )

                    score[test] = (
                        fitted.decision_function(
                            X[test]
                        )
                    )

                    fold_id[test] = fold

                    fold_aucs.append(
                        float(
                            roc_auc_score(
                                y[test],
                                score[test]
                            )
                        )
                    )

                    # Test nonoverlapping 60s blocks
                    # using the already fitted model.
                    # Never train on held-out blocks.

                    if (
                        repeat == 0
                        and duration == 60
                        and name == "RBF_SVM"
                    ):

                        for subject_idx in test:

                            sid = ids[int(subject_idx)]

                            for b in range(1, 6):

                                r = blocks[(sid, b)]

                                record = {
                                    "subject": sid,
                                    "group":
                                        "A" if y[subject_idx] else "C",
                                    "fold": fold,
                                    "block": b,
                                    "clean_epochs": r["clean_epochs"]
                                }

                                if r["theta_alpha_ratio"] == "":

                                    record.update({
                                        "predicted_ad": "",
                                        "decision_score": ""
                                    })

                                else:

                                    xb = np.array([
                                        [
                                            r[f]
                                            for f in FEATURES
                                        ]
                                    ], dtype=float)

                                    record.update({
                                        "predicted_ad": int(
                                            fitted.predict(xb)[0]
                                        ),
                                        "decision_score": float(
                                            fitted.decision_function(
                                                xb
                                            )[0]
                                        )
                                    })

                                heldout_blocks.append(
                                    record
                                )

                if (
                    not np.isfinite(score).all()
                    or (pred < 0).any()
                    or (fold_id == 0).any()
                ):
                    raise RuntimeError(
                        "Incomplete CV predictions"
                    )

                tn, fp, fn, tp = [
                    int(v)
                    for v in confusion_matrix(
                        y,
                        pred,
                        labels=[0, 1]
                    ).ravel()
                ]

                results.append({
                    "repeat": repeat + 1,
                    "model": name,
                    "duration_sec": duration,
                    "accuracy": float(
                        accuracy_score(y, pred)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y, pred)
                    ),
                    "mean_fold_auc": float(
                        np.mean(fold_aucs)
                    ),
                    "sensitivity": tp / (tp + fn),
                    "specificity": tn / (tn + fp),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp
                })

                for j, sid in enumerate(ids):

                    predictions.append({
                        "repeat": repeat + 1,
                        "subject": sid,
                        "group": "A" if y[j] else "C",
                        "model": name,
                        "duration_sec": duration,
                        "fold": int(fold_id[j]),
                        "true_label": int(y[j]),
                        "predicted_label": int(pred[j]),
                        "decision_score": float(score[j])
                    })

        print(
            f"Completed repeat {repeat + 1}/{REPEATS}",
            flush=True
        )

    save_csv(
        OUT / "nested_cv_runs.csv",
        results
    )

    save_csv(
        OUT / "held_out_predictions.csv",
        predictions
    )

    save_csv(
        OUT / "held_out_60s_blocks.csv",
        heldout_blocks
    )

    print(
        "\n===== DURATION CLASSIFICATION RESULTS ====="
    )

    for name in models:

        print("MODEL:", name)

        for duration in DURATIONS:

            rows = [
                r for r in results
                if r["model"] == name
                and r["duration_sec"] == duration
            ]

            bacc = np.array([
                r["balanced_accuracy"]
                for r in rows
            ])

            auc = np.array([
                r["mean_fold_auc"]
                for r in rows
            ])

            print(
                f"  {duration}s: "
                f"BACC mean={bacc.mean():.4f}, "
                f"SD={bacc.std(ddof=1):.4f}, "
                f"mean fold AUC={auc.mean():.4f}"
            )

    return results, predictions, heldout_blocks


# ==========================================
# PAIRED DURATION COMPARISONS
# ==========================================

def paired_changes(predictions, ids, y):

    lookup = {
        (
            r["repeat"],
            r["subject"],
            r["model"],
            r["duration_sec"]
        ): r
        for r in predictions
    }

    rng = np.random.default_rng(20261107)

    ad = np.flatnonzero(y == 1)
    hc = np.flatnonzero(y == 0)

    rows = []

    print(
        "\n===== PAIRED DURATION CHANGES ====="
    )

    for model in (
        "RBF_SVM",
        "LogisticRegression"
    ):

        for short in (60, 180):

            diff = []

            for j, sid in enumerate(ids):

                changes = []

                for repeat in range(1, REPEATS + 1):

                    a = (
                        lookup[
                            (repeat, sid, model, 300)
                        ]["predicted_label"] == int(y[j])
                    )

                    b = (
                        lookup[
                            (repeat, sid, model, short)
                        ]["predicted_label"] == int(y[j])
                    )

                    changes.append(
                        int(a) - int(b)
                    )

                diff.append(
                    float(np.mean(changes))
                )

            diff = np.array(diff)

            observed = (
                diff[ad].mean()
                + diff[hc].mean()
            ) / 2

            boot = []

            for _ in range(3000):

                da = rng.choice(
                    ad,
                    size=len(ad),
                    replace=True
                )

                dc = rng.choice(
                    hc,
                    size=len(hc),
                    replace=True
                )

                boot.append(
                    (
                        diff[da].mean()
                        + diff[dc].mean()
                    ) / 2
                )

            lo, hi = np.percentile(
                boot,
                [2.5, 97.5]
            )

            rows.append({
                "model": model,
                "comparison": f"300s minus {short}s",
                "delta_balanced_accuracy": observed,
                "conditional_ci_low": float(lo),
                "conditional_ci_high": float(hi)
            })

            print(
                f"{model}: 300s - {short}s "
                f"BACC={observed:+.4f}, "
                f"conditional paired 95% CI "
                f"[{lo:+.4f}, {hi:+.4f}]"
            )

    save_csv(
        OUT / "duration_paired_changes.csv",
        rows
    )


# ==========================================
# TEMPORAL AND PREDICTION STABILITY
# ==========================================

def block_stability(block_rows, heldout_blocks):

    by_id = {}

    for r in block_rows:

        if r["theta_alpha_ratio"] != "":

            by_id.setdefault(
                r["subject"],
                []
            ).append(r)

    summary = []

    all_ids = sorted({
        r["subject"]
        for r in block_rows
    })

    for sid in all_ids:

        rs = by_id.get(sid, [])

        group = next(
            r["group"]
            for r in block_rows
            if r["subject"] == sid
        )

        if len(rs) >= 3:

            logtar = np.log(
                np.array([
                    r["theta_alpha_ratio"]
                    for r in rs
                ], dtype=float)
            )

            sd = float(
                logtar.std(ddof=1)
            )

        else:
            sd = float("nan")

        summary.append({
            "subject": sid,
            "group": group,
            "usable_blocks": len(rs),
            "sd_log_theta_alpha_ratio_across_blocks": sd
        })

    save_csv(
        OUT / "theta_alpha_five_block_stability.csv",
        summary
    )

    a = np.array([
        r["sd_log_theta_alpha_ratio_across_blocks"]
        for r in summary
        if r["group"] == "A"
        and np.isfinite(
            r["sd_log_theta_alpha_ratio_across_blocks"]
        )
    ])

    c = np.array([
        r["sd_log_theta_alpha_ratio_across_blocks"]
        for r in summary
        if r["group"] == "C"
        and np.isfinite(
            r["sd_log_theta_alpha_ratio_across_blocks"]
        )
    ])

    print(
        "\n===== FIVE-BLOCK THETA/ALPHA STABILITY ====="
    )

    print(
        "AD usable subjects:", len(a),
        "Healthy usable subjects:", len(c)
    )

    print(
        f"AD median SD(log ratio): {np.median(a):.4f}; "
        f"Healthy median: {np.median(c):.4f}"
    )

    if len(a) >= 10 and len(c) >= 10:

        rng = np.random.default_rng(1301)

        observed = float(
            a.mean() - c.mean()
        )

        combined = np.r_[a, c]
        extreme = 0

        for _ in range(5000):

            permutation = rng.permutation(
                combined
            )

            x = (
                permutation[:len(a)].mean()
                - permutation[len(a):].mean()
            )

            extreme += (
                abs(x) >= abs(observed)
            )

        p = (extreme + 1) / 5001

        print(
            "Exploratory mean AD - Healthy difference:",
            f"{observed:+.4f}",
            "label-permutation p=",
            f"{p:.4f}"
        )

        save_csv(
            OUT / "theta_alpha_group_comparison.csv",
            [{
                "n_ad": len(a),
                "n_healthy": len(c),
                "mean_difference_ad_minus_healthy":
                    observed,
                "permutation_p_exploratory": p
            }]
        )

    by_subject = {}

    for r in heldout_blocks:

        if r["predicted_ad"] != "":

            by_subject.setdefault(
                r["subject"],
                []
            ).append(
                int(r["predicted_ad"])
            )

    usable = {
        sid: values
        for sid, values in by_subject.items()
        if len(values) >= 3
    }

    flips = sum(
        len(set(values)) > 1
        for values in usable.values()
    )

    print(
        "\n===== HELD-OUT 60s BLOCK "
        "PREDICTION STABILITY ====="
    )

    print(
        "Participants with >=3 usable blocks:",
        len(usable)
    )

    print(
        "Participants whose predicted class "
        "changes across blocks:",
        flips
    )

    save_csv(
        OUT / "block_prediction_stability_summary.csv",
        [{
            "participants_with_3plus_blocks": len(usable),
            "participants_with_class_changes": flips,
            "fraction_changed":
                flips / len(usable)
                if usable else float("nan")
        }]
    )


# ==========================================
# GENERATE DURATION FIGURE
# ==========================================

def plot(results):

    fig, ax = plt.subplots(
        figsize=(7.2, 4.3)
    )

    for name, marker in (
        ("RBF_SVM", "o"),
        ("LogisticRegression", "s")
    ):

        means = []
        deviations = []

        for duration in DURATIONS:

            vals = np.array([
                r["balanced_accuracy"]
                for r in results
                if r["model"] == name
                and r["duration_sec"] == duration
            ])

            means.append(vals.mean())
            deviations.append(
                vals.std(ddof=1)
            )

        ax.errorbar(
            DURATIONS,
            means,
            yerr=deviations,
            marker=marker,
            capsize=4,
            label=name
        )

    ax.set(
        xlabel="Retained usable EEG duration (seconds)",
        ylabel="Participant-level balanced accuracy",
        title="Matched repeated nested CV (5 repetitions)",
        xticks=DURATIONS,
        ylim=(0, 1)
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "duration_balanced_accuracy.png",
        dpi=180
    )

    plt.close(fig)


# ==========================================
# MAIN
# ==========================================

def main():

    manifest = [
        r for r in read_csv(
            REPORTS / "openneuro_5min_manifest_v2.csv"
        )
        if r["status"] == "CANDIDATE"
        and r["group"] in ("A", "C")
    ]

    if (
        len(manifest) != 46
        or Counter(
            r["group"] for r in manifest
        ) != {"A": 23, "C": 23}
    ):
        raise RuntimeError(
            "Unexpected 46-person cohort"
        )

    ids = [
        r["subject"]
        for r in manifest
    ]

    if len(set(ids)) != 46:
        raise RuntimeError(
            "Duplicate participant identifiers"
        )

    y = np.array([
        int(r["group"] == "A")
        for r in manifest
    ])

    legacy_rows = read_csv(
        REPORTS / "baseline_features_v2.csv"
    )

    legacy = {
        r["subject"]: r
        for r in legacy_rows
    }

    if (
        len(legacy) != 46
        or set(legacy) != set(ids)
    ):
        raise RuntimeError(
            "Historical feature subject mismatch"
        )

    # Record the protocol explicitly.
    protocol = {
        "cohort_n": 46,
        "diagnosis_positive": "A (AD)",
        "durations_seconds": DURATIONS,
        "sampling_rate_hz": 500,
        "epoch_seconds": 2,
        "welch_nperseg": 1000,
        "welch_noverlap": 0,
        "reference_band_hz": [1, 45],
        "frequency_bands_hz": BANDS,
        "band_power":
            "Sum Welch PSD bins per epoch/channel "
            "divided by total 1-45 Hz power",
        "subject_aggregation":
            "Median across retained epoch-channel values",
        "theta_alpha_ratio":
            "Aggregated theta divided by aggregated alpha",
        "artifact_rules":
            "Exclude 2-second epoch if >=3 channels "
            "ptp>200uV, any channel ptp>300uV, "
            "or any channel std<0.5uV",
        "minimum_clean_fraction": 0.5,
        "outer_folds": 5,
        "inner_folds": 3,
        "repeats": REPEATS,
        "evaluation_unit": "participant",
        "primary_metric": "balanced accuracy",
        "note":
            "Separate duration protocol; not claimed "
            "identical to prior 300-second baseline"
    }

    (
        OUT / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print(
        "===== LOCKED DURATION STUDY ====="
    )

    print(
        "Subjects: 46 (23 AD, 23 Healthy)"
    )

    print(
        "Conditions: 60, 180, 300 seconds"
    )

    print(
        "CV repetitions:",
        REPEATS,
        flush=True
    )

    main_rows, block_rows = extract(
        manifest,
        legacy
    )

    print(
        "\n===== CLEAN EPOCH QUALITY BY DURATION ====="
    )

    for duration in DURATIONS:

        for group in ("A", "C"):

            values = np.array([
                r["clean_fraction"]
                for r in main_rows
                if r["duration_sec"] == duration
                and r["group"] == group
            ])

            print(
                f"{duration}s {group}: "
                f"median={np.median(values):.3f}, "
                f"min={values.min():.3f}"
            )

    results, predictions, blocks = evaluate(
        main_rows,
        block_rows,
        ids,
        y
    )

    paired_changes(
        predictions,
        ids,
        y
    )

    block_stability(
        block_rows,
        blocks
    )

    plot(results)

    print(
        "\n===== DURATION STUDY COMPLETE ====="
    )

    print(
        "Output directory:",
        OUT
    )

    print(
        "Original EEG and historical model "
        "reports unchanged."
    )

    print(
        "Duration features use a separately "
        "defined consistent spectral protocol."
    )

    print(
        "Repeated CV provides internal "
        "evaluation, not external validation."
    )


if __name__ == "__main__":
    main()

