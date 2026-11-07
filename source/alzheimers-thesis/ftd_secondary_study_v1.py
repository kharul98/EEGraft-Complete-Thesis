#!/usr/bin/env python3
"""Locked secondary FTD-versus-HC study: participant-level duration and temporal reliability."""

from pathlib import Path
from collections import Counter
import csv
import json
import hashlib

import numpy as np
import mne
from scipy.signal import welch

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
R = ROOT / "reports"
S = R / "thesis_final"
OUT = S / "ftd_extension_v1" / "locked_study"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = (
    "delta", "theta", "alpha",
    "beta", "gamma", "theta_alpha_ratio"
)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

DURATIONS = (60, 180, 300)
REPEATS = 5
NBOOT = 3000
SEED = 20261107


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def save(name, rows):
    if not rows:
        raise RuntimeError("No rows: " + name)

    cols = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with (OUT / name).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=cols
        )
        w.writeheader()
        w.writerows(rows)


def indexed(rows, keyfn):
    result = {}

    for row in rows:
        key = keyfn(row)

        if key in result:
            raise RuntimeError(
                "Duplicate key: " + str(key)
            )

        result[key] = row

    return result


def bacc(y, p):
    if not (
        np.any(y == 0)
        and np.any(y == 1)
    ):
        raise RuntimeError(
            "Both classes required"
        )

    sensitivity = (p[y == 1] == 1).mean()
    specificity = (p[y == 0] == 0).mean()

    return float(
        (sensitivity + specificity) / 2
    )


def pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        )
    ])


# =========================================
# 1. LOCK PROTOCOL
# =========================================

PROTOCOL = {
    "status": "Post-hoc exploratory secondary study",
    "positive_label": "F (FTD)",
    "negative_label": "C (healthy)",
    "cohort":
        "All frozen CANDIDATE FTD participants "
        "and the same selected controls as AD study",
    "signal":
        "300-second manifest windows, 500Hz, "
        "19 EEG channels, microvolts",
    "artifact_screen":
        "Reject a 2-second epoch if >=3 channels "
        "ptp>200uV, any channel ptp>300uV, "
        "or any channel std<0.5uV",
    "bands_hz": BANDS,
    "reference_band_hz": [1, 45],
    "welch":
        "Hann, nperseg=1000, noverlap=0, "
        "constant detrend, density",
    "aggregation":
        "Relative band-bin PSD sums; "
        "median over retained epoch-channel cells",
    "minimum_usable":
        "At least 50 percent clean epochs "
        "and at least 10 epochs per block",
    "model":
        "StandardScaler + class-balanced RBF-SVM",
    "outer_cv":
        "5 subject-stratified folds x 5 repeats",
    "inner_cv":
        "3 subject-stratified folds on outer training only",
    "grid": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1]
    },
    "primary":
        "First-minute-trained model tested on "
        "five held-out minute positions; "
        "within-recording class reversals",
    "secondary":
        "60/180/300-second nested-CV comparison",
    "uncertainty":
        "Participant-stratified bootstrap, "
        "preserving repetitions and shared controls",
    "limitation":
        "Previously examined dataset; "
        "no external validation; not preregistered"
}

protocol_path = OUT / "locked_protocol.json"

encoded = (
    json.dumps(
        PROTOCOL,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if protocol_path.exists():
    if protocol_path.read_text() != encoded:
        raise RuntimeError(
            "Locked protocol has changed"
        )
else:
    protocol_path.write_text(encoded)


# =========================================
# 2. LOAD AND LOCK INPUTS
# =========================================

manifest_path = (
    R / "openneuro_5min_manifest_v2.csv"
)

prior_feature = (
    S / "duration_locked_v1/duration_features.csv"
)

prior_blocks = (
    S / "duration_locked_v1/five_60s_blocks.csv"
)

prior_ad = (
    S / "temporal_repeated_v1/"
    "all_repeats_heldout_block_predictions.csv"
)

preflight_path = (
    S / "ftd_extension_v1/"
    "ftd_recording_preflight.csv"
)

inputs = [
    manifest_path,
    prior_feature,
    prior_blocks,
    prior_ad,
    preflight_path
]

for path in inputs:
    if not path.is_file():
        raise FileNotFoundError(path)

hashes = {
    str(path.relative_to(ROOT)):
        hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for path in inputs
}

hash_path = OUT / "source_checksums.json"

hash_text = (
    json.dumps(
        hashes,
        indent=2,
        sort_keys=True
    ) + "\n"
)

if hash_path.exists():
    if hash_path.read_text() != hash_text:
        raise RuntimeError(
            "Source inputs changed. Review before rerunning."
        )
else:
    hash_path.write_text(hash_text)


print(
    "===== LOCKED FTD-VS-HEALTHY EXPERIMENT =====",
    flush=True
)

manifest = read(manifest_path)

if (
    len(manifest) != 88
    or len({
        r["subject"] for r in manifest
    }) != 88
):
    raise RuntimeError(
        "Invalid dataset manifest"
    )

if Counter(
    r["group"] for r in manifest
) != Counter({
    "A": 36,
    "F": 23,
    "C": 29
}):
    raise RuntimeError(
        "Unexpected original dataset distribution"
    )

ftd = [
    r for r in manifest
    if r["group"] == "F"
    and r["status"] == "CANDIDATE"
]

hc = [
    r for r in manifest
    if r["group"] == "C"
    and r["status"] == "CANDIDATE"
]

if len(ftd) != 18 or len(hc) != 23:
    raise RuntimeError(
        "Frozen eligibility counts changed"
    )

chosen = sorted(
    ftd + hc,
    key=lambda r: r["subject"]
)

ids = [
    r["subject"]
    for r in chosen
]

y = np.array([
    int(r["group"] == "F")
    for r in chosen
], dtype=int)

if (
    len(ids) != 41
    or Counter(y.tolist()) != {
        0: 23,
        1: 18
    }
):
    raise RuntimeError(
        "Unexpected FTD study cohort"
    )

preflight = indexed(
    read(preflight_path),
    lambda r: r["subject"]
)

for r in ftd:
    q = preflight[r["subject"]]

    if (
        q["header_status"] != "HEADER_COMPATIBLE"
        or q["manifest_status"] != "CANDIDATE"
    ):
        raise RuntimeError(
            "Invalid FTD preflight: " + r["subject"]
        )

print(
    "Frozen cohort: 18 FTD + 23 shared healthy = 41",
    flush=True
)

save(
    "locked_participants.csv",
    [{
        k: r.get(k, "")
        for k in (
            "subject", "group", "status",
            "file", "start_sec", "end_sec",
            "duration_sec", "multi200_pct",
            "any300_pct", "flat_pct"
        )
    } for r in chosen]
)

save(
    "ftd_exclusions.csv",
    [{
        "subject": r["subject"],
        "status": r["status"],
        "note": r.get("note", "")
    } for r in manifest
      if r["group"] == "F"
      and r["status"] != "CANDIDATE"]
)


# =========================================
# 3. EXTRACT MATCHED FEATURES
# =========================================

old_duration = indexed(
    read(prior_feature),
    lambda r: (
        r["subject"],
        int(r["duration_sec"])
    )
)

old_blocks = indexed(
    read(prior_blocks),
    lambda r: (
        r["subject"],
        int(r["block_number"])
    )
)

old_hc = {
    r["subject"]
    for r in manifest
    if r["group"] == "C"
    and r["status"] == "CANDIDATE"
}

if {
    sid for sid in ids
    if sid in old_hc
} != old_hc:
    raise RuntimeError(
        "Healthy controls differ from the AD study"
    )

main = []
blocks = []
screen = []

max_hc_diff = 0.0
common_channels = None

for k, r in enumerate(chosen, 1):

    sid = r["subject"]

    source = Path(r["file"])

    if not source.is_absolute():
        source = ROOT / source

    if not source.is_file():
        raise FileNotFoundError(source)

    start = float(r["start_sec"])
    end = float(r["end_sec"])
    sf = 500

    if abs(end - start - 300) > 1e-6:
        raise RuntimeError(
            f"{sid}: selected window is not 300 seconds"
        )

    raw = mne.io.read_raw_eeglab(
        str(source),
        preload=False,
        verbose="ERROR"
    )

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[]
    )

    if (
        abs(raw.info["sfreq"] - 500) > 1e-6
        or len(picks) != 19
    ):
        raise RuntimeError(
            f"{sid}: unexpected EEG metadata"
        )

    channels = {
        raw.ch_names[i].strip().lower()
        for i in picks
    }

    if len(channels) != 19:
        raise RuntimeError(
            f"{sid}: duplicate channel labels"
        )

    if common_channels is None:
        common_channels = channels

    if channels != common_channels:
        raise RuntimeError(
            f"{sid}: electrode montage mismatch"
        )

    a = int(round(start * sf))
    b = a + 150000

    if a < 0 or b > raw.n_times:
        raise RuntimeError(
            f"{sid}: window exceeds recording length"
        )

    internal_boundaries = [
        float(t)
        for t, desc in zip(
            raw.annotations.onset,
            raw.annotations.description
        )
        if (
            "boundary" in str(desc).lower()
            and start + 1/sf < float(t) < end - 1/sf
        )
    ]

    if internal_boundaries:
        raise RuntimeError(
            f"{sid}: selected window crosses boundary "
            f"at {internal_boundaries[:4]}"
        )

    signal = raw.get_data(
        picks=picks,
        start=a,
        stop=b
    ) * 1e6

    if (
        signal.shape != (19, 150000)
        or not np.isfinite(signal).all()
    ):
        raise RuntimeError(
            f"{sid}: invalid EEG samples"
        )

    epochs = signal.reshape(
        19, 150, 1000
    ).transpose(1, 0, 2)

    p2p = np.ptp(epochs, axis=-1)
    sd = np.std(epochs, axis=-1)

    clean = ~(
        ((p2p > 200).sum(axis=1) >= 3)
        | (p2p > 300).any(axis=1)
        | (sd < 0.5).any(axis=1)
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

    denominator = psd[
        ..., (freqs >= 1) & (freqs <= 45)
    ].sum(axis=-1)

    if np.any(denominator <= 0):
        raise RuntimeError(
            f"{sid}: invalid reference spectral power"
        )

    relative = {}

    for name, (lo, hi) in BANDS.items():

        mask = (
            (freqs >= lo)
            & (
                (freqs <= hi)
                if name == "gamma"
                else (freqs < hi)
            )
        )

        relative[name] = (
            psd[..., mask].sum(axis=-1)
            / denominator
        )

    def extract_record(lo, hi, seconds, block):

        retained = clean[lo:hi]

        n = int(retained.sum())
        total = hi - lo

        if n < max(10, total // 2):
            raise RuntimeError(
                f"{sid}: insufficient clean EEG "
                f"for block={block}, duration={seconds}: "
                f"{n}/{total}"
            )

        values = {
            name: float(
                np.median(
                    data[lo:hi][retained]
                )
            )
            for name, data in relative.items()
        }

        values["theta_alpha_ratio"] = (
            values["theta"]
            / max(values["alpha"], 1e-12)
        )

        return {
            "subject": sid,
            "group": r["group"],
            "duration_sec": seconds,
            "block_number": block,
            "clean_epochs": n,
            "clean_fraction": n / total,
            **values
        }

    for duration in DURATIONS:

        item = extract_record(
            0, duration // 2, duration, ""
        )

        main.append(item)

        if r["group"] == "C":

            ref = old_duration[
                (sid, duration)
            ]

            if (
                int(ref["clean_epochs"])
                != item["clean_epochs"]
            ):
                raise RuntimeError(
                    "HC clean count changed: " + sid
                )

            for feature in FEATURES:

                max_hc_diff = max(
                    max_hc_diff,
                    abs(
                        float(ref[feature])
                        - item[feature]
                    )
                )

    for block in range(1, 6):

        item = extract_record(
            (block - 1) * 30,
            block * 30,
            60,
            block
        )

        blocks.append(item)

        if r["group"] == "C":

            ref = old_blocks[
                (sid, block)
            ]

            if (
                int(ref["clean_epochs"])
                != item["clean_epochs"]
            ):
                raise RuntimeError(
                    "HC block clean count changed: " + sid
                )

            for feature in FEATURES:

                max_hc_diff = max(
                    max_hc_diff,
                    abs(
                        float(ref[feature])
                        - item[feature]
                    )
                )

    screen.append({
        "subject": sid,
        "group": r["group"],
        "clean_300s": int(clean.sum()),
        "min_60s_clean": min(
            int(
                clean[j:j + 30].sum()
            )
            for j in range(0, 150, 30)
        ),
        "manifest_multi200_pct":
            r.get("multi200_pct", ""),
        "internal_boundaries": 0
    })

    if k % 10 == 0 or k == len(chosen):
        print(
            f"Feature audit: {k}/{len(chosen)}",
            flush=True
        )

if len(main) != 123 or len(blocks) != 205:
    raise RuntimeError(
        "Unexpected feature row counts"
    )

if max_hc_diff > 1e-10:
    raise RuntimeError(
        "Shared controls differ from AD study: "
        f"{max_hc_diff}"
    )

save("duration_features.csv", main)
save("five_60s_blocks.csv", blocks)
save("window_quality_audit.csv", screen)

print(
    "PASS: Same 23 control windows/features as AD study; "
    f"maximum feature difference={max_hc_diff:.3g}"
)

print(
    "FTD minimum clean epochs in any 60s block:",
    min(
        z["min_60s_clean"]
        for z in screen
        if z["group"] == "F"
    ),
    "/30"
)


# =========================================
# 4. PREPARE PARTICIPANT FEATURE ARRAYS
# =========================================

F = indexed(
    main,
    lambda z: (
        z["subject"],
        z["duration_sec"]
    )
)

B = indexed(
    blocks,
    lambda z: (
        z["subject"],
        z["block_number"]
    )
)

XD = {
    duration: np.asarray([
        [
            F[
                (sid, duration)
            ][feature]
            for feature in FEATURES
        ]
        for sid in ids
    ], dtype=float)
    for duration in DURATIONS
}

XB = np.asarray([
    [
        [
            B[
                (sid, block)
            ][feature]
            for feature in FEATURES
        ]
        for block in range(1, 6)
    ]
    for sid in ids
], dtype=float)

if not np.allclose(
    XD[60],
    XB[:, 0, :],
    atol=1e-12,
    rtol=0
):
    raise RuntimeError(
        "First-block features differ from 60s features"
    )


# =========================================
# 5. REPEATED NESTED SUBJECT-LEVEL CV
# =========================================

predD = {
    duration: np.full(
        (REPEATS, 41),
        -1,
        dtype=int
    )
    for duration in DURATIONS
}

scoreD = {
    duration: np.full(
        (REPEATS, 41),
        np.nan
    )
    for duration in DURATIONS
}

pt = np.full(
    (REPEATS, 41, 5),
    -1,
    dtype=int
)

st = np.full(
    (REPEATS, 41, 5),
    np.nan
)

fold_map = np.full(
    (REPEATS, 41),
    -1,
    dtype=int
)

fold_hyper = []

fold_aucs = {
    duration: [
        [] for _ in range(REPEATS)
    ]
    for duration in DURATIONS
}

for repeat in range(REPEATS):

    outer = StratifiedKFold(
        5,
        shuffle=True,
        random_state=42 + 100 * repeat
    )

    for fold, (train, test) in enumerate(
        outer.split(XD[60], y),
        1
    ):

        if set(train).intersection(test):
            raise RuntimeError(
                "Outer participant leakage"
            )

        fold_map[repeat, test] = fold

        for duration in DURATIONS:

            inner = StratifiedKFold(
                3,
                shuffle=True,
                random_state=(
                    1000 + 100 * repeat + fold
                )
            )

            search = GridSearchCV(
                pipeline(),
                {
                    "clf__C": [0.1, 1.0, 10.0],
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
                XD[duration][train],
                y[train]
            )

            model = search.best_estimator_

            predD[duration][
                repeat, test
            ] = model.predict(
                XD[duration][test]
            )

            scoreD[duration][
                repeat, test
            ] = model.decision_function(
                XD[duration][test]
            )

            fold_aucs[duration][repeat].append(
                float(
                    roc_auc_score(
                        y[test],
                        scoreD[duration][repeat, test]
                    )
                )
            )

            fold_hyper.append({
                "repeat": repeat + 1,
                "fold": fold,
                "duration_sec": duration,
                "C": search.best_params_["clf__C"],
                "gamma": search.best_params_["clf__gamma"],
                "train_subjects": len(train),
                "test_subjects": len(test)
            })

            # Only the first-minute-trained SVM
            # is transferred to all five blocks.
            if duration == 60:

                test_block_features = XB[test].reshape(
                    -1,
                    len(FEATURES)
                )

                p = model.predict(
                    test_block_features
                ).reshape(
                    len(test), 5
                )

                scores = model.decision_function(
                    test_block_features
                ).reshape(
                    len(test), 5
                )

                pt[repeat, test, :] = p
                st[repeat, test, :] = scores

                if not np.array_equal(
                    p[:, 0],
                    predD[60][repeat, test]
                ):
                    raise RuntimeError(
                        "First-minute prediction mismatch"
                    )

                if not np.allclose(
                    scores[:, 0],
                    scoreD[60][repeat, test],
                    rtol=0,
                    atol=1e-8
                ):
                    raise RuntimeError(
                        "First-minute decision-score mismatch"
                    )

    print(
        f"Nested subject-level CV completed: "
        f"{repeat + 1}/{REPEATS}",
        flush=True
    )

if (
    (fold_map < 1).any()
    or (pt < 0).any()
    or not np.isfinite(st).all()
):
    raise RuntimeError(
        "Incomplete block predictions"
    )

for duration in DURATIONS:
    if (
        (predD[duration] < 0).any()
        or not np.isfinite(scoreD[duration]).all()
    ):
        raise RuntimeError(
            "Incomplete duration predictions"
        )

save(
    "fold_hyperparameters.csv",
    fold_hyper
)

duration_predictions = [
    {
        "repeat": rep + 1,
        "subject": sid,
        "group": "F" if y[i] else "C",
        "true_label": int(y[i]),
        "fold": int(fold_map[rep, i]),
        "duration_sec": duration,
        "predicted_label": int(
            predD[duration][rep, i]
        ),
        "decision_score": float(
            scoreD[duration][rep, i]
        )
    }
    for rep in range(REPEATS)
    for duration in DURATIONS
    for i, sid in enumerate(ids)
]

save(
    "held_out_duration_predictions.csv",
    duration_predictions
)

block_predictions = [
    {
        "repeat": rep + 1,
        "subject": sid,
        "group": "F" if y[i] else "C",
        "true_label": int(y[i]),
        "fold": int(fold_map[rep, i]),
        "block": block + 1,
        "predicted_label": int(
            pt[rep, i, block]
        ),
        "decision_score": float(
            st[rep, i, block]
        )
    }
    for rep in range(REPEATS)
    for i, sid in enumerate(ids)
    for block in range(5)
]

save(
    "held_out_60s_block_predictions.csv",
    block_predictions
)


# =========================================
# 6. FTD DURATION METRICS
# =========================================

run_rows = []

for rep in range(REPEATS):

    for duration in DURATIONS:

        prediction = predD[duration][rep]

        run_rows.append({
            "repeat": rep + 1,
            "duration_sec": duration,
            "n_subjects": 41,
            "FTD_n": 18,
            "HC_n": 23,
            "BACC": bacc(y, prediction),
            "accuracy": float(
                (prediction == y).mean()
            ),
            "FTD_sensitivity": float(
                (prediction[y == 1] == 1).mean()
            ),
            "HC_specificity": float(
                (prediction[y == 0] == 0).mean()
            ),
            "mean_outer_fold_AUROC": float(
                np.mean(
                    fold_aucs[duration][rep]
                )
            )
        })

save(
    "duration_nested_cv_runs.csv",
    run_rows
)

print(
    "\n===== FTD-VS-HC DURATION RESULTS ====="
)

for duration in DURATIONS:

    values = np.array([
        bacc(y, predD[duration][rep])
        for rep in range(REPEATS)
    ])

    print(
        f"{duration}s: "
        f"mean BACC={values.mean():.4f}, "
        f"repeat SD={values.std(ddof=1):.4f}, "
        f"range=[{values.min():.4f}, "
        f"{values.max():.4f}]"
    )


# =========================================
# 7. MINUTE-WISE TESTING
# =========================================

print(
    "\n===== FTD-VS-HC MINUTE-WISE PERFORMANCE ====="
)

position_rows = []

for block in range(5):

    z = pt[:, :, block]

    mean_bacc = float(
        np.mean([
            bacc(y, z[rep])
            for rep in range(REPEATS)
        ])
    )

    sensitivity = float(
        (z[:, y == 1] == 1).mean()
    )

    specificity = float(
        (z[:, y == 0] == 0).mean()
    )

    print(
        f"Minute {block + 1}: "
        f"BACC={mean_bacc:.4f}, "
        f"FTD sensitivity={sensitivity:.4f}, "
        f"HC specificity={specificity:.4f}"
    )

    position_rows.append({
        "minute": block + 1,
        "mean_BACC": mean_bacc,
        "FTD_sensitivity": sensitivity,
        "HC_specificity": specificity
    })

save(
    "minute_wise_metrics.csv",
    position_rows
)


# =========================================
# 8. PREDICTION STABILITY AND VOTING
# =========================================

flips = (
    pt != pt[:, :, :1]
).any(axis=2)

vote3 = (
    pt[:, :, :3].sum(axis=2) >= 2
).astype(int)

vote5 = (
    pt.sum(axis=2) >= 3
).astype(int)

repeat_rows = []

for rep in range(REPEATS):

    repeat_rows.append({
        "repeat": rep + 1,
        "class_reversal_n": int(
            flips[rep].sum()
        ),
        "class_reversal_rate": float(
            flips[rep].mean()
        ),
        "FTD_reversal_rate": float(
            flips[rep, y == 1].mean()
        ),
        "HC_reversal_rate": float(
            flips[rep, y == 0].mean()
        ),
        "first_minute_BACC": bacc(
            y, pt[rep, :, 0]
        ),
        "fifth_minute_BACC": bacc(
            y, pt[rep, :, 4]
        ),
        "vote3_BACC": bacc(
            y, vote3[rep]
        ),
        "vote5_BACC": bacc(
            y, vote5[rep]
        )
    })

save(
    "temporal_repeat_results.csv",
    repeat_rows
)

print(
    "\n===== FTD-VS-HC TEMPORAL RELIABILITY ====="
)

for row in repeat_rows:

    print(
        f"Repeat {row['repeat']}: "
        f"class reversals="
        f"{row['class_reversal_n']}/41; "
        f"vote3 BACC={row['vote3_BACC']:.4f}; "
        f"vote5 BACC={row['vote5_BACC']:.4f}"
    )

print(
    f"Mean reversal rate={flips.mean():.4f}; "
    f"FTD-only={flips[:, y == 1].mean():.4f}; "
    f"healthy-only={flips[:, y == 0].mean():.4f}"
)

print(
    "Participants changing class in >=1 repetition:",
    int(flips.any(axis=0).sum()),
    "/41"
)


# =========================================
# 9. LOAD EXISTING AD PREDICTIONS
# =========================================

previous = read(prior_ad)

ad_ids = sorted({
    row["subject"]
    for row in previous
})

ad_labels = {
    row["subject"]: int(row["true_label"])
    for row in previous
}

if (
    len(ad_ids) != 46
    or Counter(ad_labels.values())
    != {1: 23, 0: 23}
):
    raise RuntimeError(
        "Historical AD cohort mismatch"
    )

new_control_ids = {
    ids[i]
    for i in range(len(ids))
    if y[i] == 0
}

old_control_ids = {
    sid
    for sid in ad_ids
    if ad_labels[sid] == 0
}

if new_control_ids != old_control_ids:
    raise RuntimeError(
        "Historical healthy participants differ"
    )

ad_index = indexed(
    previous,
    lambda z: (
        int(z["repeat"]),
        z["subject"],
        int(z["block"])
    )
)

if len(ad_index) != 5 * 46 * 5:
    raise RuntimeError(
        "Incomplete historical AD predictions"
    )

ad_pred = np.array([
    [
        [
            int(
                ad_index[
                    (rep, sid, block)
                ]["predicted_label"]
            )
            for block in range(1, 6)
        ]
        for sid in ad_ids
    ]
    for rep in range(1, REPEATS + 1)
], dtype=int)

y_ad = np.array([
    ad_labels[sid]
    for sid in ad_ids
])

if np.any(
    (ad_pred < 0) | (ad_pred > 1)
):
    raise RuntimeError(
        "Invalid historical AD predictions"
    )

ad_flips = (
    ad_pred != ad_pred[:, :, :1]
).any(axis=2)


# =========================================
# 10. CROSS-DIAGNOSIS COMPARISON
# =========================================

fti = np.where(y == 1)[0]
hci = np.where(y == 0)[0]
adi = np.where(y_ad == 1)[0]

ad_map = {
    sid: i
    for i, sid in enumerate(ad_ids)
}

rng = np.random.default_rng(SEED)


def task_bacc(pred, positive_indices, negative_indices):

    sensitivity = (
        pred[:, positive_indices] == 1
    ).mean()

    specificity = (
        pred[:, negative_indices] == 0
    ).mean()

    return float(
        (sensitivity + specificity) / 2
    )


def compare_tasks(ftd_indices, ad_indices, hc_indices):

    # Shared HC IDs are aligned across tasks.
    ad_hc_indices = np.array([
        ad_map[ids[i]]
        for i in hc_indices
    ])

    f_first = task_bacc(
        pt[:, :, 0],
        ftd_indices,
        hc_indices
    )

    a_first = task_bacc(
        ad_pred[:, :, 0],
        ad_indices,
        ad_hc_indices
    )

    f_last = task_bacc(
        pt[:, :, 4],
        ftd_indices,
        hc_indices
    )

    a_last = task_bacc(
        ad_pred[:, :, 4],
        ad_indices,
        ad_hc_indices
    )

    f_reversal = 0.5 * (
        flips[:, ftd_indices].mean()
        + flips[:, hc_indices].mean()
    )

    a_reversal = 0.5 * (
        ad_flips[:, ad_indices].mean()
        + ad_flips[:, ad_hc_indices].mean()
    )

    return np.array([
        f_first - a_first,
        f_last - a_last,
        f_reversal - a_reversal,
        f_last - f_first,
        a_last - a_first
    ])


observed = compare_tasks(
    fti, adi, hci
)

boot = []

for _ in range(NBOOT):

    sample_ftd = rng.choice(
        fti,
        len(fti),
        replace=True
    )

    sample_ad = rng.choice(
        adi,
        len(adi),
        replace=True
    )

    sample_hc = rng.choice(
        hci,
        len(hci),
        replace=True
    )

    boot.append(
        compare_tasks(
            sample_ftd,
            sample_ad,
            sample_hc
        )
    )

boot = np.array(boot)

names = [
    "FTD_minus_AD_first_minute_BACC",
    "FTD_minus_AD_fifth_minute_BACC",
    "FTD_minus_AD_class_balanced_reversal_rate",
    "FTD_fifth_minus_first_minute_BACC",
    "AD_fifth_minus_first_minute_BACC"
]

print(
    "\n===== CROSS-DIAGNOSIS COMPARISON "
    "(SHARED HEALTHY CONTROLS) ====="
)

comparison_rows = []

for k, name in enumerate(names):

    lo, hi = np.percentile(
        boot[:, k],
        [2.5, 97.5]
    )

    value = float(observed[k])

    comparison_rows.append({
        "contrast": name,
        "estimate": value,
        "conditional_CI_low": float(lo),
        "conditional_CI_high": float(hi),
        "note":
            "Exploratory; shared controls "
            "resampled jointly"
    })

    print(
        f"{name}: {value:+.4f}; "
        f"conditional 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )

save(
    "cross_diagnosis_comparisons.csv",
    comparison_rows
)


# =========================================
# 11. FTD REVERSAL UNCERTAINTY
# =========================================

boot_flips = []

for _ in range(NBOOT):

    sampled = np.r_[
        rng.choice(
            fti,
            len(fti),
            replace=True
        ),
        rng.choice(
            hci,
            len(hci),
            replace=True
        )
    ]

    boot_flips.append(
        float(
            flips[:, sampled].mean()
        )
    )

lo, hi = np.percentile(
    boot_flips,
    [2.5, 97.5]
)

print(
    "FTD/HC reversal-rate conditional 95% CI:",
    f"[{lo:.4f}, {hi:.4f}]"
)


# =========================================
# 12. COMPARISON FIGURE
# =========================================

fig, ax = plt.subplots(
    figsize=(7.3, 4.5)
)

ad_by_minute = [
    np.mean([
        bacc(
            y_ad,
            ad_pred[rep, :, block]
        )
        for rep in range(REPEATS)
    ])
    for block in range(5)
]

ftd_by_minute = [
    row["mean_BACC"]
    for row in position_rows
]

ax.plot(
    range(1, 6),
    ad_by_minute,
    "o-",
    label="AD vs healthy (23 + 23)"
)

ax.plot(
    range(1, 6),
    ftd_by_minute,
    "s-",
    label="FTD vs healthy (18 + 23)"
)

ax.set(
    xlabel="Held-out 60-second EEG block",
    ylabel="Mean participant-level balanced accuracy",
    title="First-minute-trained SVM across diagnostic tasks",
    xticks=range(1, 6),
    ylim=(0, 1)
)

ax.grid(
    axis="y",
    alpha=0.2
)

ax.legend()
fig.tight_layout()

fig.savefig(
    OUT / "ftd_ad_minute_comparison.png",
    dpi=180
)

plt.close(fig)


print(
    "\n===== FTD SECONDARY STUDY COMPLETE ====="
)

print(
    "Results:",
    OUT
)

print(
    "Secondary exploratory cohort; "
    "no independent external-site validation."
)

