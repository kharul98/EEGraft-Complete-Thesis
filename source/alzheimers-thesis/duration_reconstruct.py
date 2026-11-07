#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
from functools import lru_cache

import csv
import json

import numpy as np
import mne

from scipy.signal import welch


ROOT = Path.cwd()
REPORTS = ROOT / "reports"

OUT = REPORTS / "thesis_final" / "duration"
OUT.mkdir(parents=True, exist_ok=True)

FS = 500

BAND_NAMES = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma"
)

FEATURES = BAND_NAMES + (
    "theta_alpha_ratio",
)


def load_csv(path):

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):

    if not rows:
        raise ValueError("Empty table")

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


# =====================================
# LOAD SELECTED EEG
# =====================================

def load_subject(row):

    path = Path(row["file"])

    if not path.is_absolute():
        path = ROOT / path

    raw = mne.io.read_raw_eeglab(
        str(path),
        preload=False,
        verbose="ERROR"
    )

    sf = float(raw.info["sfreq"])

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[]
    )

    if abs(sf - FS) > 0.01 or len(picks) != 19:
        raise ValueError(
            f"Wrong sampling rate or channels: "
            f"{row['subject']}"
        )

    start = int(
        round(
            float(row["start_sec"]) * sf
        )
    )

    stop = start + 300 * FS

    if start < 0 or stop > raw.n_times:
        raise ValueError(
            f"Invalid 300-second window: "
            f"{row['subject']}"
        )

    # MNE returns EEG in volts.
    # Convert to microvolts.

    data = raw.get_data(
        picks=picks,
        start=start,
        stop=stop
    ) * 1e6

    if (
        data.shape != (19, 150000)
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            f"Invalid EEG: {row['subject']}"
        )

    # 150 non-overlapping 2-second epochs.

    epochs = data.reshape(
        19, 150, 1000
    ).transpose(1, 0, 2)

    ptp = np.ptp(
        epochs,
        axis=-1
    )

    std = np.std(
        epochs,
        axis=-1
    )

    good = ~(
        ((ptp > 200).sum(axis=1) >= 3)
        |
        (ptp > 300).any(axis=1)
        |
        (std < 0.5).any(axis=1)
    )

    return epochs, good


# =====================================
# POWER SPECTRAL DENSITY
# =====================================

def compute_psd(epochs, nperseg):

    freqs, spectrum = welch(
        epochs,
        fs=FS,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1
    )

    return freqs, spectrum


def power(
    freq,
    psd,
    lo,
    hi,
    upper_inclusive,
    method
):

    if upper_inclusive:

        mask = (
            (freq >= lo)
            & (freq <= hi)
        )

    else:

        mask = (
            (freq >= lo)
            & (freq < hi)
        )

    if not mask.any():
        raise ValueError(
            "No frequency bins selected"
        )

    if method == "sum":

        return psd[
            ..., mask
        ].sum(axis=-1)

    if mask.sum() < 2:

        return np.zeros(
            psd.shape[:-1]
        )

    return np.trapezoid(
        psd[..., mask],
        freq[mask],
        axis=-1
    )


def build_power_cache(freq, psd):

    @lru_cache(maxsize=150)
    def get(lo, hi, inclusive, method):

        return power(
            freq,
            psd,
            lo,
            hi,
            inclusive,
            method
        )

    return get


# =====================================
# FEATURE AGGREGATION OPTIONS
# =====================================

def aggregate(x, name):

    if name == "median_all":
        return float(np.median(x))

    if name == "mean_all":
        return float(np.mean(x))

    if name == "median_epoch_mean_channel":
        return float(
            np.median(
                x.mean(axis=1)
            )
        )

    if name == "mean_epoch_median_channel":
        return float(
            np.mean(
                np.median(x, axis=1)
            )
        )

    if name == "median_channel_mean_epoch":
        return float(
            np.median(
                x.mean(axis=0)
            )
        )

    if name == "mean_channel_median_epoch":
        return float(
            np.mean(
                np.median(x, axis=0)
            )
        )

    if name == "median_epoch_median_channel":
        return float(
            np.median(
                np.median(x, axis=1)
            )
        )

    if name == "median_channel_median_epoch":
        return float(
            np.median(
                np.median(x, axis=0)
            )
        )

    raise ValueError(name)


AGG_NAMES = (
    "median_all",
    "mean_all",
    "median_epoch_mean_channel",
    "mean_epoch_median_channel",
    "median_channel_mean_epoch",
    "mean_channel_median_epoch",
    "median_epoch_median_channel",
    "median_channel_median_epoch"
)


# =====================================
# EXTRACT FEATURES USING A CONFIGURATION
# =====================================

def features_for(
    good,
    get,
    conf,
    cut_epoch=150,
    return_epochs=False
):

    (
        nseg,
        ref_lo,
        ref_hi,
        delta_lo,
        gamma_hi,
        ref_inclusive,
        band_inclusive,
        method,
        aggregation
    ) = conf

    denominator = get(
        ref_lo,
        ref_hi,
        ref_inclusive,
        method
    )

    if np.any(denominator <= 0):
        raise ValueError(
            "Nonpositive reference power"
        )

    bands = (
        (delta_lo, 4),
        (4, 8),
        (8, 13),
        (13, 30),
        (30, gamma_hi)
    )

    relative = []

    for lo, hi in bands:

        numerator = get(
            lo,
            hi,
            band_inclusive,
            method
        )

        relative.append(
            numerator / denominator
        )

    selected_good = good[:cut_epoch]

    clean = int(
        np.sum(selected_good)
    )

    if clean < 10:
        raise ValueError(
            f"Only {clean} clean epochs"
        )

    values = [
        aggregate(
            band[:cut_epoch][selected_good],
            aggregation
        )
        for band in relative
    ]

    # Ratio of aggregated spectral features.

    values.append(
        values[1] / max(
            values[2],
            1e-20
        )
    )

    result = np.array(
        values,
        dtype=float
    )

    if not return_epochs:
        return result

    # Exploratory temporal variability
    # of the epoch-level theta/alpha ratio.

    theta = relative[1][
        :cut_epoch
    ][selected_good]

    alpha = relative[2][
        :cut_epoch
    ][selected_good]

    epoch_ratios = np.median(
        theta / np.maximum(alpha, 1e-20),
        axis=1
    )

    log_ratios = np.log(
        np.maximum(
            epoch_ratios,
            1e-20
        )
    )

    center = np.median(
        log_ratios
    )

    mad = (
        1.4826
        * np.median(
            np.abs(
                log_ratios - center
            )
        )
    )

    sd = np.std(
        log_ratios,
        ddof=1
    )

    return (
        result,
        clean,
        float(mad),
        float(sd)
    )


# =====================================
# MAIN
# =====================================

def main():

    manifest = [
        r for r in load_csv(
            REPORTS /
            "openneuro_5min_manifest_v2.csv"
        )
        if r["group"] in ("A", "C")
        and r["status"] == "CANDIDATE"
    ]

    original = {
        r["subject"]: r
        for r in load_csv(
            REPORTS /
            "baseline_features_v2.csv"
        )
    }

    if (
        len(manifest) != 46
        or Counter(
            r["group"] for r in manifest
        ) != {"A": 23, "C": 23}
    ):
        raise RuntimeError(
            "Incorrect primary cohort"
        )

    if set(original) != {
        r["subject"] for r in manifest
    }:
        raise RuntimeError(
            "Baseline subject mismatch"
        )

    print(
        "===== RECONSTRUCTING ORIGINAL "
        "SPECTRAL FEATURES =====",
        flush=True
    )

    # Numerically calibrate the extraction
    # formula, without using diagnosis labels.

    ids = [
        "sub-001",
        "sub-002",
        "sub-004"
    ]

    by_id = {
        r["subject"]: r
        for r in manifest
    }

    subject_data = {}

    for sid in ids:

        epochs, good = load_subject(
            by_id[sid]
        )

        saved_count = int(
            float(
                original[sid]["clean_epochs"]
            )
        )

        print(
            f"{sid}: computed clean epochs="
            f"{int(good.sum())}, "
            f"saved={saved_count}",
            flush=True
        )

        subject_data[sid] = (
            epochs,
            good
        )

    # =====================================
    # NUMERICAL PARAMETER SEARCH
    # =====================================

    sid = ids[0]

    epochs, good = subject_data[sid]

    target = np.array([
        float(original[sid][name])
        for name in FEATURES
    ])

    candidates = []

    for nseg in (
        1000, 500, 512, 250
    ):

        freq, psd = compute_psd(
            epochs,
            nseg
        )

        get = build_power_cache(
            freq,
            psd
        )

        for ref_lo, ref_hi in (
            (1, 45),
            (0.5, 45),
            (1, 40),
            (0.5, 40),
            (1, 50),
            (0.5, 50)
        ):

            for delta_lo in (1, 0.5):

                for gamma_hi in (40, 45):

                    for ref_inc in (
                        False, True
                    ):

                        for band_inc in (
                            False, True
                        ):

                            for method in (
                                "sum",
                                "trapezoid"
                            ):

                                for aggregation in AGG_NAMES:

                                    conf = (
                                        nseg,
                                        ref_lo,
                                        ref_hi,
                                        delta_lo,
                                        gamma_hi,
                                        ref_inc,
                                        band_inc,
                                        method,
                                        aggregation
                                    )

                                    try:

                                        result = features_for(
                                            good,
                                            get,
                                            conf
                                        )

                                    except ValueError:
                                        continue

                                    error = float(
                                        np.max(
                                            np.abs(
                                                (result - target)
                                                /
                                                np.maximum(
                                                    np.abs(target),
                                                    1e-12
                                                )
                                            )
                                        )
                                    )

                                    candidates.append(
                                        (error, conf)
                                    )

        print(
            f"Examined Welch nperseg={nseg}",
            flush=True
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    if not candidates:
        raise RuntimeError(
            "No matching configurations generated"
        )

    print(
        "\n===== TOP NUMERICAL MATCHES ====="
    )

    for error, conf in candidates[:8]:

        print(
            f"Relative error={error:.7g} "
            f"settings={conf}"
        )

    # =====================================
    # VERIFY TOP CANDIDATES ON THREE PEOPLE
    # =====================================

    refined = []

    validation_cache = {}

    for _, conf in candidates[:20]:

        errors = []

        for sid in ids:

            epochs, good = (
                subject_data[sid]
            )

            key = (
                sid,
                conf[0]
            )

            if key not in validation_cache:

                freq, psd = compute_psd(
                    epochs,
                    conf[0]
                )

                validation_cache[key] = (
                    build_power_cache(
                        freq,
                        psd
                    )
                )

            get = validation_cache[key]

            calc = features_for(
                good,
                get,
                conf
            )

            reference = np.array([
                float(original[sid][name])
                for name in FEATURES
            ])

            errors.append(
                float(
                    np.max(
                        np.abs(
                            calc - reference
                        )
                    )
                )
            )

        refined.append(
            (max(errors), conf)
        )

    refined.sort(
        key=lambda x: x[0]
    )

    best_difference, best = refined[0]

    print(
        "\n===== THREE-PARTICIPANT CALIBRATION ====="
    )

    print(
        "Best configuration:",
        best
    )

    print(
        "Maximum absolute feature error:",
        f"{best_difference:.10g}"
    )

    if best_difference > 1e-6:

        print(
            "REPRODUCTION NOT CONFIRMED."
        )

        print(
            "Existing baseline was not modified."
        )

        print(
            "Send the TOP NUMERICAL MATCHES "
            "and THREE-PARTICIPANT CALIBRATION "
            "output for review."
        )

        return

    print(
        "First three participants: "
        "FEATURE REPRODUCTION PASS"
    )

    # =====================================
    # VERIFY ALL 46 AND EXTRACT DURATIONS
    # =====================================

    duration_rows = []
    errors = []
    count_errors = []

    for i, row in enumerate(
        manifest,
        1
    ):

        sid = row["subject"]

        epochs, good = load_subject(
            row
        )

        saved_count = int(
            float(
                original[sid]["clean_epochs"]
            )
        )

        if int(good.sum()) != saved_count:

            count_errors.append(
                (
                    sid,
                    int(good.sum()),
                    saved_count
                )
            )

        freq, psd = compute_psd(
            epochs,
            best[0]
        )

        get = build_power_cache(
            freq,
            psd
        )

        for duration in (
            60, 180, 300
        ):

            (
                calc,
                clean,
                mad,
                sd
            ) = features_for(
                good,
                get,
                best,
                duration // 2,
                True
            )

            record = {
                "subject": sid,
                "group": row["group"],
                "duration_sec": duration,
                "clean_epochs": clean,
                "clean_fraction":
                    clean / (duration // 2),
                "theta_alpha_log_mad": mad,
                "theta_alpha_log_sd": sd
            }

            record.update(
                dict(
                    zip(
                        FEATURES,
                        map(float, calc)
                    )
                )
            )

            duration_rows.append(
                record
            )

            if duration == 300:

                reference = np.array([
                    float(
                        original[sid][name]
                    )
                    for name in FEATURES
                ])

                errors.append(
                    (
                        sid,
                        float(
                            np.max(
                                np.abs(
                                    calc - reference
                                )
                            )
                        )
                    )
                )

        if (
            i % 10 == 0
            or i == 46
        ):
            print(
                f"Verified {i}/46 participants",
                flush=True
            )

    largest_error = max(
        error for _, error in errors
    )

    print(
        "\n===== ALL-SUBJECT BASELINE REPRODUCTION ====="
    )

    print(
        "Largest 300s absolute feature difference:",
        f"{largest_error:.12g}"
    )

    print(
        "Epoch-count mismatches:",
        len(count_errors)
    )

    for item in count_errors[:10]:
        print(
            "EPOCH COUNT MISMATCH:",
            item
        )

    if (
        largest_error > 1e-6
        or count_errors
    ):

        print(
            "FULL-COHORT REPRODUCTION FAILED."
        )

        print(
            "Duration features were not saved "
            "as final results."
        )

        for sid, error in sorted(
            errors,
            key=lambda x: -x[1]
        )[:10]:

            print(
                sid,
                error
            )

        return

    # =====================================
    # SAVE VERIFIED FEATURES
    # =====================================

    output_file = (
        OUT /
        "duration_features_reproduced.csv"
    )

    write_csv(
        output_file,
        duration_rows
    )

    settings = dict(
        zip(
            (
                "welch_nperseg",
                "reference_low_hz",
                "reference_high_hz",
                "delta_low_hz",
                "gamma_high_hz",
                "reference_upper_inclusive",
                "band_upper_inclusive",
                "power_method",
                "aggregate_method"
            ),
            best
        )
    )

    settings_file = (
        OUT /
        "feature_extraction_locked.json"
    )

    settings_file.write_text(
        json.dumps(
            settings,
            indent=2
        ) + "\n"
    )

    print(
        "PASS: All 46 participants reproduce "
        "saved 300-second features."
    )

    print(
        "Duration features:",
        output_file
    )

    print(
        "Locked settings:",
        settings_file
    )

    print(
        "Original EEG and previous "
        "reports unchanged."
    )

    print(
        "\n===== FEATURE RECONSTRUCTION COMPLETE ====="
    )


if __name__ == "__main__":
    main()

