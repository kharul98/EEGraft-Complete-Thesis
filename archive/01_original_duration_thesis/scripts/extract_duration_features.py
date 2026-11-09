from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT
    / "data"
    / "eeg"
    / "on004504_v1.0.0"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "eeg_duration_protocol.json"
)

OUT_DIR = ROOT / "results" / "duration_features"

OUT_DIR.mkdir(parents=True, exist_ok=True)


REGIONS = {
    "frontal": [
        "Fp1", "Fp2", "F7", "F3",
        "Fz", "F4", "F8"
    ],
    "central": [
        "C3", "Cz", "C4"
    ],
    "temporal": [
        "T3", "T4", "T5", "T6"
    ],
    "parietal": [
        "P3", "Pz", "P4"
    ],
    "occipital": [
        "O1", "O2"
    ],
}


def subject_from_path(path: Path) -> str:
    match = re.search(
        r"(sub-[A-Za-z0-9]+)",
        str(path)
    )

    if not match:
        raise RuntimeError(
            f"Cannot determine subject from {path}"
        )

    return match.group(1)


def integrate_band(
    psd: np.ndarray,
    freqs: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:

    if high == 45:
        mask = (
            (freqs >= low)
            & (freqs <= high)
        )
    else:
        mask = (
            (freqs >= low)
            & (freqs < high)
        )

    return np.trapezoid(
        psd[:, mask],
        freqs[mask],
        axis=1,
    )


def get_window(
    data: np.ndarray,
    sfreq: float,
    duration: int,
    anchor: str,
) -> tuple[np.ndarray, float, float]:

    n_total = data.shape[1]

    n_window = int(
        round(duration * sfreq)
    )

    if n_window > n_total:
        raise RuntimeError(
            f"Requested {duration}s but only "
            f"{n_total / sfreq:.2f}s available"
        )

    if anchor == "start":
        start = 0

    elif anchor == "center":
        start = (
            n_total - n_window
        ) // 2

    elif anchor == "end":
        start = (
            n_total - n_window
        )

    else:
        raise ValueError(anchor)

    stop = start + n_window

    return (
        data[:, start:stop],
        start / sfreq,
        stop / sfreq,
    )


def extract_features(
    segment: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    bands: dict[str, list[float]],
    welch_seconds: float,
    overlap_fraction: float,
) -> dict[str, float]:

    nperseg = int(
        round(welch_seconds * sfreq)
    )

    noverlap = int(
        round(nperseg * overlap_fraction)
    )

    freqs, psd = welch(
        segment,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=1,
    )

    total_mask = (
        (freqs >= 1.0)
        & (freqs <= 45.0)
    )

    total_power = np.trapezoid(
        psd[:, total_mask],
        freqs[total_mask],
        axis=1,
    )

    eps = np.finfo(float).eps

    relative = {}

    for band_name, (low, high) in bands.items():
        power = integrate_band(
            psd,
            freqs,
            float(low),
            float(high),
        )

        relative[band_name] = (
            power
            / np.maximum(total_power, eps)
        )

    ch_index = {
        name: i
        for i, name in enumerate(ch_names)
    }

    output = {}

    for band_name, values in relative.items():
        output[
            f"global_{band_name}"
        ] = float(np.mean(values))

        for region_name, channels in REGIONS.items():
            idx = [
                ch_index[ch]
                for ch in channels
            ]

            output[
                f"{region_name}_{band_name}"
            ] = float(
                np.mean(values[idx])
            )

    delta = output["global_delta"]
    theta = output["global_theta"]
    alpha = output["global_alpha"]
    beta = output["global_beta"]

    output[
        "global_slowing_ratio"
    ] = float(
        (delta + theta)
        / max(alpha + beta, eps)
    )

    output[
        "global_alpha_theta_ratio"
    ] = float(
        alpha
        / max(theta, eps)
    )

    global_psd = np.mean(
        psd,
        axis=0,
    )

    alpha_mask = (
        (freqs >= 7.0)
        & (freqs <= 13.0)
    )

    alpha_freqs = freqs[alpha_mask]
    alpha_psd = global_psd[alpha_mask]

    output[
        "global_paf_hz"
    ] = float(
        alpha_freqs[
            np.argmax(alpha_psd)
        ]
    )

    return output


def process_subject(
    set_path: Path,
    participant_row: dict,
    config: dict,
    durations: list[int],
    anchors: list[str],
) -> list[dict]:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw = mne.io.read_raw_eeglab(
            set_path,
            preload=True,
            verbose="ERROR",
        )

    sfreq = float(
        raw.info["sfreq"]
    )

    if sfreq != float(
        config["sampling_rate_hz"]
    ):
        raise RuntimeError(
            f"Unexpected sfreq {sfreq} "
            f"for {set_path}"
        )

    data = raw.get_data()

    records = []

    for duration in durations:

        for anchor in anchors:

            (
                segment,
                start_s,
                stop_s,
            ) = get_window(
                data,
                sfreq,
                duration,
                anchor,
            )

            features = extract_features(
                segment=segment,
                ch_names=raw.ch_names,
                sfreq=sfreq,
                bands=config["bands"],
                welch_seconds=float(
                    config[
                        "welch_window_seconds"
                    ]
                ),
                overlap_fraction=float(
                    config[
                        "welch_overlap_fraction"
                    ]
                ),
            )

            row = {
                "participant_id":
                    participant_row[
                        "participant_id"
                    ],
                "Group":
                    participant_row["Group"],
                "Age":
                    participant_row["Age"],
                "Gender":
                    participant_row["Gender"],
                "MMSE":
                    participant_row["MMSE"],
                "duration_seconds":
                    duration,
                "anchor":
                    anchor,
                "window_start_seconds":
                    start_s,
                "window_stop_seconds":
                    stop_s,
                "recording_seconds":
                    raw.n_times / sfreq,
            }

            row.update(features)

            records.append(row)

    return records


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sanity",
        action="store_true",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    config = json.loads(
        CONFIG_PATH.read_text()
    )

    participants = pd.read_csv(
        DATA_ROOT / "participants.tsv",
        sep="\t",
    )

    set_files = sorted(
        (
            DATA_ROOT
            / "derivatives"
        ).rglob("*.set")
    )

    file_map = {
        subject_from_path(path): path
        for path in set_files
    }

    missing = sorted(
        set(
            participants[
                "participant_id"
            ]
        )
        - set(file_map)
    )

    if missing:
        raise RuntimeError(
            f"Missing derivative files: {missing}"
        )

    if args.sanity:
        participants = (
            participants
            .iloc[:3]
            .copy()
        )

        durations = [30, 300]
        anchors = ["center"]

        output_path = (
            OUT_DIR
            / "sanity_features.csv"
        )

    else:
        durations = list(
            config[
                "durations_seconds"
            ]
        )

        anchors = list(
            config["anchors"]
        )

        output_path = (
            OUT_DIR
            / "eeg_duration_features.csv"
        )

    rows = Parallel(
        n_jobs=args.jobs,
        verbose=10,
    )(
        delayed(
            process_subject
        )(
            file_map[
                row["participant_id"]
            ],
            row,
            config,
            durations,
            anchors,
        )
        for row in participants.to_dict(
            orient="records"
        )
    )

    flat = [
        item
        for subject_rows in rows
        for item in subject_rows
    ]

    result = pd.DataFrame(flat)

    feature_cols = [
        c
        for c in result.columns
        if c.startswith(
            (
                "global_",
                "frontal_",
                "central_",
                "temporal_",
                "parietal_",
                "occipital_",
            )
        )
    ]

    if result[
        feature_cols
    ].isna().any().any():

        bad = (
            result[
                feature_cols
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            "NaNs detected:\n"
            + bad[
                bad > 0
            ].to_string()
        )

    if not args.sanity:
        expected = (
            len(participants)
            * len(durations)
            * len(anchors)
        )

        if len(result) != expected:
            raise RuntimeError(
                f"Expected {expected} rows, "
                f"got {len(result)}"
            )

        counts = (
            result
            .groupby(
                [
                    "duration_seconds",
                    "anchor",
                ]
            )[
                "participant_id"
            ]
            .nunique()
        )

        if not (
            counts == len(participants)
        ).all():

            raise RuntimeError(
                "Participant count mismatch:\n"
                + counts.to_string()
            )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "FEATURE EXTRACTION COMPLETE"
    )

    print(
        "Output:",
        output_path,
    )

    print(
        "Rows:",
        len(result),
    )

    print(
        "Participants:",
        result[
            "participant_id"
        ].nunique(),
    )

    print(
        "Feature columns:",
        len(feature_cols),
    )

    print(
        "NaN feature values:",
        int(
            result[
                feature_cols
            ]
            .isna()
            .sum()
            .sum()
        ),
    )

    print()
    print(
        "Rows by duration/anchor:"
    )

    print(
        result
        .groupby(
            [
                "duration_seconds",
                "anchor",
            ]
        )
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
