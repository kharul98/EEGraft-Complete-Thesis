#!/usr/bin/env python3
"""Exploratory EEG duration versus recording-position control."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfilt, welch

ROOT = Path(__file__).resolve().parents[1]

INV = ROOT / "results/p1_raw_recording_inventory.csv"
OLD = ROOT / "results/p2_prefix_robust_features.csv"
REF = ROOT / "results/p2_disjoint_reference_features.csv"
OLD_TV = ROOT / "results/p2_disjoint_reference_reliability.csv"

OUT = ROOT / "results/p2_duration_anchor_control.csv"
OUT_PAIRED = ROOT / "results/p2_duration_anchor_control_pairs.csv"
OUT_JSON = ROOT / "docs/audit/p2_duration_anchor_control_summary.json"

FS = 500
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))
NAMES = ("delta", "theta", "alpha", "beta", "gamma")

SOS = butter(
    4, [1, 40],
    btype="bandpass",
    fs=FS,
    output="sos"
)

WINDOWS = (
    [("prefix", 0, n) for n in (30, 60, 120, 180)]
    + [
        ("fixed_end_180", 180-n, 180)
        for n in (30, 60, 120, 180)
    ]
    + [
        ("placement_60", s, s+60)
        for s in (0, 60, 120, 180)
    ]
)


def spectral_vector(x):
    f, p = welch(
        x,
        fs=FS,
        nperseg=4*FS,
        noverlap=2*FS,
        axis=-1
    )

    mask = (f >= 1) & (f <= 40)
    total = np.trapezoid(
        p[:, mask], f[mask], axis=-1
    )

    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("Invalid spectral power")

    values = []

    for low, high in BANDS:
        mask = (f >= low) & (f <= high)

        power = np.trapezoid(
            p[:, mask], f[mask], axis=-1
        )

        values.append(power / total)

    return np.stack(values, axis=-1)


def tv(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        a.min() < 0
        or b.min() < 0
        or a.sum() <= 0
        or b.sum() <= 0
    ):
        raise ValueError("Invalid spectral vectors")

    return float(
        0.5 * np.abs(
            a/a.sum() - b/b.sum()
        ).sum()
    )


def bootstrap_delta(values, groups, seed):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(groups)

    if len(values) != 88 or not np.isfinite(values).all():
        raise ValueError("Invalid paired values")

    rng = np.random.default_rng(seed)
    samples = []

    for label in ("A", "C", "F"):
        indices = np.where(labels == label)[0]

        samples.append(
            rng.choice(
                indices,
                size=(5000, len(indices)),
                replace=True
            )
        )

    indices = np.concatenate(samples, axis=1)
    boot = values[indices].mean(axis=1)

    low, high = np.quantile(
        boot, [0.025, 0.975]
    )

    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "fraction_improved": float(
            np.mean(values < 0)
        ),
        "n": 88
    }


def main():
    inventory = pd.read_csv(INV)
    previous = pd.read_csv(OLD)
    reference = pd.read_csv(REF)
    old_tv = pd.read_csv(OLD_TV)

    assert len(inventory) == 88
    assert inventory.participant_id.nunique() == 88

    assert len(previous) == 440
    assert len(reference) == 88
    assert len(old_tv) == 352

    previous = previous.set_index(
        ["participant_id", "seconds"],
        verify_integrity=True
    )

    reference = reference.set_index(
        "participant_id",
        verify_integrity=True
    )

    old_tv = old_tv.set_index(
        ["participant_id", "seconds"],
        verify_integrity=True
    )

    records = []
    max_feature_error = 0.0
    max_reference_error = 0.0
    max_tv_error = 0.0

    print(
        "===== P2 DURATION / ANCHOR CONTROL =====",
        flush=True
    )

    for i, item in enumerate(
        inventory.itertuples(index=False), 1
    ):
        subject = str(item.participant_id)

        raw = mne.io.read_raw_eeglab(
            str(ROOT / item.path),
            preload=False,
            verbose="ERROR"
        )

        x = raw.get_data(
            picks="eeg",
            start=0,
            stop=300*FS
        )

        assert float(raw.info["sfreq"]) == FS
        assert x.shape == (19, 300*FS)
        assert np.isfinite(x).all()

        state = np.zeros(
            (len(SOS), 19, 2)
        )

        epochs = []

        for e in range(30):
            start = e * 10 * FS

            filtered, state = sosfilt(
                SOS,
                x[:, start:start+10*FS],
                axis=-1,
                zi=state
            )

            epochs.append(
                spectral_vector(filtered)
            )

        epochs = np.stack(epochs)

        late = np.median(
            epochs[24:30],
            axis=(0, 1)
        )

        stored_reference = reference.loc[subject]

        assert str(stored_reference.group) == str(item.group)

        expected_late = np.array([
            stored_reference[
                f"late_median_relative_{band}"
            ]
            for band in NAMES
        ])

        max_reference_error = max(
            max_reference_error,
            float(np.max(np.abs(late - expected_late)))
        )

        for design, start_s, end_s in WINDOWS:
            duration = end_s - start_s

            features = np.median(
                epochs[start_s//10:end_s//10],
                axis=(0, 1)
            )

            distance = tv(features, late)

            if design == "prefix":
                old = previous.loc[
                    (subject, duration)
                ]

                expected = np.array([
                    old[f"median_relative_{band}"]
                    for band in NAMES
                ])

                max_feature_error = max(
                    max_feature_error,
                    float(
                        np.max(np.abs(features - expected))
                    )
                )

                old_distance = float(
                    old_tv.loc[
                        (subject, duration),
                        "tv_to_late_240_300"
                    ]
                )

                max_tv_error = max(
                    max_tv_error,
                    abs(distance - old_distance)
                )

            records.append({
                "participant_id": subject,
                "group": str(item.group),
                "design": design,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration,
                "gap_to_reference_s": 240-end_s,
                "tv_to_late_240_300": distance
            })

        if i % 22 == 0:
            print(
                f"Processed {i}/88 participants",
                flush=True
            )

    if max(
        max_feature_error,
        max_reference_error,
        max_tv_error
    ) > 1e-8:
        raise ValueError("Historical reproduction check failed")

    df = pd.DataFrame(records)

    assert len(df) == 1056
    assert df.participant_id.nunique() == 88
    assert df.tv_to_late_240_300.between(0, 1).all()
    assert df.gap_to_reference_s.ge(0).all()

    ids = sorted(df.participant_id.unique())

    groups = (
        inventory.set_index("participant_id")
        .loc[ids, "group"]
        .to_numpy()
    )

    def values(design, duration, start):
        subset = df.loc[
            df.design.eq(design)
            & df.duration_s.eq(duration)
            & df.start_s.eq(start)
        ].set_index("participant_id")

        assert len(subset) == 88

        return subset.loc[
            ids,
            "tv_to_late_240_300"
        ].to_numpy(dtype=float)

    contrasts = {
        "prefix_120_minus_60":
            values("prefix", 120, 0)
            - values("prefix", 60, 0),

        "prefix_180_minus_60":
            values("prefix", 180, 0)
            - values("prefix", 60, 0),

        "fixed_end_120_minus_60":
            values("fixed_end_180", 120, 60)
            - values("fixed_end_180", 60, 120),

        "fixed_end_180_minus_60":
            values("fixed_end_180", 180, 0)
            - values("fixed_end_180", 60, 120),

        "placement_60_start120_minus_start0":
            values("placement_60", 60, 120)
            - values("placement_60", 60, 0),

        "placement_60_start180_minus_start0":
            values("placement_60", 60, 180)
            - values("placement_60", 60, 0)
    }

    contrasts["duration_effect_difference_120"] = (
        contrasts["fixed_end_120_minus_60"]
        - contrasts["prefix_120_minus_60"]
    )

    contrasts["duration_effect_difference_180"] = (
        contrasts["fixed_end_180_minus_60"]
        - contrasts["prefix_180_minus_60"]
    )

    paired = pd.DataFrame([
        {
            "comparison": name,
            **bootstrap_delta(
                difference,
                groups,
                20261107 + j
            )
        }
        for j, (name, difference)
        in enumerate(contrasts.items())
    ])

    medians = (
        df.groupby(
            ["design", "start_s", "end_s", "duration_s"],
            as_index=False
        )
        .tv_to_late_240_300
        .median()
    )

    report = {
        "status": "EXPLORATORY",
        "participants": 88,
        "window_rows": len(df),
        "reference_interval_s": [240, 300],
        "reproduction_errors": {
            "prefix_features": max_feature_error,
            "late_reference": max_reference_error,
            "prefix_tv": max_tv_error
        },
        "medians": medians.to_dict("records"),
        "contrasts": paired.to_dict("records"),
        "limitations": [
            "Same recording and same participant.",
            "Late reference is not ground truth.",
            "Window-placement comparisons retain causal filter state from earlier samples.",
            "The fixed endpoint controls timing relative to the reference, not all temporal confounding.",
            "Post-hoc exploratory intervals; no multiplicity correction."
        ]
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUT, index=False, float_format="%.12g"
    )

    paired.to_csv(
        OUT_PAIRED, index=False, float_format="%.12g"
    )

    OUT_JSON.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("===== DURATION AND ANCHOR MEDIANS =====")
    print(
        medians.round(6).to_string(index=False)
    )

    print()
    print("===== PAIRED ANCHOR CONTRASTS =====")
    print(
        paired.round(6).to_string(index=False)
    )

    print()
    print("===== REPRODUCTION CHECK =====")
    print("Prefix feature error:", max_feature_error)
    print("Late reference error:", max_reference_error)
    print("Historical TV error:", max_tv_error)
    print("Participants: 88")
    print("Windows: 1056")
    print("Reference overlap: NONE")
    print("P2 ANCHOR CONTROL COMPLETE")


if __name__ == "__main__":
    main()
