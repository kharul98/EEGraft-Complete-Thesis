#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

from scipy.signal import butter, sosfilt, welch


ROOT = Path(__file__).resolve().parents[1]

FIG = ROOT / "figures/2026-11-07_p2_evidence"
SOURCE = ROOT / "results/figure_sources"

FIG.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

INPUTS = [
    "results/p2_paired_uncertainty.csv",
    "results/p2_quality_control_metrics.csv",
    "results/p2_adaptive_pilot_repeat_metrics.csv",
    "results/p1_raw_recording_inventory.csv",
]

for name in INPUTS:
    if not (ROOT / name).is_file():
        raise FileNotFoundError(name)

code_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True
).strip()

records = []


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def save(
    fig, figure_id, name, purpose,
    sources, caveat, section, experiment,
    vector=True
):
    png = FIG / f"{name}.png"
    svg = FIG / f"{name}.svg"

    if not png.exists():
        fig.savefig(
            png,
            dpi=300,
            bbox_inches="tight"
        )
    else:
        print("Preserving existing:", png.name)

    if vector and not svg.exists():
        fig.savefig(
            svg,
            bbox_inches="tight"
        )

    plt.close(fig)

    outputs = [png]

    if vector and svg.exists():
        outputs.append(svg)

    records.append({
        "figure_id": figure_id,
        "section": section,
        "purpose": purpose,
        "dataset": "NEMAR on004504 v1.0.0",
        "source_experiment": experiment,
        "input_artifacts": sources,
        "script": "scripts/p2_generate_evidence_figures.py",
        "source_code_commit": code_commit,
        "selection_rule": (
            "First sorted participant for single-recording figures; "
            "all eligible participants for group-level figures"
        ),
        "outputs": [
            str(p.relative_to(ROOT))
            for p in outputs
        ],
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in outputs
        },
        "status": "GENERATED_AWAITING_VISUAL_QA",
        "interpretation": purpose,
        "caveat": caveat
    })

    print("FIGURE GENERATED:", figure_id, png.name)


# ==================================================
# Figure 1: Actual paired uncertainty
# ==================================================

paired = pd.read_csv(ROOT / INPUTS[0])

assert len(paired) == 4

paired = paired.iloc[::-1].reset_index(drop=True)

labels = [
    f"{r.task}: {r.candidate} vs {r.reference}"
    for r in paired.itertuples()
]

fig, ax = plt.subplots(figsize=(11, 4.8))

for index, row in enumerate(paired.itertuples()):

    ax.plot(
        [
            row.delta_ba_ci_low * 100,
            row.delta_ba_ci_high * 100
        ],
        [index, index],
        linewidth=2
    )

    ax.plot(
        row.delta_ba * 100,
        index,
        "o"
    )

ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1
)

ax.set_yticks(range(len(labels)), labels)

ax.set_xlabel(
    "Balanced-accuracy difference (percentage points)"
)

ax.set_title(
    "P2: Paired exploratory differences and 95% intervals"
)

ax.grid(axis="x", alpha=0.2)

fig.text(
    0.04, 0.01,
    "Post-hoc; participant-clustered bootstrap; "
    "conditional on saved OOF predictions; "
    "no multiplicity correction.",
    fontsize=8
)

fig.tight_layout(rect=(0, 0.055, 1, 1))

save(
    fig,
    "P2-F01",
    "p2_posthoc_effects",
    "Display all paired differences, including intervals crossing zero.",
    [INPUTS[0]],
    "Exploratory comparisons; full retraining uncertainty not captured.",
    "Results",
    "P2 paired uncertainty"
)


# ==================================================
# Figure 2: EEG quality-only negative control
# ==================================================

quality = pd.read_csv(ROOT / INPUTS[1])

assert len(quality) == 18

quality_summary = (
    quality.groupby(
        ["task", "seconds"],
        as_index=False
    )["balanced_accuracy"]
    .mean()
)

quality_summary.to_csv(
    SOURCE / "p2_quality_ba_plot_values.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 5))

for task, group in quality_summary.groupby("task"):

    group = group.sort_values("seconds")

    ax.plot(
        group.seconds,
        group.balanced_accuracy,
        marker="o",
        label=task
    )

ax.axhline(
    0.5,
    color="black",
    linestyle="--",
    alpha=0.6,
    label="Chance reference"
)

ax.set_xticks([30, 60, 300])
ax.set_ylim(0, 1)

ax.set_xlabel("Analyzed EEG prefix (seconds)")
ax.set_ylabel("Balanced accuracy")

ax.set_title(
    "Exploratory EEG quality-only negative control"
)

ax.legend()
ax.grid(alpha=0.2)

fig.text(
    0.04, 0.01,
    "Quality descriptors only; not validated artifact labels. "
    "Means across correlated CV repeats.",
    fontsize=8
)

fig.tight_layout(rect=(0, 0.05, 1, 1))

save(
    fig,
    "P2-F02",
    "p2_quality_negative_control",
    "Show whether amplitude-based quality descriptors predict diagnosis.",
    [
        INPUTS[1],
        "results/figure_sources/p2_quality_ba_plot_values.csv"
    ],
    "Near-chance prediction does not rule out acquisition confounding.",
    "Results",
    "P2 quality negative control"
)


# ==================================================
# Figure 3: Actual stopping-policy trade-off
# ==================================================

metrics = pd.read_csv(ROOT / INPUTS[2])

assert len(metrics) == 60

policy_summary = (
    metrics.loc[metrics.task == "AD_vs_CN"]
    .groupby("policy", as_index=False)[
        ["balanced_accuracy", "mean_seconds"]
    ]
    .mean()
)

assert len(policy_summary) == 10

policy_summary.to_csv(
    SOURCE / "p2_policy_tradeoff_plot_values.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(10, 5))

for row in policy_summary.itertuples():

    marker = (
        "s" if row.policy.startswith("fixed_")
        else "o"
    )

    ax.scatter(
        row.mean_seconds,
        row.balanced_accuracy,
        label=row.policy,
        marker=marker,
        s=58
    )

ax.set_xlim(0, 320)
ax.set_ylim(0, 1)

ax.set_xlabel("Mean recording duration (seconds)")
ax.set_ylabel("Balanced accuracy")

ax.set_title(
    "AD versus controls: exploratory acquisition trade-off"
)

ax.grid(alpha=0.2)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    fontsize=8
)

fig.text(
    0.03, 0.01,
    "All 10 policies; correlated CV repeats; "
    "uncalibrated probabilities. Descriptive, not confirmatory.",
    fontsize=8
)

fig.tight_layout(rect=(0, 0.05, 0.78, 1))

save(
    fig,
    "P2-F03",
    "p2_policy_tradeoff",
    "Compare all stopping policies on observed accuracy and recording time.",
    [
        INPUTS[2],
        "results/figure_sources/p2_policy_tradeoff_plot_values.csv"
    ],
    "Repeated-CV estimates and exploratory policy choices.",
    "Results",
    "P2 adaptive stopping feasibility"
)


# ==================================================
# Real EEG example: deterministic participant choice
# ==================================================

inventory = pd.read_csv(ROOT / INPUTS[3])

assert inventory.participant_id.nunique() == 88

example = (
    inventory.sort_values("participant_id")
    .iloc[0]
)

raw_path = ROOT / example["path"]

if not raw_path.is_file():
    raise FileNotFoundError(raw_path)

raw = mne.io.read_raw_eeglab(
    str(raw_path),
    preload=False,
    verbose="ERROR"
)

fs = float(raw.info["sfreq"])

assert fs == 500.0

channels = ["Fp1", "Cz", "O1"]

assert all(
    channel in raw.ch_names
    for channel in channels
)


# ==================================================
# Figure 4: Genuine unprocessed EEG signal
# ==================================================

excerpt = raw.get_data(
    picks=channels,
    start=0,
    stop=8 * int(fs)
) * 1e6

assert np.isfinite(excerpt).all()

time = np.arange(
    excerpt.shape[1]
) / fs

fig, axes = plt.subplots(
    3, 1,
    figsize=(10, 6),
    sharex=True
)

for channel, signal, ax in zip(
    channels, excerpt, axes
):

    ax.plot(
        time,
        signal,
        linewidth=0.7
    )

    ax.set_ylabel(f"{channel} (µV)")
    ax.grid(alpha=0.2)

axes[-1].set_xlabel(
    "Time from recording start (seconds)"
)

fig.suptitle(
    "Genuine raw EEG example E01 — 500 Hz"
)

fig.text(
    0.02, 0.01,
    "NEMAR on004504 v1.0.0. First sorted subject; "
    "raw and not artifact-corrected. "
    "Single-participant illustration.",
    fontsize=8
)

fig.tight_layout(rect=(0, 0.045, 1, 0.95))

save(
    fig,
    "P0-F01",
    "p0_raw_eeg_excerpt",
    "Show actual raw EEG waveforms and physical amplitude units.",
    [
        str(raw_path.relative_to(ROOT)),
        INPUTS[3]
    ],
    "One reproducibly selected recording; not population evidence.",
    "Dataset and methods",
    "P1 raw dataset audit",
    vector=False
)


# ==================================================
# Figure 5: Real recording and nested duration prefixes
# ==================================================

x = raw.get_data(
    picks=["Fp1"],
    start=0,
    stop=300 * int(fs)
)[0] * 1e6

rms = np.sqrt(
    np.mean(
        x.reshape(300, int(fs)) ** 2,
        axis=1
    )
)

assert np.isfinite(rms).all()

pd.DataFrame({
    "second": np.arange(300),
    "fp1_rms_uv": rms
}).to_csv(
    SOURCE / "p0_prefix_rms_plot_values.csv",
    index=False
)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 6),
    sharex=True,
    gridspec_kw={
        "height_ratios": [2, 3]
    }
)

ax1.plot(
    np.arange(300) + 0.5,
    rms
)

ax1.set_ylabel(
    "Fp1 RMS (µV per 1 s)"
)

ax1.grid(alpha=0.2)

for duration in [30, 60, 120, 180, 300]:

    ax2.barh(
        f"{duration} s",
        duration,
        left=0,
        height=0.7
    )

ax2.set_xlim(0, 300)

ax2.set_xlabel(
    "Recording time (seconds)"
)

ax2.set_ylabel(
    "Available EEG prefix"
)

ax2.grid(axis="x", alpha=0.2)

fig.suptitle(
    "Real recording context and nested START prefixes"
)

fig.text(
    0.02, 0.01,
    "P2 causal-prefix construction; all shorter durations "
    "overlap the 300-second reference. "
    "Not the older center/end crop design.",
    fontsize=8
)

fig.tight_layout(rect=(0, 0.045, 1, 0.95))

save(
    fig,
    "P0-F02",
    "p0_prefix_duration_construction",
    "Show how the actual 30–300-second acquisition prefixes overlap.",
    [
        str(raw_path.relative_to(ROOT)),
        "results/figure_sources/p0_prefix_rms_plot_values.csv"
    ],
    "Nested segments are dependent observations.",
    "Methods",
    "P2 prefix extraction"
)


# ==================================================
# Figure 6: Actual causal EEG power spectrum
# ==================================================

signal = raw.get_data(
    picks=["Cz"],
    start=0,
    stop=300 * int(fs)
)[0] * 1e6

sos = butter(
    4,
    [1, 40],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfilt(
    sos,
    signal
)

psd_values = {
    "frequency_hz": None
}

fig, ax = plt.subplots(figsize=(9, 5))

for seconds in [60, 300]:

    frequencies, power = welch(
        filtered[:int(seconds * fs)],
        fs=fs,
        nperseg=int(4 * fs),
        noverlap=int(2 * fs)
    )

    if psd_values["frequency_hz"] is None:
        psd_values["frequency_hz"] = frequencies

    psd_values[
        f"psd_{seconds}s_uv2_hz"
    ] = power

    selected = (
        (frequencies >= 1)
        & (frequencies <= 40)
    )

    ax.semilogy(
        frequencies[selected],
        power[selected],
        label=f"{seconds}-second prefix"
    )

ax.set_xlabel("Frequency (Hz)")

ax.set_ylabel(
    "Welch PSD (µV²/Hz)"
)

ax.set_title(
    "Actual causally filtered EEG spectrum — Cz"
)

ax.grid(alpha=0.2)
ax.legend()

fig.text(
    0.02, 0.01,
    "Single example E01; 1–40 Hz causal Butterworth filter; "
    "no artifact correction. The prefixes overlap.",
    fontsize=8
)

fig.tight_layout(rect=(0, 0.05, 1, 1))

pd.DataFrame(
    psd_values
).to_csv(
    SOURCE / "p0_psd_plot_values.csv",
    index=False
)

save(
    fig,
    "P0-F03",
    "p0_causal_psd_example",
    "Display real 60-second and 300-second EEG spectral estimates.",
    [
        str(raw_path.relative_to(ROOT)),
        "results/figure_sources/p0_psd_plot_values.csv"
    ],
    "Single participant; no disease-specific claim.",
    "Methods",
    "P2 causal preprocessing"
)


# ==================================================
# Persistent figure provenance registry
# ==================================================

registry = ROOT / "docs/FIGURE_REGISTRY.md"

existing = (
    registry.read_text()
    if registry.exists()
    else "# Thesis Figure Registry\n"
)

heading = (
    "## Figures generated from existing evidence "
    "(2026-11-07)"
)

if heading not in existing:

    with registry.open(
        "a",
        encoding="utf-8"
    ) as output:

        output.write(
            "\n" + heading + "\n\n"
        )

        for record in records:

            output.write(
                f"### {record['figure_id']}\n\n"
            )

            for key, value in record.items():

                output.write(
                    f"- **{key}:** "
                    f"{json.dumps(value, ensure_ascii=False)}\n"
                )

            output.write("\n")

print()
print("===== SCIENTIFIC FIGURE GENERATION =====")
print("Figures:", len(records))
print("Source-code commit:", code_commit)
print("Status: GENERATED - VISUAL QA REQUIRED")
print("FIGURE EVIDENCE BATCH COMPLETE")
