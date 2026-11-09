#!/usr/bin/env python3
"""EEGraft: export 10 thesis figures from archived aggregates and frozen reports.

Never reads raw EEG, per-recording CAUEEG test predictions, or model checkpoints.
Outputs PNG, vector PDF, and a provenance JSON. No fitting/tuning.

Usage:
 python eegraft_export_10_figures.py [--zip /path/to/Overleaf.zip] [--work ~/research/eegraft-caueeg] [--out DIR]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE = "#24577c"
TEAL = "#2f9f9b"
PURPLE = "#875b98"
GREY = "#8a98a5"
ORANGE = "#cf8837"
RED = "#a14c4c"
LIGHT = "#dce4eb"
DARK = "#20364b"

ARCHIVED_NUMERIC_SOURCE_SHA256 = "f46620b2746c9bcde2da566a9b230479dae99838789e67d9380b2a38cf146242"
ARCHIVED_DURATION_IMAGE_SHA256 = "6c3db4aabf0e87c7ef8c7288ccca3e88a0a44e5af828352860f63d4a5a8c6cb9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#778a9c", "text.color": DARK,
    "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "none"
})


def fail(msg):
    raise RuntimeError(msg)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_json(p):
    if not p.is_file():
        fail(f"Required aggregate report missing: {p}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def check(cond, msg):
    if not cond:
        fail(msg)


def ensure_fraction(x, expected, name, tolerance=0.003):
    x = float(x)
    check(np.isfinite(x) and 0 <= x <= 1, f"Invalid fraction: {name}")
    check(abs(x - expected) < tolerance,
          f"Research metric changed ({name}): got {x}, expected approximately {expected}. Stop and investigate.")
    return x


def find_archive(given):
    if given:
        paths = [Path(given).expanduser()]
    else:
        home = Path.home()
        paths = list((home / "Downloads").glob("*Overleaf*.zip"))
        paths += list((home / "research").glob("*Overleaf*.zip"))
        paths += list(home.glob("*Overleaf*.zip"))
        for wpath in Path("/mnt/c/Users").glob("*/Downloads/*Overleaf*.zip"):
            paths.append(wpath)
        # Also accept an unzipped user's thesis project if no archive found.
        roots = [home / "alzheimers-thesis", home / "research"]
        for root in roots:
            if root.is_dir():
                paths += list(root.glob("**/EEGraft_Final_*Overleaf.zip"))[:20]
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                entries = z.namelist()
                script = next(n for n in entries if n.endswith("source/rebuild_numeric_figures.py"))
                duration = next(n for n in entries if n.endswith("images/fig06_duration.png"))
                if sha_bytes(z.read(script)) == ARCHIVED_NUMERIC_SOURCE_SHA256 and \
                   sha_bytes(z.read(duration)) == ARCHIVED_DURATION_IMAGE_SHA256:
                    return p.resolve(), script, duration
        except (OSError, zipfile.BadZipFile, StopIteration):
            continue
    fail("Cannot find the verified existing Overleaf ZIP. Download EEGraft_Final_Extended_Overleaf.zip "
         "to Windows Downloads, or specify --zip '/mnt/c/Users/YOUR_USER/Downloads/EEGraft_Final_Extended_Overleaf.zip'. "
         "We will not invent missing historical figure data.")


def setup_plot(title, subtitle=None, size=(7.0, 4.6)):
    fig, ax = plt.subplots(figsize=size)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985, color=DARK)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top", color="#516679", fontsize=8.6)
    return fig, ax


def note(fig, text):
    fig.text(.035, .015, text, ha="left", va="bottom", fontsize=7.7, color="#5d6d7c")


def export(fig, out, tag, manifest, detail, adjust=0.12):
    fig.tight_layout(rect=[0.035, adjust, .975, .875])
    png = out / (tag + ".png")
    pdf = out / (tag + ".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    manifest[tag] = {"png": str(png), "pdf": str(pdf), "method": detail,
                     "png_sha256": sha_file(png), "pdf_sha256": sha_file(pdf)}
    print("[OK]", tag)


def old_61(out, m):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.suptitle("6.1  Temporal classification and persistent errors", fontsize=12, fontweight="bold", y=.985)
    labs = ["AD vs CN", "FTD vs CN"]
    yy = [1, 0]
    left.scatter([73.48, 73.72], yy, s=105, color=BLUE)
    for y, x in zip(yy, [73.48, 73.72]):
        left.text(x + 3.5, y, f"{x:.2f}%", va="center", fontsize=8)
    left.set_yticks(yy, labs); left.set_xlim(0, 100); left.set_ylim(-.6, 1.6)
    left.set_title("First-minute BAcc", fontsize=10)
    left.set_xlabel("Balanced accuracy (%)")
    for j, (values, label, color) in enumerate([
        ([44.35, 33.66], "Changed over 5 min", TEAL),
        ([8.70, 13.17], "Stable but wrong", PURPLE)
    ]):
        dy = .12 if j == 0 else -.12
        right.scatter(values, [z + dy for z in yy], s=85, color=color, label=label)
        for v, y in zip(values, yy):
            right.text(v + 3, y + dy, f"{v:.1f}%", fontsize=7.5, va="center")
    right.set_yticks(yy, labs); right.set_xlim(0, 65); right.set_ylim(-.6, 1.6)
    right.set_title("Longitudinal prediction states", fontsize=10)
    right.set_xlabel("Participant-repeat states (%)")
    right.legend(fontsize=7.5, loc="lower right", frameon=False)
    right.grid(axis="x", color=LIGHT, alpha=.65)
    fig.text(.03, .02,
        "Change/error denominators: 230 AD/CN and 205 FTD/CN participant-repeat states; not independent people.",
        fontsize=7.3, color="#5d6d7c")
    export(fig, out, "Fig_6_1_Temporal_Reliability", m,
           "Historical Study A; archived source/rebuild_numeric_figures.py SHA-verified; aggregate values only", .15)


def old_62(out, m, z, member):
    raw = z.read(member)
    image = out / "Fig_6_2_Duration_Original_Archived.png"
    image.write_bytes(raw)
    # Keep chart pixels unchanged. This PDF only places the archived raster on a white page.
    im = Image.open(image).convert("RGB")
    fig = plt.figure(figsize=(7.1, 5.15))
    ax = fig.add_axes([.03, .05, .94, .90])
    ax.imshow(im)
    ax.axis("off")
    pdf = out / "Fig_6_2_Duration_Archived.pdf"
    fig.savefig(pdf, dpi=240, facecolor="white")
    plt.close(fig)
    m["Fig_6_2_Duration"] = {
        "png": str(image), "pdf": str(pdf),
        "method": "Preserved original Figure 06 bitmap, pixel-unchanged; underlying duration plotting records not bundled in Overleaf archive; DO NOT describe as fresh recomputation",
        "png_sha256": sha_file(image), "pdf_sha256": sha_file(pdf)}
    print("[OK] Fig_6_2_Duration (original authentic archived chart retained)")


def old_63(out, m):
    models = ["DSRF", "XGBoost", "EEGNet-inspired", "Random Forest", "High only", "SVM", "No skip", "Low only"]
    scores = np.array([77.06, 76.61, 74.89, 72.71, 72.26, 71.81, 71.44, 62.74])
    fig, ax = setup_plot("6.3  Source-cohort model development", "Five-fold pooled subject-level results (52 development participants)", (7.0, 5.0))
    ys = np.arange(len(models))
    for i, (name, value) in enumerate(zip(models, scores)):
        color = BLUE if i == 0 else GREY
        ax.hlines(i, 0, value, color=color, linewidth=2, alpha=.85)
        ax.scatter(value, i, color=color, s=50, zorder=3)
        ax.text(value + 1.1, i, f"{value:.2f}%", fontsize=8.2, va="center")
    ax.invert_yaxis(); ax.set_yticks(ys, models)
    ax.set_xlim(0, 90); ax.set_xlabel("Balanced accuracy (%)")
    note(fig, "Descriptive development comparison; these rankings do not establish statistical superiority.")
    export(fig, out, "Fig_6_3_Development_Models", m,
           "Archived numeric generator SHA-verified; five-fold source development aggregate results", .09)


def old_64(out, m):
    names = ["DSRF residual", "Capacity-matched gated blend", "High only", "No skip", "Low only"]
    values = [77.06, 73.16, 72.26, 71.44, 62.74]
    fig, ax = setup_plot("6.4  EEGraft decoder ablations", "Pooled development BAcc; higher is better", (7.0, 4.4))
    ypos = np.arange(len(names))[::-1]
    for i, (y, v) in enumerate(zip(ypos, values)):
        ax.plot([50, v], [y, y], linewidth=2.0, color=LIGHT)
        ax.scatter(v, y, s=90, color=BLUE if i == 0 else PURPLE if i == 1 else GREY, zorder=3)
        ax.text(v + .8, y, f"{v:.2f}%", fontsize=8.5, va="center")
    ax.set_yticks(ypos, names); ax.set_xlim(50, 85)
    ax.set_xlabel("Balanced accuracy (%)")
    note(fig, "The gated blend is capacity-matched; other ablations are component comparisons.")
    export(fig, out, "Fig_6_4_Decoder_Ablations", m,
           "Archived numeric generator SHA-verified; source 52-person development aggregate results", .12)


def old_65(out, m):
    names = ["DSRF", "XGBoost"]
    points = [70.2381, 77.38095]
    intervals = [(46.4286, 92.8571), (53.5714, 100.0)]
    fig, ax = setup_plot("6.5  Locked source-cohort holdout", "13 participants; exploratory stratified participant bootstrap", (7.0, 4.0))
    y = np.array([1, 0])
    for i, (v, bounds) in enumerate(zip(points, intervals)):
        col = BLUE if i == 0 else TEAL
        ax.hlines(y[i], bounds[0], bounds[1], color=col, linewidth=3)
        ax.plot([bounds[0], bounds[0]], [y[i]-.09, y[i]+.09], color=col)
        ax.plot([bounds[1], bounds[1]], [y[i]-.09, y[i]+.09], color=col)
        ax.scatter(v, y[i], s=115, color=col, zorder=3)
        ax.text(v + 1.5, y[i] + .12, f"{v:.2f}%", fontsize=9)
    ax.axvline(50, ls="--", color=GREY)
    ax.set_yticks(y, names); ax.set_xlim(35, 110); ax.set_ylim(-.65, 1.55)
    ax.set_xlabel("Balanced accuracy (%)")
    note(fig, "DSRF - XGBoost: -7.14 pp; paired 95% CI [-28.57, +14.29] pp (crosses zero).")
    export(fig, out, "Fig_6_5_Source_Locked_Holdout", m,
           "Archived numeric generator SHA-verified; 13-person source holdout and reported bootstraps", .15)


def external_66(out, m, test):
    check(test.get("status") == "LOCKED_EXTERNAL_TEST_EVALUATION_COMPLETED", "Code 19 report not complete")
    check(test.get("primary_binary_recordings") == 57, "Expected exactly 57 locked binary test recordings")
    variants = test["model_variants"]
    dsrf = ensure_fraction(variants["DSRF"]["original"]["balanced_accuracy"], .6838, "DSRF external BAcc")
    xgb = ensure_fraction(variants["XGBoost"]["original"]["balanced_accuracy"], .5857, "XGBoost external BAcc")
    bacc = [dsrf * 100, xgb * 100]
    # Bootstrap endpoints from the supplied Code19 terminal output, not recomputed from patient records.
    cis = [(55.26, 80.06), (52.86, 65.71)]
    aucs = [.7481, .7221]
    check(test["model_variants"]["XGBoost"].get("threshold") is not None,
          "Missing frozen secondary XGBoost threshold result")
    fig, ax = setup_plot("6.6  CAUEEG locked external evaluation", "57 recordings: 35 Normal / 22 Dementia; original threshold 0.50", (7.0, 4.5))
    for i, (v, (lo, hi), auc) in enumerate(zip(bacc, cis, aucs)):
        y = 1 - i
        col = BLUE if i == 0 else TEAL
        ax.hlines(y, lo, hi, color=col, lw=3)
        ax.vlines([lo, hi], y-.09, y+.09, color=col, lw=1.5)
        ax.scatter([v], [y], s=120, color=col, zorder=3)
        ax.text(v+1.5, y+.12, f"BAcc {v:.2f}%   AUROC {auc:.4f}", fontsize=8)
    ax.set_yticks([1, 0], ["Frozen EEGraft DSRF", "Frozen XGBoost"])
    ax.set_xlim(40, 103); ax.set_ylim(-.55, 1.55)
    ax.axvline(50, ls="--", lw=1, color=GREY)
    ax.set_xlabel("Recording-level balanced accuracy (%)")
    note(fig, "95% stratified recording bootstrap; paired DSRF-XGBoost difference CI [-3.31, +22.08] pp.")
    export(fig, out, "Fig_6_6_CAUEEG_External_Test", m,
           "Code19 frozen original threshold results; CIs and AUROC from user-supplied Code19 aggregate terminal log", .18)


def threshold_71(out, m, confound):
    check(confound.get("status") == "RELIABILITY_CONFOUND_AUDIT_COMPLETED", "Code23 report incomplete")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5), sharex=True)
    fig.suptitle("7.1  Threshold changes apparent prediction stability", fontsize=12, fontweight="bold", y=.98)
    for ax, split, expected_n in zip(axes, ("train", "validation"), (404, 42)):
        variants = confound["results"][split]["threshold_sensitivity"]["variants"]
        items = [
            ("DSRF 0.50", "DSRF_original", BLUE),
            ("XGBoost 0.50", "XGBoost_original", PURPLE),
            ("XGBoost 0.70", "XGBoost_training_threshold", TEAL),
        ]
        yy = [2, 1, 0]
        for y, (name, key, col) in zip(yy, items):
            d = variants[key]
            check(d["n_recordings"] == expected_n, f"Code23 {split} cohort count changed")
            frac = float(d["condition_flip_fraction"])
            check(0 <= frac <= 1, "Invalid flip fraction")
            ax.scatter(frac*100, y, s=70, color=col, zorder=3)
            ax.text(frac*100+2.4, y, f"{frac:.1%}", va="center", fontsize=8)
        ax.set_yticks(yy, [r[0] for r in items], fontsize=8)
        ax.set_xlim(0, 52); ax.set_ylim(-.65, 2.65)
        ax.set_title(f"{split.title()} (n={expected_n})", fontsize=10)
        ax.grid(axis="x", color=LIGHT)
        ax.set_xlabel("Eye-state label changes (%)", fontsize=9)
    fig.text(.03, .015, "The same saved scores are relabeled at 0.50 vs train-selected 0.70; exploratory paired eye-state subset.",
             fontsize=7.3, color="#5d6d7c")
    export(fig, out, "Fig_7_1_Threshold_Dependent_Stability", m,
           "Code23 report, train n404/validation n42; unchanged original models", .14)


def stable_72(out, m, test):
    check(test.get("primary_binary_recordings") == 57, "Code19 binary denominator changed")
    # Code19 reported aggregates for unanimous 24-window labels (not recomputed from raw/test signals).
    names = ["Frozen EEGraft DSRF", "Frozen XGBoost"]
    correct = [4, 13]; wrong = [0, 12]
    nonunanimous = [57 - c - w for c, w in zip(correct, wrong)]
    fig, ax = setup_plot("7.2  Consistent predictions can still be wrong", "CAUEEG locked binary test; 24 windows per recording, n = 57", (7.0, 4.1))
    yy = np.array([1, 0])
    for i, (value, label, color) in enumerate([
        (correct, "Unanimous and correct", TEAL),
        (wrong, "Unanimous but wrong", RED),
        (nonunanimous, "Window labels not unanimous", LIGHT)
    ]):
        left = np.zeros(2) if i == 0 else (np.asarray(correct) if i == 1 else np.asarray(correct)+np.asarray(wrong))
        bars = ax.barh(yy, value, left=left, height=.44, label=label, color=color, edgecolor="white")
        for bar, n, start in zip(bars, value, left):
            if n:
                ax.text(start+n/2, bar.get_y()+bar.get_height()/2, str(n), va="center", ha="center", fontsize=9,
                        color="white" if i < 2 else DARK, fontweight="bold")
    ax.set_xlim(0, 60); ax.set_yticks(yy, names)
    ax.set_xlabel("Number of recordings (57 per model)")
    fig.text(.5, .84, "Teal: unanimous correct    |    Red: unanimous wrong    |    Gray: not unanimous",
             ha="center", fontsize=8, color=DARK)
    note(fig, "Unanimous = all 24 window labels agree at original threshold 0.50; no inference of clinical safety.")
    export(fig, out, "Fig_7_2_Unanimous_Correct_Wrong", m,
           "Code19 user-supplied aggregate terminal: DSRF 4 unanimous correct/0 wrong, XGBoost 13/12", .21)


def condition_73(out, m, c23):
    fig, ax = setup_plot("7.3  Recording condition and elapsed time", "Exploratory within-recording fixed effects; open - closed model score", (7.0, 4.9))
    records = []
    for split in ("train", "validation"):
        cases = c23["results"][split]
        for model in ("DSRF", "XGBoost"):
            fit = cases["time_adjustment"][model]
            matched = cases["close_pair_sensitivity"][model]
            point = float(fit["eye_state_coefficient_adjusted_for_linear_time"])
            ci = fit["eye_state_coefficient_95ci"]
            check(float(ci[0]) <= point <= float(ci[1]), "Invalid fixed-effects CI")
            records.append((split, model, point, ci, matched.get("mean_recording_score_difference"),
                            matched.get("recordings_with_close_opposite_state_pairs", 0)))
    yy = np.arange(4)[::-1]
    for y, (split, model, point, ci, near, n) in zip(yy, records):
        col = BLUE if model == "DSRF" else PURPLE
        ax.hlines(y, *ci, color=col, lw=2.5)
        ax.vlines([ci[0], ci[1]], y-.07, y+.07, color=col)
        ax.scatter(point, y, color=col, s=70, marker="o", zorder=3)
        if near is not None:
            ax.scatter(float(near), y, color=ORANGE, s=60, marker="D", zorder=3)
        ax.text(.175, y, f"matched n={n}", fontsize=7.8, va="center")
    ax.set_yticks(yy, [f"{split.title()}  {model}" for split,model,*_ in records])
    ax.set_xlim(-.09, .27); ax.set_xlabel("Change in frozen dementia score (open - closed)")
    ax.axvline(0, ls="--", color=GREY)
    fig.text(.5, .845, "Circle + interval: linear time-adjusted estimate and 95% CI    |    Diamond: nearby-window mean",
             ha="center", fontsize=7.8, color=DARK)
    note(fig, "Time-adjusted means linear elapsed-time adjustment, not causal identification. Validation CIs cross zero.")
    export(fig, out, "Fig_7_3_Eye_Condition_Time_Audit", m,
           "Code23 fixed-effects cluster CIs and <=60s matched-window sensitivity; observational post-hoc", .16)


def age_74(out, m, age, incr):
    check(age.get("status") == "AGE_LABEL_AUDIT_COMPLETED", "Code24 incomplete")
    check(incr.get("status") == "AGE_INCREMENTAL_AUDIT_COMPLETED", "Code25 incomplete")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 7.0), gridspec_kw={"height_ratios": [1, 1.4]})
    fig.suptitle("7.4  Age imbalance and incremental EEG scores", fontsize=12, fontweight="bold", y=.985)
    top.set_title("(A) Mean age by class: Normal versus Dementia", fontsize=10)
    for i, split in enumerate(("train", "validation")):
        classes = age["demographic_summary"][split]["by_class"]
        n, d = (float(classes[c]["mean_age"]) for c in ("normal", "dementia"))
        top.plot([n, d], [i, i], color=LIGHT, linewidth=4)
        top.scatter(n, i, s=85, color=BLUE, label="Normal" if i == 0 else None)
        top.scatter(d, i, s=85, color=PURPLE, label="Dementia" if i == 0 else None)
        top.text(n, i+.15, f"{n:.1f} y", ha="center", fontsize=8)
        top.text(d, i+.15, f"{d:.1f} y", ha="center", fontsize=8)
    top.set_yticks([0,1], ["Training", "Validation"])
    top.set_xlim(57, 88); top.set_ylim(-.45, 1.55)
    top.set_xlabel("Mean recorded age (years)")
    fig.text(.5, .872, "Navy: Normal    |    Purple: Dementia", ha="center", fontsize=8.2, color=DARK)
    top.invert_yaxis()
    bottom.set_title("(B) AUROC gain from adding an EEG score to age-only model", fontsize=10)
    comp = incr["paired_incremental_information"]["validation"]
    keys = ["linear_DSRF_added_to_age", "linear_XGBoost_added_to_age",
            "spline_DSRF_added_to_age", "spline_XGBoost_added_to_age"]
    labels = ["Linear age + DSRF", "Linear age + XGBoost", "Spline age + DSRF", "Spline age + XGBoost"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        d = comp[key]
        point = float(d["auroc_improvement"])
        ci = [float(v) for v in d["auroc_difference_95ci"]]
        check(ci[0] <= 0 <= ci[1], f"Unexpected age increment CI: {key}")
        yy = 3-i
        color = BLUE if "DSRF" in key else TEAL
        bottom.hlines(yy, *ci, lw=2.3, color=color)
        bottom.vlines(ci, yy-.07, yy+.07, color=color)
        bottom.scatter(point, yy, s=65, color=color)
        bottom.text(.118, yy, f"{point:+.3f}", fontsize=8, va="center")
    bottom.set_yticks(np.arange(4)[::-1], labels)
    bottom.axvline(0, color=GREY, ls="--")
    bottom.set_xlim(-.12, .15)
    bottom.set_xlabel("Incremental AUROC on CAUEEG validation (n=54)")
    fig.text(.03, .015, "Post-hoc; recording-bootstrap 95% intervals condition on fitted scores; all validation intervals include zero.",
             fontsize=7.4, color="#5d6d7c")
    export(fig, out, "Fig_7_4_Age_Confounding_Increment", m,
           "Code24 age means and Code25 paired validation AUROC increments; post-hoc aggregate only", .11)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=None, help="Latest/old corrected Overleaf source ZIP (auto-search if omitted)")
    ap.add_argument("--work", default=str(Path.home()/"research/eegraft-caueeg"), help="Ubuntu CAUEEG work directory")
    ap.add_argument("--out", default=str(Path.home()/"research/eegraft-figure-exports-v1"), help="Output directory")
    args = ap.parse_args()
    work = Path(args.work).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print("\n1. Verifying original evidence archive")
    archive, script_entry, duration_entry = find_archive(args.zip)
    print("[VERIFIED]", archive)
    print("[VERIFIED] historic numerical plotting source SHA-256")
    print("[VERIFIED] original duration figure SHA-256")

    print("\n2. Verifying frozen external report files")
    sources = {
        "19": work/"results/19_locked_test_evaluation.json",
        "22": work/"results/22_condition_analysis.json",
        "23": work/"results/23_reliability_confound_audit.json",
        "24": work/"results/24_age_label_audit.json",
        "25": work/"results/25_age_incremental_information_audit.json",
        "plan": work/"protocol/caueeg_final_test_plan_v1.json"
    }
    docs = {k: load_json(p) for k, p in sources.items()}
    for k, status in (
        ("19", "LOCKED_EXTERNAL_TEST_EVALUATION_COMPLETED"),
        ("22", "CONDITION_ANALYSIS_COMPLETED"),
        ("23", "RELIABILITY_CONFOUND_AUDIT_COMPLETED"),
        ("24", "AGE_LABEL_AUDIT_COMPLETED"),
        ("25", "AGE_INCREMENTAL_AUDIT_COMPLETED")
    ):
        check(docs[k].get("status")==status, f"Unexpected Code{k} status, refusing to plot")
    plan_hash = sha_file(sources["plan"])
    check(docs["19"].get("final_test_plan_sha256") == plan_hash, "Code19 plan identity mismatch")
    check(docs["23"]["provenance"].get("code22_report_sha256") == sha_file(sources["22"]),
          "Code22/23 provenance mismatch")
    check(docs["25"]["source_hashes"].get("code24_report_sha256") == sha_file(sources["24"]),
          "Code24/25 provenance mismatch")
    check(docs["25"].get("test_predictions_accessed")==0, "Unexpected Code25 test access")
    print("[VERIFIED] frozen Code19, Code22, Code23, Code24, Code25 reports")

    with zipfile.ZipFile(archive) as z:
        m = {}
        print("\n3. Exporting the ten figures")
        old_61(out, m)
        old_62(out, m, z, duration_entry)
        old_63(out, m)
        old_64(out, m)
        old_65(out, m)
        external_66(out, m, docs["19"])
        threshold_71(out, m, docs["23"])
        stable_72(out, m, docs["19"])
        condition_73(out, m, docs["23"])
        age_74(out, m, docs["24"], docs["25"])

    check(len(m)==10, "Not all 10 figures generated")
    manifest = {
        "status": "TEN_FIGURE_EXPORT_COMPLETED",
        "scientific_basis": "Archive source aggregates + previously frozen external reports; no raw EEG read",
        "source_overleaf_archive": str(archive),
        "archive_sha256": sha_file(archive),
        "report_hashes": {key:sha_file(path) for key,path in sources.items()},
        "original_plotting_script_sha256": ARCHIVED_NUMERIC_SOURCE_SHA256,
        "archived_duration_figure_sha256": ARCHIVED_DURATION_IMAGE_SHA256,
        "limitations": [
            "Old figures 6.1, 6.3, 6.4, 6.5 are redrawn from verified archived aggregate literals, not raw EEG recomputation.",
            "Figure 6.2 is preserved as the existing unmodified bitmap; raw plotted duration arrays were not bundled.",
            "Figures 6.6 and 7.2 rely on reported Code19 terminal aggregate intervals/unanimous counts checked against the frozen report identity.",
            "Figures 7.1, 7.3, 7.4 use the archived Code23-25 aggregate values.",
            "External analyses of recording condition and age were exploratory and do not imply causal effects.",
            "Recording-level confidence intervals cannot assume independently verified participants.",
            "No new model fitting, test re-selection, event-file reads, EEG reads, or raw test predictions."
        ],
        "figures": m
    }
    manifest_path = out / "FIGURE_PROVENANCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("[SAVED]", manifest_path)
    print("\nFIGURE_EXPORT_STATUS: COMPLETED (10 / 10)")
    print("OUTPUT_DIRECTORY:", out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[STOPPED - NO INVENTED DATA]", type(exc).__name__, str(exc), file=sys.stderr)
        sys.exit(1)
