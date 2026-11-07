"""Create honest thesis result tables from finished runs, never fabricated numbers."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]


def paired_delta(path_a, path_b, n=1000, seed=42):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    joined = a.merge(b, on="subject_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError("Comparison requires EXACTLY identical held-out patient IDs")
    if not (joined.true_label_a == joined.true_label_b).all():
        raise ValueError("Ground-truth labels differ between runs")
    yy = joined.true_label_a.to_numpy()
    pred_a = joined.predicted_label_a.to_numpy()
    pred_b = joined.predicted_label_b.to_numpy()
    classes = np.unique(yy)
    groups = [np.where(yy == c)[0] for c in classes]
    delta = float(balanced_accuracy_score(yy, pred_a) - balanced_accuracy_score(yy, pred_b))
    if min(map(len, groups)) < 2:
        return delta, [None, None]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        sub = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        da = balanced_accuracy_score(yy[sub], pred_a[sub])
        db = balanced_accuracy_score(yy[sub], pred_b[sub])
        deltas.append(da - db)
    return delta, list(np.quantile(deltas, [0.025, 0.975]))


def main(args):
    rows, prediction_paths = [], {}
    for path in sorted((ROOT / "results").glob("*/metrics.json")):
        j = json.loads(path.read_text())
        metric = j["test"]
        name = path.parent.name
        rows.append({"run": name, "model": j.get("model"),
                     "pretrained": j.get("trained_with_pretrained_weights", "not applicable"),
                     "n_subjects": metric["n_subjects"],
                     "accuracy": metric["accuracy"],
                     "balanced_accuracy": metric["balanced_accuracy"],
                     "macro_f1": metric["macro_f1"],
                     "macro_auroc": metric.get("auroc_ovr_macro"),
                     "brier": metric["multiclass_brier"],
                     "ece": metric["ece_10_bins"],
                     "majority_prediction_fraction": metric["majority_rate"],
                     "noise_label_flip_rate": j.get("stability", {}).get(
                         "noise_10pct_rms", {}).get("label_flip_rate"),
                     "one_channel_blank_flip_rate": j.get("stability", {}).get(
                         "one_electrode_zeroed", {}).get("label_flip_rate"),
                     })
        prediction_paths[name] = path.parent / "test_subject_predictions.csv"
    if not rows:
        raise SystemExit("No trained runs detected. Run training scripts first.")
    result = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    out = ROOT / "results" / "comparison_table.csv"
    result.to_csv(out, index=False)
    print("Result table:", out)
    print(result[["run", "n_subjects", "balanced_accuracy", "macro_f1", "ece"]].to_string(index=False))
    if args.reference:
        if args.reference not in prediction_paths:
            raise SystemExit("Reference model missing. Available: " + ", ".join(prediction_paths))
        comparisons = []
        for name, pred_file in prediction_paths.items():
            if name == args.reference:
                continue
            try:
                delta, ci = paired_delta(prediction_paths[args.reference], pred_file)
            except ValueError as exc:
                print("Skip invalid paired comparison to", name, ":", exc)
                continue
            comparisons.append({"reference": args.reference, "competitor": name,
                                "balanced_acc_delta": delta, "paired_95ci_low": ci[0],
                                "paired_95ci_high": ci[1]})
        compare = pd.DataFrame(comparisons)
        compare.to_csv(ROOT / "results" / "paired_delta_comparisons.csv", index=False)
        print("Paired subject-level model-difference table saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="proposed_fusion")
    main(parser.parse_args())
