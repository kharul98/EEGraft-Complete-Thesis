"""Standard signal-feature baselines (SVM, Random Forest, optional XGBoost).

Apply EXACT subject partitions and preprocessing as neural models.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import assign_splits, build_windows, load_manifest
from src.metrics import aggregate_subjects, summarize_subject_metrics, stratified_bootstrap_ci


def features(x):
    """Hand-engineered EEG band power, log absolute power, Hjorth time features."""
    # windows [N,C,P,200] -> [N,C,P*200], all at 200Hz
    z = x.reshape(x.shape[0], x.shape[1], -1).astype(np.float64)
    fs = 200
    freq, psd = welch(z, fs=fs, nperseg=min(256, z.shape[-1]), axis=-1)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    total = trapezoid(psd[(..., (freq >= 0.5) & (freq < 40))],
                         x=freq[(freq >= 0.5) & (freq < 40)], axis=-1) + 1e-12
    extracted = []
    for lo, hi in bands:
        ix = (freq >= lo) & (freq < hi)
        power = trapezoid(psd[..., ix], x=freq[ix], axis=-1)
        extracted.extend([np.log10(power + 1e-12), power / total])
    dx = np.diff(z, axis=-1)
    ddx = np.diff(dx, axis=-1)
    a = np.var(z, axis=-1) + 1e-12
    b = np.var(dx, axis=-1) + 1e-12
    c = np.var(ddx, axis=-1) + 1e-12
    mob = np.sqrt(b / a)
    complexity = np.sqrt(c / b) / mob
    extracted.extend([np.log10(a), mob, complexity])
    return np.stack(extracted, axis=-1).reshape(len(z), -1).astype(np.float32)


def run(args):
    df = assign_splits(load_manifest(args.manifest), seed=args.seed)
    classes = sorted(df.label.unique().tolist())
    stores = {s: build_windows(df, s, classes=classes, seconds=args.seconds,
                                high_hz=args.high_hz, apply_car=args.car,
                                max_per_subject=args.max_windows_per_subject)
              for s in ("train", "val", "test")}
    x = {k: features(v.x) for k, v in stores.items()}
    if args.model == "svm":
        model = Pipeline([("scale", StandardScaler()),
                          ("estimator", SVC(C=1., kernel="rbf", probability=True,
                                            class_weight="balanced", random_state=args.seed))])
    elif args.model == "random_forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                       class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1)
    elif args.model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("pip install xgboost required for --model xgboost")
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                              objective="binary:logistic" if len(classes) == 2 else "multi:softprob",
                              num_class=None if len(classes) == 2 else len(classes),
                              random_state=args.seed, n_jobs=2)
    else:
        raise ValueError(args.model)
    sub_counts = pd.Series(stores["train"].subject_ids).value_counts()
    sw = np.asarray([1. / sub_counts[k] for k in stores["train"].subject_ids])
    # Equal subject influence. For SVM, class_weight='balanced' also balances classes.
    if args.model == "svm":
        model.fit(x["train"], stores["train"].y, estimator__sample_weight=sw)
    else:
        model.fit(x["train"], stores["train"].y, sample_weight=sw)
    report = {"model": args.model, "classes": classes, "seed": args.seed, "subjects": {}}
    pred_df = None
    for s in ("val", "test"):
        prob = model.predict_proba(x[s])
        sid, sy, sp = aggregate_subjects(stores[s].y, prob, stores[s].subject_ids)
        report[s] = summarize_subject_metrics(sy, sp, classes)
        report["subjects"][s] = len(sid)
        if s == "test":
            report[s]["balanced_accuracy_95ci_subject_bootstrap"] = stratified_bootstrap_ci(
                sy, sp, classes, seed=args.seed, n=300)
            pred_df = pd.DataFrame({"subject_id": sid,
                                    "true_label": [classes[i] for i in sy],
                                    "predicted_label": [classes[i] for i in sp.argmax(1)]})
            for j, name in enumerate(classes):
                pred_df[f"p_{name}"] = sp[:, j]
    out = ROOT / "results" / f"classical_{args.model}_{args.seed}"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out}. Re-run with --overwrite to replace")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "subject_splits.csv", index=False)
    pred_df.to_csv(out / "test_subject_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print("Saved", out / "metrics.json")
    print("Subject-level TEST metrics:", report["test"]["balanced_accuracy"], report["test"]["macro_f1"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default="svm", choices=["svm", "random_forest", "xgboost"])
    p.add_argument("--seconds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--high-hz", type=float, default=40)
    p.add_argument("--max-windows-per-subject", type=int, default=100)
    p.add_argument("--car", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())
