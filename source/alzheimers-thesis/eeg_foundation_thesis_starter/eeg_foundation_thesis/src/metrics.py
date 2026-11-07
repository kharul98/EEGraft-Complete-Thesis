"""Subject-level metrics. Windows from one patient count as ONE evaluated case."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score)


def aggregate_subjects(y_true, probabilities, subject_ids):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    ids = np.asarray(subject_ids, dtype=str)
    if probs.ndim != 2 or len(probs) != len(ids) or len(y_true) != len(ids):
        raise ValueError("Invalid array lengths/shapes for subject aggregation")
    uniq = np.unique(ids)
    sub_y, sub_p = [], []
    for subject in uniq:
        ix = np.where(ids == subject)[0]
        if len(np.unique(y_true[ix])) != 1:
            raise ValueError(f"Conflicting labels for subject {subject}")
        sub_y.append(int(y_true[ix[0]]))
        sub_p.append(probs[ix].mean(axis=0))  # soft voting; not per-window accuracy
    return uniq, np.asarray(sub_y), np.stack(sub_p)


def expected_calibration_error(y_true, prob, n_bins=10):
    conf = prob.max(axis=1)
    ok = (prob.argmax(axis=1) == y_true)
    result = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        belongs = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if belongs.any():
            result += belongs.mean() * abs(float(conf[belongs].mean()) - float(ok[belongs].mean()))
    return float(result)


def summarize_subject_metrics(y_true, probs, classes):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)
    K = len(classes)
    result = {
        "n_subjects": int(len(y_true)),
        "n_classes": K,
        "classes": list(classes),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(K)), average="macro", zero_division=0)),
        "majority_rate": float(np.bincount(pred, minlength=K).max() / max(1, len(pred))),
        "ece_10_bins": expected_calibration_error(y_true, probs),
        "multiclass_brier": float(np.mean(np.sum((probs - np.eye(K)[y_true]) ** 2, axis=1))),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(K))).tolist(),
        "classification_report": classification_report(y_true, pred, labels=list(range(K)),
                                                        target_names=classes, output_dict=True,
                                                        zero_division=0),
    }
    if len(np.unique(y_true)) == K:
        try:
            result["auroc_ovr_macro"] = float(roc_auc_score(
                y_true, probs[:, 1] if K == 2 else probs,
                multi_class="ovr", average="macro"))
        except ValueError:
            result["auroc_ovr_macro"] = None
    else:
        result["auroc_ovr_macro"] = None
    return result


def stratified_bootstrap_ci(y_true, probs, classes, field="balanced_accuracy", n=1000, seed=42):
    """Nonparametric CI on subjects, resampling within each class."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == i) for i in range(len(classes))]
    if min(map(len, groups)) < 2:
        return None
    vals = []
    for _ in range(n):
        take = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        vals.append(summarize_subject_metrics(y_true[take], probs[take], classes)[field])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
