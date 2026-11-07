#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

REPORTS = Path("reports")
N_BOOT = 2000
SEED = 2026


def csv_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction report: {path}"
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def index_rows(path):
    data = csv_rows(path)
    indexed = {
        r["subject"]: r for r in data
    }

    if len(indexed) != len(data):
        raise RuntimeError(
            f"Duplicate subjects in {path}"
        )

    return indexed


def values(
    indexed, ids, pred_key,
    prob_key, y, folds, label
):
    if set(indexed) != set(ids):
        raise RuntimeError(
            f"{label}: subject list mismatch"
        )

    for j, sid in enumerate(ids):
        row = indexed[sid]

        expected = "A" if y[j] else "C"

        if row.get("group") != expected:
            raise RuntimeError(
                f"{label}: diagnosis mismatch for {sid}"
            )

        if int(row["fold"]) != folds[j]:
            raise RuntimeError(
                f"{label}: fold mismatch for {sid}"
            )

        if row.get("true_label") not in (None, ""):
            if int(float(row["true_label"])) != y[j]:
                raise RuntimeError(
                    f"{label}: label mismatch for {sid}"
                )

    pred = np.array([
        int(float(indexed[s][pred_key]))
        for s in ids
    ])

    prob = np.array([
        float(indexed[s][prob_key])
        for s in ids
    ])

    if (
        not np.isin(pred, [0, 1]).all()
        or not np.isfinite(prob).all()
        or ((prob < 0) | (prob > 1)).any()
    ):
        raise RuntimeError(
            f"{label}: invalid predictions"
        )

    return pred, prob


def metric(y, pred, prob):
    positive = y == 1
    negative = y == 0

    sensitivity = (
        pred[positive] == 1
    ).mean()

    specificity = (
        pred[negative] == 0
    ).mean()

    return {
        "accuracy": float(
            np.mean(pred == y)
        ),
        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),
        "roc_auc": float(
            roc_auc_score(y, prob)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)
    }


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    baseline = index_rows(
        REPORTS / "baseline_predictions_v2.csv"
    )

    ids = sorted(baseline)

    y = np.array([
        int(baseline[s]["group"] == "A")
        for s in ids
    ])

    folds = np.array([
        int(baseline[s]["fold"])
        for s in ids
    ])

    if (
        len(ids) != 46
        or Counter(y) != {0: 23, 1: 23}
    ):
        raise RuntimeError(
            "Expected 23 AD and 23 Healthy subjects"
        )

    data = {}

    classical = [
        ("Logistic Regression", "LogisticRegression"),
        ("SVM", "SVM"),
        ("Random Forest", "RandomForest"),
        ("Nested Selection", "NestedSelection")
    ]

    for name, prefix in classical:
        data[name] = values(
            baseline,
            ids,
            prefix + "_prediction",
            prefix + "_ad_probability",
            y,
            folds,
            name
        )

    advanced = [
        (
            "Basic CNN",
            "eeg_cnn_cv_predictions.csv"
        ),
        (
            "EEGNet v1",
            "eegnet_cv_predictions.csv"
        ),
        (
            "EEGNet v2",
            "eegnet_v2_cv_predictions.csv"
        ),
        (
            "EEG-Conformer",
            "eeg_conformer_cv_predictions.csv"
        ),
        (
            "CBraMod frozen",
            "cbramod_frozen_cv_predictions.csv"
        )
    ]

    for name, filename in advanced:
        indexed = index_rows(
            REPORTS / filename
        )

        data[name] = values(
            indexed,
            ids,
            "predicted_label",
            "ad_probability",
            y,
            folds,
            name
        )

    print(
        "===== MODEL COMPARISON INTEGRITY ====="
    )

    print("Subjects:", len(ids))
    print("AD:", int(y.sum()))
    print("Healthy:", int(len(y) - y.sum()))

    print(
        "Identical subjects, labels and "
        "outer folds: PASS"
    )

    svm_pred, svm_prob = data["SVM"]

    svm_threshold_difference = int(
        (
            (svm_prob >= 0.5).astype(int)
            != svm_pred
        ).sum()
    )

    print(
        "SVM native predictions differing "
        "from probability threshold:",
        svm_threshold_difference
    )

    print(
        "Accuracy uses saved predicted labels; "
        "ROC-AUC uses probabilities."
    )

    # =====================================
    # STRATIFIED SUBJECT-LEVEL BOOTSTRAP
    # =====================================

    rng = np.random.default_rng(SEED)

    healthy = np.flatnonzero(y == 0)
    alz = np.flatnonzero(y == 1)

    samples = []

    for _ in range(N_BOOT):
        sample = np.concatenate([
            rng.choice(
                healthy,
                len(healthy),
                replace=True
            ),
            rng.choice(
                alz,
                len(alz),
                replace=True
            )
        ])
        samples.append(sample)

    print(
        "\n===== MODEL METRICS WITH 95% CIs ====="
    )

    summaries = []
    bootstrap = {}

    keys = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc"
    ]

    for name, (pred, prob) in data.items():
        observed = metric(y, pred, prob)

        boot_values = []

        for idx in samples:
            result = metric(
                y[idx],
                pred[idx],
                prob[idx]
            )

            boot_values.append([
                result[k] for k in keys
            ])

        boot_values = np.asarray(
            boot_values
        )

        bootstrap[name] = boot_values

        bounds = np.percentile(
            boot_values,
            [2.5, 97.5],
            axis=0
        )

        row = {
            "model": name,
            "n_subjects": len(y)
        }

        row.update({
            k: round(v, 5)
            for k, v in observed.items()
        })

        for j, key in enumerate(keys):
            row[key + "_ci_low"] = round(
                float(bounds[0, j]), 5
            )
            row[key + "_ci_high"] = round(
                float(bounds[1, j]), 5
            )

        summaries.append(row)

        print(
            f"\n{name}"
            f"\n  Accuracy: "
            f"{observed['accuracy']:.3f}"
            f"\n  Balanced accuracy: "
            f"{observed['balanced_accuracy']:.3f}"
            f" [{bounds[0, 1]:.3f}, "
            f"{bounds[1, 1]:.3f}]"
            f"\n  ROC-AUC: "
            f"{observed['roc_auc']:.3f}"
            f" [{bounds[0, 2]:.3f}, "
            f"{bounds[1, 2]:.3f}]"
        )

    write_csv(
        REPORTS / "model_metrics_with_ci.csv",
        summaries
    )

    # =====================================
    # PAIRED COMPARISONS AGAINST SVM
    # =====================================

    print(
        "\n===== PAIRED COMPARISONS VERSUS SVM ====="
    )

    reference_correct = (
        data["SVM"][0] == y
    )

    paired = []

    for name, (pred, prob) in data.items():
        if name == "SVM":
            continue

        candidate_correct = pred == y

        svm_only = int(
            (
                reference_correct
                & ~candidate_correct
            ).sum()
        )

        other_only = int(
            (
                ~reference_correct
                & candidate_correct
            ).sum()
        )

        discordant = svm_only + other_only

        if discordant:
            p_value = float(
                binomtest(
                    min(svm_only, other_only),
                    n=discordant,
                    p=0.5
                ).pvalue
            )
        else:
            p_value = 1.0

        # Paired bootstrap:
        # identical resampled subjects for both.
        difference = (
            bootstrap[name] - bootstrap["SVM"]
        )

        bounds = np.percentile(
            difference,
            [2.5, 97.5],
            axis=0
        )

        accuracy_difference = (
            np.mean(pred == y)
            - np.mean(reference_correct)
        )

        auc_difference = (
            roc_auc_score(y, prob)
            - roc_auc_score(y, data["SVM"][1])
        )

        row = {
            "model": name,
            "accuracy_difference_vs_svm":
                round(float(accuracy_difference), 5),
            "accuracy_diff_ci_low":
                round(float(bounds[0, 0]), 5),
            "accuracy_diff_ci_high":
                round(float(bounds[1, 0]), 5),
            "auc_difference_vs_svm":
                round(float(auc_difference), 5),
            "auc_diff_ci_low":
                round(float(bounds[0, 2]), 5),
            "auc_diff_ci_high":
                round(float(bounds[1, 2]), 5),
            "svm_only_correct": svm_only,
            "other_only_correct": other_only,
            "mcnemar_exact_p_uncorrected":
                round(p_value, 5)
        }

        paired.append(row)

        print(
            f"\n{name}"
            f"\n  Accuracy difference: "
            f"{accuracy_difference:+.3f}"
            f"\n  95% paired CI: "
            f"[{bounds[0, 0]:+.3f}, "
            f"{bounds[1, 0]:+.3f}]"
            f"\n  SVM-only correct: {svm_only}"
            f"\n  Other-only correct: {other_only}"
            f"\n  Exact McNemar p: {p_value:.4f}"
        )

    write_csv(
        REPORTS / "model_vs_svm_paired.csv",
        paired
    )

    print("\n===== COMPARISON COMPLETE =====")

    print(
        "Summary: reports/model_metrics_with_ci.csv"
    )

    print(
        "Paired tests: reports/model_vs_svm_paired.csv"
    )

    print("No models retrained.")
    print("Original EEG data unchanged.")

    print(
        "CIs are conditional on the existing "
        "46 out-of-fold predictions."
    )

    print(
        "Prior tuning, subject-selection bias, "
        "and multiple comparisons remain limitations."
    )


if __name__ == "__main__":
    main()

