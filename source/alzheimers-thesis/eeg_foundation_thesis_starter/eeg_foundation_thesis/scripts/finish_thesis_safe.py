"""Frozen final assessment and thesis-results package. No model training occurs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_windows, load_manifest
from src.metrics import aggregate_subjects, summarize_subject_metrics
from src.model import FoundationClassifier, sha256_file
from scripts.run_classical import features
from scripts.train import get_metrics

P = ROOT / "results/final_protocol_2026.json"
CANONICAL = ROOT / "manifests/ds004504_ad_cn_preprocessed.csv"
DEV = ROOT / "manifests/final_dev52_only.csv"
PRETRAINED = ROOT / "checkpoints/cbramod_official.pth"
FITTED = ROOT / "results/final_development_models_2026"
HOLDOUT = ROOT / "results/locked_final_test_2026_v1"
PACKAGE = ROOT / "results/thesis_execution_package_2026_v1"

WEIGHTS_SHA = (
    "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
)

CLASSES = ["AD", "CN"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def ensure(ok, why):
    if not ok:
        raise RuntimeError(why)


def pct(v):
    return f"{100 * float(v):.2f}%"


def protocol_and_models():

    ensure(ROOT.name == "eeg_foundation_thesis", "Wrong project root")

    p = json.loads(P.read_text())

    ensure(
        p["version"] == "development_locked_2026_v1"
        and p["heldout_status"] == "NOT_EVALUATED",
        "Protocol must remain frozen"
    )

    ensure(
        p["development_subjects"] == 52
        and p["heldout_subjects"] == 13
        and p["classes"] == CLASSES,
        "Unexpected cohort"
    )

    ensure(
        p["primary_architecture"] == "fusion",
        "Unexpected primary model"
    )

    ensure(
        sha(CANONICAL) == p["original_manifest_sha256"],
        "Canonical manifest changed"
    )

    ensure(
        sha(DEV) == p["final_development_manifest_sha256"],
        "Development manifest changed"
    )

    ensure(
        Path(p["final_development_manifest"]).resolve()
        == DEV.resolve(),
        "Wrong development manifest"
    )

    ensure(
        p["pretrained_weights_sha256"] == WEIGHTS_SHA
        and sha256_file(PRETRAINED) == WEIGHTS_SHA,
        "Wrong CBraMod weights"
    )

    ensure(
        p["windows"] == {
            "seconds": 4,
            "max_per_subject": 24,
            "sampling_hz": 200,
            "apply_car": False,
            "high_hz": 40.0,
            "artifact_limit_uv": None
        },
        "Window specification changed"
    )

    ensure(
        p["subject_aggregation"]
        == "mean_window_softmax_probabilities",
        "Unexpected aggregation"
    )

    raw = pd.read_csv(
        CANONICAL,
        dtype={"subject_id": str}
    )

    dev = pd.read_csv(
        DEV,
        dtype={"subject_id": str}
    )

    ensure(
        len(raw) == 65
        and raw.subject_id.nunique() == 65
        and raw.split.value_counts().to_dict() == {
            "train": 39,
            "val": 13,
            "test": 13
        },
        "Original patient split changed"
    )

    ensure(
        len(dev) == 52
        and dev.subject_id.nunique() == 52
        and set(dev.split) == {"train"},
        "Wrong final training population"
    )

    test_rows = raw[raw.split == "test"].copy()
    train_rows = raw[raw.split != "test"].copy()

    ensure(
        test_rows.label.value_counts().to_dict()
        == {"AD": 7, "CN": 6},
        "Wrong held-out labels"
    )

    ensure(
        set(dev.subject_id) == set(train_rows.subject_id)
        and not set(dev.subject_id) & set(test_rows.subject_id),
        "Subject leakage"
    )

    columns = [
        "subject_id", "path", "label", "fs_hz", "unit"
    ]

    pd.testing.assert_frame_equal(
        dev[columns].sort_values(
            "subject_id"
        ).reset_index(drop=True),
        train_rows[columns].sort_values(
            "subject_id"
        ).reset_index(drop=True),
        check_dtype=False
    )

    subject_hash = hashlib.sha256(
        "\n".join(
            sorted(test_rows.subject_id)
        ).encode()
    ).hexdigest()

    ensure(
        subject_hash == p["heldout_subject_ids_sha256"],
        "Held-out subject IDs changed"
    )

    # File existence only; no held-out EEG signal loading.
    for f in test_rows.path:
        ensure(
            (CANONICAL.parent / str(f)).resolve().is_file(),
            "Held-out EEG file missing"
        )

    pf = sha(P)

    for key, expected in [
        ("dsrf", "DSRF"),
        ("xgboost", "XGBoost")
    ]:

        meta = json.loads(
            (FITTED / key / "training_metadata.json").read_text()
        )

        ensure(
            meta["model"] == expected
            and meta["protocol_sha256"] == pf,
            "Model protocol hash mismatch"
        )

        ensure(
            meta["manifest_sha256"] == sha(DEV)
            and meta["n_development_subjects"] == 52,
            "Model training population mismatch"
        )

        ensure(
            meta["final_test_evaluated"] is False,
            "Unexpected test-use flag"
        )

    ckpt = torch.load(
        FITTED / "dsrf/final.pt",
        map_location="cpu",
        weights_only=True
    )

    ensure(
        ckpt["model_name"] == "fusion"
        and ckpt["train_mode"] == "frozen"
        and ckpt["classes"] == CLASSES,
        "Wrong DSRF architecture"
    )

    ensure(
        ckpt["epochs"] == p["dsrf"]["epochs"] == 5
        and ckpt["seed"] == 42,
        "Wrong final DSRF schedule"
    )

    ensure(
        ckpt["checkpoint_sha256"] == WEIGHTS_SHA
        and ckpt["selection_protocol_sha256"] == pf,
        "Wrong DSRF hashes"
    )

    ensure(
        Path(ckpt["manifest"]).resolve() == DEV.resolve(),
        "DSRF training manifest mismatch"
    )

    xpkg = joblib.load(
        FITTED / "xgboost/final.joblib"
    )

    ensure(
        xpkg["classes"] == CLASSES
        and xpkg["protocol_sha256"] == pf
        and xpkg["manifest_sha256"] == sha(DEV),
        "XGBoost provenance mismatch"
    )

    xgb = xpkg["model"]

    ensure(
        np.array_equal(
            np.asarray(xgb.classes_),
            [0, 1]
        ),
        "Wrong XGBoost probability class order"
    )

    ensure(
        int(xgb.max_depth)
        == int(p["xgboost"]["max_depth"]),
        "Wrong XGBoost depth"
    )

    ensure(
        int(xgb.n_estimators)
        == int(p["xgboost"]["n_estimators"])
        and int(xgb.random_state) == 42,
        "Wrong XGBoost settings"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dsrf = FoundationClassifier(
        2,
        str(PRETRAINED),
        decoder="fusion",
        freeze_encoder=True
    )

    dsrf.load_state_dict(
        ckpt["model_state"],
        strict=True
    )

    dsrf.to(device).eval()

    # Synthetic EEG checks compatibility without viewing test EEG.
    with torch.no_grad():
        output = dsrf(
            torch.zeros(
                (1, 19, 4, 200),
                device=device
            )
        )

    ensure(
        output.shape == (1, 2)
        and torch.isfinite(output).all(),
        "Synthetic model check failed"
    )

    n_features = features(
        np.zeros(
            (1, 19, 4, 200),
            dtype=np.float32
        )
    ).shape[1]

    ensure(
        int(xgb.n_features_in_) == n_features,
        "Feature dimensionality changed"
    )

    signatures = {
        "protocol_sha256": pf,
        "original_manifest_sha256": sha(CANONICAL),
        "dsrf_checkpoint_sha256": sha(
            FITTED / "dsrf/final.pt"
        ),
        "xgboost_checkpoint_sha256": sha(
            FITTED / "xgboost/final.joblib"
        )
    }

    return p, test_rows, dsrf, xgb, device, signatures


def stats_interval(n_success, n_total):

    r = binomtest(
        n_success,
        n_total
    ).proportion_ci(
        confidence_level=0.95,
        method="exact"
    )

    return [float(r.low), float(r.high)]


def bootstrap(y, p1, p2):

    a = p1.argmax(1) == y
    b = p2.argmax(1) == y

    rng = np.random.default_rng(20261108)

    aa, bb = [], []

    for klass in (0, 1):

        group = np.flatnonzero(y == klass)

        ix = rng.choice(
            group,
            size=(10000, len(group)),
            replace=True
        )

        aa.append(a[ix].mean(axis=1))
        bb.append(b[ix].mean(axis=1))

    a_boot = (aa[0] + aa[1]) / 2
    b_boot = (bb[0] + bb[1]) / 2

    return {
        "dsrf_bacc_bootstrap_95ci":
            np.quantile(
                a_boot, [0.025, 0.975]
            ).tolist(),

        "xgboost_bacc_bootstrap_95ci":
            np.quantile(
                b_boot, [0.025, 0.975]
            ).tolist(),

        "paired_delta_bacc_bootstrap_95ci":
            np.quantile(
                a_boot - b_boot,
                [0.025, 0.975]
            ).tolist(),

        "bootstrap_replicates": 10000,

        "note": (
            "Exploratory within-cohort stratified "
            "patient bootstrap; 13 held-out patients."
        )
    }


def frame(ids, targets, probs, counts):

    return pd.DataFrame({
        "subject_id": ids,
        "true_label": np.array(CLASSES)[targets],
        "predicted_label": np.array(CLASSES)[
            probs.argmax(axis=1)
        ],
        "p_AD": probs[:, 0],
        "p_CN": probs[:, 1],
        "n_windows": [
            counts[s] for s in ids
        ]
    })


def test_once():

    # Never perform another evaluation if outputs exist.
    if HOLDOUT.exists():

        status_file = HOLDOUT / "status.json"

        if status_file.exists():

            status = json.loads(
                status_file.read_text()
            )

            if status.get("state") == "complete":

                for name, fingerprint in status[
                    "output_sha256"
                ].items():

                    ensure(
                        sha(HOLDOUT / name) == fingerprint,
                        "Saved test output was altered"
                    )

                print(
                    "FINAL TEST ALREADY COMPLETED. "
                    "Not evaluating again."
                )

                display_test(
                    json.loads(
                        (
                            HOLDOUT / "final_metrics.json"
                        ).read_text()
                    )
                )

                return

        raise RuntimeError(
            "Incomplete locked-test folder detected. "
            "Preserve it for investigation; do not delete "
            "or repeat: " + str(HOLDOUT)
        )

    p, expected, dsrf, xgb, device, signatures = (
        protocol_and_models()
    )

    print(
        "ALL PRE-TEST CHECKS PASSED. "
        "Frozen models, 13 locked subjects, no tuning."
    )

    # Start marker prevents unintentional repeat evaluations.
    HOLDOUT.mkdir(
        parents=True,
        exist_ok=False
    )

    (
        HOLDOUT / "status.json"
    ).write_text(
        json.dumps({
            "state": "started",
            "signatures": signatures,
            "if_interrupted": (
                "Do not delete or rerun until "
                "the failure is audited."
            )
        }, indent=2)
    )

    # First intentional access to the held-out EEG signals.
    raw = load_manifest(CANONICAL)

    holdout = raw[
        raw.split == "test"
    ].copy()

    ensure(
        len(holdout) == 13
        and set(holdout.subject_id)
        == set(expected.subject_id),
        "Unexpected held-out patient IDs"
    )

    opts = p["windows"]

    store = build_windows(
        holdout,
        "test",
        classes=CLASSES,
        seconds=opts["seconds"],
        max_per_subject=opts["max_per_subject"],
        apply_car=opts["apply_car"],
        high_hz=opts["high_hz"],
        artifact_limit_uv=opts["artifact_limit_uv"]
    )

    window_ids = store.subject_ids.tolist()

    counts = {
        str(s): window_ids.count(s)
        for s in set(window_ids)
    }

    ensure(
        set(counts) == set(expected.subject_id)
        and all(
            0 < n <= 24
            for n in counts.values()
        ),
        "Missing test EEG windows"
    )

    # Frozen DSRF inference.
    m1, (ids, y, p1) = get_metrics(
        dsrf,
        store,
        device,
        CLASSES,
        8
    )

    # Frozen XGBoost inference on identical windows.
    x = features(store.x)

    ensure(
        x.shape[1] == int(xgb.n_features_in_),
        "XGBoost input features mismatch"
    )

    b_ids, by, p2 = aggregate_subjects(
        store.y,
        xgb.predict_proba(x),
        store.subject_ids
    )

    ensure(
        np.array_equal(ids, b_ids)
        and np.array_equal(y, by),
        "Model patient ordering differs"
    )

    ensure(
        len(ids) == 13
        and set(ids) == set(expected.subject_id),
        "Wrong test patient count"
    )

    ensure(
        int((y == 0).sum()) == 7
        and int((y == 1).sum()) == 6,
        "Wrong test class counts"
    )

    for prob in (p1, p2):

        ensure(
            prob.shape == (13, 2)
            and np.isfinite(prob).all()
            and np.allclose(
                prob.sum(axis=1),
                1,
                atol=2e-4
            ),
            "Bad model probabilities"
        )

    m2 = summarize_subject_metrics(
        y,
        p2,
        CLASSES
    )

    for metrics, prob in [
        (m1, p1),
        (m2, p2)
    ]:

        pred = prob.argmax(axis=1)

        metrics["accuracy_exact_95ci"] = (
            stats_interval(
                int((pred == y).sum()),
                13
            )
        )

        for i, label in enumerate(CLASSES):

            metrics[f"{label}_recall_exact_95ci"] = (
                stats_interval(
                    int(
                        (
                            (y == i)
                            & (pred == i)
                        ).sum()
                    ),
                    int((y == i).sum())
                )
            )

    a = frame(ids, y, p1, counts)
    b = frame(ids, y, p2, counts)

    paired = a[
        [
            "subject_id",
            "true_label",
            "predicted_label",
            "p_AD",
            "p_CN"
        ]
    ].rename(columns={
        "predicted_label": "dsrf_label",
        "p_AD": "dsrf_p_AD",
        "p_CN": "dsrf_p_CN"
    }).merge(
        b[
            [
                "subject_id",
                "predicted_label",
                "p_AD",
                "p_CN"
            ]
        ].rename(columns={
            "predicted_label": "xgboost_label",
            "p_AD": "xgboost_p_AD",
            "p_CN": "xgboost_p_CN"
        }),
        on="subject_id",
        validate="one_to_one"
    )

    right_a = (
        paired.dsrf_label == paired.true_label
    )

    right_b = (
        paired.xgboost_label == paired.true_label
    )

    better = int(
        (right_a & ~right_b).sum()
    )

    worse = int(
        (~right_a & right_b).sum()
    )

    disagreement = better + worse

    pair_stats = {
        "bacc_delta_dsrf_minus_xgboost":
            float(
                m1["balanced_accuracy"]
                - m2["balanced_accuracy"]
            ),

        "dsrf_correct_only": better,

        "xgboost_correct_only": worse,

        "discordant_n": disagreement,

        "mcnemar_exact_two_sided_p_exploratory":
            float(
                binomtest(
                    better,
                    disagreement,
                    0.5
                ).pvalue
            )
            if disagreement else 1.0,

        **bootstrap(y, p1, p2)
    }

    report = {
        "models": {
            "DSRF": m1,
            "XGBoost": m2
        },

        "paired": pair_stats,

        "audit": {
            "heldout_subjects": 13,
            "heldout_AD": 7,
            "heldout_CN": 6,
            "development_subjects": 52,
            "evaluation_type": (
                "one-time within-cohort holdout"
            ),
            "no_test_training": True,
            "no_test_tuning": True,
            "windows_per_subject": counts,
            "signatures": signatures,
            "limitations": [
                "Small held-out sample",
                "No independent external cohort",
                "Not a clinical diagnostic validation"
            ]
        }
    }

    a.to_csv(
        HOLDOUT / "dsrf_subjects.csv",
        index=False
    )

    b.to_csv(
        HOLDOUT / "xgboost_subjects.csv",
        index=False
    )

    paired.to_csv(
        HOLDOUT / "paired_subjects.csv",
        index=False
    )

    (
        HOLDOUT / "final_metrics.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False
        )
    )

    files = [
        "dsrf_subjects.csv",
        "xgboost_subjects.csv",
        "paired_subjects.csv",
        "final_metrics.json"
    ]

    (
        HOLDOUT / "status.json"
    ).write_text(
        json.dumps({
            "state": "complete",
            "test_evaluated_once": True,
            "signatures": signatures,
            "output_sha256": {
                f: sha(HOLDOUT / f)
                for f in files
            }
        }, indent=2)
    )

    display_test(report)


def display_test(report):

    print(
        "\n========== FINAL 13-PATIENT "
        "HELD-OUT RESULTS =========="
    )

    for name in [
        "DSRF",
        "XGBoost"
    ]:

        m = report["models"][name]

        print(
            f"{name}: "
            f"BAcc={pct(m['balanced_accuracy'])}, "
            f"accuracy={pct(m['accuracy'])}, "
            f"macro-F1={pct(m['macro_f1'])}, "
            f"AUROC={m['auroc_ovr_macro']}, "
            f"confusion [AD,CN]={m['confusion_matrix']}"
        )

    q = report["paired"]

    print(
        "DSRF minus XGBoost BAcc:",
        f"{100 * q['bacc_delta_dsrf_minus_xgboost']:+.2f} pp"
    )

    print(
        "Exploratory exact McNemar p:",
        q["mcnemar_exact_two_sided_p_exploratory"]
    )

    print("Archived results:", HOLDOUT)

    print(
        "No model refitting or test-based changes "
        "are permitted."
    )


def check_test_archive():

    status = json.loads(
        (
            HOLDOUT / "status.json"
        ).read_text()
    )

    ensure(
        status["state"] == "complete"
        and status["test_evaluated_once"] is True,
        "Final test not completed"
    )

    for name, fingerprint in status[
        "output_sha256"
    ].items():

        ensure(
            sha(HOLDOUT / name) == fingerprint,
            "Final test archive checksum failure"
        )

    result = json.loads(
        (
            HOLDOUT / "final_metrics.json"
        ).read_text()
    )

    signatures = result["audit"]["signatures"]

    ensure(
        signatures["protocol_sha256"] == sha(P),
        "Locked protocol changed after test"
    )

    ensure(
        signatures["original_manifest_sha256"]
        == sha(CANONICAL),
        "Original manifest changed after test"
    )

    return result


def markdown_table(df):

    cols = df.columns.tolist()

    result = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(
            ["---"] * len(cols)
        ) + " |"
    ]

    for _, row in df.iterrows():

        result.append(
            "| " + " | ".join(
                str(row[c])
                for c in cols
            ) + " |"
        )

    return "\n".join(result)


def development_results(canonical):

    group = {
        "DSRF": ("neural", "fusion"),
        "No skip": ("neural", "no_skip"),
        "High only": ("neural", "high_only"),
        "Low only": ("neural", "low_only"),
        "SVM": ("classical", "svm"),
        "Random Forest": ("classical", "random_forest"),
        "XGBoost": ("classical", "xgboost"),
        "EEGNet-inspired": ("neural", "eegnet_like")
    }

    dev_ids = set(
        canonical.loc[
            canonical.split != "test",
            "subject_id"
        ]
    )

    test_ids = set(
        canonical.loc[
            canonical.split == "test",
            "subject_id"
        ]
    )

    truth = dict(
        zip(
            canonical.subject_id,
            canonical.label
        )
    )

    rows = []

    for name, (kind, key) in group.items():

        parts = []
        scores = []

        for k in range(5):

            if kind == "neural":

                folder = ROOT / (
                    f"results/nestedcv5_f{k}_{key}_s42"
                )

            else:

                folder = ROOT / (
                    f"results/devcv_classical_2026/"
                    f"fold_{k}/{key}"
                )

            meta = json.loads(
                (
                    folder / "outer_dev_metrics.json"
                ).read_text()
            )

            ensure(
                meta.get(
                    "original_final_test_evaluated"
                ) is False,
                "Development results include test?"
            )

            x = pd.read_csv(
                folder /
                "outer_dev_subject_predictions.csv",
                dtype={"subject_id": str}
            )

            fold = pd.read_csv(
                ROOT /
                f"manifests/nested_cv5_dev2026/fold_{k}.csv",
                dtype={"subject_id": str}
            )

            ids = set(
                fold.loc[
                    fold.split == "test",
                    "subject_id"
                ]
            )

            ensure(
                len(x) == len(ids)
                and set(x.subject_id) == ids
                and not (ids & test_ids),
                "Development fold mismatch"
            )

            ensure(
                all(
                    truth[s] == lab
                    for s, lab in zip(
                        x.subject_id,
                        x.true_label
                    )
                ),
                "Development label mismatch"
            )

            scores.append(
                balanced_accuracy_score(
                    x.true_label,
                    x.predicted_label
                )
            )

            parts.append(x)

        pred = pd.concat(
            parts,
            ignore_index=True
        )

        ensure(
            len(pred) == 52
            and pred.subject_id.nunique() == 52
            and set(pred.subject_id) == dev_ids,
            "Development subject coverage failure"
        )

        rows.append({
            "Model": name,

            "Pooled BAcc (%)":
                round(
                    100 * balanced_accuracy_score(
                        pred.true_label,
                        pred.predicted_label
                    ),
                    2
                ),

            "Mean fold BAcc (%)":
                round(
                    100 * np.mean(scores),
                    2
                ),

            "Fold SD (pp)":
                round(
                    100 * np.std(
                        scores,
                        ddof=1
                    ),
                    2
                ),

            "Macro-F1 (%)":
                round(
                    100 * f1_score(
                        pred.true_label,
                        pred.predicted_label,
                        average="macro"
                    ),
                    2
                ),

            "Correct /52":
                f"{int((pred.true_label == pred.predicted_label).sum())}/52"
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "Pooled BAcc (%)",
        ascending=False
    ).reset_index(drop=True)


def report_only():

    if PACKAGE.exists():

        manifest = PACKAGE / "package_manifest.json"

        ensure(
            manifest.is_file(),
            "Incomplete thesis report package preserved"
        )

        old = json.loads(
            manifest.read_text()
        )

        for name, hashvalue in old[
            "files_sha256"
        ].items():

            ensure(
                sha(PACKAGE / name) == hashvalue,
                "Report package checksum mismatch"
            )

        print(
            "Existing report package verified. "
            "No files overwritten:",
            PACKAGE
        )

        return

    test = check_test_archive()

    raw = pd.read_csv(
        CANONICAL,
        dtype={"subject_id": str}
    )

    ensure(
        len(raw) == 65
        and raw.subject_id.nunique() == 65,
        "Cohort mismatch"
    )

    dev_table = development_results(raw)

    robust = pd.read_csv(
        ROOT /
        "results/devcv_robustness_repeated_v1/"
        "summary_repeated.csv"
    )

    ensure(
        len(robust) == 9,
        "Expected nine repeated stress-test rows"
    )

    ablate = []

    for rule in [
        "residual",
        "blend"
    ]:

        all_preds = []

        for fold in range(5):

            f = ROOT / (
                f"results/devcv_matched_skip_2026/"
                f"fold_{fold}/{rule}"
            )

            meta = json.loads(
                (
                    f / "outer_dev_metrics.json"
                ).read_text()
            )

            ensure(
                meta[
                    "original_final_test_evaluated"
                ] is False,
                "Ablation included test?"
            )

            all_preds.append(
                pd.read_csv(
                    f /
                    "outer_dev_subject_predictions.csv",
                    dtype={"subject_id": str}
                )
            )

        preds = pd.concat(
            all_preds,
            ignore_index=True
        )

        ensure(
            len(preds) == 52
            and preds.subject_id.nunique() == 52,
            "Ablation patient count mismatch"
        )

        ablate.append({
            "Rule": rule,

            "Pooled BAcc (%)":
                round(
                    100 * balanced_accuracy_score(
                        preds.true_label,
                        preds.predicted_label
                    ),
                    2
                ),

            "Macro-F1 (%)":
                round(
                    100 * f1_score(
                        preds.true_label,
                        preds.predicted_label,
                        average="macro"
                    ),
                    2
                ),

            "Correct /52":
                f"{int((preds.true_label == preds.predicted_label).sum())}/52"
        })

    ablation = pd.DataFrame(ablate)

    final_rows = []

    for name in [
        "DSRF",
        "XGBoost"
    ]:

        m = test["models"][name]

        cm = m["confusion_matrix"]

        final_rows.append({
            "Model": name,

            "BAcc (%)":
                round(
                    100 * m["balanced_accuracy"], 2
                ),

            "Accuracy (%)":
                round(
                    100 * m["accuracy"], 2
                ),

            "Macro-F1 (%)":
                round(
                    100 * m["macro_f1"], 2
                ),

            "AD recall (%)":
                round(
                    100 * cm[0][0] / sum(cm[0]), 2
                ),

            "CN recall (%)":
                round(
                    100 * cm[1][1] / sum(cm[1]), 2
                ),

            "AUROC":
                round(
                    m["auroc_ovr_macro"], 3
                )
                if m["auroc_ovr_macro"] is not None
                else "N/A",

            "Brier":
                round(
                    m["multiclass_brier"], 4
                ),

            "Correct /13":
                f"{cm[0][0] + cm[1][1]}/13"
        })

    final_table = pd.DataFrame(
        final_rows
    )

    explain_dir = (
        ROOT /
        "results/devcv_dsr_fusion_interpret_full24_2026"
    )

    if (
        explain_dir / "per_subject_channel.csv"
    ).is_file():

        source = "Full-window (up to 24)"

    else:

        explain_dir = (
            ROOT /
            "results/devcv_dsr_fusion_interpret_2026"
        )

        source = "Four-window exploratory"

    explanation = (
        "No interpretability data available."
    )

    if (
        explain_dir / "per_subject_channel.csv"
    ).is_file():

        e = pd.read_csv(
            explain_dir / "per_subject_channel.csv",
            dtype={"subject_id": str}
        )

        ensure(
            len(e) == 52 * 19
            and e.subject_id.nunique() == 52,
            "Explanation patient coverage mismatch"
        )

        selections = []

        for label in [
            "AD",
            "CN"
        ]:

            ranks = (
                e.loc[
                    e.true_label == label
                ]
                .groupby("channel")
                .occlusion_delta_p_true
                .mean()
                .nlargest(5)
            )

            selections.append(
                label + ": " + ", ".join(
                    f"{ch} ({100 * v:.2f} pp)"
                    for ch, v in ranks.items()
                )
            )

        explanation = (
            source
            + " channel-occlusion analysis on 52 "
            + "development patients: "
            + "; ".join(selections)
            + ". Masking sensitivity and gates are "
            + "descriptive, not physiological causal evidence."
        )

        if source.startswith("Full"):

            refs = []

            for k in range(5):

                refs.append(
                    pd.read_csv(
                        ROOT /
                        f"results/nestedcv5_f{k}_fusion_s42/"
                        "outer_dev_subject_predictions.csv",
                        dtype={"subject_id": str}
                    )
                )

            reference = pd.concat(
                refs
            ).set_index(
                "subject_id"
            ).sort_index()

            explained = e.drop_duplicates(
                "subject_id"
            ).set_index(
                "subject_id"
            ).sort_index()

            ensure(
                reference.index.equals(
                    explained.index
                ),
                "Full-window explanation IDs mismatch"
            )

            matches = int(
                (
                    reference.predicted_label
                    == explained.predicted_label
                ).sum()
            )

            explanation += (
                f" Same predictions as full CV: {matches}/52."
            )

    pair = test["paired"]

    ca = pair["dsrf_bacc_bootstrap_95ci"]
    cb = pair["xgboost_bacc_bootstrap_95ci"]
    cd = pair["paired_delta_bacc_bootstrap_95ci"]

    lines = [
        "# Thesis execution and results chapter draft",

        "**Working title:** Reliability and Interpretability "
        "of a Pretrained EEG Foundation Model for "
        "Alzheimer's Disease Classification",

        "## Abstract",

        f"This study evaluated Alzheimer's disease (AD) "
        f"versus healthy controls (CN) using a pretrained "
        f"CBraMod EEG encoder and a custom dual-scale "
        f"residual fusion decoder. The cohort comprised "
        f"65 participants: 52 for development and 13 "
        f"for a locked within-dataset assessment. Five "
        f"patient-disjoint development folds compared eight "
        f"neural and classical model variants. DSRF achieved "
        f"{dev_table.loc[dev_table.Model == 'DSRF', 'Pooled BAcc (%)'].iloc[0]:.2f}% "
        f"pooled development balanced accuracy; XGBoost achieved "
        f"{dev_table.loc[dev_table.Model == 'XGBoost', 'Pooled BAcc (%)'].iloc[0]:.2f}%. "
        f"Fixed final models achieved held-out balanced accuracies "
        f"of {pct(test['models']['DSRF']['balanced_accuracy'])} "
        f"(DSRF) and {pct(test['models']['XGBoost']['balanced_accuracy'])} "
        f"(XGBoost). Synthetic noise and channel-removal tests "
        f"revealed that clean classification and reliability "
        f"do not coincide. Results are exploratory and do "
        f"not establish clinical diagnostic validity.",

        "## Methods and reproducibility",

        "Original OpenNeuro ds004504 EEG records were analyzed "
        "for 36 AD and 29 CN patients. A fixed 39/13/13 "
        "train/validation/test partition was used; 52 "
        "development subjects were further divided into five "
        "stratified outer folds with separate inner validation. "
        "The locked 13-person test (7 AD, 6 CN) was used only "
        "after final DSRF and XGBoost selection and fitting. "
        "Subjects, not windows, defined partitions.",

        "The pipeline used 19 channels, a 0.3-40 Hz bandpass, "
        "200 Hz sampling rate, 4-second EEG windows (up to 24 "
        "per person), and subject-level aggregation of mean "
        "window softmax probabilities. CBraMod pretrained weights "
        "were strictly loaded and the encoder frozen. DSRF fused "
        "high-level pretrained tokens with multiscale convolutional "
        "raw-signal features through a learned gate and residual. "
        "Baselines included SVM, Random Forest and XGBoost on "
        "spectral and Hjorth features, plus an EEGNet-inspired "
        "network trained from scratch.",

        "## Table 1 - Five-fold development model comparison",

        markdown_table(dev_table),

        "DSRF achieved the highest pooled development balanced "
        "accuracy but XGBoost was very close, with both predicting "
        "40 of 52 development patients correctly. All between-model "
        "comparisons were exploratory; the best model was selected "
        "after examining development results.",

        "## Table 2 - Capacity-matched skip-connection study",

        markdown_table(ablation),

        "Both matched variants used identical modules and effective "
        "parameter counts. The comparison contrasted an additive "
        "identity residual against a gated blend. The observed "
        "residual advantage was not statistically conclusive.",

        "## Table 3 - Repeated synthetic perturbation robustness",

        markdown_table(
            robust[
                [
                    "model",
                    "condition",
                    "mean_BAcc_pct",
                    "BAcc_min_pct",
                    "BAcc_max_pct",
                    "mean_flip_pct",
                    "mean_Brier"
                ]
            ]
        ),

        "Robustness included five seeded perturbation realizations "
        "on the SAME 52 development patients. Repeats are not "
        "independent patient cohorts. Noise magnitudes were "
        "relative to within-window signal standard deviation; "
        "channel-removal blanked one electrode per EEG window. "
        "Label-flip rates must be interpreted alongside balanced "
        "accuracy and Brier scores. A constant-label classifier "
        "has zero flips but only 50% binary balanced accuracy.",

        "## Explainability",

        explanation,

        "The explanation procedure identifies sensitivity to "
        "model inputs rather than clinically validated brain "
        "biomarkers. Interpretation is sensitive to the masking "
        "design and the number of explained windows.",

        "## Table 4 - Final one-time held-out results (13 patients)",

        markdown_table(final_table),

        f"DSRF patient-level BAcc exploratory 95% stratified "
        f"bootstrap interval: {pct(ca[0])}-{pct(ca[1])}; "
        f"XGBoost interval: {pct(cb[0])}-{pct(cb[1])}. "
        f"Paired DSRF-minus-XGBoost difference: "
        f"{100 * pair['bacc_delta_dsrf_minus_xgboost']:+.2f} "
        f"percentage points (interval {100 * cd[0]:+.2f} "
        f"to {100 * cd[1]:+.2f} pp). Exploratory exact "
        f"McNemar p={pair['mcnemar_exact_two_sided_p_exploratory']:.5f}, "
        f"with {pair['dsrf_correct_only']} patients correct "
        f"only for DSRF and {pair['xgboost_correct_only']} "
        f"correct only for XGBoost.",

        "## Discussion and limitations",

        "The contribution is a documented patient-level "
        "evaluation of an existing pretrained EEG encoder "
        "with a custom, interpretable decoder and a "
        "reliability-focused stress-test design. Small "
        "held-out sample size, one source cohort, exploratory "
        "model-selection comparisons, possible demographic "
        "or recording confounding, artificial rather than "
        "real-world acquisition corruption, and the absence "
        "of external validation limit generalization. "
        "Calibration measurements on only 13 held-out subjects "
        "are especially uncertain.",

        "## Conclusion",

        "Pretrained EEG representations with dual-scale residual "
        "fusion can yield competitive exploratory "
        "Alzheimer's-versus-control discrimination, but prediction "
        "stability under corruption is distinct from clean "
        "performance. Findings do not predict future "
        "Alzheimer's risk or establish a clinical diagnostic "
        "system. Independent prospective validation would "
        "be required.",

        "## Reproducibility and writing tasks remaining",

        "The five outer development folds, model weights, "
        "final training protocol, final held-out predictions, "
        "paired statistics, robustness records and explainability "
        "outputs have been stored. Add verified citations, "
        "ethics/data-use statements and institution-specific "
        "thesis formatting separately. Do not claim external "
        "validation or clinical applicability."
    ]

    PACKAGE.mkdir(
        parents=True,
        exist_ok=False
    )

    (
        PACKAGE / "thesis_results_ready.md"
    ).write_text(
        "\n\n".join(lines) + "\n"
    )

    dev_table.to_csv(
        PACKAGE / "table_1_eight_models.csv",
        index=False
    )

    ablation.to_csv(
        PACKAGE / "table_2_matched_skip.csv",
        index=False
    )

    robust.to_csv(
        PACKAGE / "table_3_repeated_robustness.csv",
        index=False
    )

    final_table.to_csv(
        PACKAGE / "table_4_locked_holdout.csv",
        index=False
    )

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        dev_table.Model.iloc[::-1],
        dev_table["Pooled BAcc (%)"].iloc[::-1]
    )

    ax.set(
        xlabel="Pooled out-of-fold balanced accuracy (%)",
        xlim=(0, 100),
        title="Development model comparison"
    )

    fig.tight_layout()

    fig.savefig(
        PACKAGE / "figure_1_dev_models.png",
        dpi=160
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.bar(
        ["DSRF", "XGBoost"],
        [
            100 * test["models"][m]["balanced_accuracy"]
            for m in ["DSRF", "XGBoost"]
        ]
    )

    ax.set(
        ylabel="Patient balanced accuracy (%)",
        ylim=(0, 100),
        title="Final within-cohort holdout (n=13)"
    )

    fig.tight_layout()

    fig.savefig(
        PACKAGE / "figure_2_final_holdout.png",
        dpi=160
    )

    plt.close(fig)

    package_files = [
        f for f in PACKAGE.iterdir()
        if f.is_file()
    ]

    manifest = {
        "test_metrics_sha256":
            sha(HOLDOUT / "final_metrics.json"),

        "protocol_sha256": sha(P),

        "no_additional_test_inference": True,

        "files_sha256": {
            f.name: sha(f)
            for f in package_files
        }
    }

    (
        PACKAGE / "package_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2
        )
    )

    print(
        "\n========== THESIS REPORT PACKAGE COMPLETE =========="
    )

    print(
        "Report:",
        PACKAGE / "thesis_results_ready.md"
    )

    print(
        "Four CSV tables, two figures "
        "and package_manifest.json saved."
    )

    print(
        "\nFINAL HELD-OUT MODEL COMPARISON:"
    )

    print(
        final_table.to_string(index=False)
    )

    print(
        "No additional test inference "
        "or model training was performed."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--preflight",
        action="store_true"
    )

    group.add_argument(
        "--test-once",
        action="store_true"
    )

    group.add_argument(
        "--report-only",
        action="store_true"
    )

    mode = parser.parse_args()

    if mode.preflight:

        ensure(
            not HOLDOUT.exists(),
            "A held-out results directory already exists; "
            "do not preflight for a repeat"
        )

        p, t, model, xgb, device, hashes = (
            protocol_and_models()
        )

        print(
            "PRE-TEST CHECK PASSED. Device:",
            device
        )

        print(
            "DSRF final epochs:",
            p["dsrf"]["epochs"],
            "| XGBoost depth:",
            p["xgboost"]["max_depth"]
        )

        print(
            "52 development patients | "
            "13 reserved patients (7 AD, 6 CN)"
        )

        print(
            "No test signals were loaded or evaluated."
        )

    elif mode.test_once:

        test_once()

    else:

        report_only()
