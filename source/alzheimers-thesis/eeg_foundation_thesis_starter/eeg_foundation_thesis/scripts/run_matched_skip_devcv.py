"""Parameter-matched residual versus gated-blend study."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import TensorWindows, build_windows, load_manifest
from src.model import DualScaleResidualDecoder, load_encoder, sha256_file
from scripts.train import get_metrics, seed_everything, train_epoch

CLASSES = ["AD", "CN"]
WEIGHTS = ROOT / "checkpoints/cbramod_official.pth"
OUTROOT = ROOT / "results/devcv_matched_skip_2026"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MatchedDecoder(DualScaleResidualDecoder):

    def __init__(self, mode):
        if mode not in ("residual", "blend"):
            raise ValueError(mode)

        super().__init__(num_classes=2, mode="fusion")
        self.fusion_rule = mode

    def forward(self, x, high_tokens, *, return_extras=False):

        if high_tokens.shape != (*x.shape[:3], 200):
            raise ValueError("Invalid CBraMod token shape")

        h = self.high(high_tokens)
        low = self.low(x)

        gate = torch.sigmoid(
            self.gate(torch.cat([h, low], dim=-1))
        )

        if self.fusion_rule == "residual":
            values = h + gate * low
        else:
            values = (1.0 - gate) * h + gate * low

        z = self.dropout(self.fused_norm(values))

        b, c, p, d = z.shape
        z = z.reshape(b, c * p, d)

        attention = (
            self.class_attention(z)
            .transpose(1, 2)
            .softmax(-1)
        )

        evidence = torch.einsum(
            "bkn,bnd->bkd", attention, z
        )

        logits = (
            evidence * self.class_weight[None]
        ).sum(-1) + self.class_bias

        if return_extras:
            return logits, {
                "gate_map": gate.mean(-1),
                "class_attention": attention.view(b, 2, c, p)
            }

        return logits


class MatchedModel(nn.Module):

    def __init__(self, rule):
        super().__init__()

        self.encoder = load_encoder(str(WEIGHTS))

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.decoder = MatchedDecoder(rule)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x, *, return_extras=False):

        with torch.no_grad():
            tokens = self.encoder(x)

        return self.decoder(
            x, tokens, return_extras=return_extras
        )


def check_manifest(fold):

    canonical = pd.read_csv(
        ROOT / "manifests/ds004504_ad_cn_preprocessed.csv",
        dtype={"subject_id": str}
    )

    if len(canonical) != 65 or canonical.subject_id.nunique() != 65:
        raise RuntimeError("Canonical manifest changed")

    reserved = set(
        canonical.loc[
            canonical.split == "test", "subject_id"
        ]
    )

    dev = canonical[
        canonical.split.isin(["train", "val"])
    ].copy()

    if len(reserved) != 13 or len(dev) != 52:
        raise RuntimeError("Original patient split changed")

    path = (
        ROOT / f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
    ).resolve()

    df = load_manifest(path)

    if len(df) != 52 or df.subject_id.nunique() != 52:
        raise RuntimeError("Wrong development fold size")

    if (
        set(df.subject_id) != set(dev.subject_id)
        or reserved.intersection(df.subject_id)
    ):
        raise RuntimeError("FINAL TEST LEAKAGE DETECTED")

    columns = ["path", "label", "fs_hz", "unit"]

    pd.testing.assert_frame_equal(
        dev.set_index("subject_id")[columns].sort_index(),
        df.set_index("subject_id")[columns].sort_index()
    )

    for split in ("train", "val", "test"):
        if set(
            df.loc[df.split == split, "label"]
        ) != set(CLASSES):
            raise RuntimeError(f"Missing class in {split}")

    outer_ids = set(
        df.loc[df.split == "test", "subject_id"]
    )

    if not 10 <= len(outer_ids) <= 11:
        raise RuntimeError("Unexpected outer-fold size")

    return path, df, outer_ids


def run_one(fold, rule, device):

    manifest, df, outer_ids = check_manifest(fold)

    dest = OUTROOT / f"fold_{fold}" / rule

    report_file = dest / "outer_dev_metrics.json"
    prediction_file = (
        dest / "outer_dev_subject_predictions.csv"
    )

    manifest_sha = digest(manifest)

    if dest.exists():

        if not (
            report_file.is_file()
            and prediction_file.is_file()
        ):
            raise RuntimeError(
                f"Incomplete run preserved: {dest}"
            )

        old = json.loads(report_file.read_text())

        ids = set(
            pd.read_csv(
                prediction_file,
                dtype={"subject_id": str}
            ).subject_id
        )

        if (
            old.get("manifest_sha256") != manifest_sha
            or old.get("fusion_rule") != rule
            or old.get("original_final_test_evaluated") is not False
            or ids != outer_ids
        ):
            raise RuntimeError(
                f"Existing result verification failed: {dest}"
            )

        print("VERIFIED COMPLETE:", dest, flush=True)
        return

    seed_everything(42)

    options = dict(
        classes=CLASSES,
        seconds=4,
        max_per_subject=24,
        apply_car=False,
        high_hz=40.0,
        artifact_limit_uv=None
    )

    train = build_windows(df, "train", **options)
    val = build_windows(df, "val", **options)

    subject_counts = Counter(
        train.subject_ids.tolist()
    )

    label_by_subject = dict(
        zip(train.subject_ids, train.y)
    )

    class_counts = Counter(
        label_by_subject.values()
    )

    weights = [
        1.0 / (
            subject_counts[sid]
            * class_counts[label_by_subject[sid]]
        )
        for sid in train.subject_ids
    ]

    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42)
    )

    loader = DataLoader(
        TensorWindows(train),
        batch_size=8,
        sampler=sampler,
        num_workers=0
    )

    model = MatchedModel(rule).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.01
    )

    dest.mkdir(parents=True, exist_ok=False)

    best_score = -1.0
    stale = 0
    history = []

    print(
        f"\nFOLD {fold} | RULE {rule} | "
        f"TRAIN WINDOWS {len(train)} | "
        f"VAL WINDOWS {len(val)}",
        flush=True
    )

    for epoch in range(1, 16):

        loss = train_epoch(
            model, loader, optimizer, device
        )

        val_metrics, _ = get_metrics(
            model, val, device, CLASSES, 8
        )

        score = val_metrics["balanced_accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "inner_balanced_accuracy": score,
            "inner_macro_f1": val_metrics["macro_f1"]
        })

        print(
            f"Fold={fold} Rule={rule} "
            f"Epoch={epoch:02d} "
            f"Loss={loss:.4f} "
            f"ValBAcc={score:.4f}",
            flush=True
        )

        if score > best_score + 1e-8:

            best_score = score
            stale = 0

            torch.save({
                "model_state": model.state_dict(),
                "fusion_rule": rule,
                "manifest_sha256": manifest_sha,
                "weights_sha256": sha256_file(WEIGHTS),
                "best_epoch": epoch
            }, dest / "best.pt")

        else:
            stale += 1

            if stale >= 5:
                break

    pd.DataFrame(history).to_csv(
        dest / "history.csv",
        index=False
    )

    saved = torch.load(
        dest / "best.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        saved["model_state"],
        strict=True
    )

    model.to(device).eval()

    # Outer-development patients are loaded only
    # after inner-validation checkpoint selection.
    outer = build_windows(
        df, "test", **options
    )

    metrics, (
        ids, targets, probabilities
    ) = get_metrics(
        model, outer, device, CLASSES, 8
    )

    if (
        set(ids) != outer_ids
        or len(ids) != len(outer_ids)
    ):
        raise RuntimeError(
            "Outer-development subject mismatch"
        )

    labels = np.asarray(CLASSES)

    predictions = pd.DataFrame({
        "subject_id": ids,
        "true_label": labels[targets],
        "predicted_label": labels[
            probabilities.argmax(axis=1)
        ],
        "p_AD": probabilities[:, 0],
        "p_CN": probabilities[:, 1]
    })

    predictions.to_csv(
        prediction_file,
        index=False
    )

    report = {
        "fold": fold,
        "fusion_rule": rule,
        "manifest_sha256": manifest_sha,
        "weights_sha256": saved["weights_sha256"],
        "best_epoch": saved["best_epoch"],
        "inner_best_balanced_accuracy": best_score,
        "outer_development_metrics": metrics,
        "n_outer_subjects": len(ids),
        "original_final_test_evaluated": False,
        "seed": 42,
        "comparison": "Parameter-matched residual versus gated blend"
    }

    report_file.write_text(
        json.dumps(report, indent=2)
    )

    print(
        f"OUTER FOLD {fold} {rule}: "
        f"BAcc={metrics['balanced_accuracy']:.4f} "
        f"MacroF1={metrics['macro_f1']:.4f}",
        flush=True
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds", default="0,1,2,3,4"
    )

    parser.add_argument(
        "--rules", default="residual,blend"
    )

    args = parser.parse_args()

    folds = [
        int(x) for x in args.folds.split(",")
    ]

    rules = [
        x.strip() for x in args.rules.split(",")
    ]

    if (
        not folds
        or len(folds) != len(set(folds))
        or not set(folds).issubset(range(5))
    ):
        raise ValueError("Invalid fold selection")

    if (
        not rules
        or len(rules) != len(set(rules))
        or not set(rules).issubset({"residual", "blend"})
    ):
        raise ValueError("Invalid fusion rule")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Original final-test patients excluded.")

    for fold in folds:
        for rule in rules:
            run_one(fold, rule, device)

    print("\nMATCHED-SKIP EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    main()
