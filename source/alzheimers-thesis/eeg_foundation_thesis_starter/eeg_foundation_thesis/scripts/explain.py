"""Generate channel/patch sensitivity from a REAL trained checkpoint (not demo output).

Attention is only a visualization; corroborate with occlusion and grad*x scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CHANNELS, build_windows, load_manifest
from src.model import EEGNetLike, FoundationClassifier


def main(args):
    run_dir = Path(args.run_dir).resolve()
    saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    df = load_manifest(run_dir / "subject_splits.csv")
    label_names = saved["classes"]
    test = build_windows(df, "test", classes=label_names, seconds=int(saved["seconds"]),
                         apply_car=bool(saved["car"]), high_hz=float(saved["high_hz"]))
    available = np.unique(test.subject_ids)
    subject = args.subject or available[0]
    subset = np.where(test.subject_ids == subject)[0][:args.max_windows]
    if not len(subset):
        raise ValueError(f"Subject not in held-out test split: {subject}")
    weights_path = args.weights or str(ROOT / "checkpoints" / "cbramod_official.pth")
    if saved["model_name"] == "eegnet_like":
        model = EEGNetLike(len(CHANNELS), len(label_names))
    else:
        model = FoundationClassifier(
            len(label_names), None if saved["train_mode"] == "scratch" else weights_path,
            decoder=saved["model_name"], freeze_encoder=saved["train_mode"] == "frozen",
            unfreeze_last_n=2 if saved["train_mode"] == "last2" else 0)
    model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    xx = torch.from_numpy(test.x[subset]).to(device)
    with torch.no_grad():
        logits, extra = model(xx, return_extras=True)
        avg_prob = logits.softmax(dim=-1).mean(dim=0)
        predicted_cls = int(avg_prob.argmax().item())
        target = predicted_cls if args.target_class is None else label_names.index(args.target_class)
        clean_probs = logits.softmax(-1)[:, target]
    grad_importances = []
    # Gradient*x on each window; averaged AFTER per-window channel reduction.
    for x in xx:
        x = x.unsqueeze(0).detach().clone().requires_grad_(True)
        score = model(x)[0, target]
        g = torch.autograd.grad(score, x)[0]
        grad_importances.append((x * g).abs().mean(dim=(-1, -2)).detach().cpu().numpy()[0])
    gradient = np.mean(np.stack(grad_importances), axis=0)
    # One-at-a-time per-channel occlusion, using the same channel's mean value (NOT a causal intervention).
    occlusion = []
    with torch.no_grad():
        for channel in range(len(CHANNELS)):
            modified = xx.clone()
            modified[:, channel] = modified[:, channel].mean(dim=(1, 2), keepdim=True)
            changed = model(modified).softmax(-1)[:, target]
            occlusion.append(float((clean_probs - changed).mean().cpu()))
    attention_map = None
    if "class_attention" in extra:
        attention_map = extra["class_attention"][:, target].mean(dim=0).detach().cpu().numpy()
    table = pd.DataFrame({"channel": CHANNELS, "gradient_x_input": gradient,
                          "occlusion_delta_probability": occlusion})
    table["gradient_x_input_normalized"] = gradient / (gradient.sum() + 1e-12)
    # Plot used for exploratory insight, not for clinical localization claims.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(np.arange(len(CHANNELS)) - 0.2, table.gradient_x_input_normalized, width=0.4,
           label="Normalized |gradient x signal|")
    # Signed delta retained in the CSV, visualization may show negative (counter-evidence) sensitivity.
    ax.bar(np.arange(len(CHANNELS)) + 0.2, table.occlusion_delta_probability, width=0.4,
           label="Occlusion delta P(target)")
    ax.axhline(0, linewidth=0.7)
    ax.set_xticks(np.arange(len(CHANNELS)), CHANNELS, rotation=45)
    ax.set_ylabel("Attribution (different scales)")
    ax.set_title(f"Held-out EEG channel sensitivity: target={label_names[target]}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    prefix = f"explanation_{subject.replace('/', '_').replace(' ', '_')}"
    out_path = run_dir / f"{prefix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    table.to_csv(run_dir / f"{prefix}.csv", index=False)
    if "gate_map" in extra:
        # Descriptive strength of our *learned* residual route. High gate does
        # not by itself establish diagnostic relevance or a causal mechanism.
        gate_map = extra["gate_map"].mean(dim=0).detach().cpu().numpy()
        table["mean_residual_gate"] = gate_map.mean(axis=1)
        table.to_csv(run_dir / f"{prefix}.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(gate_map, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(CHANNELS)), CHANNELS)
        ax.set_xlabel("One-second patches")
        ax.set_title("Gated residual skip strength (descriptive)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"{prefix}_gate.png", dpi=180)
        plt.close(fig)
    if attention_map is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(attention_map, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(CHANNELS)), CHANNELS)
        ax.set_xlabel("One-second patches")
        ax.set_title(f"Decoder attention (NOT causal): class {label_names[target]}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"{prefix}_attention.png", dpi=180)
        plt.close(fig)
    (run_dir / f"{prefix}.json").write_text(json.dumps({
        "subject_id": subject, "n_windows": len(subset), "target_class": label_names[target],
        "prediction": label_names[predicted_cls], "mean_target_probability": float(avg_prob[target].cpu()),
        "methodology_caveat": "Model sensitivity, not clinically validated channel localization."
    }, indent=2))
    print("Wrote explanation files for held-out subject to", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--max-windows", type=int, default=3)
    ap.add_argument("--target-class", default=None)
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
