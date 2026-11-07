"""CPU smoke test with SYNTHETIC signals and RANDOM encoder (not thesis metrics)."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CHANNELS, assign_splits, build_windows, load_manifest
from src.model import CBraMod, FoundationClassifier, DualScaleResidualDecoder, load_encoder


def main():
    torch.manual_seed(19)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        # Verify original model's architecture can strict-load a state_dict.
        encoder = CBraMod(n_layer=2)
        ckpt = folder / "random_weights_for_test_only.pth"
        torch.save(encoder.state_dict(), ckpt)
        pretrained_compatibility_test = load_encoder(ckpt, layers=2)
        assert isinstance(pretrained_compatibility_test.proj_out, torch.nn.Identity)
        del encoder, pretrained_compatibility_test

        model = FoundationClassifier(num_classes=3, checkpoint=None,
                                     decoder="fusion", freeze_encoder=True, encoder_layers=2)
        x = torch.randn(2, 19, 2, 200) * 0.1
        logits, extras = model(x, return_extras=True)
        assert tuple(logits.shape) == (2, 3)
        assert extras["class_attention"].shape == (2, 3, 19, 2)
        assert extras["gate_map"].shape == (2, 19, 2)
        assert torch.logical_and(extras["gate_map"] >= 0, extras["gate_map"] <= 1).all()
        torch.testing.assert_close(extras["class_attention"].sum((-1, -2)), torch.ones(2, 3))
        ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
        ce.backward()
        assert model.decoder.class_weight.grad is not None
        assert model.decoder.low.short.weight.grad is not None
        assert all(p.grad is None for p in model.encoder.parameters())
        print("PASS: original upstream backbone output; gated skips; class attention; frozen gradients")

        # Check ablations and both class cardinalities using same encoder output.
        with torch.no_grad():
            high = model.encoder(x)
        for mode in ("fusion", "no_skip", "high_only", "low_only"):
            for k in (2, 3):
                d = DualScaleResidualDecoder(k, mode=mode)
                result = d(x, high)
                assert result.shape == (2, k) and torch.isfinite(result).all()
        print("PASS: no-skip / high-only / low-only ablations; binary and multiclass shapes")

        # Synthetic CSVs only; examples are random, carry no diagnostic signal.
        rng = np.random.default_rng(12)
        rows = []
        for cl in ("healthy", "mild", "severe"):
            for sid in range(5):
                filename = f"{cl}_{sid}.csv"
                # Samples x channels, mismatched CSV column order tested.
                arr = rng.normal(size=(450, len(CHANNELS))) * 10
                pd.DataFrame(arr, columns=CHANNELS[::-1]).to_csv(folder / filename, index=False)
                rows.append({"path": filename, "subject_id": filename.replace(".csv", ""),
                             "label": cl, "fs_hz": 200, "unit": "uV"})
        pd.DataFrame(rows).to_csv(folder / "manifest.csv", index=False)
        df = assign_splits(load_manifest(folder / "manifest.csv"), seed=42)
        assert all(df.groupby("subject_id").split.nunique() == 1)
        assert set(df.split) == {"train", "val", "test"}
        for split in ("train", "val", "test"):
            store = build_windows(df, split, classes=["healthy", "mild", "severe"], seconds=2)
            assert store.x.ndim == 4 and store.x.shape[1:] == (19, 2, 200)
            assert len(np.unique(store.subject_ids)) > 0
        print("PASS: CSV parsing/units/resampling/windowing/subject-level stratified partition")
    print("All local smoke tests passed. NOTE: official weights and real EEG were NOT tested here.")


if __name__ == "__main__":
    main()
