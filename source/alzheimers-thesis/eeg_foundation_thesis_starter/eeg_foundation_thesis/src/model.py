"""Original research decoder + wrappers around upstream CBraMod, not a new encoder.

Input contract: scaled EEG x [batch, channels, one-second patches, 200 samples].
A checkpoint must load strictly before a model is described as pretrained.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "vendor" / "CBraMod"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))
from models.cbramod import CBraMod  # noqa: E402 - upstream original source


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(checkpoint: Optional[str] = None, *, layers: int = 12) -> CBraMod:
    """Only use non-pretrained encoder (no checkpoint) for explicit scratch/control tests."""
    encoder = CBraMod(n_layer=layers)
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint missing: {path}. Run scripts/fetch_weights.py")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load trusted checkpoint {path}: {exc}") from exc
        if not isinstance(state, dict):
            raise TypeError("Checkpoint is not a PyTorch state_dict")
        # Fail closed on bad checkpoints / wrong architecture, no silent strict=False.
        encoder.load_state_dict(state, strict=True)
        print(f"Loaded CBraMod-format state_dict (strict=True), SHA256={sha256_file(path)}")
    encoder.proj_out = nn.Identity()  # return high-level token representations
    return encoder


class RawMultiScaleStem(nn.Module):
    """Shared across electrodes and patches: short/long temporal filters retain local evidence."""
    def __init__(self, hidden: int = 128, stem_channels: int = 16):
        super().__init__()
        self.short = nn.Conv1d(1, stem_channels, 9, padding=4)
        self.long = nn.Conv1d(1, stem_channels, 41, padding=20)
        self.norm = nn.GroupNorm(8, stem_channels * 2)
        self.projection = nn.Sequential(nn.Linear(stem_channels * 2, hidden), nn.LayerNorm(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, p, s = x.shape
        if s != 200:
            raise ValueError("CBraMod supports 200 points per patch for this checkpoint")
        z = x.contiguous().view(b * c * p, 1, s)
        z = torch.cat([self.short(z), self.long(z)], dim=1)
        z = F.gelu(self.norm(z)).mean(-1)
        return self.projection(z).view(b, c, p, -1)


class DualScaleResidualDecoder(nn.Module):
    """DSRF: high-level pretrained tokens + low-level raw signal with gated residual skip.

    Modes support properly controlled removal of low-level path and/or skip connection:
    fusion, no_skip, high_only, low_only. This is a proposed, unvalidated thesis design.
    """
    def __init__(self, num_classes: int, mode: str = "fusion", dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        if mode not in ("fusion", "no_skip", "high_only", "low_only"):
            raise ValueError(f"Bad decoder mode: {mode}")
        self.mode = mode
        self.num_classes = num_classes
        self.low = RawMultiScaleStem(hidden=dim)
        self.high = nn.Sequential(nn.Linear(200, dim), nn.LayerNorm(dim))
        self.gate = nn.Linear(2 * dim, dim)
        self.no_skip_fuser = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.fused_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # Each class learns an evidence distribution over electrode x time patches.
        self.class_attention = nn.Linear(dim, num_classes)
        self.class_weight = nn.Parameter(torch.empty(num_classes, dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.class_weight)

    def forward(self, x: torch.Tensor, high_tokens: torch.Tensor, *,
                return_extras: bool = False):
        if high_tokens.ndim != 4 or high_tokens.shape[:3] != x.shape[:3] or high_tokens.shape[-1] != 200:
            raise ValueError("Expected pretrained encoder tokens [B,C,P,200] matching EEG patches")
        h = self.high(high_tokens)
        low = self.low(x)
        g = torch.sigmoid(self.gate(torch.cat([h, low], dim=-1)))
        if self.mode == "fusion":
            fused = self.fused_norm(h + g * low)  # proposed residual skip
        elif self.mode == "no_skip":
            fused = self.fused_norm(self.no_skip_fuser(torch.cat([h, low], dim=-1)))
        elif self.mode == "high_only":
            fused = self.fused_norm(h)
        else:
            fused = self.fused_norm(low)
        fused = self.dropout(fused)
        b, c, p, d = fused.shape
        tokens = fused.reshape(b, c * p, d)
        attention = self.class_attention(tokens).transpose(1, 2).softmax(dim=-1)  # [B,K,C*P]
        evidence = torch.einsum("bkn,bnd->bkd", attention, tokens)
        logits = (evidence * self.class_weight[None, :, :]).sum(dim=-1) + self.class_bias
        if return_extras:
            extras = {"class_attention": attention.view(b, self.num_classes, c, p)}
            if self.mode == "fusion":
                extras["gate_map"] = g.mean(dim=-1).view(b, c, p)
            return logits, extras
        return logits


class PooledBackboneHead(nn.Module):
    """Frozen / finetuned CBraMod with simple standard linear or MLP decoder."""
    def __init__(self, num_classes: int, head: str = "linear"):
        super().__init__()
        if head == "linear":
            self.fc = nn.Linear(200, num_classes)
        elif head == "mlp":
            self.fc = nn.Sequential(nn.Linear(200, 128), nn.GELU(),
                                    nn.Dropout(0.2), nn.Linear(128, num_classes))
        else:
            raise ValueError(head)

    def forward(self, x: torch.Tensor, high_tokens: torch.Tensor, *, return_extras=False):
        pooled = high_tokens.mean(dim=(1, 2))
        logits = self.fc(pooled)
        if return_extras:
            return logits, {}
        return logits


class FoundationClassifier(nn.Module):
    def __init__(self, num_classes: int, checkpoint: Optional[str],
                 decoder: str = "fusion", freeze_encoder: bool = True,
                 unfreeze_last_n: int = 0, encoder_layers: int = 12):
        super().__init__()
        self.encoder = load_encoder(checkpoint, layers=encoder_layers)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder and unfreeze_last_n:
            raise ValueError("Set freeze_encoder=False to unfreeze final layers")
        for p in self.encoder.parameters():
            p.requires_grad = not freeze_encoder and unfreeze_last_n == 0
        if not freeze_encoder and unfreeze_last_n > 0:
            if unfreeze_last_n > encoder_layers:
                raise ValueError("unfreeze_last_n exceeds encoder layer count")
            for layer in self.encoder.encoder.layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.decoder_name = decoder
        if decoder in ("linear", "mlp"):
            self.decoder = PooledBackboneHead(num_classes, head=decoder)
        else:
            self.decoder = DualScaleResidualDecoder(num_classes, mode=decoder)

    def train(self, mode: bool = True):
        super().train(mode)
        # When frozen, CBraMod dropout must stay disabled even while decoder trains.
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor, *, return_extras=False):
        if x.ndim != 4 or x.shape[-1] != 200:
            raise ValueError("Input shape [B,C,P,200] required")
        if self.freeze_encoder and not x.requires_grad:
            with torch.no_grad():
                high_tokens = self.encoder(x)
        else:
            high_tokens = self.encoder(x)
        return self.decoder(x, high_tokens, return_extras=return_extras)


class EEGNetLike(nn.Module):
    """EEGNet-inspired architecture for a reproducible FROM-SCRATCH neural baseline.

    Not an exact reproduction of the EEGNet paper / Braindecode implementation.
    """
    def __init__(self, n_channels: int, num_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            nn.Conv2d(16, 16, (1, 15), groups=16, padding=(0, 7), bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(16, num_classes)
        )

    def forward(self, x, *, return_extras=False):
        if x.shape[1] != self.n_channels:
            raise ValueError("Mismatched n_channels")
        logits = self.network(x.flatten(2).unsqueeze(1))
        return (logits, {}) if return_extras else logits
