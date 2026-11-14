"""Build denoisers from configs and load trained checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from glassdiff.models.base import Denoiser


def build_model(model_cfg: dict[str, Any]) -> Denoiser:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    cfg.pop("rotation_augmentation", None)  # a training option, not a model argument
    if name == "egnn_pbc":
        from glassdiff.models.egnn_pbc import EGNNPBC

        return EGNNPBC(**cfg)
    if name == "mpnn":
        from glassdiff.models.mpnn import MPNN

        return MPNN(**cfg)
    if name == "flat_mlp":
        from glassdiff.models.mlp import FlatMLP

        return FlatMLP(**cfg)
    raise ValueError(f"unknown model {name!r}")


def load_denoiser(path: str | Path, device: str | torch.device = "cpu") -> Denoiser:
    """Load a checkpoint written by scripts/train.py (uses the EMA weights)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["ema"])
    return model.to(device).eval()


def training_sigma_max(path: str | Path) -> float | None:
    """Largest noise level a checkpoint was trained on (None for checkpoints without it)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sigma_max = ckpt.get("cfg", {}).get("noise", {}).get("sigma_max")
    return None if sigma_max is None else float(sigma_max)
