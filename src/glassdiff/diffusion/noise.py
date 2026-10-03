"""Training-time noise and loss (DM2 formulation, docs/PROJECT_PLAN.md §3.1).  [Ticket M-3]

sigma ~ U(sigma_min, sigma_max) per structure (or log-uniform, as an ablation)
x_sigma = x_0 + sigma * eps         (pinned atoms get no noise)
loss = mean over graphs of the per-graph MSE over unpinned atoms and both coordinates,
       between model(x_sigma) and the added displacement sigma * eps
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from glassdiff.types import Structures


def sample_sigma(
    batch_size: int,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.5,
    generator: torch.Generator | None = None,
    distribution: str = "uniform",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """[B] noise levels: uniform as in DM2, or log-uniform."""
    u = torch.rand(batch_size, generator=generator, dtype=dtype)
    if distribution == "uniform":
        return sigma_min + (sigma_max - sigma_min) * u
    if distribution == "log_uniform":
        return torch.exp(math.log(sigma_min) + (math.log(sigma_max) - math.log(sigma_min)) * u)
    raise ValueError(f"unknown sigma distribution {distribution!r}")


def add_noise(
    s: Structures,
    sigma: Tensor,
    pin: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Structures, Tensor]:
    """Return (noisy structures, displacement [B, N, 2]); displacement is 0 on pinned atoms."""
    eps = torch.randn(s.pos.shape, generator=generator, dtype=s.pos.dtype).to(s.pos.device)
    disp = sigma.to(s.pos)[:, None, None] * eps
    if pin is not None:
        disp = disp * (~pin)[..., None]
    return s.with_pos(s.pos + disp), disp


def masked_displacement_loss(pred: Tensor, target: Tensor, pin: Tensor | None = None) -> Tensor:
    """Scalar loss: mean over graphs of the per-graph MSE over unpinned atoms and both
    coordinates (DM2 uses F.mse_loss, i.e. a mean over coordinates as well)."""
    sq = ((pred - target) ** 2).sum(-1)  # [B, N]
    weight = torch.ones_like(sq) if pin is None else (~pin).to(sq.dtype)
    per_graph = (sq * weight).sum(-1) / (2 * weight.sum(-1).clamp_min(1))
    return per_graph.mean()
