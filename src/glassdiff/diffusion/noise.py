"""Training-time noise and loss (DM2 formulation, docs/PROJECT_PLAN.md §3.1).  [Ticket M-3]

sigma ~ U(sigma_min, sigma_max) per structure
x_sigma = x_0 + sigma * eps         (pinned atoms get no noise)
loss = per-graph mean over unpinned atoms of |model(x_sigma) - sigma * eps|^2
"""

from __future__ import annotations

import torch
from torch import Tensor

from glassdiff.types import Structures


def sample_sigma(
    batch_size: int,
    sigma_min: float = 1e-3,
    sigma_max: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """[B] noise levels, uniform as in DM2 (try log-uniform as an ablation)."""
    raise NotImplementedError("Ticket M-3")


def add_noise(
    s: Structures,
    sigma: Tensor,
    pin: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Structures, Tensor]:
    """Return (noisy structures, displacement [B, N, 2]); displacement is 0 on pinned atoms."""
    raise NotImplementedError("Ticket M-3")


def masked_displacement_loss(pred: Tensor, target: Tensor, pin: Tensor | None = None) -> Tensor:
    """Scalar loss: mean over graphs of the per-graph MSE over unpinned atoms and both
    coordinates (DM2 uses F.mse_loss, i.e. a mean over coordinates as well)."""
    raise NotImplementedError("Ticket M-3")
