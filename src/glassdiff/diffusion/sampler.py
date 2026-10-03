"""DM2 score-dynamics sampler, batched.  [Ticket S-1]

Reproduces DM2's demo_generating/denoise_generate_*.py:
    start from uniform random positions in the box
    for k in 1..n_noisy:   sigma_k linear from sigma_start to sigma_end
        s ~ U(sigma_min, sigma_k) per coordinate;  eta ~ N(0, s^2)
        x <- x - model(x, sigma_k) - eta
    for k in 1..n_final:   x <- x - model(x, sigma_end)          (no extra noise)
Conditioning methods plug in through a Strategy (diffusion/conditioning.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from glassdiff.diffusion.conditioning import Strategy
from glassdiff.models.base import Denoiser
from glassdiff.types import Request, Structures


@dataclass
class ScoreDynamicsSchedule:
    n_noisy: int = 2900
    n_final: int = 100
    sigma_start: float = 1.0  # DM2 used 1.0 Angstrom for Si-O ~1.6 A; tune in LJ units
    sigma_end: float = 1e-3
    sigma_min: float = 1e-3

    def sigmas(self) -> Tensor:
        """[n_noisy] linearly decreasing noise levels."""
        return torch.linspace(self.sigma_start, self.sigma_end, self.n_noisy)


def random_init(types: Tensor, box: Tensor, generator: torch.Generator | None = None) -> Structures:
    """Uniform random positions in the box for the given species and boxes."""
    raise NotImplementedError("Ticket S-1")


@torch.no_grad()
def sample(
    model: Denoiser,
    init: Structures,
    schedule: ScoreDynamicsSchedule,
    strategy: Strategy | None = None,
    request: Request | None = None,
    generator: torch.Generator | None = None,
    save_every: int = 0,
) -> tuple[Structures, list[Tensor]]:
    """Run the sampler; returns (final structures, trajectory snapshots every save_every steps)."""
    raise NotImplementedError("Ticket S-1")
