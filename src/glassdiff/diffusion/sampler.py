"""DM2 score-dynamics sampler, batched.  [Ticket S-1]

Reproduces DM2's demo_generating/denoise_generate_*.py:
    start from uniform random positions in the box
    for k in 1..n_noisy:   sigma_k linear from sigma_start to sigma_end
        s ~ U(sigma_min, sigma_k) per coordinate;  eta ~ N(0, s^2)
        x <- x - model(x, sigma_k) - eta
    for k in 1..n_final:   x <- x - model(x, sigma_end)          (no extra noise)
Conditioning methods plug in through a Strategy (diffusion/conditioning.py).

A sigma-aware model only saw sigma <= its training sigma_max, so the sigma passed to the
model is clipped at ``model_sigma_max`` (the DM2 model ignores it).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from glassdiff.diffusion.conditioning import RePaint, Strategy, Unconditional
from glassdiff.models.base import Denoiser
from glassdiff.types import Request, Structures


@dataclass
class ScoreDynamicsSchedule:
    n_noisy: int = 2900
    n_final: int = 100
    sigma_start: float = 1.0  # DM2 used 1.0 Angstrom for Si-O ~1.6 A; tune in LJ units
    sigma_end: float = 1e-3
    sigma_min: float = 1e-3
    model_sigma_max: float = 0.5  # training sigma_max of a sigma-aware model

    def sigmas(self) -> Tensor:
        """[n_noisy] linearly decreasing noise levels."""
        return torch.linspace(self.sigma_start, self.sigma_end, self.n_noisy, dtype=torch.float64)


def random_init(types: Tensor, box: Tensor, generator: torch.Generator | None = None) -> Structures:
    """Uniform random positions in the box for the given species and boxes."""
    u = torch.rand(types.shape + (2,), generator=generator, dtype=box.dtype).to(box.device)
    return Structures(u * box[:, None, :], types, box)


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
    strategy = strategy or Unconditional()
    x = strategy.initialize(init, request)
    frozen = strategy.frozen(request)
    keep = None if frozen is None else (~frozen)[..., None].to(x.pos.dtype)
    sigmas = schedule.sigmas().tolist()
    b_size, dev, dtype = x.batch_size, x.pos.device, x.pos.dtype
    traj: list[Tensor] = []

    def step(x: Structures, sigma: float, noisy: bool, n_done: int) -> Structures:
        sig_model = torch.full(
            (b_size,), min(sigma, schedule.model_sigma_max), dtype=dtype, device=dev
        )
        disp = strategy.displacement(model, x, sig_model, request)
        new = x.pos - disp
        if noisy:
            scale = schedule.sigma_min + (sigma - schedule.sigma_min) * torch.rand(
                x.pos.shape, generator=generator, dtype=dtype
            )
            eta = (scale * torch.randn(x.pos.shape, generator=generator, dtype=dtype)).to(dev)
            new = new - (eta if keep is None else eta * keep)
        x = strategy.after_step(x.with_pos(new), sigma if noisy else 0.0, request, generator)
        if save_every and n_done % save_every == 0:
            traj.append(x.pos.detach().cpu().clone())
        return x

    repaint = isinstance(strategy, RePaint) and strategy.n_resample > 1
    resampled: dict[int, int] = defaultdict(int)
    k, n_done = 0, 0
    while k < len(sigmas):
        n_done += 1
        x = step(x, sigmas[k], True, n_done)
        k += 1
        if repaint and k % strategy.jump == 0 and resampled[k] < strategy.n_resample - 1:
            resampled[k] += 1
            lo, hi = sigmas[k - 1], sigmas[k - strategy.jump]
            z = torch.randn(x.pos.shape, generator=generator, dtype=dtype).to(dev)
            renoise = (hi**2 - lo**2) ** 0.5 * z
            unknown = (~request.pin_mask)[..., None]
            x = x.with_pos(x.pos + renoise * unknown)
            k -= strategy.jump
    for _ in range(schedule.n_final):
        n_done += 1
        x = step(x, schedule.sigma_end, False, n_done)
    return x, traj


def sampler_from_config(cfg: dict) -> ScoreDynamicsSchedule:
    return ScoreDynamicsSchedule(**cfg)
