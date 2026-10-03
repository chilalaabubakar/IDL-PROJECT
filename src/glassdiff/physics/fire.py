"""Batched FIRE energy minimization ("relaxation").  [Ticket P-3]

Bitzek et al., PRL 97, 170201 (2006), in the form used by ASE. Each structure in the batch
keeps its own dt, alpha and step counter, and stops once converged, so a structure relaxes
identically alone or in a batch. Frozen atoms (e.g. the host in Task B) never move.
Positions are never wrapped, so ``displacement`` is the true relaxation displacement.
Acceptance: tests/test_fire.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from glassdiff.physics.ka_potential import ka_forces
from glassdiff.types import Structures


@dataclass
class FireResult:
    structures: Structures
    displacement: Tensor  # [B, N, 2] displacement from the input
    max_force: Tensor  # [B] largest per-atom force norm at the end
    n_steps: Tensor  # [B] long, steps taken until convergence (or max_steps)
    converged: Tensor  # [B] bool


def _max_force(f: Tensor) -> Tensor:
    return f.norm(dim=-1).amax(dim=-1)


@torch.no_grad()
def fire_minimize(
    s: Structures,
    frozen: Tensor | None = None,  # [B, N] bool
    fmax: float = 1e-6,
    max_steps: int = 100_000,
    dt_start: float = 0.005,
    dt_max: float = 0.05,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    max_move: float = 0.1,
) -> FireResult:
    """Minimize the Kob–Andersen energy of every structure in the batch."""
    batch = s.batch_size
    dev, dtype = s.pos.device, s.pos.dtype
    free = (
        torch.ones(s.pos.shape[:2], dtype=torch.bool, device=dev) if frozen is None else ~frozen
    )[..., None]
    pos = s.pos.clone()
    v = torch.zeros_like(pos)
    dt = torch.full((batch,), dt_start, dtype=dtype, device=dev)
    alpha = torch.full((batch,), alpha_start, dtype=dtype, device=dev)
    n_pos = torch.zeros(batch, dtype=torch.long, device=dev)
    n_steps = torch.zeros(batch, dtype=torch.long, device=dev)
    active = torch.ones(batch, dtype=torch.bool, device=dev)

    f = ka_forces(s.with_pos(pos)) * free
    for _ in range(max_steps):
        active &= _max_force(f) > fmax
        if not active.any():
            break
        n_steps += active
        p = (f * v).sum(dim=(1, 2))
        uphill = p <= 0
        # Velocity mixing towards the force direction (per structure, global norms).
        v_norm = v.norm(dim=(1, 2), keepdim=True)
        f_norm = f.norm(dim=(1, 2), keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        a = alpha[:, None, None]
        v = torch.where(
            uphill[:, None, None], torch.zeros_like(v), (1 - a) * v + a * f / f_norm * v_norm
        )
        grow = ~uphill & (n_pos > n_min)
        dt = torch.where(uphill, dt * f_dec, torch.where(grow, (dt * f_inc).clamp_max(dt_max), dt))
        alpha = torch.where(
            uphill, torch.full_like(alpha, alpha_start), torch.where(grow, alpha * f_alpha, alpha)
        )
        n_pos = torch.where(uphill, torch.zeros_like(n_pos), n_pos + 1)
        # Semi-implicit Euler step with per-atom move limit; inactive structures stay put.
        v = v + dt[:, None, None] * f
        dx = dt[:, None, None] * v
        step = dx.norm(dim=-1, keepdim=True)
        dx = dx * (max_move / step.clamp_min(max_move))
        dx = dx * active[:, None, None] * free
        v = v * active[:, None, None]
        pos = pos + dx
        f = ka_forces(s.with_pos(pos)) * free

    max_f = _max_force(f)
    return FireResult(
        structures=s.with_pos(pos),
        displacement=pos - s.pos,
        max_force=max_f,
        n_steps=n_steps,
        converged=max_f <= fmax,
    )
