"""Batched FIRE energy minimization ("relaxation").  [Ticket P-3]

Bitzek et al., PRL 97, 170201 (2006). Each structure in the batch keeps its own dt and
alpha, and converges independently. Frozen atoms (e.g. the host in Task B) never move.
Acceptance: minimizing dataset thermal snapshots reproduces the LAMMPS inherent-structure
energy within 1e-4 per atom; frozen atoms have exactly zero displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from glassdiff.types import Structures


@dataclass
class FireResult:
    structures: Structures
    displacement: Tensor  # [B, N, 2] minimum-image displacement from the input
    max_force: Tensor  # [B]
    n_steps: Tensor  # [B] long
    converged: Tensor  # [B] bool


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
) -> FireResult:
    raise NotImplementedError("Ticket P-3")
