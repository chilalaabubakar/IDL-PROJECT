"""Melt-quench generation of 2D Kob–Andersen 65:35 glasses.  [Ticket P-1]

Protocol and quality gates: docs/DATASET.md §1.2 and §1.4. Working reference code:
docs/pilot/pilot_ka2d.py (start from its ``make_glass``; keep the lattice initialisation,
because random insertion lost atoms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuenchProtocol:
    rho: float = 1.2
    frac_a: float = 0.65
    t_melt: float = 2.0
    melt_steps: int = 20_000
    t_final: float = 0.01
    rate: float = 1e-2  # temperature drop per tau
    dt: float = 0.005
    tdamp: float = 0.1
    min_ftol: float = 1e-8


@dataclass
class Glass:
    pos: np.ndarray  # [N, 2] wrapped into [0, L)
    types: np.ndarray  # [N] int8, 0 = A, 1 = B
    box: np.ndarray  # [2]
    pe_atom: np.ndarray  # [N] per-atom potential energy of the inherent structure
    seed: int
    thermal_pos: np.ndarray | None = None  # [N, 2] snapshot at t_final before minimization


def make_glass(n_atoms: int, seed: int, protocol: QuenchProtocol) -> Glass:
    """Run melt -> linear quench -> FIRE minimization for one glass.

    ``n_atoms`` must be a perfect square (lattice start). Use one MPI rank and one thread so
    a seed always reproduces the same glass.
    """
    raise NotImplementedError("Ticket P-1")


def passes_quality_gates(glass: Glass) -> tuple[bool, dict[str, float]]:
    """Composition, global |psi6| < 0.3, min pair distance > 0.7 sigma_AB, PE outlier check."""
    raise NotImplementedError("Ticket P-1")
