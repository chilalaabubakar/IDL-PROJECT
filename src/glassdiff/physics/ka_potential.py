"""Kob–Andersen pair potential in PyTorch.  [Ticket P-2]

Must reproduce LAMMPS ``pair_style lj/cut 2.5`` with ``pair_modify shift yes`` and the
coefficients below (cutoff 2.5 * sigma_ab, energy shifted to zero at the cutoff).
Acceptance: tests/test_potential.py.
"""

from __future__ import annotations

from torch import Tensor

from glassdiff.types import Structures

# Plain floats: build tensors in the positions' dtype (torch.tensor(KA_SIGMA, dtype=pos.dtype)).
# A float32 table cast to float64 gives 0.800000012 and fails the 1e-8 analytic tests.
KA_SIGMA = ((1.0, 0.8), (0.8, 0.88))
KA_EPSILON = ((1.0, 1.5), (1.5, 0.5))
KA_CUTOFF_FACTOR = 2.5


def ka_energy(s: Structures, per_atom: bool = False) -> Tensor:
    """Potential energy per structure [B], or per atom [B, N] (each pair split equally).

    Use all minimum-image pairs (glassdiff.geometry.pair_vectors); N <= 1024 fits easily.
    Must be differentiable with respect to ``s.pos``.
    """
    raise NotImplementedError("Ticket P-2")


def ka_forces(s: Structures) -> Tensor:
    """Forces [B, N, 2] = -dE/dpos (autograd or analytic; tests compare to LAMMPS)."""
    raise NotImplementedError("Ticket P-2")
