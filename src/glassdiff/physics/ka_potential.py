"""Kob–Andersen pair potential in PyTorch.  [Ticket P-2]

Reproduces LAMMPS ``pair_style lj/cut 2.5`` with ``pair_modify shift yes`` and the
coefficients below (cutoff 2.5 * sigma_ab, energy shifted to zero at the cutoff).
Dense over all minimum-image pairs, which is fast for N <= ~1024; works in float32 and
float64. Acceptance: tests/test_potential.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

from glassdiff.geometry import pair_vectors
from glassdiff.types import Structures

# Plain floats: build tensors in the positions' dtype (torch.tensor(KA_SIGMA, dtype=pos.dtype)).
# A float32 table cast to float64 gives 0.800000012 and fails the 1e-8 analytic tests.
KA_SIGMA = ((1.0, 0.8), (0.8, 0.88))
KA_EPSILON = ((1.0, 1.5), (1.5, 0.5))
KA_CUTOFF_FACTOR = 2.5


def _pair_terms(s: Structures) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Minimum-image vectors, r^2, sigma^2, epsilon and the within-cutoff mask, all [B, N, N]."""
    vec = pair_vectors(s.pos, s.box)
    r2 = (vec * vec).sum(-1)
    sigma = torch.tensor(KA_SIGMA, dtype=s.pos.dtype, device=s.pos.device)
    eps = torch.tensor(KA_EPSILON, dtype=s.pos.dtype, device=s.pos.device)
    ti, tj = s.types[:, :, None], s.types[:, None, :]
    sig2, eps_ij = sigma[ti, tj] ** 2, eps[ti, tj]
    eye = torch.eye(s.n_atoms, dtype=torch.bool, device=s.pos.device)
    mask = (r2 < KA_CUTOFF_FACTOR**2 * sig2) & ~eye
    return vec, torch.where(mask, r2, torch.ones_like(r2)), sig2, eps_ij, mask


def ka_energy(s: Structures, per_atom: bool = False) -> Tensor:
    """Potential energy per structure [B], or per atom [B, N] (each pair split equally).

    Differentiable with respect to ``s.pos``.
    """
    _, r2, sig2, eps, mask = _pair_terms(s)
    x3 = (sig2 / r2) ** 3
    xc3 = KA_CUTOFF_FACTOR**-6
    e = 4 * eps * (x3 * x3 - x3) - 4 * eps * (xc3 * xc3 - xc3)
    e_atom = 0.5 * torch.where(mask, e, torch.zeros_like(e)).sum(-1)
    return e_atom if per_atom else e_atom.sum(-1)


def ka_forces(s: Structures) -> Tensor:
    """Forces [B, N, 2] = -dE/dpos (analytic)."""
    vec, r2, sig2, eps, mask = _pair_terms(s)
    x3 = (sig2 / r2) ** 3
    # x3 = (sigma/r)^6: de/d(r^2) = -12 eps (2 x3^2 - x3) / r^2 ; F_i = sum_j 2 de/d(r^2) r_ij
    de_dr2 = -12 * eps * (2 * x3 * x3 - x3) / r2
    coef = torch.where(mask, 2 * de_dr2, torch.zeros_like(de_dr2))
    return (coef[..., None] * vec).sum(2)
