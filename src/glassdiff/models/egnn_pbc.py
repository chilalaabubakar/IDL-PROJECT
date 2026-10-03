"""EGNN with periodic boundaries: the main backbone.  [Ticket M-2]

Satorras et al. (2021), with one change that matters: messages and coordinate updates use
minimum-image edge vectors from glassdiff.geometry.neighbor_list, never raw x_i - x_j
(DM2/graphite's EGNN does the latter, which breaks under periodic boundaries).

    m_ij = phi_e(h_i, h_j, |r_ij|^2)          r_ij = minimum-image r_j - r_i
    h_i <- h_i + phi_h(h_i, sum_j m_ij)
    output_i = sum_j r_ij * phi_x(m_ij)       (translation-invariant, rotation-equivariant)

Acceptance: tests/test_equivariance.py.
"""

from __future__ import annotations

from torch import Tensor

from glassdiff.models.base import Denoiser
from glassdiff.types import Structures


class EGNNPBC(Denoiser):
    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 4,
        cutoff: float = 2.5,
        k_max: int = 48,
        sigma_aware: bool = True,
    ) -> None:
        super().__init__()
        self.sigma_aware = sigma_aware
        raise NotImplementedError("Ticket M-2")

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError("Ticket M-2")
