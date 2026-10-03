"""Message-passing GNN (Gilmer et al. 2017): permutation symmetry only.  [Ticket M-5]

Edge features are the raw minimum-image vector (dx, dy) plus a distance RBF, so the model
is NOT rotation-equivariant; that is the point of the ablation. Train with and without
random 90-degree-rotation augmentation as a control.
"""

from __future__ import annotations

from torch import Tensor

from glassdiff.models.base import Denoiser
from glassdiff.types import Structures


class MPNN(Denoiser):
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
        raise NotImplementedError("Ticket M-5")

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError("Ticket M-5")
