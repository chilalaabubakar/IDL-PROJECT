"""MLP on flattened coordinates: no built-in symmetry, fixed N = 64.  [Ticket M-6]

Input per atom: sin/cos of 2*pi*fractional coordinates (keeps periodicity), species,
pin flag, label token; atoms ordered by species, then index. Output: [B, N, 2].
"""

from __future__ import annotations

from torch import Tensor

from glassdiff.models.base import Denoiser
from glassdiff.types import Structures


class FlatMLP(Denoiser):
    def __init__(
        self, n_atoms: int = 64, hidden: int = 512, n_layers: int = 4, sigma_aware: bool = True
    ) -> None:
        super().__init__()
        self.sigma_aware = sigma_aware
        raise NotImplementedError("Ticket M-6")

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        raise NotImplementedError("Ticket M-6")
