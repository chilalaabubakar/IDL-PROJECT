"""Node embeddings shared by all denoisers.  [Ticket M-1]"""

from __future__ import annotations

from torch import Tensor, nn

from glassdiff.types import NUM_LABEL_TOKENS


class SigmaEmbedding(nn.Module):
    """Gaussian random Fourier features of log(sigma), then an MLP -> [B, dim]."""

    def __init__(self, dim: int, n_features: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        raise NotImplementedError("Ticket M-1")

    def forward(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Ticket M-1")


class NodeEmbedding(nn.Module):
    """Species embedding + pin flag + LabelToken embedding (NUM_LABEL_TOKENS entries,
    including UNLABELLED and NULL) + broadcast sigma embedding -> [B, N, dim]."""

    num_label_tokens = NUM_LABEL_TOKENS

    def __init__(self, dim: int, n_species: int = 2, sigma_aware: bool = True) -> None:
        super().__init__()
        raise NotImplementedError("Ticket M-1")

    def forward(
        self, types: Tensor, sigma: Tensor, pin: Tensor | None, label: Tensor | None
    ) -> Tensor:
        raise NotImplementedError("Ticket M-1")
