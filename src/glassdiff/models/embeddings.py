"""Node embeddings shared by all denoisers.  [Ticket M-1]"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from glassdiff.models.layers import mlp
from glassdiff.types import NUM_LABEL_TOKENS, LabelToken


class SigmaEmbedding(nn.Module):
    """Gaussian random Fourier features of log(sigma), then an MLP -> [B, dim]."""

    def __init__(self, dim: int, n_features: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(n_features // 2) * scale)
        self.net = mlp([n_features, dim, dim])

    def forward(self, sigma: Tensor) -> Tensor:
        x = 2 * math.pi * torch.log(sigma)[:, None] * self.freqs
        return self.net(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


class NodeEmbedding(nn.Module):
    """Species + pin flag + LabelToken (incl. UNLABELLED and NULL) + broadcast sigma
    embedding -> [B, N, dim]."""

    num_label_tokens = NUM_LABEL_TOKENS

    def __init__(self, dim: int, n_species: int = 2, sigma_aware: bool = True) -> None:
        super().__init__()
        self.species = nn.Embedding(n_species, dim)
        self.pin = nn.Embedding(2, dim)
        self.label = nn.Embedding(NUM_LABEL_TOKENS, dim)
        self.sigma = SigmaEmbedding(dim) if sigma_aware else None
        self.out = mlp([dim, dim, dim])

    def forward(
        self, types: Tensor, sigma: Tensor, pin: Tensor | None, label: Tensor | None
    ) -> Tensor:
        h = self.species(types)
        pin_idx = torch.zeros_like(types) if pin is None else pin.long()
        lab_idx = torch.full_like(types, int(LabelToken.UNLABELLED)) if label is None else label
        h = h + self.pin(pin_idx) + self.label(lab_idx)
        if self.sigma is not None:
            h = h + self.sigma(sigma)[:, None, :]
        return self.out(h)
