"""Small building blocks shared by the denoisers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def mlp(sizes: list[int], act: type[nn.Module] = nn.SiLU, final_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or final_act:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianRBF(nn.Module):
    """Gaussian radial basis on [0, cutoff]."""

    def __init__(self, cutoff: float, n_rbf: int = 16) -> None:
        super().__init__()
        self.register_buffer("centres", torch.linspace(0.0, cutoff, n_rbf))
        self.width = cutoff / n_rbf

    def forward(self, d: Tensor) -> Tensor:
        return torch.exp(-(((d[..., None] - self.centres) / self.width) ** 2))


def cosine_envelope(d: Tensor, cutoff: float) -> Tensor:
    """Smooth cutoff: 1 at d = 0, 0 at d >= cutoff, so messages switch on continuously."""
    return torch.where(d < cutoff, 0.5 * (torch.cos(math.pi * d / cutoff) + 1), torch.zeros_like(d))
