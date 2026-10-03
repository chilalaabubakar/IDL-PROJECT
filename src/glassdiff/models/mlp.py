"""MLP on flattened coordinates: no built-in symmetry, fixed N.  [Ticket M-6]

Input per atom: sin/cos of 2*pi*fractional coordinates (keeps periodicity), species, pin
flag and label token (one-hot); atoms are ordered by species (stable sort) before
flattening and the output is put back in the original order. The sigma embedding is
concatenated to the flat input. Output: [B, N, 2].
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from glassdiff.models.base import Denoiser
from glassdiff.models.embeddings import SigmaEmbedding
from glassdiff.models.layers import mlp
from glassdiff.types import NUM_LABEL_TOKENS, LabelToken, Structures

_PER_ATOM = 4 + 2 + 1 + NUM_LABEL_TOKENS


class FlatMLP(Denoiser):
    def __init__(
        self,
        n_atoms: int = 64,
        hidden: int = 512,
        n_layers: int = 4,
        sigma_aware: bool = True,
        sigma_dim: int = 64,
    ) -> None:
        super().__init__()
        self.sigma_aware = sigma_aware
        self.n_atoms = n_atoms
        self.sigma = SigmaEmbedding(sigma_dim) if sigma_aware else None
        n_in = n_atoms * _PER_ATOM + (sigma_dim if sigma_aware else 0)
        self.net = mlp([n_in] + [hidden] * n_layers + [n_atoms * 2])

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        if s.n_atoms != self.n_atoms:
            raise ValueError(f"FlatMLP was built for N={self.n_atoms}, got N={s.n_atoms}")
        order = torch.sort(s.types, dim=-1, stable=True).indices
        inv = torch.argsort(order, dim=-1)

        def take(t: Tensor) -> Tensor:
            return torch.gather(t, 1, order if t.dim() == 2 else order[..., None].expand_as(t))

        frac = 2 * math.pi * take(s.pos) / s.box[:, None, :]
        pin_f = torch.zeros_like(s.types, dtype=s.pos.dtype) if pin is None else take(pin).to(s.pos)
        lab = torch.full_like(s.types, int(LabelToken.UNLABELLED)) if label is None else take(label)
        feats = torch.cat(
            [
                torch.sin(frac),
                torch.cos(frac),
                nn.functional.one_hot(take(s.types), 2).to(s.pos),
                pin_f[..., None],
                nn.functional.one_hot(lab, NUM_LABEL_TOKENS).to(s.pos),
            ],
            dim=-1,
        ).flatten(1)
        if self.sigma is not None:
            feats = torch.cat([feats, self.sigma(sigma)], dim=-1)
        out = self.net(feats).view(s.batch_size, self.n_atoms, 2)
        out = torch.gather(out, 1, inv[..., None].expand_as(out))
        return out if pin is None else out * (~pin)[..., None]
