"""EGNN with periodic boundaries: the main backbone.  [Ticket M-2]

Satorras et al. (2021), adapted to a periodic box. Every geometric quantity comes from
minimum-image edge vectors (glassdiff.geometry.neighbor_list), never raw x_i - x_j, which
is what DM2/graphite's EGNN does and which breaks under periodic boundaries.

Positions stay fixed inside the network; each layer contributes an equivariant vector:

    m_ij  = phi_e(h_i, h_j, rbf(|r_ij|)) * env(|r_ij|)        r_ij = minimum-image r_j - r_i
    h_i  <- h_i + phi_h(LN(h_i), sum_j gate(m_ij) m_ij / K0)
    out_i += sum_j r_ij * phi_x(m_ij) * env(|r_ij|) / K0

so the output is translation-invariant, rotation-equivariant and permutation-equivariant.
env is a smooth cosine cutoff (messages switch on continuously) and K0 a fixed average
neighbour count. Pinned atoms always get zero displacement. Acceptance:
tests/test_equivariance.py.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from glassdiff.geometry import NeighborList, neighbor_list
from glassdiff.models.base import Denoiser
from glassdiff.models.embeddings import NodeEmbedding
from glassdiff.models.layers import GaussianRBF, cosine_envelope, mlp
from glassdiff.types import Structures


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, n_rbf: int) -> None:
        super().__init__()
        self.phi_e = mlp([2 * hidden + n_rbf, hidden, hidden], final_act=True)
        self.gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden)
        self.phi_h = mlp([2 * hidden, hidden, hidden])
        self.phi_x = mlp([hidden, hidden, 1])

    def forward(
        self, h: Tensor, nl: NeighborList, rbf: Tensor, env: Tensor, inv_k0: float
    ) -> tuple[Tensor, Tensor]:
        bidx = torch.arange(h.shape[0], device=h.device)[:, None, None]
        h_i = h[:, :, None, :].expand(-1, -1, nl.idx.shape[2], -1)
        h_j = h[bidx, nl.idx]
        m = self.phi_e(torch.cat([h_i, h_j, rbf], dim=-1)) * env[..., None]
        agg = (self.gate(m) * m).sum(2) * inv_k0
        h = h + self.phi_h(torch.cat([self.norm(h), agg], dim=-1))
        vec = (nl.vec * self.phi_x(m) * env[..., None]).sum(2) * inv_k0
        return h, vec


class EGNNPBC(Denoiser):
    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 4,
        cutoff: float = 2.5,
        k_max: int = 64,
        sigma_aware: bool = True,
        n_rbf: int = 16,
        avg_neighbors: float = 24.0,
    ) -> None:
        super().__init__()
        self.sigma_aware = sigma_aware
        self.cutoff = cutoff
        self.k_max = k_max
        self.inv_k0 = 1.0 / avg_neighbors
        self.embed = NodeEmbedding(hidden, sigma_aware=sigma_aware)
        self.rbf = GaussianRBF(cutoff, n_rbf)
        self.layers = nn.ModuleList(EGNNLayer(hidden, n_rbf) for _ in range(n_layers))

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        nl = neighbor_list(s.pos, s.box, self.cutoff, self.k_max)
        env = cosine_envelope(nl.dist, self.cutoff) * nl.mask
        rbf = self.rbf(nl.dist)
        h = self.embed(s.types, sigma, pin, label)
        out = torch.zeros_like(s.pos)
        for layer in self.layers:
            h, vec = layer(h, nl, rbf, env, self.inv_k0)
            out = out + vec
        if pin is not None:
            out = out * (~pin)[..., None]
        return out
