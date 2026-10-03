"""Message-passing GNN (Gilmer et al. 2017): permutation symmetry only.  [Ticket M-5]

Edge features are the raw minimum-image vector (dx, dy) plus a distance RBF, and the
output is a plain per-atom MLP readout, so the model is translation-invariant and
permutation-equivariant but NOT rotation-equivariant; that is the point of the ablation.
Train with ``rotation_augmentation: true`` (random 90-degree rotations) as a control.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from glassdiff.geometry import neighbor_list
from glassdiff.models.base import Denoiser
from glassdiff.models.embeddings import NodeEmbedding
from glassdiff.models.layers import GaussianRBF, cosine_envelope, mlp
from glassdiff.types import Structures


class MPNNLayer(nn.Module):
    def __init__(self, hidden: int, n_edge: int) -> None:
        super().__init__()
        self.phi_e = mlp([2 * hidden + n_edge, hidden, hidden], final_act=True)
        self.norm = nn.LayerNorm(hidden)
        self.phi_h = mlp([2 * hidden, hidden, hidden])

    def forward(self, h: Tensor, idx: Tensor, edge: Tensor, env: Tensor, inv_k0: float) -> Tensor:
        bidx = torch.arange(h.shape[0], device=h.device)[:, None, None]
        h_i = h[:, :, None, :].expand(-1, -1, idx.shape[2], -1)
        m = self.phi_e(torch.cat([h_i, h[bidx, idx], edge], dim=-1)) * env[..., None]
        return h + self.phi_h(torch.cat([self.norm(h), m.sum(2) * inv_k0], dim=-1))


class MPNN(Denoiser):
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
        self.cutoff, self.k_max, self.inv_k0 = cutoff, k_max, 1.0 / avg_neighbors
        self.embed = NodeEmbedding(hidden, sigma_aware=sigma_aware)
        self.rbf = GaussianRBF(cutoff, n_rbf)
        self.layers = nn.ModuleList(MPNNLayer(hidden, n_rbf + 2) for _ in range(n_layers))
        self.readout = mlp([hidden, hidden, 2])

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        nl = neighbor_list(s.pos, s.box, self.cutoff, self.k_max)
        env = cosine_envelope(nl.dist, self.cutoff) * nl.mask
        edge = torch.cat([self.rbf(nl.dist), nl.vec], dim=-1)
        h = self.embed(s.types, sigma, pin, label)
        for layer in self.layers:
            h = layer(h, nl.idx, edge, env, self.inv_k0)
        out = self.readout(h)
        return out if pin is None else out * (~pin)[..., None]
