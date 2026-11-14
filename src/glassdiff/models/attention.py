"""A transformer finder: attention over all atoms, with the geometry as an attention bias.

The comparison it serves (docs/POINT_DEFECTS.md, "Other finders"): does a model that sees the
whole glass at once, through attention, find defects better or worse than the EGNN, whose
messages pass between neighbours within 2.5 sigma (reach 10 sigma after four layers)?

Design, chosen so that the answer cannot depend on things that do not matter physically:

    atoms       a set: no positional encoding of the atom index, so renumbering the atoms
                renumbers the output (permutation equivariance)
    geometry    only the distances between atoms, taken across the periodic box (minimum
                image), so shifting or turning the glass, or moving an atom by a box vector,
                changes nothing (as in Graphormer's 3D variant: Ying et al., NeurIPS 2021;
                Shi et al., 2022)
    input       species embedding + the species-resolved radial density of the neighbours
                within env_cutoff (a learned local descriptor, like a radial distribution
                function around each atom)
    layers      pre-norm transformer blocks; every head adds a learned function of the
                distance to its attention scores (attention bias), so the model can learn
                both local and long-range attention
    output      per-atom logits of the five atom roles, as EGNNAtomClassifier

Cost: all N^2 pairs per glass (256 atoms: 65,536 pairs), which is cheap on a GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from glassdiff.geometry import pair_vectors
from glassdiff.models.base import Denoiser
from glassdiff.models.layers import GaussianRBF, cosine_envelope, mlp
from glassdiff.types import Structures


class DistanceBiasBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, ffn: int, pair_dim: int) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.bias = nn.Linear(pair_dim, n_heads)  # attention bias per head from the distance
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = mlp([hidden, ffn, hidden])

    def forward(self, h: Tensor, pair: Tensor) -> Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.n_heads, d // self.n_heads).unbind(2)
        bias = self.bias(pair).permute(0, 3, 1, 2)  # [B, heads, N, N]
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias
        )
        h = h + self.out(att.transpose(1, 2).reshape(b, n, d))
        return h + self.ffn(self.norm2(h))


class AtomTransformerClassifier(Denoiser):
    """Per-atom role logits [B, N, n_classes] from a distance-biased transformer."""

    sigma_aware = False

    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn: int = 256,
        n_rbf: int = 32,
        r_max: float = 10.5,  # half the diagonal of the 14.6-sigma box is 10.3
        env_cutoff: float = 2.5,
        n_env_rbf: int = 16,
        n_species: int = 2,
        n_classes: int = 5,
        pair_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_species, self.n_classes, self.env_cutoff = n_species, n_classes, env_cutoff
        self.species = nn.Embedding(n_species, hidden)
        self.env_rbf = GaussianRBF(env_cutoff, n_env_rbf)
        self.env = mlp([n_species * n_env_rbf, hidden, hidden])
        self.rbf = GaussianRBF(r_max, n_rbf)
        self.pair = mlp([n_rbf, pair_dim, pair_dim], final_act=True)  # shared by the blocks
        self.blocks = nn.ModuleList(
            DistanceBiasBlock(hidden, n_heads, ffn, pair_dim) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = mlp([hidden, hidden, n_classes])

    def forward(
        self,
        s: Structures,
        sigma: Tensor | None = None,
        pin: Tensor | None = None,
        label: Tensor | None = None,
    ) -> Tensor:
        dist = pair_vectors(s.pos, s.box).norm(dim=-1)  # [B, N, N], minimum image
        n = dist.shape[1]
        other = ~torch.eye(n, dtype=torch.bool, device=dist.device)
        # local environment: radial density of each species around each atom
        weight = cosine_envelope(dist, self.env_cutoff) * other  # [B, N, N]
        onehot = F.one_hot(s.types, self.n_species).to(dist.dtype)  # [B, N, S]
        env = torch.einsum("bij,bijr,bjs->bisr", weight, self.env_rbf(dist), onehot)
        h = self.species(s.types) + self.env(env.flatten(2))
        pair = self.pair(self.rbf(dist))  # [B, N, N, pair_dim]
        for block in self.blocks:
            h = block(h, pair)
        return self.head(self.norm(h))
