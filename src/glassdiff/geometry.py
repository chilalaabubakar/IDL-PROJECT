"""Periodic geometry: minimum-image vectors and fixed-size neighbour lists.

Every model, descriptor and the potential build on these two functions, so periodic
boundaries are handled in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def minimum_image(vec: Tensor, box: Tensor) -> Tensor:
    """Wrap displacement vectors [B, ..., 2] into the minimum-image convention for box [B, 2]."""
    b = box.view((box.shape[0],) + (1,) * (vec.dim() - 2) + (2,))
    return vec - b * torch.round(vec / b)


def pair_vectors(pos: Tensor, box: Tensor) -> Tensor:
    """All minimum-image pair vectors: out[b, i, j] = r_j - r_i, shape [B, N, N, 2]."""
    return minimum_image(pos[:, None, :, :] - pos[:, :, None, :], box)


@dataclass
class NeighborList:
    """Up to K neighbours per atom; padded slots have mask False and zero vec/dist."""

    idx: Tensor  # [B, N, K] long, neighbour index j
    vec: Tensor  # [B, N, K, 2] minimum-image r_j - r_i
    dist: Tensor  # [B, N, K]
    mask: Tensor  # [B, N, K] bool


def neighbor_list(pos: Tensor, box: Tensor, cutoff: float, k_max: int) -> NeighborList:
    """Neighbours within ``cutoff`` under periodic boundaries, sorted by distance.

    Dense O(N^2) per structure, which is fast on a GPU for N up to a few thousand.
    K = min(k_max, N - 1). Raises instead of silently truncating when an atom has more than
    ``k_max`` neighbours, because dropping neighbours would break permutation symmetry.
    """
    if cutoff >= 0.5 * float(box.min()):
        raise ValueError(f"cutoff {cutoff} must be < half the smallest box side {float(box.min())}")
    n_atoms = pos.shape[1]
    vec = pair_vectors(pos, box)
    dist = vec.norm(dim=-1)
    self_pair = torch.eye(n_atoms, dtype=torch.bool, device=pos.device)
    dist = dist.masked_fill(self_pair, float("inf"))

    n_within = (dist <= cutoff).sum(-1)
    if n_atoms > 1 and int(n_within.max()) > k_max:
        raise ValueError(
            f"an atom has {int(n_within.max())} neighbours within {cutoff}, more than k_max={k_max}"
        )

    k = min(k_max, n_atoms - 1)
    d, idx = torch.topk(dist, k, dim=-1, largest=False)
    mask = d <= cutoff
    v = torch.gather(vec, 2, idx[..., None].expand(-1, -1, -1, 2))
    return NeighborList(
        idx=idx,
        vec=v * mask[..., None],
        dist=d.masked_fill(~mask, 0.0),
        mask=mask,
    )
