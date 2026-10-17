"""Periodic neighbour graphs in 3D (orthorhombic boxes, minimum image).

Edges are built from the current (noisy) positions at every step, so no precomputed large
cutoff is needed. Dense per structure, in row chunks to bound memory: ~10 ms for 576 atoms
and ~0.3 s for 3,000 atoms on a CPU, much less on a GPU.
"""

from __future__ import annotations

import torch
from torch import Tensor


def minimum_image(d: Tensor, box: Tensor) -> Tensor:
    return d - box * torch.round(d / box)


def radius_graph_pbc(
    pos: Tensor,  # [N, 3]
    box: Tensor,  # [3]
    cutoff: float,
    chunk: int = 1024,
) -> tuple[Tensor, Tensor]:
    """(edge_index [2, E], vec [E, 3]) for all ordered pairs (i, j), i != j, closer than
    ``cutoff``; vec = r_j - r_i under minimum image. Needs cutoff < min(box) / 2."""
    if cutoff >= 0.5 * float(box.min()):
        raise ValueError(f"cutoff {cutoff} must be < half the smallest box side {float(box.min())}")
    n = pos.shape[0]
    src, dst, vecs = [], [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = minimum_image(pos[None, :, :] - pos[a:b, None, :], box)  # [c, N, 3]
        r2 = (d * d).sum(-1)
        rows = torch.arange(a, b, device=pos.device)
        r2[rows - a, rows] = float("inf")  # no self-edges
        i, j = torch.nonzero(r2 < cutoff * cutoff, as_tuple=True)
        src.append(i + a)
        dst.append(j)
        vecs.append(d[i, j])
    return torch.stack([torch.cat(src), torch.cat(dst)]), torch.cat(vecs)


def batched_radius_graph(
    pos: Tensor,  # [M, 3] all atoms of a batch, structure after structure
    box: Tensor,  # [G, 3]
    batch: Tensor,  # [M] structure index of each atom (sorted)
    cutoff: float,
) -> tuple[Tensor, Tensor]:
    counts = torch.bincount(batch, minlength=box.shape[0]).tolist()
    edges, vecs, offset = [], [], 0
    for g, n in enumerate(counts):
        e, v = radius_graph_pbc(pos[offset : offset + n], box[g], cutoff)
        edges.append(e + offset)
        vecs.append(v)
        offset += n
    return torch.cat(edges, dim=1), torch.cat(vecs)
