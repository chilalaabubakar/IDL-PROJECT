"""Structure statistics for realism metrics.  [Ticket E-4]"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from glassdiff.geometry import pair_vectors
from glassdiff.physics.ka_potential import KA_SIGMA
from glassdiff.types import Structures


def _dist(s: Structures) -> Tensor:
    d = pair_vectors(s.pos, s.box).norm(dim=-1)
    eye = torch.eye(s.n_atoms, dtype=torch.bool, device=s.pos.device)
    return d.masked_fill(eye, float("inf"))


def radial_distribution(
    s: Structures, pair: tuple[int, int], r_max: float = 5.0, n_bins: int = 100
) -> tuple[Tensor, Tensor]:
    """Partial g_ab(r) averaged over the batch: (bin centres [n_bins], g [n_bins]).

    Normalised so g -> 1 for an uncorrelated fluid; r_max must stay below L / 2.
    """
    a, b = pair
    d = _dist(s)
    edges = torch.linspace(0.0, r_max, n_bins + 1, dtype=s.pos.dtype)
    hist = torch.zeros(n_bins, dtype=s.pos.dtype)
    norm = 0.0
    for k in range(s.batch_size):
        ia, ib = s.types[k] == a, s.types[k] == b
        sel = d[k][ia][:, ib].flatten()
        hist += torch.histogram(sel[sel < r_max].cpu(), bins=edges.cpu())[0].to(hist)
        area = float(s.box[k].prod())
        norm += float(ia.sum()) * float(ib.sum()) / area
    shell = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), hist / (norm * shell)


def pair_distances(
    s: Structures, pair: tuple[int, int], r_max: float = 2.5, centre_mask: Tensor | None = None
) -> list[Tensor]:
    """Per structure: a-b pair distances below r_max (each unordered pair once), optionally
    only pairs with at least one atom in ``centre_mask`` [B, N]."""
    a, b = pair
    d = _dist(s)
    out = []
    for k in range(s.batch_size):
        ia, ib = s.types[k] == a, s.types[k] == b
        keep = ia[:, None] & ib[None, :] & (d[k] < r_max)
        if a == b:  # rows a, columns b: for a != b each unordered pair appears once already
            keep &= torch.ones_like(keep).triu(1)
        if centre_mask is not None:
            m = centre_mask[k]
            keep &= m[:, None] | m[None, :]
        out.append(d[k][keep])
    return out


def bond_angles(
    s: Structures, cn_factor: float = 1.4, centre_mask: Tensor | None = None
) -> list[Tensor]:
    """Per structure: angles (radians) between every pair of first-shell bonds at each atom
    (first shell = within cn_factor * sigma_ab), optionally only at atoms in centre_mask."""
    vec = pair_vectors(s.pos, s.box)
    d = _dist(s)
    sigma = torch.tensor(KA_SIGMA, dtype=s.pos.dtype, device=s.pos.device)
    rc = cn_factor * sigma[s.types[:, :, None], s.types[:, None, :]]
    shell = d < rc
    out = []
    for k in range(s.batch_size):
        atoms = range(s.n_atoms) if centre_mask is None else torch.nonzero(centre_mask[k]).flatten()
        angles = []
        for i in atoms:
            v = vec[k, int(i)][shell[k, int(i)]]
            if len(v) < 2:
                continue
            u = v / v.norm(dim=-1, keepdim=True)
            cos = (u @ u.T).clamp(-1, 1)
            iu = torch.triu_indices(len(v), len(v), 1)
            angles.append(torch.arccos(cos[iu[0], iu[1]]))
        out.append(torch.cat(angles) if angles else torch.zeros(0, dtype=s.pos.dtype))
    return out
