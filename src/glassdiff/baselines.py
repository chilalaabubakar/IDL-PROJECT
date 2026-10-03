"""Non-learned baselines.  [Ticket B-2]

B2, hand insertion ("current practice"): paste a library patch into a host glass at the
target. Each patch atom replaces the nearest unused host atom of the same species, so the
atom count and composition are unchanged; every other host atom stays where it was, so
overlaps at the patch edge are left for relaxation to resolve.
"""

from __future__ import annotations

import torch
from torch import Tensor

from glassdiff.data.patches import Patch
from glassdiff.geometry import minimum_image
from glassdiff.types import Structures


def hand_insert(host: Structures, target: Tensor, patches: list[Patch]) -> Structures:
    """Insert ``patches[b]`` centred at ``target[b]`` into ``host`` structure b."""
    pos = host.pos.clone()
    for b, patch in enumerate(patches):
        dist = minimum_image(host.pos[b] - target[b], host.box[b : b + 1]).norm(dim=-1)
        used = torch.zeros(host.n_atoms, dtype=torch.bool)
        for rel, sp in zip(patch.rel_pos, patch.types.tolist()):
            cand = torch.where(
                (host.types[b] == sp) & ~used, dist, torch.full_like(dist, float("inf"))
            )
            j = int(cand.argmin())
            used[j] = True
            pos[b, j] = target[b] + rel.to(pos)
    return host.with_pos(pos)
