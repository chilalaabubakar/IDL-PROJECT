"""Reference-defect patch library, built from the train split only.  [Ticket E-3]

Each patch is a defect atom plus its neighbours within ``radius`` (default 1.5), stored as
coordinates relative to the centre with their species and the centre's DefectClass. Used by
the hand-insertion, noise-patch, clamping and RePaint baselines and by "patch" requests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from glassdiff.geometry import pair_vectors
from glassdiff.types import DefectClass, Structures


@dataclass
class Patch:
    rel_pos: Tensor  # [P, 2], centre atom first, at (0, 0)
    types: Tensor  # [P]
    defect: int

    def rotated(self, angle: float) -> Patch:
        c, s = math.cos(angle), math.sin(angle)
        rot = torch.tensor([[c, -s], [s, c]], dtype=self.rel_pos.dtype)
        return Patch(self.rel_pos @ rot.T, self.types.clone(), self.defect)


class PatchLibrary:
    def __init__(self, patches: list[Patch]) -> None:
        self.patches = patches
        self._by_class: dict[int, list[int]] = {}
        for i, p in enumerate(patches):
            self._by_class.setdefault(p.defect, []).append(i)

    def __len__(self) -> int:
        return len(self.patches)

    def counts(self) -> dict[str, int]:
        return {DefectClass(c).name: len(v) for c, v in sorted(self._by_class.items())}

    @classmethod
    def build(cls, s: Structures, labels: Tensor, radius: float = 1.5) -> PatchLibrary:
        patches = []
        for b in range(s.batch_size):
            centres = torch.nonzero(labels[b] != DefectClass.NONE).flatten()
            if len(centres) == 0:
                continue
            vec = pair_vectors(s.pos[b : b + 1], s.box[b : b + 1])[0]  # [N, N, 2], r_j - r_i
            dist = vec.norm(dim=-1)
            for i in centres.tolist():
                nb = torch.nonzero((dist[i] <= radius) & (torch.arange(s.n_atoms) != i)).flatten()
                order = torch.cat([torch.tensor([i]), nb[torch.argsort(dist[i, nb])]])
                patches.append(
                    Patch(
                        rel_pos=vec[i, order].clone(),
                        types=s.types[b, order].clone(),
                        defect=int(labels[b, i]),
                    )
                )
        return cls(patches)

    def save(self, path: str | Path) -> None:
        sizes = np.array([len(p.types) for p in self.patches])
        np.savez_compressed(
            path,
            rel_pos=torch.cat([p.rel_pos for p in self.patches]).double().numpy(),
            types=torch.cat([p.types for p in self.patches]).numpy().astype(np.int8),
            defect=np.array([p.defect for p in self.patches], dtype=np.int8),
            sizes=sizes,
        )

    @classmethod
    def load(cls, path: str | Path) -> PatchLibrary:
        with np.load(path) as f:
            offsets = np.concatenate([[0], np.cumsum(f["sizes"])])
            rel, types, defect = f["rel_pos"], f["types"], f["defect"]
            return cls(
                [
                    Patch(
                        torch.from_numpy(rel[a:b]).double(),
                        torch.from_numpy(types[a:b]).long(),
                        int(d),
                    )
                    for a, b, d in zip(offsets[:-1], offsets[1:], defect)
                ]
            )

    def sample(self, defect: int, generator: torch.Generator | None = None) -> Patch:
        """Random patch of the given class, randomly rotated about its centre."""
        options = self._by_class.get(int(defect))
        if not options:
            raise KeyError(f"no patches of class {DefectClass(defect).name}")
        pick = options[int(torch.randint(len(options), (1,), generator=generator))]
        angle = float(torch.rand(1, generator=generator)) * 2 * math.pi
        return self.patches[pick].rotated(angle)
