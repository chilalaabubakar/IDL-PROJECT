"""Reference-defect patch library, built from the train split only.  [Ticket E-3]

Each patch: centre atom plus its neighbours within ``radius`` (default 1.5), as coordinates
relative to the centre, with species and the centre's DefectClass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from glassdiff.types import Structures


@dataclass
class Patch:
    rel_pos: Tensor  # [P, 2], centre atom first, at (0, 0)
    types: Tensor  # [P]
    defect: int


class PatchLibrary:
    def __init__(self, patches: list[Patch]) -> None:
        self.patches = patches

    @classmethod
    def build(cls, s: Structures, labels: Tensor, radius: float = 1.5) -> PatchLibrary:
        raise NotImplementedError("Ticket E-3")

    @classmethod
    def load(cls, path: str | Path) -> PatchLibrary:
        raise NotImplementedError("Ticket E-3")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Ticket E-3")

    def sample(self, defect: int, generator: torch.Generator | None = None) -> Patch:
        """Random patch of the given class, randomly rotated about its centre."""
        raise NotImplementedError("Ticket E-3")
