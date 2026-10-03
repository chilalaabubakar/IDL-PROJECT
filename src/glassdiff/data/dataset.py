"""Dataset splits on disk (``data/<set>/{train,val,test}.npz``).  [Ticket P-1]

File format: docs/DATASET.md §1.7. Keys: pos [M, N, 2] float32, types [M, N] int8,
box [M, 2] float32, pe_atom [M, N] float32, labels [M, N] int8, seed [M] int64.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from glassdiff.types import Structures


@dataclass
class GlassSplit:
    structures: Structures
    pe_atom: Tensor  # [M, N]
    labels: Tensor | None  # [M, N] DefectClass; None until thresholds are frozen
    seed: Tensor  # [M]


def load_split(path: str | Path) -> GlassSplit:
    raise NotImplementedError("Ticket P-1")


def save_split(path: str | Path, split: GlassSplit) -> None:
    raise NotImplementedError("Ticket P-1")


class GlassDataset(torch.utils.data.Dataset):
    """Indexes glasses of one split. Batching stacks along dim 0, since N is fixed per set."""

    def __init__(self, split: GlassSplit) -> None:
        self.split = split

    def __len__(self) -> int:
        return self.split.structures.batch_size

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        raise NotImplementedError("Ticket M-3")
