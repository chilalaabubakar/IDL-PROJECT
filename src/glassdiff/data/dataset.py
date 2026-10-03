"""Dataset splits on disk (``data/<set>/{train,val,test}.npz``).  [Ticket P-1]

File format: docs/DATASET.md §1.7. Keys: pos [M, N, 2] float64, types [M, N] int8,
box [M, 2] float64, pe_atom [M, N] float64, seed [M] int64, optional thermal_pos
[M, N, 2] float64, and labels [M, N] int8 once thresholds are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from glassdiff.types import Structures


@dataclass
class GlassSplit:
    structures: Structures
    pe_atom: Tensor  # [M, N]
    labels: Tensor | None  # [M, N] DefectClass; None until thresholds are frozen
    seed: Tensor  # [M]
    thermal_pos: Tensor | None = None  # [M, N, 2]

    def __len__(self) -> int:
        return self.structures.batch_size

    def subset(self, idx: Tensor | slice) -> GlassSplit:
        s = self.structures
        return GlassSplit(
            structures=Structures(s.pos[idx], s.types[idx], s.box[idx]),
            pe_atom=self.pe_atom[idx],
            labels=None if self.labels is None else self.labels[idx],
            seed=self.seed[idx],
            thermal_pos=None if self.thermal_pos is None else self.thermal_pos[idx],
        )


def load_split(path: str | Path) -> GlassSplit:
    with np.load(path) as f:
        structures = Structures(
            torch.from_numpy(f["pos"]).double(),
            torch.from_numpy(f["types"]).long(),
            torch.from_numpy(f["box"]).double(),
        )
        return GlassSplit(
            structures=structures,
            pe_atom=torch.from_numpy(f["pe_atom"]).double(),
            labels=torch.from_numpy(f["labels"]).long() if "labels" in f else None,
            seed=torch.from_numpy(f["seed"]).long(),
            thermal_pos=torch.from_numpy(f["thermal_pos"]).double() if "thermal_pos" in f else None,
        )


def save_split(path: str | Path, split: GlassSplit) -> None:
    s = split.structures
    arrays = {
        "pos": s.pos.detach().cpu().double().numpy(),
        "types": s.types.cpu().numpy().astype(np.int8),
        "box": s.box.detach().cpu().double().numpy(),
        "pe_atom": split.pe_atom.detach().cpu().double().numpy(),
        "seed": split.seed.cpu().numpy().astype(np.int64),
    }
    if split.labels is not None:
        arrays["labels"] = split.labels.cpu().numpy().astype(np.int8)
    if split.thermal_pos is not None:
        arrays["thermal_pos"] = split.thermal_pos.detach().cpu().double().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class GlassDataset(torch.utils.data.Dataset):
    """Indexes glasses of one split. The default collate stacks along dim 0 (N is fixed)."""

    def __init__(self, split: GlassSplit, dtype: torch.dtype = torch.float32) -> None:
        self.split = split
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        s = self.split.structures
        item = {
            "pos": s.pos[i].to(self.dtype),
            "types": s.types[i],
            "box": s.box[i].to(self.dtype),
        }
        if self.split.labels is not None:
            item["labels"] = self.split.labels[i]
        return item


def batch_to_structures(batch: dict[str, Tensor]) -> Structures:
    return Structures(batch["pos"], batch["types"], batch["box"])
