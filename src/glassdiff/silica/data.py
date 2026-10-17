"""Silica glass splits: one .npz per split plus a meta.json per dataset.

pos           [B, N, 3] float32, wrapped into [0, L) per axis (Angstrom)
types         [B, N]    int64, 0 = O, 1 = Si
box           [B, 3]    float32, orthorhombic cell lengths (Angstrom)
cooling_rate  [B]       float32, K/ps (the melt-quench rate the glass was made with)
labels        [B, N]    int64 SilicaDefect labels (optional until labelled)
source        [B]       str, where each glass came from (file name or generator seed)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SPECIES = ("O", "Si")
ATOMIC_NUMBER = {"O": 8, "Si": 14}
MASS = {"O": 15.9994, "Si": 28.0855}  # g/mol


@dataclass
class SilicaSplit:
    pos: Tensor
    types: Tensor
    box: Tensor
    cooling_rate: Tensor
    labels: Tensor | None = None
    source: np.ndarray | None = None

    def __len__(self) -> int:
        return self.pos.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.pos.shape[1]

    def subset(self, idx) -> SilicaSplit:
        return SilicaSplit(
            self.pos[idx],
            self.types[idx],
            self.box[idx],
            self.cooling_rate[idx],
            None if self.labels is None else self.labels[idx],
            None if self.source is None else self.source[idx],
        )

    def density(self) -> Tensor:
        """[B] mass density in g/cm^3."""
        mass = torch.where(self.types == 1, MASS["Si"], MASS["O"]).sum(-1)
        return mass / 0.602214 / self.box.prod(-1)


def save_split(path: str | Path, split: SilicaSplit) -> None:
    arrays = {
        "pos": split.pos.numpy().astype(np.float32),
        "types": split.types.numpy().astype(np.int64),
        "box": split.box.numpy().astype(np.float32),
        "cooling_rate": split.cooling_rate.numpy().astype(np.float32),
    }
    if split.labels is not None:
        arrays["labels"] = split.labels.numpy().astype(np.int64)
    if split.source is not None:
        arrays["source"] = np.asarray(split.source, dtype=str)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_split(path: str | Path) -> SilicaSplit:
    with np.load(path, allow_pickle=False) as f:
        return SilicaSplit(
            pos=torch.from_numpy(f["pos"]).float(),
            types=torch.from_numpy(f["types"]).long(),
            box=torch.from_numpy(f["box"]).float(),
            cooling_rate=torch.from_numpy(f["cooling_rate"]).float(),
            labels=torch.from_numpy(f["labels"]).long() if "labels" in f else None,
            source=f["source"] if "source" in f else None,
        )


def stack(glasses: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, str]]) -> SilicaSplit:
    """Glasses as (pos [N, 3], types [N], box [3], cooling rate, source); all the same N."""
    n = {len(g[1]) for g in glasses}
    if len(n) != 1:
        raise ValueError(f"all glasses of a split must have the same atom count, got {n}")
    return SilicaSplit(
        pos=torch.tensor(np.stack([np.mod(g[0], g[2]) for g in glasses]), dtype=torch.float32),
        types=torch.tensor(np.stack([g[1] for g in glasses]), dtype=torch.long),
        box=torch.tensor(np.stack([g[2] for g in glasses]), dtype=torch.float32),
        cooling_rate=torch.tensor([g[3] for g in glasses], dtype=torch.float32),
        source=np.asarray([g[4] for g in glasses], dtype=str),
    )


def read_lammps_data(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pos [N, 3] wrapped into the box, types [N] with 0 = O / 1 = Si, box [3]) from a
    LAMMPS data file with atom_style charge (DM2's glasses and ours)."""
    import ase.io

    atoms = ase.io.read(path, format="lammps-data", atom_style="charge")
    if not np.allclose(atoms.cell.angles(), 90.0):
        raise ValueError(f"{path}: only orthorhombic cells are supported")
    box = atoms.cell.lengths()
    numbers = atoms.numbers
    if not set(np.unique(numbers)) <= {8, 14}:
        raise ValueError(f"{path}: expected only O and Si, got atomic numbers {set(numbers)}")
    return np.mod(atoms.positions, box), (numbers == 14).astype(np.int64), box
