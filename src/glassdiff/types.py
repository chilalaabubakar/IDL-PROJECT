"""Core data containers and label conventions shared by every module.

Conventions (docs/IMPLEMENTATION_PLAN.md §1):
- LJ units: lengths in sigma_AA, energies in eps_AA, time in tau.
- Positions are float tensors of shape [B, N, 2]. They may lie outside the box; wrap only
  when needed (graph building uses minimum-image vectors, so wrapping is never required).
- Boxes are orthorhombic and periodic in both directions: side lengths of shape [B, 2].
- All structures in one batch have the same number of atoms N.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import torch
from torch import Tensor


class Species(IntEnum):
    A = 0  # large
    B = 1  # small


class DefectClass(IntEnum):
    """Per-atom defect labels stored in the dataset (docs/DATASET.md §1.5)."""

    NONE = 0
    D1_MINUS = 1  # under-coordinated A atom
    D1_PLUS = 2  # over-coordinated A atom
    D2 = 3  # B atom with too many B neighbours


class LabelToken(IntEnum):
    """Per-atom condition tokens fed to the denoiser.

    Most atoms are UNLABELLED. NULL replaces a dropped request label (classifier-free
    guidance). A requested defect class c is encoded as ``defect_token(c)``.
    """

    UNLABELLED = 0
    NULL = 1


_FIRST_DEFECT_TOKEN = 2
NUM_LABEL_TOKENS = _FIRST_DEFECT_TOKEN + len(DefectClass)


def defect_token(defect: int | Tensor) -> int | Tensor:
    """Map a DefectClass value (or tensor of values) to its LabelToken id."""
    return defect + _FIRST_DEFECT_TOKEN


@dataclass
class Structures:
    """A batch of periodic 2D configurations with equal atom counts."""

    pos: Tensor  # [B, N, 2] float
    types: Tensor  # [B, N] long, Species values
    box: Tensor  # [B, 2] float, side lengths

    def __post_init__(self) -> None:
        if self.pos.dim() != 3 or self.pos.shape[-1] != 2:
            raise ValueError(f"pos must be [B, N, 2], got {tuple(self.pos.shape)}")
        if self.types.shape != self.pos.shape[:2]:
            raise ValueError(
                f"types must be [B, N] = {tuple(self.pos.shape[:2])}, got {tuple(self.types.shape)}"
            )
        if self.box.shape != (self.pos.shape[0], 2):
            raise ValueError(f"box must be [B, 2], got {tuple(self.box.shape)}")

    @property
    def batch_size(self) -> int:
        return self.pos.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.pos.shape[1]

    def to(self, device: torch.device | str) -> Structures:
        return Structures(self.pos.to(device), self.types.to(device), self.box.to(device))

    def clone(self) -> Structures:
        return Structures(self.pos.clone(), self.types.clone(), self.box.clone())

    def with_pos(self, pos: Tensor) -> Structures:
        return replace(self, pos=pos)

    def wrapped(self) -> Structures:
        """Copy with positions wrapped into [0, L)."""
        return self.with_pos(torch.remainder(self.pos, self.box[:, None, :]))


@dataclass
class Request:
    """A batch of defect requests: "defect class k at location p".

    ``pin_mask``/``pin_pos`` describe atoms held fixed: the centre atom, a reference patch,
    or the host glass in Task B. ``label`` holds per-atom LabelToken ids; the centre atom
    gets ``defect_token(k)`` and every other atom is UNLABELLED.
    """

    target: Tensor  # [B, 2] requested location p
    defect: Tensor  # [B] long, DefectClass values
    pin_mask: Tensor  # [B, N] bool
    pin_pos: Tensor  # [B, N, 2], meaningful where pin_mask is True
    label: Tensor  # [B, N] long, LabelToken ids

    def to(self, device: torch.device | str) -> Request:
        return Request(
            *(
                t.to(device)
                for t in (self.target, self.defect, self.pin_mask, self.pin_pos, self.label)
            )
        )
