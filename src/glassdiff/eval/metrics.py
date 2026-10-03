"""Evaluation metrics (docs/PROJECT_PLAN.md §4).  [Ticket E-4]

Every metric is computed on RELAXED samples (physics.fire.fire_minimize, no pins), and
returns one value per sample, so bootstrap CIs resample samples, never atoms.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from glassdiff.analysis.defects import DefectThresholds
from glassdiff.types import Request, Structures


def success(
    relaxed: Structures, request: Request, th: DefectThresholds, r_tol: float = 0.5
) -> Tensor:
    """[B] bool: the atom of the requested species nearest the target lies within r_tol of it
    and carries the requested defect label after relaxation."""
    raise NotImplementedError("Ticket E-4")


@dataclass
class LocalFeatures:
    """Per-sample feature lists inside the window of radius r_loc around the target."""

    pe_atom: list[Tensor]
    pair_dist: dict[str, list[Tensor]]  # "AA", "AB", "BB"
    bond_angle: list[Tensor]
    relax_disp: list[Tensor]


def local_features(
    relaxed: Structures,
    centres: Tensor,  # [B, 2]
    relax_disp: Tensor,  # [B, N, 2]
    r_loc: float = 3.0,
) -> LocalFeatures:
    raise NotImplementedError("Ticket E-4")


def wasserstein1(x: Tensor, y: Tensor) -> Tensor:
    """1D Wasserstein-1 distance between two empirical samples."""
    raise NotImplementedError("Ticket E-4")


def local_realism(generated: LocalFeatures, reference: LocalFeatures) -> dict[str, float]:
    """W1 per feature (and tail metrics) of generated vs natural-defect windows."""
    raise NotImplementedError("Ticket E-4")


def global_realism(generated: Structures, reference: Structures) -> dict[str, float]:
    """g_ab(r) distances, PE/atom W1, coordination histograms, global |psi6|."""
    raise NotImplementedError("Ticket E-4")


def diversity(generated: LocalFeatures, train_patches) -> dict[str, float]:
    """Mean pairwise descriptor distance; nearest-training-patch distance (memorization)."""
    raise NotImplementedError("Ticket E-4")


def cost_per_success(seconds: float, n_success: int) -> float:
    return float("inf") if n_success == 0 else seconds / n_success
