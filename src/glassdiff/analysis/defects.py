"""Defect detectors D1-, D1+, D2 (and D3, stretch).  [Ticket E-2]

Definitions and pilot frequencies: docs/DATASET.md §1.5. Thresholds are fitted on the
train split by scripts/freeze_thresholds.py, written to data/<set>/meta.json, and never
changed after model training starts.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from glassdiff.types import Structures


@dataclass(frozen=True)
class DefectThresholds:
    cn_factor: float = 1.4  # coordination cutoff = cn_factor * sigma_ab
    d1_minus_max_cn: int = 5  # A atom with CN <= this -> D1_MINUS
    d1_plus_min_cn: int = 8  # A atom with CN >= this -> D1_PLUS
    d2_rc: float = 1.2  # B-B neighbour cutoff
    d2_min_bb: int = 2  # B atom with >= this many B neighbours -> D2
    d3_min_radius: float | None = None  # train-split 99.9% circumradius quantile (stretch)


def detect_defects(s: Structures, th: DefectThresholds) -> Tensor:
    """[B, N] long DefectClass labels. An atom matching no definition is NONE."""
    raise NotImplementedError("Ticket E-2")


def defect_rates(labels: Tensor, types: Tensor) -> dict[str, float]:
    """Fraction of atoms (and of the relevant species) carrying each label."""
    raise NotImplementedError("Ticket E-2")
