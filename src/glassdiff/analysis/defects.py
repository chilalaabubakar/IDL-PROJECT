"""Defect detectors D1-, D1+, D2 (and D3, stretch).  [Ticket E-2]

Definitions and pilot frequencies: docs/DATASET.md §1.5. Thresholds live in
configs/defects/*.yaml; scripts/freeze_thresholds.py checks them on the train split,
writes them (with a hash) to data/<set>/meta.json, and labels every split. They are never
changed after model training starts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from glassdiff.analysis.descriptors import cutoff_coordination, same_species_neighbors
from glassdiff.types import DefectClass, Species, Structures


@dataclass(frozen=True)
class DefectThresholds:
    cn_factor: float = 1.4  # coordination cutoff = cn_factor * sigma_ab
    d1_minus_max_cn: int = 5  # A atom with CN <= this -> D1_MINUS
    d1_plus_min_cn: int = 8  # A atom with CN >= this -> D1_PLUS
    d2_rc: float = 1.2  # B-B neighbour cutoff
    d2_min_bb: int = 2  # B atom with >= this many B neighbours -> D2
    d3_min_radius: float | None = None  # train-split circumradius quantile (stretch)

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> DefectThresholds:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def detect_defects(s: Structures, th: DefectThresholds) -> Tensor:
    """[B, N] long DefectClass labels. An atom matching no definition is NONE."""
    cn = cutoff_coordination(s, th.cn_factor)
    bb = same_species_neighbors(s, Species.B, th.d2_rc)
    is_a = s.types == Species.A
    labels = torch.full_like(cn, int(DefectClass.NONE))
    labels[is_a & (cn <= th.d1_minus_max_cn)] = DefectClass.D1_MINUS
    labels[is_a & (cn >= th.d1_plus_min_cn)] = DefectClass.D1_PLUS
    labels[(s.types == Species.B) & (bb >= th.d2_min_bb)] = DefectClass.D2
    return labels


def defect_rates(labels: Tensor, types: Tensor) -> dict[str, float]:
    """Fraction of all atoms carrying each label, and of the species it applies to."""
    n_all = labels.numel()
    n_a = int((types == Species.A).sum())
    n_b = int((types == Species.B).sum())
    rates: dict[str, float] = {}
    for c in DefectClass:
        if c == DefectClass.NONE:
            continue
        count = int((labels == c).sum())
        per_species = n_b if c == DefectClass.D2 else n_a
        rates[f"{c.name}_of_all"] = count / n_all
        rates[f"{c.name}_of_species"] = count / max(per_species, 1)
        rates[f"{c.name}_per_glass"] = count / labels.shape[0]
    return rates
