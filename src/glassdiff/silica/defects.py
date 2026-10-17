"""Defect detectors for amorphous silica (docs/SILICA_3D_PLAN.md section 2.3).

Bonds are Si-O pairs closer than ``si_o_cutoff`` (2.2 A, near the first minimum of g_SiO).
Two Si atoms are linked when they share a bonded O.

    SI_III  Si with 3 O neighbours            (precursor of the E' centre)
    SI_V    Si with 5 or more O neighbours    (stress, densification)
    NBO     O with 1 Si neighbour             (non-bridging oxygen)
    O_III   O with 3 or more Si neighbours    (tricluster oxygen)
    RING2   Si in an edge-sharing pair: two Si sharing two O (a 2-membered ring)
    RING3   Si in a 3-membered ring: three Si linked pairwise through three distinct O

An atom matching several definitions keeps the first in this order (coordination defects
before rings, RING2 before RING3). Thresholds are frozen once on the training split
(scripts/silica/label_defects.py) and never changed afterwards.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import IntEnum
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

OXYGEN, SILICON = 0, 1  # species indices


class SilicaDefect(IntEnum):
    NONE = 0
    SI_III = 1
    SI_V = 2
    NBO = 3
    O_III = 4
    RING2 = 5
    RING3 = 6


# species that carries each defect label
DEFECT_SPECIES = {
    SilicaDefect.SI_III: SILICON,
    SilicaDefect.SI_V: SILICON,
    SilicaDefect.NBO: OXYGEN,
    SilicaDefect.O_III: OXYGEN,
    SilicaDefect.RING2: SILICON,
    SilicaDefect.RING3: SILICON,
}


@dataclass(frozen=True)
class SilicaThresholds:
    si_o_cutoff: float = 2.2

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def si_o_bonds(pos: np.ndarray, types: np.ndarray, box: np.ndarray, cutoff: float) -> np.ndarray:
    """[K, 2] (si, o) index pairs closer than cutoff under periodic boundaries."""
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = types[a] != types[b]
    a, b = a[keep], b[keep]
    si = np.where(types[a] == SILICON, a, b)
    o = np.where(types[a] == SILICON, b, a)
    return np.stack([si, o], axis=1)


def detect(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, th: SilicaThresholds = SilicaThresholds()
) -> np.ndarray:
    """[N] int64 SilicaDefect labels for one glass."""
    n = len(types)
    bonds = si_o_bonds(pos, types, box, th.si_o_cutoff)
    cn = np.bincount(bonds[:, 0], minlength=n) + np.bincount(bonds[:, 1], minlength=n)
    si_of_o: dict[int, list[int]] = defaultdict(list)
    for s, o in bonds:
        si_of_o[int(o)].append(int(s))
    # Si-Si links and the O atoms through which they are linked
    shared: dict[tuple[int, int], set[int]] = defaultdict(set)
    for o, sis in si_of_o.items():
        for a, b in combinations(sorted(sis), 2):
            shared[(a, b)].add(o)
    ring2 = {si for pair, os in shared.items() if len(os) >= 2 for si in pair}
    neighbours: dict[int, set[int]] = defaultdict(set)
    for a, b in shared:
        neighbours[a].add(b)
        neighbours[b].add(a)
    ring3: set[int] = set()
    for a, b in shared:
        for c in neighbours[a] & neighbours[b]:
            if c <= b:  # each triangle once (a < b < c)
                continue
            if _distinct_bridges(shared[(a, b)], shared[_key(b, c)], shared[_key(a, c)]):
                ring3.update((a, b, c))

    labels = np.zeros(n, dtype=np.int64)
    is_si, is_o = types == SILICON, types == OXYGEN
    labels[is_si & np.isin(np.arange(n), list(ring3))] = SilicaDefect.RING3
    labels[is_si & np.isin(np.arange(n), list(ring2))] = SilicaDefect.RING2
    labels[is_o & (cn >= 3)] = SilicaDefect.O_III
    labels[is_o & (cn == 1)] = SilicaDefect.NBO
    labels[is_si & (cn >= 5)] = SilicaDefect.SI_V
    labels[is_si & (cn == 3)] = SilicaDefect.SI_III
    return labels


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _distinct_bridges(ab: set[int], bc: set[int], ac: set[int]) -> bool:
    """True when one O can be chosen per edge with all three different (a real 3-ring, not
    three Si around one tricluster O)."""
    return any(x != y and y != z and x != z for x in ab for y in bc for z in ac)


def label_glasses(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, th: SilicaThresholds = SilicaThresholds()
) -> np.ndarray:
    """[B, N] labels for a stack of glasses (pos [B, N, 3], types [B, N], box [B, 3])."""
    return np.stack([detect(p, t, b, th) for p, t, b in zip(pos, types, box)])


def gate_counts(labels: np.ndarray, groups: dict[str, list[str]]) -> dict[str, int]:
    """Training instances per gate group, e.g. {"SI_V_or_III": ["SI_V", "SI_III"]}."""
    return {
        g: int(np.isin(labels, [SilicaDefect[c] for c in classes]).sum())
        for g, classes in groups.items()
    }


def coordination_summary(labels: np.ndarray, types: np.ndarray) -> dict[str, float]:
    """Count of each class per glass and its fraction of the species that carries it."""
    out = {}
    n_glass = labels.shape[0] if labels.ndim == 2 else 1
    for c in SilicaDefect:
        if c is SilicaDefect.NONE:
            continue
        count = int((labels == c).sum())
        n_species = int((types == DEFECT_SPECIES[c]).sum())
        out[f"{c.name}_count"] = count
        out[f"{c.name}_per_glass"] = count / n_glass
        out[f"{c.name}_of_species"] = count / max(n_species, 1)
    return out
