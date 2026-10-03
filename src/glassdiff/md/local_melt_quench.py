"""Baseline B6 (Task B): re-melt and quench only a disk inside a frozen host glass.  [Ticket P-4]

Freeze every atom farther than ``r_melt`` from the target, heat the inner disk to T_melt,
quench at the dataset rate, minimize, and check for the requested defect. Repeat with new
seeds until success; the cost per success is the measured wall-clock time.
"""

from __future__ import annotations

from glassdiff.md.lammps_quench import QuenchProtocol
from glassdiff.types import Structures


def local_melt_quench(
    host: Structures,
    target: tuple[float, float],
    r_melt: float,
    protocol: QuenchProtocol,
    seed: int,
) -> Structures:
    """One local melt-quench attempt on a single host (batch size 1)."""
    raise NotImplementedError("Ticket P-4")
