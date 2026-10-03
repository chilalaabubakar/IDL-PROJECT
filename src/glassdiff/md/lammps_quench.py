"""Melt-quench generation of 2D Kob–Andersen 65:35 glasses.  [Ticket P-1]

Protocol and quality gates: docs/DATASET.md §1.2 and §1.4. Ported from the feasibility
pilot (docs/pilot/pilot_ka2d.py). One LAMMPS instance per glass, single rank and single
thread, so a seed always reproduces the same glass.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# LAMMPS potential, matching glassdiff.physics.ka_potential (type 1 = A, type 2 = B).
_PAIR_COMMANDS = """
pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 1.0 1.0  2.5
pair_coeff 1 2 1.5 0.8  2.0
pair_coeff 2 2 0.5 0.88 2.2
"""


@dataclass
class QuenchProtocol:
    rho: float = 1.2
    frac_a: float = 0.65
    t_melt: float = 2.0
    melt_steps: int = 20_000
    t_final: float = 0.01
    rate: float = 1e-2  # temperature drop per tau
    dt: float = 0.005
    tdamp: float = 0.1
    min_ftol: float = 1e-8

    def quench_steps(self) -> int:
        return int(round((self.t_melt - self.t_final) / self.rate / self.dt))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityGates:
    max_global_psi6: float = 0.3
    min_pair_dist: float = 0.56  # 0.7 * sigma_AB
    max_force: float = 1e-6


@dataclass
class Glass:
    pos: np.ndarray  # [N, 2] wrapped into [0, L)
    types: np.ndarray  # [N] int8, 0 = A, 1 = B
    box: np.ndarray  # [2]
    pe_atom: np.ndarray  # [N] per-atom potential energy of the inherent structure
    seed: int
    max_force: float
    thermal_pos: np.ndarray | None = None  # [N, 2] snapshot at t_final before minimization


def n_species(n_atoms: int, frac_a: float) -> tuple[int, int]:
    n_a = int(round(frac_a * n_atoms))
    return n_a, n_atoms - n_a


def make_glass(n_atoms: int, seed: int, protocol: QuenchProtocol) -> Glass:
    """Run lattice start -> melt -> linear quench -> FIRE minimization for one glass."""
    from lammps import lammps

    side = int(round(math.sqrt(n_atoms)))
    if side * side != n_atoms:
        raise ValueError(f"n_atoms must be a perfect square, got {n_atoms}")
    _, n_b = n_species(n_atoms, protocol.frac_a)
    # LAMMPS seeds must be positive and distinct per random stream.
    seed_types, seed_vel = 3 * seed + 1, 3 * seed + 2
    p = protocol

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        script = f"""
units lj
dimension 2
atom_style atomic
boundary p p p
atom_modify map array
lattice sq {p.rho}
region box block 0 {side} 0 {side} -0.5 0.5
create_box 2 box
create_atoms 1 box
set type 1 type/subset 2 {n_b} {seed_types}
mass * 1.0
{_PAIR_COMMANDS}
neighbor 0.3 bin
fix e2d all enforce2d
compute pea all pe/atom
thermo_style custom step temp pe fmax
min_style fire
minimize 0 1e-6 10000 100000
timestep {p.dt}
velocity all create {p.t_melt} {seed_vel} dist gaussian
fix nvt all nvt temp {p.t_melt} {p.t_melt} {p.tdamp}
run {p.melt_steps}
unfix nvt
fix nvt all nvt temp {p.t_melt} {p.t_final} {p.tdamp}
run {p.quench_steps()}
unfix nvt
"""
        lmp.commands_string(script)
        thermal = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2].copy()
        lmp.commands_string(f"minimize 0 {p.min_ftol} 100000 1000000\nrun 0")
        pos = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2]
        types = np.array(lmp.gather_atoms("type", 0, 1), dtype=np.int8) - 1
        pe_atom = np.array(lmp.gather("c_pea", 1, 1), dtype=np.float64)
        max_force = float(lmp.get_thermo("fmax"))
        lo, hi = lmp.extract_box()[:2]
        box = np.array([hi[0] - lo[0], hi[1] - lo[1]])
    finally:
        lmp.close()
    return Glass(
        pos=np.mod(pos, box),
        types=types,
        box=box,
        pe_atom=pe_atom,
        seed=seed,
        max_force=max_force,
        thermal_pos=np.mod(thermal, box),
    )


def passes_quality_gates(
    glass: Glass, frac_a: float, gates: QualityGates
) -> tuple[bool, dict[str, float]]:
    """Composition, crystallinity (global |psi6|), minimum pair distance, residual force.

    The ensemble-level PE outlier gate is applied by scripts/make_dataset.py.
    """
    import torch

    from glassdiff.analysis.descriptors import psi6_global
    from glassdiff.geometry import pair_vectors
    from glassdiff.types import Structures

    n_a, n_b = n_species(len(glass.types), frac_a)
    s = Structures(
        torch.tensor(glass.pos)[None],
        torch.tensor(glass.types, dtype=torch.long)[None],
        torch.tensor(glass.box)[None],
    )
    dist = pair_vectors(s.pos, s.box).norm(dim=-1)[0]
    dist.fill_diagonal_(float("inf"))
    stats = {
        "n_b": float((glass.types == 1).sum()),
        "psi6_global": float(psi6_global(s)[0]),
        "min_pair_dist": float(dist.min()),
        "max_force": glass.max_force,
        "pe_mean": float(glass.pe_atom.mean()),
    }
    ok = (
        stats["n_b"] == n_b
        and stats["psi6_global"] < gates.max_global_psi6
        and stats["min_pair_dist"] > gates.min_pair_dist
        and stats["max_force"] < gates.max_force
    )
    return ok, stats
