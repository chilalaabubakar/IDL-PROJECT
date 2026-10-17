"""LAMMPS melt-quench of amorphous silica (docs/SILICA_3D_PLAN.md section 2.1).

Potential: Bouhadja, Jakse and Pasturel, J. Chem. Phys. 138, 224510 (2013), the
Born-Mayer-Huggins potential DM2's silica data was made with. Parameters from Table III of
arXiv:1403.1640 (kcal/mol, A), charges Si +2.4 and O -1.2, Coulomb by damped shifted force
(alpha 0.25 1/A), 8 A cutoffs. LAMMPS ``born/coul/dsf``:

    E(r) = A exp((sigma - r) / rho) - C / r^6 + D / r^8 + DSF Coulomb

Protocol (DM2 Methods; LAMMPS units real, 1 fs timestep):
    1. random placement at ``rho_init`` with a minimum distance, then energy minimisation
    2. 50 ps NVT and 50 ps NPT at 300 K
    3. 50 ps NPT at 5000 K (melt)
    4. NPT quench 5000 K -> 300 K at the chosen rate (K/ps)
    5. 100 ps NPT at 300 K
One LAMMPS instance per glass, one rank and one thread, so a seed always gives the same glass.
LAMMPS atom types: 1 = O, 2 = Si (as in DM2's files); our species index is type - 1.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from glassdiff.silica.data import MASS
from glassdiff.silica.defects import SilicaThresholds, si_o_bonds

# pair_coeff i j A rho sigma C D   (kcal/mol, A); 1 = O, 2 = Si
JAKSE_COEFFS = {
    (1, 1): (0.276344, 0.2630, 3.6430, 1959.372, 0.0),
    (1, 2): (0.16120, 0.1560, 2.5419, 1066.0667, 0.0),
    (2, 2): (0.0276344, 0.0460, 1.4408, 580.030, 0.0),
}
CHARGE = {1: -1.2, 2: 2.4}


def pair_commands(alpha: float = 0.25, cutoff: float = 8.0) -> list[str]:
    cmds = [f"pair_style born/coul/dsf {alpha} {cutoff}"]
    for (i, j), c in JAKSE_COEFFS.items():
        cmds.append(f"pair_coeff {i} {j} " + " ".join(f"{v:g}" for v in c))
    cmds += [f"set type {t} charge {q}" for t, q in CHARGE.items()]
    return cmds


@dataclass
class SilicaProtocol:
    n_sio2: int = 192  # 576 atoms
    rate: float = 10.0  # K/ps
    rho_init: float = 1.5  # g/cm^3 for the random placement
    min_dist: float = 1.8  # A, random placement
    dt: float = 1.0  # fs
    t_low: float = 300.0  # K
    t_melt: float = 5000.0  # K
    equil_nvt_ps: float = 50.0
    equil_npt_ps: float = 50.0
    melt_ps: float = 50.0
    final_ps: float = 100.0
    pressure: float = 1.0  # atm
    tdamp: float = 100.0  # fs
    pdamp: float = 1000.0  # fs
    alpha: float = 0.25  # 1/A, DSF damping
    cutoff: float = 8.0  # A

    def steps(self, ps: float) -> int:
        return int(round(ps * 1000.0 / self.dt))

    def quench_ps(self) -> float:
        return (self.t_melt - self.t_low) / self.rate

    def total_steps(self) -> int:
        ps = self.equil_nvt_ps + self.equil_npt_ps + self.melt_ps + self.quench_ps()
        return self.steps(ps + self.final_ps)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SilicaGlass:
    pos: np.ndarray  # [N, 3] wrapped into [0, L)
    types: np.ndarray  # [N] int64, 0 = O, 1 = Si
    box: np.ndarray  # [3]
    rate: float
    seed: int
    stats: dict = field(default_factory=dict)


def box_length(n_sio2: int, rho: float) -> float:
    mass = n_sio2 * (MASS["Si"] + 2 * MASS["O"])  # g/mol
    return (mass / 0.602214 / rho) ** (1 / 3)  # A


def _new_lammps():
    from lammps import lammps

    return lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def _setup(lmp, p: SilicaProtocol) -> None:
    lmp.commands_list(
        [
            "units real",
            "atom_style charge",
            "boundary p p p",
            "atom_modify map array",
        ]
    )


def _force_field(lmp, p: SilicaProtocol) -> None:
    lmp.commands_list(
        [
            f"mass 1 {MASS['O']}",
            f"mass 2 {MASS['Si']}",
            *pair_commands(p.alpha, p.cutoff),
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            f"timestep {p.dt}",
            "thermo 0",
        ]
    )


def _run_stage(lmp, fix: str, steps: int) -> None:
    if steps <= 0:
        return
    lmp.command(f"fix md all {fix}")
    lmp.command(f"run {steps}")
    lmp.command("unfix md")


def _snapshot(lmp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box_lo, box_hi, *_ = lmp.extract_box()
    box = np.array(box_hi) - np.array(box_lo)
    n = lmp.get_natoms()
    pos = np.array(lmp.gather_atoms("x", 1, 3), dtype=np.float64).reshape(n, 3)
    types = np.array(lmp.gather_atoms("type", 0, 1), dtype=np.int64)
    return np.mod(pos - np.array(box_lo), box), types - 1, box


def make_glass(seed: int, p: SilicaProtocol) -> SilicaGlass:
    """Melt-quench one glass. Takes ~0.6 h at 10 K/ps for 576 atoms on one core."""
    t0 = time.time()
    lmp = _new_lammps()
    try:
        _setup(lmp, p)
        length = box_length(p.n_sio2, p.rho_init)
        lmp.commands_list(
            [
                f"region cell block 0 {length} 0 {length} 0 {length}",
                "create_box 2 cell",
                f"create_atoms 2 random {p.n_sio2} {seed} NULL overlap {p.min_dist} maxtry 1000",
                f"create_atoms 1 random {2 * p.n_sio2} {seed + 7919} NULL "
                f"overlap {p.min_dist} maxtry 1000",
            ]
        )
        if lmp.get_natoms() != 3 * p.n_sio2:
            raise RuntimeError(f"placed {lmp.get_natoms()} of {3 * p.n_sio2} atoms")
        _force_field(lmp, p)
        lmp.command("minimize 1.0e-6 1.0e-8 1000 10000")
        lmp.command(f"velocity all create {p.t_low} {seed} dist gaussian mom yes rot yes")
        iso = f"iso {p.pressure} {p.pressure} {p.pdamp}"
        lo, hi = p.t_low, p.t_melt
        _run_stage(lmp, f"nvt temp {lo} {lo} {p.tdamp}", p.steps(p.equil_nvt_ps))
        _run_stage(lmp, f"npt temp {lo} {lo} {p.tdamp} {iso}", p.steps(p.equil_npt_ps))
        _run_stage(lmp, f"npt temp {hi} {hi} {p.tdamp} {iso}", p.steps(p.melt_ps))
        _run_stage(lmp, f"npt temp {hi} {lo} {p.tdamp} {iso}", p.steps(p.quench_ps()))
        _run_stage(lmp, f"npt temp {lo} {lo} {p.tdamp} {iso}", p.steps(p.final_ps))
        pe = lmp.get_thermo("pe") / lmp.get_natoms()  # kcal/mol per atom
        temp = lmp.get_thermo("temp")
        pos, types, box = _snapshot(lmp)
    finally:
        lmp.close()
    stats = {"pe_per_atom": pe, "temp": temp, "seconds": time.time() - t0}
    return SilicaGlass(pos, types, box, p.rate, seed, stats)


def npt_check(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, p: SilicaProtocol, ps: float, seed: int = 1
) -> SilicaGlass:
    """Run ``ps`` of NPT at t_low from a given glass (scripts/silica/check_potential.py)."""
    t0 = time.time()
    lmp = _new_lammps()
    try:
        _setup(lmp, p)
        lmp.commands_list(
            [
                f"region cell block 0 {box[0]} 0 {box[1]} 0 {box[2]}",
                "create_box 2 cell",
            ]
        )
        n = len(types)
        lmp.create_atoms(
            n,
            None,
            [int(t) + 1 for t in types],
            np.asarray(pos, dtype=np.float64).reshape(-1).tolist(),
        )
        _force_field(lmp, p)
        lmp.command(f"velocity all create {p.t_low} {seed} dist gaussian mom yes rot yes")
        iso = f"iso {p.pressure} {p.pressure} {p.pdamp}"
        _run_stage(lmp, f"npt temp {p.t_low} {p.t_low} {p.tdamp} {iso}", p.steps(ps))
        pe = lmp.get_thermo("pe") / lmp.get_natoms()
        temp = lmp.get_thermo("temp")
        out_pos, out_types, out_box = _snapshot(lmp)
    finally:
        lmp.close()
    stats = {"pe_per_atom": pe, "temp": temp, "seconds": time.time() - t0}
    return SilicaGlass(out_pos, out_types, out_box, p.rate, seed, stats)


def density(types: np.ndarray, box: np.ndarray) -> float:
    mass = np.where(types == 1, MASS["Si"], MASS["O"]).sum()
    return float(mass / 0.602214 / np.prod(box))


def structure_stats(
    pos: np.ndarray, types: np.ndarray, box: np.ndarray, th: SilicaThresholds = SilicaThresholds()
) -> dict:
    """Density, coordination fractions and the closest pair distance of one glass."""
    bonds = si_o_bonds(pos, types, box, th.si_o_cutoff)
    n = len(types)
    cn = np.bincount(bonds[:, 0], minlength=n) + np.bincount(bonds[:, 1], minlength=n)
    si, o = types == 1, types == 0
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.mod(pos, box), boxsize=box).query(np.mod(pos, box), k=2)
    return {
        "density": density(types, box),
        "si_cn4": float((cn[si] == 4).mean()),
        "o_cn2": float((cn[o] == 2).mean()),
        "min_dist": float(d[:, 1].min()),
    }


@dataclass
class SilicaGates:
    density_range: tuple[float, float] = (2.0, 2.4)  # g/cm^3 (DM2 glasses: 2.13-2.15)
    min_si_cn4: float = 0.95
    min_dist: float = 1.2  # A (Si-O bonds are 1.6 A)


def passes_gates(stats: dict, gates: SilicaGates) -> bool:
    lo, hi = gates.density_range
    return (
        lo <= stats["density"] <= hi
        and stats["si_cn4"] >= gates.min_si_cn4
        and stats["min_dist"] >= gates.min_dist
    )
