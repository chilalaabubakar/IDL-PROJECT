"""Thermal snapshots of 2D Kob-Andersen structures with LAMMPS (docs/NPJ_EXTENSIONS.md, item 3).

A relaxed structure is given velocities at temperature T and run at constant temperature
(Nose-Hoover, as the dataset's quench in glassdiff.md.lammps_quench) for a few time units. The
positions at the end are a snapshot of the vibrating structure. The snapshot is then
minimised, which tells whether the structure stayed in the same energy minimum (the
vibration around it) or moved to another one.

Positions are returned unwrapped in the frame of the input, as glassdiff.md.minimize does.
One LAMMPS instance per structure, one rank, one thread, so a result does not depend on how
the work is split across processes.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from glassdiff.md.lammps_quench import _PAIR_COMMANDS


def _unwrap(new: np.ndarray, wrapped: np.ndarray, pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    shift = new - wrapped
    return pos + shift - box * np.round(shift / box)


def lammps_thermal_snapshot(
    pos: np.ndarray,  # [N, 2], a relaxed structure
    types: np.ndarray,  # [N] 0 = A, 1 = B
    box: np.ndarray,  # [2]
    temperature: float,
    steps: int,
    seed: int,
    dt: float = 0.005,
    tdamp: float = 0.1,
    ftol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(snapshot [N, 2], minimised snapshot [N, 2], temperature at the end, final max force)."""
    from lammps import lammps

    if not 0 < seed < 900_000_000:
        raise ValueError("LAMMPS seeds must lie in (0, 9e8)")
    pos = np.asarray(pos, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    wrapped = np.mod(pos, box)
    xyz = np.zeros((len(pos), 3))
    xyz[:, :2] = wrapped
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units lj
dimension 2
atom_style atomic
boundary p p p
atom_modify map array
region box block 0 {box[0]} 0 {box[1]} -0.5 0.5
create_box 2 box
mass * 1.0
"""
        )
        lmp.create_atoms(
            len(pos), None, (np.asarray(types) + 1).astype(int).tolist(), xyz.reshape(-1).tolist()
        )
        lmp.commands_string(
            f"""{_PAIR_COMMANDS}
neighbor 0.3 bin
fix e2d all enforce2d
thermo_style custom step temp pe fmax
timestep {dt}
velocity all create {temperature} {seed} dist gaussian mom yes rot yes
fix nvt all nvt temp {temperature} {temperature} {tdamp}
run {steps}
"""
        )
        snap = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2].copy()
        temp_end = float(lmp.get_thermo("temp"))
        lmp.commands_string(f"unfix nvt\nmin_style fire\nminimize 0 {ftol} 100000 1000000\nrun 0")
        minimised = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2]
        ids = np.array(lmp.gather_atoms("id", 0, 1))
        fmax = float(lmp.get_thermo("fmax"))
    finally:
        lmp.close()
    if not np.array_equal(ids, np.arange(1, len(pos) + 1)):
        raise RuntimeError("LAMMPS reordered or lost atoms")
    return _unwrap(snap, wrapped, pos, box), _unwrap(minimised, wrapped, pos, box), temp_end, fmax


def _job(args):
    return lammps_thermal_snapshot(*args)


def thermal_snapshots_many(
    jobs: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]],
    workers: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Many (pos, types, box, temperature, steps, seed) runs in parallel processes; keeps the order."""
    workers = workers or os.cpu_count() or 1
    if workers == 1 or len(jobs) <= 1:
        return [_job(a) for a in jobs]
    ctx = multiprocessing.get_context("spawn")  # as glassdiff.md.minimize: no fork under torch
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        return list(pool.map(_job, jobs, chunksize=max(1, len(jobs) // (8 * workers))))
