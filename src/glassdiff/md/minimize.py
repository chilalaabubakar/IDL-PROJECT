"""Energy minimisation of given 2D Kob-Andersen structures with LAMMPS (FIRE).

Same potential and minimiser as the dataset's inherent structures (glassdiff.md.lammps_quench).
About 0.2 s per 256-atom glass on one core, roughly 15 times faster than the batched PyTorch
FIRE on four cores; both reach the same minimum (checked on defected test glasses).

Positions are returned unwrapped in the frame of the input: ``out - pos`` is the true
relaxation displacement. One LAMMPS instance per structure, one rank, one thread, so a result
does not depend on how the work is split across processes.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from glassdiff.md.lammps_quench import _PAIR_COMMANDS


def lammps_minimize(
    pos: np.ndarray,  # [N, 2]
    types: np.ndarray,  # [N] 0 = A, 1 = B
    box: np.ndarray,  # [2]
    ftol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Minimise all atoms; returns (positions [N, 2] in the input frame, final max force)."""
    from lammps import lammps

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
thermo_style custom step pe fmax
min_style fire
minimize 0 {ftol} 100000 1000000
run 0
"""
        )
        new = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2]
        ids = np.array(lmp.gather_atoms("id", 0, 1))
        fmax = float(lmp.get_thermo("fmax"))
    finally:
        lmp.close()
    if not np.array_equal(ids, np.arange(1, len(pos) + 1)):
        raise RuntimeError("LAMMPS reordered or lost atoms")
    shift = new - wrapped
    return pos + shift - box * np.round(shift / box), fmax


def _job(args):
    return lammps_minimize(*args)


def minimize_many(
    jobs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    workers: int | None = None,
    ftol: float = 1e-8,
) -> list[tuple[np.ndarray, float]]:
    """Minimise many (pos, types, box) structures, in parallel processes; keeps the order."""
    workers = workers or os.cpu_count() or 1
    args = [(p, t, b, ftol) for p, t, b in jobs]
    if workers == 1 or len(args) <= 1:
        return [_job(a) for a in args]
    # spawned, not forked: the parent may already run torch threads, and forking a
    # multi-threaded process can deadlock the child
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        return list(pool.map(_job, args, chunksize=max(1, len(args) // (8 * workers))))
