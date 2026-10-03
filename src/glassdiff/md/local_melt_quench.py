"""Baseline B6 (Task B): re-melt and quench only a disk inside a frozen host glass.  [Ticket P-4]

Atoms farther than ``r_melt`` from the target are frozen (zero force and velocity). The
inner disk is heated to T_melt, quenched at the dataset rate, and the whole structure is
minimized with the host still frozen. Each call is one attempt; the cost per success is
the measured wall-clock time divided by the number of attempts that produce the defect.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from glassdiff.md.lammps_quench import _PAIR_COMMANDS, QuenchProtocol


def _write_data(path: Path, pos: np.ndarray, types: np.ndarray, box: np.ndarray) -> None:
    lines = [
        "local melt-quench input",
        "",
        f"{len(pos)} atoms",
        "2 atom types",
        "",
        f"0 {box[0]} xlo xhi",
        f"0 {box[1]} ylo yhi",
        "-0.5 0.5 zlo zhi",
        "",
        "Masses",
        "",
        "1 1.0",
        "2 1.0",
        "",
        "Atoms # atomic",
        "",
    ]
    wrapped = np.mod(pos, box)
    lines += [
        f"{i + 1} {t + 1} {x:.12f} {y:.12f} 0.0"
        for i, (t, (x, y)) in enumerate(zip(types, wrapped))
    ]
    path.write_text("\n".join(lines) + "\n")


def local_melt_quench(
    pos: np.ndarray,  # [N, 2]
    types: np.ndarray,  # [N] 0 = A, 1 = B
    box: np.ndarray,  # [2]
    target: np.ndarray,  # [2]
    r_melt: float,
    protocol: QuenchProtocol,
    seed: int,
    melt_steps: int | None = None,
) -> np.ndarray:
    """One attempt; returns the new positions [N, 2] (host atoms unchanged)."""
    from lammps import lammps

    d = pos - target
    d -= box * np.round(d / box)
    mobile = np.nonzero(np.linalg.norm(d, axis=1) <= r_melt)[0] + 1  # LAMMPS ids
    p = protocol
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "host.data"
        _write_data(data, pos, types, box)
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        try:
            lmp.commands_string(
                f"""
units lj
dimension 2
atom_style atomic
boundary p p p
atom_modify map array
read_data {data}
{_PAIR_COMMANDS}
neighbor 0.3 bin
fix e2d all enforce2d
group mobile id {" ".join(map(str, mobile))}
group host subtract all mobile
fix freeze host setforce 0.0 0.0 0.0
timestep {p.dt}
velocity mobile create {p.t_melt} {3 * seed + 7} dist gaussian
fix nvt mobile nvt temp {p.t_melt} {p.t_melt} {p.tdamp}
run {melt_steps if melt_steps is not None else p.melt_steps}
unfix nvt
fix nvt mobile nvt temp {p.t_melt} {p.t_final} {p.tdamp}
run {p.quench_steps()}
unfix nvt
min_style fire
minimize 0 {p.min_ftol} 100000 1000000
"""
            )
            new = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2]
        finally:
            lmp.close()
    # Undo LAMMPS wrapping relative to the input so host atoms keep their exact coordinates.
    shift = new - np.mod(pos, box)
    new = pos + shift - box * np.round(shift / box)
    return new
