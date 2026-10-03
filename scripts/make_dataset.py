"""Generate a glass dataset with LAMMPS.  [Ticket P-1]

    python scripts/make_dataset.py --config configs/data/ka2d_256.yaml [workers=4]

For each split: generate glasses in parallel (one LAMMPS process per glass), apply the
per-glass quality gates and the ensemble PE-outlier gate, replace rejected glasses with the
next unused seeds, then save <out_dir>/<split>.npz and <out_dir>/meta.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.dataset import GlassSplit, save_split
from glassdiff.md.lammps_quench import (
    Glass,
    QualityGates,
    QuenchProtocol,
    make_glass,
    passes_quality_gates,
)
from glassdiff.types import Structures
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import git_state


def _work(args: tuple[int, int, dict, dict]) -> tuple[Glass, bool, dict]:
    n_atoms, seed, protocol, gates = args
    p = QuenchProtocol(**protocol)
    glass = make_glass(n_atoms, seed, p)
    ok, stats = passes_quality_gates(glass, p.frac_a, QualityGates(**gates))
    return glass, ok, stats


def _pe_outliers(glasses: list[Glass], max_z: float) -> list[int]:
    """Indices whose mean PE is more than max_z robust SDs from the median."""
    pe = np.array([g.pe_atom.mean() for g in glasses])
    med = np.median(pe)
    sd = 1.4826 * np.median(np.abs(pe - med)) or pe.std() or 1.0
    return [i for i, v in enumerate(pe) if abs(v - med) / sd > max_z]


def generate_split(cfg: dict, n: int, seed0: int, pool, log: list[dict]) -> list[Glass]:
    protocol = QuenchProtocol(**cfg["protocol"]).to_dict()
    quality = dict(cfg.get("quality", {}))
    pe_max_z = quality.pop("pe_max_z", 5.0)
    gates = {k: v for k, v in quality.items() if k in ("max_global_psi6", "min_pair_dist")}
    accepted: list[Glass] = []
    next_seed = seed0
    while len(accepted) < n:
        seeds = list(range(next_seed, next_seed + n - len(accepted)))
        next_seed += len(seeds)
        jobs = [(cfg["n_atoms"], s, protocol, gates) for s in seeds]
        for glass, ok, stats in pool.imap(_work, jobs):
            if ok:
                accepted.append(glass)
            else:
                log.append({"seed": glass.seed, "reason": "quality_gate", **stats})
        if len(accepted) == n:
            for i in sorted(_pe_outliers(accepted, pe_max_z), reverse=True):
                g = accepted.pop(i)
                log.append({"seed": g.seed, "reason": "pe_outlier", "pe_mean": g.pe_atom.mean()})
    return sorted(accepted, key=lambda g: g.seed)


def to_split(glasses: list[Glass], save_thermal: bool) -> GlassSplit:
    return GlassSplit(
        structures=Structures(
            torch.tensor(np.stack([g.pos for g in glasses])),
            torch.tensor(np.stack([g.types for g in glasses]), dtype=torch.long),
            torch.tensor(np.stack([g.box for g in glasses])),
        ),
        pe_atom=torch.tensor(np.stack([g.pe_atom for g in glasses])),
        labels=None,
        seed=torch.tensor([g.seed for g in glasses]),
        thermal_pos=torch.tensor(np.stack([g.thermal_pos for g in glasses]))
        if save_thermal
        else None,
    )


def main() -> None:
    cfg = cli_config(__doc__)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    workers = int(cfg.get("workers", os.cpu_count() or 1))

    from lammps import lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lammps_version = lmp.version()
    lmp.close()

    meta = {
        "name": cfg["name"],
        "n_atoms": cfg["n_atoms"],
        "protocol": QuenchProtocol(**cfg["protocol"]).to_dict(),
        "quality": cfg.get("quality", {}),
        "lammps_version": lammps_version,
        "git": git_state(),
        "splits": {},
    }
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for split_name, spec in cfg["splits"].items():
            t0 = time.time()
            log: list[dict] = []
            glasses = generate_split(cfg, spec["n"], spec["seed0"], pool, log)
            save_split(out_dir / f"{split_name}.npz", to_split(glasses, cfg.get("save_thermal")))
            pe = np.array([g.pe_atom.mean() for g in glasses])
            meta["splits"][split_name] = {
                "n": len(glasses),
                "seeds": [int(glasses[0].seed), int(glasses[-1].seed)],
                "pe_atom_mean": float(pe.mean()),
                "pe_atom_sd": float(pe.std()),
                "rejected": log,
                "wall_seconds": round(time.time() - t0, 1),
            }
            print(
                f"{split_name}: {len(glasses)} glasses, {len(log)} rejected, "
                f"PE/atom {pe.mean():.4f} +- {pe.std():.4f}, {time.time() - t0:.0f}s",
                flush=True,
            )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    main()
