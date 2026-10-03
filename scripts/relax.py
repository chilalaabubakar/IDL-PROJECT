"""Relax samples with batched FIRE (all atoms free) and save displacements.  [Ticket P-3]

    python scripts/relax.py --config configs/eval/default.yaml run=runs/<run>

Reads <run>/samples.npz, writes <run>/relaxed.npz (positions, displacement, convergence)
and adds the relaxation time to <run>/timing.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.physics.fire import fire_minimize
from glassdiff.types import Structures
from glassdiff.utils.config import cli_config


def main() -> None:
    cfg = cli_config(__doc__)
    run = Path(cfg["run"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    relax = cfg.get("relax", {})
    batch_size = int(cfg.get("batch_size", 32))
    with np.load(run / "samples.npz") as f:
        pos = torch.from_numpy(f["pos"]).double()
        types = torch.from_numpy(f["types"]).long()
        box = torch.from_numpy(f["box"]).double()

    out = {k: [] for k in ("pos", "disp", "converged", "max_force", "n_steps")}
    t0 = time.time()
    for i in range(0, len(pos), batch_size):
        s = Structures(pos[i : i + batch_size], types[i : i + batch_size], box[i : i + batch_size])
        res = fire_minimize(
            s.to(device),
            fmax=float(relax.get("fmax", 1e-6)),
            max_steps=int(relax.get("max_steps", 100_000)),
        )
        out["pos"].append(res.structures.pos.cpu())
        out["disp"].append(res.displacement.cpu())
        out["converged"].append(res.converged.cpu())
        out["max_force"].append(res.max_force.cpu())
        out["n_steps"].append(res.n_steps.cpu())
        print(
            f"relaxed {min(i + batch_size, len(pos))}/{len(pos)}: "
            f"{int(res.converged.sum())}/{len(res.converged)} converged, "
            f"median {int(res.n_steps.median())} steps",
            flush=True,
        )
    seconds = time.time() - t0
    np.savez_compressed(run / "relaxed.npz", **{k: torch.cat(v).numpy() for k, v in out.items()})
    timing_path = run / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing.update({"relax_seconds": seconds, "relax_device": str(device)})
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"relaxation took {seconds:.0f}s -> {run / 'relaxed.npz'}")


if __name__ == "__main__":
    main()
