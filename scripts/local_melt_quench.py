"""Baseline B6 (Task B): local melt-quench attempts in test-split host glasses.  [Ticket P-4]

    python scripts/local_melt_quench.py --config configs/data/ka2d_256.yaml data=data/ka2d_256 \\
        defect=D2 n_samples=256 r_melt=4.0 workers=4

Each sample is one attempt at the same kind of request as Task B ("host" mode: the centre
is the host atom of the defect's species nearest a random point). Writes samples.npz in
the format of scripts/sample.py, plus timing.json with the summed per-attempt CPU seconds,
so scripts/relax.py and scripts/evaluate.py apply unchanged.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.dataset import load_split
from glassdiff.data.requests import make_eval_request
from glassdiff.md.lammps_quench import QuenchProtocol
from glassdiff.md.local_melt_quench import local_melt_quench
from glassdiff.types import DefectClass, Structures
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import make_run_dir, seed_everything


def _attempt(args):
    pos, types, box, target, r_melt, protocol, seed, melt_steps = args
    t0 = time.time()
    new = local_melt_quench(
        pos, types, box, target, r_melt, QuenchProtocol(**protocol), seed, melt_steps
    )
    return new, time.time() - t0


def main() -> None:
    cfg = cli_config(__doc__)
    gen = seed_everything(int(cfg.get("seed", 0)))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    defect = DefectClass[cfg["defect"]] if isinstance(cfg["defect"], str) else cfg["defect"]
    run_dir = make_run_dir(
        cfg.get("run_name") or f"local_melt_quench_{DefectClass(defect).name}",
        cfg,
        root=cfg.get("runs_root", "runs"),
    )
    test = load_split(Path(cfg["data"]) / "test.npz").structures
    n = int(cfg.get("n_samples", 256))
    idx = torch.arange(n) % test.batch_size
    host = Structures(test.pos[idx], test.types[idx], test.box[idx])
    target = torch.rand(n, 2, generator=gen, dtype=host.pos.dtype) * host.box
    r_melt = float(cfg.get("r_melt", 4.0))
    request = make_eval_request(
        int(defect), "host", host.types, host.box, host=host, target=target, host_r_out=r_melt
    )
    protocol = QuenchProtocol(**cfg["protocol"]).to_dict()
    jobs = [
        (
            host.pos[i].numpy(),
            host.types[i].numpy(),
            host.box[i].numpy(),
            request.target[i].numpy(),
            r_melt,
            protocol,
            int(cfg.get("seed", 0)) * 100_000 + i,
            cfg.get("melt_steps"),
        )
        for i in range(n)
    ]
    with mp.get_context("spawn").Pool(int(cfg.get("workers", os.cpu_count() or 1))) as pool:
        results = pool.map(_attempt, jobs)
    pos = np.stack([r[0] for r in results])
    seconds = float(sum(r[1] for r in results))
    np.savez_compressed(
        run_dir / "samples.npz",
        defect=np.int64(defect),
        mode="local_melt_quench",
        pos=pos,
        types=host.types.numpy(),
        box=host.box.numpy(),
        target=request.target.numpy(),
        pin_mask=request.pin_mask.numpy(),
        pin_pos=request.pin_pos.numpy(),
        label=request.label.numpy(),
    )
    timing = {"sampling_seconds": seconds, "seconds_per_sample": seconds / n, "device": "cpu"}
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"wrote {run_dir / 'samples.npz'} ({seconds / n:.2f} CPU-s per attempt)")


if __name__ == "__main__":
    main()
