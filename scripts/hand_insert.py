"""Baseline B2: hand insertion of library patches into test-split host glasses.  [Ticket B-2]

    python scripts/hand_insert.py --config configs/eval/default.yaml data=data/ka2d_256 \\
        defect=D2 n_samples=256

Writes a run directory with samples.npz in the same format as scripts/sample.py (positions
BEFORE relaxation), so scripts/relax.py and scripts/evaluate.py apply unchanged. Evaluated
with ``also_unrelaxed=true`` it also gives the "ceiling" row (insertion before relaxation).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.baselines import hand_insert
from glassdiff.data.dataset import load_split
from glassdiff.data.patches import PatchLibrary
from glassdiff.data.requests import make_eval_request
from glassdiff.types import DefectClass, Structures
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import make_run_dir, seed_everything


def main() -> None:
    cfg = cli_config(__doc__)
    gen = seed_everything(int(cfg.get("seed", 0)))
    defect = DefectClass[cfg["defect"]] if isinstance(cfg["defect"], str) else cfg["defect"]
    run_dir = make_run_dir(
        cfg.get("name") or f"hand_insert_{DefectClass(defect).name}",
        cfg,
        root=cfg.get("runs_root", "runs"),
    )
    data_dir = Path(cfg["data"])
    test = load_split(data_dir / "test.npz").structures
    patches = PatchLibrary.load(data_dir / "patches_train.npz")
    n = int(cfg.get("n_samples", 256))
    idx = torch.arange(n) % test.batch_size
    host = Structures(test.pos[idx], test.types[idx], test.box[idx])
    # Random target per sample; the request pins nothing (hand insertion is not generative).
    target = torch.rand(n, 2, generator=gen, dtype=host.pos.dtype) * host.box
    request = make_eval_request(int(defect), "host", host.types, host.box, host=host, target=target)
    t0 = time.time()
    chosen = [patches.sample(int(defect), generator=gen) for _ in range(n)]
    inserted = hand_insert(host, request.target, chosen)
    seconds = time.time() - t0
    np.savez_compressed(
        run_dir / "samples.npz",
        defect=np.int64(defect),
        mode="hand_insert",
        pos=inserted.pos.numpy(),
        types=inserted.types.numpy(),
        box=inserted.box.numpy(),
        target=request.target.numpy(),
        pin_mask=np.zeros((n, host.n_atoms), dtype=bool),
        pin_pos=inserted.pos.numpy(),
        label=request.label.numpy(),
    )
    (run_dir / "timing.json").write_text(
        json.dumps({"sampling_seconds": seconds, "seconds_per_sample": seconds / n}, indent=2)
    )
    print(f"wrote {run_dir / 'samples.npz'}")


if __name__ == "__main__":
    main()
