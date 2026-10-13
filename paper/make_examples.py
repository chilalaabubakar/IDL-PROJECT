"""Example Task B samples for the paper's overview and comparison figures (CPU, run from the
repository root).

    python paper/make_examples.py part=model    # ~20 min on 4 CPU cores
    python paper/make_examples.py part=hand
    python paper/make_examples.py part=lmq      # needs LAMMPS

All three parts answer the SAME D2 requests (the first n test glasses as hosts, sites drawn
with seed 7), so the figure can compare methods on one host and site. ``model`` runs the
sigma_max = 1 checkpoint at w = 6 with the paper's sampler (100 noisy + 100 final steps) and
keeps trajectory snapshots; every part relaxes its samples with FIRE (f_max 1e-6, all
atoms free) and records success. Writes paper/data/examples_D2_<part>.npz.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.data.dataset import load_split
from glassdiff.data.requests import make_eval_request
from glassdiff.eval.metrics import success
from glassdiff.physics.fire import fire_minimize
from glassdiff.types import DefectClass, Structures

OUT = Path("paper/data")
CKPT = "results/kaggle_20261006-0454Z/training/egnn_cond_smax1/ema.pt"
SNAPSHOTS = (10, 25, 50, 75, 100, 150, 200)  # sampler steps kept from the trajectory
SEED, R_OUT, W = 7, 4.0, 6.0


def requests(n: int, gen: torch.Generator):
    test = load_split("data/ka2d_256/test.npz").structures
    idx = torch.arange(n)
    types, box = test.types[idx], test.box[idx].float()
    host = Structures(test.pos[idx].float(), types, box)
    req = make_eval_request(
        int(DefectClass.D2), "host", types, box, host=host, host_r_out=R_OUT, generator=gen
    )
    return host, req


def relax_and_score(pos: torch.Tensor, host: Structures, req) -> dict:
    s = Structures(pos.double(), host.types, host.box.double())
    res = fire_minimize(s, fmax=1e-6, max_steps=100_000)
    meta = json.loads(Path("data/ka2d_256/meta.json").read_text())
    th = DefectThresholds.from_dict(meta["defects"]["thresholds"])
    return {
        "relaxed": res.structures.pos.numpy(),
        "labels_relaxed": detect_defects(res.structures, th).numpy(),
        "success": success(res.structures, req, th).numpy(),
        "converged": res.converged.numpy(),
    }


def main() -> None:
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    part, n = args.get("part", "model"), int(args.get("n", 6))
    torch.set_num_threads(int(args.get("threads", 4)))
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    host, req = requests(n, gen)
    out: dict = {
        "host": host.pos.numpy(),
        "types": host.types.numpy(),
        "box": host.box.numpy(),
        "target": req.target.numpy(),
        "pin": req.pin_mask.numpy(),
    }
    t0 = time.time()
    if part == "model":
        from glassdiff.data.requests import host_init
        from glassdiff.diffusion.conditioning import build_strategy
        from glassdiff.diffusion.guidance import CFGDenoiser
        from glassdiff.diffusion.sampler import ScoreDynamicsSchedule, sample
        from glassdiff.models.registry import load_denoiser

        model = CFGDenoiser(load_denoiser(CKPT, "cpu"), W)
        strategy = build_strategy({"name": "pinned_label", "guidance_w": W})
        schedule = ScoreDynamicsSchedule(
            n_noisy=100,
            n_final=100,
            sigma_start=1.0,
            sigma_end=1e-3,
            sigma_min=1e-3,
            model_sigma_max=1.0,
        )
        init = host_init(host, req, gen)
        final, traj = sample(model, init, schedule, strategy, req, generator=gen, save_every=1)
        out["init"] = init.pos.numpy()
        out["snap_steps"] = np.array(SNAPSHOTS)
        out["snapshots"] = torch.stack([traj[k - 1] for k in SNAPSHOTS], 1).numpy()
        out["sigmas"] = schedule.sigmas().numpy()
        pos = final.pos
    elif part == "hand":
        from glassdiff.baselines import hand_insert
        from glassdiff.data.patches import PatchLibrary

        patches = PatchLibrary.load("data/ka2d_256/patches_train.npz")
        chosen = [patches.sample(int(DefectClass.D2), generator=gen) for _ in range(n)]
        pos = hand_insert(host, req.target, chosen).pos
    elif part == "lmq":
        import yaml

        from glassdiff.md.lammps_quench import QuenchProtocol
        from glassdiff.md.local_melt_quench import local_melt_quench

        protocol = QuenchProtocol(
            **yaml.safe_load(Path("configs/data/ka2d_256.yaml").read_text())["protocol"]
        )
        pos = torch.from_numpy(
            np.stack(
                [
                    local_melt_quench(
                        host.pos[i].double().numpy(),
                        host.types[i].numpy(),
                        host.box[i].double().numpy(),
                        req.target[i].double().numpy(),
                        R_OUT,
                        protocol,
                        SEED * 100_000 + i,
                        None,
                    )
                    for i in range(n)
                ]
            )
        )
    else:
        raise ValueError(part)
    out["sample"] = pos.numpy()
    out.update(relax_and_score(pos, host, req))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"examples_D2_{part}.npz", **out)
    print(part, "success", out["success"].tolist(), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
