"""Matched D2 radius-6 repeat across three trained models and a null-label sham.

CPU pilot: python scripts/repeat_credibility.py --out runs/credibility_repeat --n 8 --workers 4
Larger GPU run: use --n 256 --workers 1 --device cuda (one model at a time).
Existing completed batch files are reused only if their complete configuration matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds
from glassdiff.data.requests import host_init
from glassdiff.diffusion.conditioning import PinnedLabel
from glassdiff.diffusion.guidance import CFGDenoiser
from glassdiff.diffusion.sampler import ScoreDynamicsSchedule, sample
from glassdiff.eval.metrics import success
from glassdiff.models.registry import load_denoiser
from glassdiff.physics.fire import fire_minimize
from glassdiff.types import LabelToken, Request, Structures
from glassdiff.utils.runs import seed_everything

SOURCE = "results/kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout6_host_D2"
MODELS = {
    "seed0": "results/kaggle_20261006-0454Z/training/egnn_cond_smax1/ema.pt",
    "seed1": "results/kaggle_20261013-1240Z/training/egnn_cond_smax1_seed1/ema.pt",
    "seed2": "results/kaggle_20261013-1750Z/training/egnn_cond_smax1_seed2/ema.pt",
    "sham_seed0": "results/kaggle_20261006-0454Z/training/egnn_cond_smax1/ema.pt",
}


def worker(job):
    name, checkpoint, n, output, device, batch_size = job
    torch.set_num_threads(1)
    out = Path(output) / name
    out.mkdir(parents=True, exist_ok=True)
    config = dict(
        name=name,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        n=n,
        device=device,
        source=SOURCE,
        batch_size=batch_size,
        n_noisy=100,
        n_final=100,
        w=2,
        radius=6,
        batch_seed_rule="1000 + batch_start",
        null_label=name.startswith("sham"),
    )
    cfgpath = out / "config.json"
    if cfgpath.exists() and json.loads(cfgpath.read_text()) != config:
        raise ValueError(f"Configuration mismatch: {out}")
    cfgpath.write_text(json.dumps(config, indent=2) + "\n")
    model = load_denoiser(checkpoint, device)
    if not config["null_label"]:
        model = CFGDenoiser(model, 2)
    with np.load(Path(SOURCE) / "samples.npz") as f:
        source = {k: f[k] for k in f.files}
    if n > len(source["pos"]):
        raise ValueError("n exceeds retained matched request count")
    threshold = DefectThresholds.from_dict(
        json.loads(Path("data/ka2d_256/meta.json").read_text())["defects"]["thresholds"]
    )
    started = time.monotonic()
    for start in range(0, n, batch_size):
        file = out / f"batch_{start:03d}.npz"
        if file.exists():
            continue
        end = min(start + batch_size, n)
        sl = slice(start, end)
        host = Structures(
            torch.from_numpy(source["pin_pos"][sl]).float().to(device),
            torch.from_numpy(source["types"][sl]).long().to(device),
            torch.from_numpy(source["box"][sl]).float().to(device),
        )
        label = torch.from_numpy(source["label"][sl]).long().to(device)
        if config["null_label"]:
            label = torch.where(
                label == LabelToken.UNLABELLED, label, torch.full_like(label, int(LabelToken.NULL))
            )
        req = Request(
            target=torch.from_numpy(source["target"][sl]).float().to(device),
            defect=torch.full((end - start,), 3, dtype=torch.long, device=device),
            pin_mask=torch.from_numpy(source["pin_mask"][sl]).bool().to(device),
            pin_pos=host.pos.clone(),
            label=label,
        )
        gen = seed_everything(1000 + start)
        init = host_init(host, req, gen)
        with torch.no_grad():
            result, _ = sample(
                model,
                init,
                ScoreDynamicsSchedule(n_noisy=100, n_final=100, model_sigma_max=1.0),
                PinnedLabel(),
                req,
                gen,
            )
            cpu = Structures(result.pos.cpu().double(), host.types.cpu(), host.box.cpu().double())
            relax = fire_minimize(cpu, fmax=1e-6)
            reqcpu = Request(
                req.target.cpu().double(),
                req.defect.cpu(),
                req.pin_mask.cpu(),
                req.pin_pos.cpu().double(),
                req.label.cpu(),
            )
            ok = success(relax.structures, reqcpu, threshold).numpy()
        if not bool(relax.converged.all()):
            raise RuntimeError(
                f"Unconverged relaxation in {name}, batch {start}: {relax.max_force.tolist()}"
            )
        np.savez_compressed(
            file,
            pos=relax.structures.pos.numpy(),
            pre_pos=cpu.pos.numpy(),
            types=cpu.types.numpy(),
            box=cpu.box.numpy(),
            target=reqcpu.target.numpy(),
            host_pos=host.pos.cpu().numpy(),
            host_ids=np.arange(start, end) % 100,
            pin_mask=reqcpu.pin_mask.numpy(),
            success=ok,
            converged=relax.converged.numpy(),
            max_force=relax.max_force.numpy(),
        )
        print(
            f"{name}: {end}/{n}; batch successes {int(ok.sum())}; elapsed {(time.monotonic() - started) / 60:.1f} min",
            flush=True,
        )
    batches = []
    for start in range(0, n, batch_size):
        with np.load(out / f"batch_{start:03d}.npz") as f:
            batches.append({k: f[k] for k in f.files})
    np.savez_compressed(
        out / "samples.npz", **{k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    )
    return name, int(sum(b["success"].sum() for b in batches)), n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    a = p.parse_args()
    if not 1 <= a.n <= 256 or a.workers < 1 or a.batch_size < 1:
        p.error("n must be 1..256 and workers positive")
    jobs = [(name, MODELS[name], a.n, a.out, a.device, a.batch_size) for name in a.models]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        result = list(pool.map(worker, jobs))
    filename = (
        "summary.json"
        if len(a.models) == len(MODELS)
        else "summary_" + "_".join(a.models) + ".json"
    )
    Path(a.out, filename).write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
