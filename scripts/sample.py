"""Generate samples with a trained model and a conditioning strategy.  [Tickets S-1..S-4, B-2]

    python scripts/sample.py --config configs/sampler/dm2.yaml ckpt=runs/<run>/ckpt.pt \\
        data=data/ka2d_256 defect=D2
    python scripts/sample.py --config configs/sampler/cfg.yaml ckpt=... defect=D1_MINUS \\
        request.mode=host strategy.guidance_w=2

Species and boxes are taken from the test split (cycled), so every sample has the dataset's
composition. Task B ("host" requests) uses test glasses as hosts. Writes a run directory
with samples.npz (positions, types, boxes, the requests) and timing.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.dataset import load_split
from glassdiff.data.patches import PatchLibrary
from glassdiff.data.requests import host_init, make_eval_request
from glassdiff.diffusion.conditioning import build_strategy
from glassdiff.diffusion.guidance import CFGDenoiser
from glassdiff.diffusion.sampler import ScoreDynamicsSchedule, random_init, sample
from glassdiff.models.registry import load_denoiser
from glassdiff.types import DefectClass, Structures
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import make_run_dir, pick_device, seed_everything


def main() -> None:
    cfg = cli_config(__doc__)
    gen = seed_everything(int(cfg.get("seed", 0)))
    device = pick_device(cfg.get("device", "auto"))
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    defect = DefectClass[cfg["defect"]] if isinstance(cfg["defect"], str) else cfg["defect"]
    mode = cfg.get("request", {}).get("mode", "centre")
    name = cfg.get("name") or f"sample_{cfg['strategy']['name']}_{mode}_{DefectClass(defect).name}"
    run_dir = make_run_dir(name, cfg, root=cfg.get("runs_root", "runs"))
    print(f"run directory: {run_dir}", flush=True)

    model = load_denoiser(cfg["ckpt"], device)
    w = float(cfg["strategy"].get("guidance_w", 0.0))
    if w:
        model = CFGDenoiser(model, w)
    strategy = build_strategy(cfg["strategy"])
    schedule = ScoreDynamicsSchedule(**cfg["schedule"])

    data_dir = Path(cfg["data"])
    test = load_split(data_dir / "test.npz").structures
    patches = PatchLibrary.load(data_dir / "patches_train.npz") if mode == "patch" else None
    n_samples, batch_size = int(cfg["n_samples"]), int(cfg["batch_size"])

    out = {k: [] for k in ("pos", "types", "box", "target", "pin_mask", "pin_pos", "label")}
    seconds = []
    for start in range(0, n_samples, batch_size):
        idx = torch.arange(start, min(start + batch_size, n_samples)) % test.batch_size
        types = test.types[idx].to(device)
        box = test.box[idx].to(device=device, dtype=torch.float32)
        host = None
        if mode == "host":
            host = Structures(test.pos[idx].to(device, torch.float32), types, box)
        request = make_eval_request(
            int(defect),
            mode,
            types,
            box,
            host=host,
            patches=patches,
            host_r_out=float(cfg.get("request", {}).get("host_r_out", 4.0)),
            generator=gen,
        )
        init = host_init(host, request, gen) if host is not None else random_init(types, box, gen)
        t0 = time.time()
        result, _ = sample(model, init, schedule, strategy, request, generator=gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.time() - t0)
        print(f"samples {start + len(idx)}/{n_samples}: {seconds[-1]:.1f}s", flush=True)
        for key, val in (
            ("pos", result.pos),
            ("types", types),
            ("box", box),
            ("target", request.target),
            ("pin_mask", request.pin_mask),
            ("pin_pos", request.pin_pos),
            ("label", request.label),
        ):
            out[key].append(val.detach().cpu())

    arrays = {k: torch.cat(v).numpy() for k, v in out.items()}
    np.savez_compressed(run_dir / "samples.npz", defect=np.int64(defect), mode=mode, **arrays)
    timing = {
        "sampling_seconds": float(sum(seconds)),
        "seconds_per_sample": float(sum(seconds)) / n_samples,
        "device": str(device),
        "n_samples": n_samples,
        "steps": schedule.n_noisy + schedule.n_final,
    }
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"wrote {run_dir / 'samples.npz'} ({timing['seconds_per_sample']:.2f} s/sample)")


if __name__ == "__main__":
    main()
