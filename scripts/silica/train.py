"""Train the 3D silica denoiser (conditional NequIP; docs/SILICA_QUICKSTART.md).

    python scripts/silica/train.py --config configs/silica/train_cond_dm2.yaml \\
        run_dir=runs/silica/dm2_s0               # on DM2's glasses: can start now
    python scripts/silica/train.py --config configs/silica/train_cond.yaml \\
        run_dir=runs/silica/cond_s0              # on our 576-atom glasses (the main model)
    python scripts/silica/train.py --config configs/silica/train_cond.yaml variant=17 \\
        n_updates=5000 run_dir=runs/silica_screen/v17   # screening variant 17 (screen.yaml)
    python scripts/silica/train.py --config configs/silica/train_cond.yaml resume=runs/<run> \\
        max_minutes=600                          # continue after a time limit or disconnect
    python scripts/silica/train.py --config configs/silica/train_cond.yaml resume=runs/<run> \\
        n_updates=15000                          # extend a screening run (the funnel's stage 2)
    python scripts/silica/train.py --config configs/silica/train_cond_dm2.yaml device=cpu \\
        n_updates=20 optim.batch_size=1 optim.accumulate=1 val_every=10 val_batches=1  # CPU smoke

Each update: a batch of clean glasses -> requests (held-fixed atoms, label at the centre,
label dropped 15% of the time) -> noise sigma ~ U(sigma_min, sigma_max) on the free atoms ->
the model predicts the added displacement -> mean squared error on the free atoms -> AdamW
step (optionally accumulated over several batches) -> EMA update.

The run directory holds config.yaml, git.txt, log.csv (train loss, validation loss of the EMA
model at fixed noise levels as a fraction of the injected noise, seconds per update), ckpt.pt
(model, EMA, optimizer, step; load with glassdiff.silica.model.load_model) and timing.json
(seconds per update and GPU name: the week-1 compute benchmark).

``run_dir=<dir>`` fixes the run directory: the first call starts the run there and any later
call with the same command continues it (what the SLURM and Kaggle launchers do).
``resume=<run dir>`` continues from ckpt.pt with the run's saved config; only the session
settings (max_minutes, device, threads, intervals) and the total ``n_updates`` can change.
Checkpoints are written atomically, so a time limit during a save cannot corrupt them.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import time
from pathlib import Path

import torch
import yaml

from glassdiff.silica.data import SilicaSplit, load_split
from glassdiff.silica.defects import SilicaThresholds
from glassdiff.silica.model import build_model
from glassdiff.silica.requests import SilicaRequestConfig, SilicaRequestSampler
from glassdiff.utils.config import cli_config, load_config
from glassdiff.utils.runs import git_state, make_run_dir, pick_device, seed_everything

SESSION_KEYS = (
    "max_minutes",
    "device",
    "threads",
    "log_every",
    "val_every",
    "ckpt_every",
    "val_batches",
)


def apply_variant(cfg: dict, k: int) -> dict:
    """Overrides of the k-th combination of the screening grid (configs/silica/screen.yaml)."""
    grid = load_config(cfg.get("screen", "configs/silica/screen.yaml"))["factors"]
    combos = list(itertools.product(*[f["levels"] for f in grid]))
    if not 0 <= k < len(combos):
        raise ValueError(f"variant {k} outside the grid of {len(combos)}")
    for level in combos[k]:
        for key, value in level["set"].items():
            *parents, leaf = key.split(".")
            node = cfg
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
    cfg["name"] = f"{cfg['name']}_v{k:02d}"
    cfg["variant_levels"] = [level["name"] for level in combos[k]]
    return cfg


def check_labels(data_dir: Path, cfg: dict, train: SilicaSplit) -> None:
    """Labels must exist and come from the frozen thresholds of cfg['defects']."""
    if train.labels is None:
        raise ValueError(f"{data_dir}: no defect labels; run scripts/silica/label_defects.py")
    meta_path = data_dir / "meta.json"
    if "defects" not in cfg or not meta_path.exists():
        return
    want = SilicaThresholds(**load_config(cfg["defects"])["thresholds"]).digest()
    have = json.loads(meta_path.read_text()).get("labels", {}).get("digest")
    if have != want:
        raise ValueError(
            f"{data_dir} was labelled with thresholds {have}, but {cfg['defects']} gives {want}; "
            "relabel with scripts/silica/label_defects.py"
        )


def flatten(split: SilicaSplit, idx: torch.Tensor, device) -> dict:
    """Batch of structures -> flat atoms with a structure index."""
    b, n = len(idx), split.n_atoms
    return {
        "pos": split.pos[idx].to(device),  # [B, N, 3]
        "types": split.types[idx].to(device),
        "box": split.box[idx].to(device),
        "rate": split.cooling_rate[idx].to(device),
        "labels": None if split.labels is None else split.labels[idx].to(device),
        "batch": torch.arange(b, device=device).repeat_interleave(n),
    }


def loss_on(model, data, requests, sigma_min, sigma_max, gen, sigma_fixed=None):
    """Noise the free atoms, predict the displacement, MSE over free atoms. Returns
    (loss, mean squared noise), so loss / msd is the fraction of the noise not removed."""
    pos, box = data["pos"], data["box"]
    b, n, _ = pos.shape
    dev = pos.device
    pin = label = None
    if requests is not None:
        req = requests(pos, box, data["labels"], generator=gen)
        pin, label = req.pin.reshape(-1), req.label.reshape(-1)
    if sigma_fixed is None:
        sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(b, generator=gen)
    else:
        sigma = torch.full((b,), float(sigma_fixed))
    sigma = sigma.to(dev)
    eps = torch.randn(b * n, 3, generator=gen).to(dev)
    dx = sigma[data["batch"], None] * eps
    if pin is not None:
        dx = dx * (~pin)[:, None]
    noisy = torch.remainder(pos.reshape(-1, 3) + dx, box[data["batch"]])
    pred = model(
        noisy,
        data["types"].reshape(-1),
        box,
        data["batch"],
        sigma,
        pin,
        label,
        data["rate"],
    )
    # as the 2D loss (and DM2's F.mse_loss): mean over free atoms and coordinates per glass,
    # then over glasses
    w = torch.ones(b, n, device=dev) if pin is None else (~pin).reshape(b, n).float()
    per_atom = lambda v: (v**2).sum(-1).reshape(b, n)  # noqa: E731
    count = 3 * w.sum(-1).clamp_min(1)
    err = ((per_atom(pred - dx) * w).sum(-1) / count).mean()
    msd = ((per_atom(dx) * w).sum(-1) / count).mean()
    return err, msd


@torch.no_grad()
def validate(model, val: SilicaSplit, cfg, requests, device) -> dict[str, float]:
    out = {}
    bs = int(cfg["optim"]["batch_size"])
    n_batches = int(cfg.get("val_batches", 4))
    for s in cfg["val_sigmas"]:
        gen = torch.Generator().manual_seed(1234)
        num = den = 0.0
        for i in range(min(n_batches, max(1, len(val) // bs))):
            idx = torch.arange(i * bs, min((i + 1) * bs, len(val)))
            err, msd = loss_on(model, flatten(val, idx, device), requests, 0, 0, gen, s)
            num, den = num + err.item(), den + msd.item()
        out[f"valn_{s}"] = num / max(den, 1e-12)
    return out


def main() -> None:
    cfg = cli_config(__doc__)
    device = pick_device(cfg.get("device", "auto"))
    ckpt = None
    if cfg.get("run_dir") and (Path(cfg["run_dir"]) / "ckpt.pt").exists():
        cfg["resume"] = cfg["run_dir"]
    if cfg.get("resume"):
        run_dir = Path(cfg["resume"])
        ckpt = torch.load(run_dir / "ckpt.pt", map_location=device, weights_only=False)
        session = {k: cfg[k] for k in SESSION_KEYS if k in cfg}
        optim = dict(ckpt["cfg"]["optim"])
        if cfg.get("n_updates") is not None:
            optim["n_updates"] = int(cfg["n_updates"])
        cfg = {**ckpt["cfg"], "optim": optim, **session}
        print(f"resuming {run_dir} at step {ckpt['step']}", flush=True)
    else:
        if cfg.get("variant") is not None:
            cfg = apply_variant(cfg, int(cfg["variant"]))
        if cfg.get("n_updates") is not None:
            cfg["optim"]["n_updates"] = int(cfg["n_updates"])
        if cfg.get("run_dir"):
            run_dir = Path(cfg["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (run_dir / "git.txt").write_text(git_state() + "\n")
        else:
            run_dir = make_run_dir(cfg["name"], cfg, root=cfg.get("runs_root", "runs"))
        print(f"run directory: {run_dir}", flush=True)
    start_step = 0 if ckpt is None else int(ckpt["step"])
    seconds_before = 0.0 if ckpt is None else float(ckpt.get("seconds", 0.0))
    seed_everything(int(cfg["seed"]) + start_step)
    gen = torch.Generator().manual_seed(int(cfg["seed"]) * 1000 + start_step)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    data_dir = Path(cfg["data"])
    train, val = load_split(data_dir / "train.npz"), load_split(data_dir / "val.npz")
    conditional = bool(cfg.get("conditional", True))
    if conditional:
        check_labels(data_dir, cfg, train)
    requests = SilicaRequestSampler(SilicaRequestConfig(**cfg["requests"])) if conditional else None
    model = build_model(cfg["model"]).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt_cfg = cfg["optim"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"])
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        del ckpt
    n_params = sum(p.numel() for p in model.parameters())
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        f"{len(train)} training glasses of {train.n_atoms} atoms; {n_params:,} parameters on {gpu}",
        flush=True,
    )

    bs, accum = int(opt_cfg["batch_size"]), int(opt_cfg.get("accumulate", 1))
    n_updates, ema_decay = int(opt_cfg["n_updates"]), float(opt_cfg["ema"])
    sigma_min, sigma_max = float(cfg["noise"]["sigma_min"]), float(cfg["noise"]["sigma_max"])
    max_seconds = float(cfg.get("max_minutes", 0)) * 60 or float("inf")
    log_path = run_dir / "log.csv"
    if start_step == 0:
        cols = ",".join(f"valn_{s}" for s in cfg["val_sigmas"])
        log_path.write_text(f"step,seconds,s_per_update,train_loss,{cols}\n")

    def save(step: int, seconds: float) -> None:
        tmp = run_dir / "ckpt.pt.tmp"
        state = {
            "model_cfg": cfg["model"],
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "seconds": seconds,
            "cfg": cfg,
        }
        torch.save(state, tmp)
        os.replace(tmp, run_dir / "ckpt.pt")

    def order():
        while True:
            yield from torch.randperm(len(train), generator=gen).split(bs)

    batches = (b for b in order() if len(b) == bs or len(train) < bs)
    step, t0 = start_step, time.time()
    train_time, run_loss, n_loss = 0.0, 0.0, 0
    done = step >= n_updates
    while not done:
        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            err, msd = loss_on(
                model, flatten(train, next(batches), device), requests, sigma_min, sigma_max, gen
            )
            (err / accum).backward()
            run_loss, n_loss = run_loss + err.item(), n_loss + 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
        opt.step()
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1 - ema_decay)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step += 1
        train_time += time.time() - t_step
        session = time.time() - t0
        elapsed = seconds_before + session
        done = step >= n_updates or session > max_seconds
        s_per_update = train_time / (step - start_step)
        if step % int(cfg.get("val_every", 500)) == 0 or done:
            vals = validate(ema, val, cfg, requests, device)
            with log_path.open("a") as f:
                f.write(f"{step},{elapsed:.0f},{s_per_update:.4f},{run_loss / n_loss:.6g},")
                f.write(",".join(f"{v:.6g}" for v in vals.values()) + "\n")
            print(
                f"step {step} {elapsed:.0f}s ({s_per_update:.3f} s/update) train "
                f"{run_loss / n_loss:.5f} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()),
                flush=True,
            )
            run_loss, n_loss = 0.0, 0
        elif step % int(cfg.get("log_every", 50)) == 0:
            print(f"step {step} {elapsed:.0f}s train {run_loss / n_loss:.5f}", flush=True)
        if step % int(cfg.get("ckpt_every", 1000)) == 0 or done:
            save(step, elapsed)
    timing = {
        "device": gpu,
        "seconds_per_update": train_time / max(step - start_step, 1),
        "batch_size": bs * accum,
        "atoms_per_structure": train.n_atoms,
        "parameters": n_params,
        "updates_this_session": step - start_step,
    }
    if step > start_step:  # a call on a finished run must not overwrite the timing
        (run_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(f"finished at step {step} -> {run_dir / 'ckpt.pt'}; {timing}")


if __name__ == "__main__":
    main()
