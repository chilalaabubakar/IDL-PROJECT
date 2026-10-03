"""Train a denoiser (unconditional or conditional).  [Tickets M-3, M-4]

    python scripts/train.py --config configs/train/uncond.yaml
    python scripts/train.py --config configs/train/cond.yaml model=configs/model/mpnn.yaml
    python scripts/train.py --config configs/train/uncond.yaml model_args.hidden=64 \\
        optim.n_updates=2000 max_minutes=20          # CPU smoke run

Loop: batch of clean glasses -> (conditional: RequestSampler) -> sample_sigma -> add_noise
-> model -> masked_displacement_loss -> AdamW step -> EMA update. Everything goes into a
run directory: config.yaml, git.txt, log.csv (train loss; val loss of the EMA model at
fixed noise levels) and ckpt.pt (model, EMA, optimizer; load with
glassdiff.models.registry.load_denoiser).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from glassdiff.data.dataset import GlassDataset, batch_to_structures, load_split
from glassdiff.data.requests import RequestSampler, RequestSamplerConfig
from glassdiff.diffusion.noise import add_noise, masked_displacement_loss, sample_sigma
from glassdiff.models.registry import build_model
from glassdiff.types import Structures
from glassdiff.utils.config import cli_config, load_config
from glassdiff.utils.runs import make_run_dir, seed_everything

VAL_SIGMAS = (0.02, 0.1, 0.3, 0.5)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rotate90(s: Structures, k: Tensor) -> Structures:
    """Rotate each structure by k[b] * 90 degrees (square boxes only; augmentation)."""
    pos = s.pos.clone()
    for _ in range(4):
        sel = k > 0
        pos[sel] = torch.stack([-pos[sel][..., 1], pos[sel][..., 0]], dim=-1)
        k = k - 1
    return s.with_pos(pos)


@torch.no_grad()
def validation_losses(model, val_loader, device, requests, n_batches: int = 8) -> dict[str, float]:
    """EMA-model loss at fixed noise levels, with fixed noise, so runs are comparable."""
    out = {}
    for sigma_val in VAL_SIGMAS:
        gen = torch.Generator().manual_seed(1234)
        total, count = 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            s = batch_to_structures(batch).to(device)
            pin = label = None
            if requests is not None:
                req = requests(s, batch["labels"].to(device), generator=gen)
                pin, label = req.pin_mask, req.label
            sigma = torch.full((s.batch_size,), sigma_val, device=device)
            noisy, disp = add_noise(s, sigma, pin, generator=gen)
            total += masked_displacement_loss(model(noisy, sigma, pin, label), disp, pin).item()
            count += 1
        out[f"val_{sigma_val}"] = total / max(count, 1)
    return out


def main() -> None:
    cfg = cli_config(__doc__)
    gen = seed_everything(int(cfg["seed"]))
    device = pick_device(cfg.get("device", "auto"))
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    model_cfg = load_config(cfg["model"]) if isinstance(cfg["model"], str) else dict(cfg["model"])
    model_cfg.update(cfg.get("model_args", {}))
    cfg["resolved_model"] = model_cfg
    run_dir = make_run_dir(cfg["name"], cfg, root=cfg.get("runs_root", "runs"))
    print(f"run directory: {run_dir}", flush=True)

    data_dir = Path(cfg["data"])
    train, val = load_split(data_dir / "train.npz"), load_split(data_dir / "val.npz")
    if cfg.get("n_train"):
        train = train.subset(slice(0, int(cfg["n_train"])))
    opt_cfg = cfg["optim"]
    loader = DataLoader(
        GlassDataset(train),
        batch_size=int(opt_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        generator=gen,
    )
    val_loader = DataLoader(GlassDataset(val), batch_size=int(opt_cfg["batch_size"]))

    model = build_model(model_cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {model_cfg['name']}: {n_params:,} parameters on {device}", flush=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"])
    )
    requests = (
        RequestSampler(RequestSamplerConfig(**cfg["requests"])) if cfg.get("conditional") else None
    )
    noise = cfg["noise"]
    augment = bool(model_cfg.get("rotation_augmentation", False))
    noise_gen = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
    n_updates, ema_decay = int(opt_cfg["n_updates"]), float(opt_cfg["ema"])
    max_seconds = float(cfg.get("max_minutes", 0)) * 60 or float("inf")

    log_path = run_dir / "log.csv"
    log_path.write_text(
        "step,seconds,train_loss," + ",".join(f"val_{v}" for v in VAL_SIGMAS) + "\n"
    )

    def save(step: int) -> None:
        torch.save(
            {
                "model_cfg": model_cfg,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": cfg,
            },
            run_dir / "ckpt.pt",
        )

    step, running, n_running, t0 = 0, 0.0, 0, time.time()
    done = False
    while not done:
        for batch in loader:
            s = batch_to_structures(batch).to(device)
            if augment:
                s = rotate90(s, torch.randint(4, (s.batch_size,), generator=noise_gen).to(device))
            pin = label = None
            if requests is not None:
                req = requests(s, batch["labels"].to(device), generator=noise_gen)
                pin, label = req.pin_mask, req.label
            sigma = sample_sigma(
                s.batch_size,
                float(noise["sigma_min"]),
                float(noise["sigma_max"]),
                generator=noise_gen,
                distribution=noise.get("distribution", "uniform"),
            ).to(device)
            noisy, disp = add_noise(s, sigma, pin, generator=noise_gen)
            loss = masked_displacement_loss(model(noisy, sigma, pin, label), disp, pin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
            opt.step()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1 - ema_decay)
            step += 1
            running, n_running = running + loss.item(), n_running + 1
            elapsed = time.time() - t0
            done = step >= n_updates or elapsed > max_seconds

            if step % int(cfg.get("val_every", 1000)) == 0 or done:
                vals = validation_losses(ema, val_loader, device, requests)
                train_loss = running / n_running
                with log_path.open("a") as f:
                    f.write(f"{step},{elapsed:.0f},{train_loss:.6g},")
                    f.write(",".join(f"{vals[f'val_{v}']:.6g}" for v in VAL_SIGMAS) + "\n")
                print(
                    f"step {step} {elapsed:.0f}s train {train_loss:.5f} "
                    + " ".join(f"{k}={v:.5f}" for k, v in vals.items()),
                    flush=True,
                )
                running, n_running = 0.0, 0
            elif step % int(cfg.get("log_every", 100)) == 0:
                print(f"step {step} {elapsed:.0f}s train {running / n_running:.5f}", flush=True)
            if step % int(cfg.get("ckpt_every", 5000)) == 0 or done:
                save(step)
            if done:
                break
    print(f"finished {step} updates in {time.time() - t0:.0f}s -> {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
