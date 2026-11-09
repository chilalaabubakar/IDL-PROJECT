"""Train the point-defect models: repair (task: heal) and detection (task: detect).

    python scripts/train_point_defects.py --config configs/train/heal_point_defects.yaml
    python scripts/train_point_defects.py --config configs/train/detect_point_defects.yaml
    python scripts/train_point_defects.py --config configs/train/heal_point_defects.yaml \\
        resume=runs/<run> max_minutes=600       # continue after a disconnect

Data: scripts/make_point_defects.py (vacancy, interstitial, Frenkel pair, antisite; see
docs/POINT_DEFECTS.md and glassdiff.data.point_defects).

task: heal. The repair model is DM2's denoiser with the corruption changed. DM2 trains on
    x = x0 + sigma * eps (a defect-free glass x0 shaken by random noise) and predicts the
    shift x - x0. Here the starting point is the repair input of a defected glass (the
    relaxed defected glass with its atom count and species put right), so
        x = heal_pos + sigma * eps,    target = x - x0  (minimum image, mean shift removed),
    and the same EGNN, told sigma, predicts the target. A fraction p_clean of the batches are
    defect-free glasses (heal_pos = x0); for those the loss is exactly DM2's. The trained
    model therefore removes noise and undoes the defect, and is used with the DM2 sampler
    from a defected glass (scripts/eval_point_defects.py --healer).
    Version 2 (configs/train/heal_point_defects_v2.yaml) fixes what kept the first repair
    model from learning the defect. A defect moves 30-40 of the 256 atoms by more than 0.1
    sigma, so in a plain mean over atoms the random noise filled almost all of the loss.
    Now atoms the defect moved by more than defect_tol get weight 1 + defect_weight;
    noise.distribution log_uniform puts most examples at small noise, where the defect is
    what is left to remove; and frenkel_input: return puts a Frenkel pair's displaced atom
    back into the gap at its site first (which gap is a separate choice in the evaluation),
    so the model cleans up the surroundings instead of learning a 1.2-3 sigma jump.
task: detect. The EGNN classifier gives every atom of a relaxed defected glass an AtomRole
    (intact, next to an empty site, extra atom, displaced atom, wrong species);
    class-weighted cross-entropy, weights (class frequency)^-class_weight_power. A fraction
    p_clean of the batches are defect-free glasses, all atoms intact.

Each batch holds one defect kind, drawn with kind_probs (vacancy and interstitial glasses
have 255 and 257 atoms). init_from loads matching weights from another checkpoint, e.g. the
trained unconditional denoiser (the classifier takes its trunk). Run directory: config.yaml,
git.txt, log.jsonl (train loss and validation metrics of the EMA model on fixed val
examples), ckpt.pt (load with glassdiff.models.registry.load_denoiser). Validation:
    heal:   per kind, one repair step at each val_sigma, x - model(x): RMS distance to the
            original over all atoms, over the defect atoms (defect_rms) and the mean
            worst-atom distance; "none_*" is the same for leaving the input as it is.
    detect: per kind, how often the atom most likely to be defective lies within 1.5 sigma
            of the defect (as eval_point_defects.py), the recall of each role, and on
            defect-free glasses the fraction with any atom above probability 0.5.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from glassdiff.data.point_defects import (
    N_ROLES,
    AtomRole,
    PointDefectSet,
    heal_target,
    load_point_defects,
    return_displaced_atom,
)
from glassdiff.models.registry import build_model
from glassdiff.types import Structures
from glassdiff.utils.config import cli_config, load_config
from glassdiff.utils.runs import make_run_dir, pick_device, seed_everything

SESSION_KEYS = ("max_minutes", "device", "threads", "log_every", "val_every", "ckpt_every")
HIT_RADIUS = 1.5


def wrap(v: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    b = box.view((box.shape[0],) + (1,) * (v.dim() - 2) + (2,))
    return v - b * torch.round(v / b)


def defect_probability(logits: torch.Tensor) -> torch.Tensor:
    """[B, N] probability that an atom is not intact."""
    return 1 - logits.softmax(-1)[..., int(AtomRole.INTACT)]


def detection_hits(
    score: torch.Tensor, pos: torch.Tensor, sites: torch.Tensor, box
) -> torch.Tensor:
    """[B] bool: the top-scored atom lies within HIT_RADIUS of either defect site."""
    top = pos[torch.arange(len(pos)), score.argmax(-1)]
    d = torch.stack([wrap(top - sites[:, k], box).norm(dim=-1) for k in range(2)]).amin(0)
    return d < HIT_RADIUS


def make_batch(task, d: PointDefectSet, idx, clean: bool, sigma, gen, device):
    """Model input and target for examples idx of one kind."""
    box = d.box[idx]
    if task == "heal":
        base = d.orig_pos[idx] if clean else d.heal_pos[idx]
        eps = torch.randn(base.shape, generator=gen, dtype=base.dtype)
        x = base + sigma.to(base.dtype)[:, None, None] * eps
        target = heal_target(x, d.orig_pos[idx], box)
        s = Structures(x, d.orig_types[idx], box)
        return s.to(device), target.float().to(device)
    if clean:
        s = Structures(d.orig_pos[idx], d.orig_types[idx], box)
        role = torch.zeros(d.orig_types[idx].shape, dtype=torch.long)
    else:
        s = Structures(d.pos[idx], d.types[idx], box)
        role = d.role[idx]
    return s.to(device), role.to(device)


def defect_atoms(d: PointDefectSet, idx, clean: bool, tol: float) -> torch.Tensor:
    """[B, N] atoms the defect moved by more than tol (none for a defect-free glass)."""
    if clean:
        return torch.zeros(d.orig_types[idx].shape, dtype=torch.bool)
    return heal_target(d.heal_pos[idx], d.orig_pos[idx], d.box[idx]).norm(dim=-1) > tol


def sample_sigma(n: int, noise: dict, gen: torch.Generator) -> torch.Tensor:
    """[n] noise levels, uniform (DM2) or log-uniform (more weight on small noise)."""
    lo, hi = float(noise.get("sigma_min", 1e-3)), float(noise.get("sigma_max", 0.5))
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    if noise.get("distribution", "uniform") == "log_uniform":
        return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)
    return lo + (hi - lo) * u


def to_float(s: Structures) -> Structures:
    return Structures(s.pos.float(), s.types, s.box.float())


@torch.no_grad()
def validate(task, model, val: dict[str, PointDefectSet], cfg, device) -> dict[str, float]:
    out: dict[str, float] = {}
    bs = int(cfg["optim"]["batch_size"])
    n_val = int(cfg.get("n_val", 200))
    for name, d in val.items():
        d = d.subset(slice(0, n_val))
        if task == "heal":
            for sig in cfg.get("val_sigma", [1e-3]):
                gen = torch.Generator().manual_seed(1234)
                rows = {k: [] for k in ("rms", "worst", "defect_rms", "none_rms", "none_worst",
                                        "none_defect_rms")}  # fmt: skip
                tol = float(cfg.get("defect_tol", 0.1))
                for i in range(0, len(d), bs):
                    idx = torch.arange(i, min(i + bs, len(d)))
                    sigma = torch.full((len(idx),), float(sig))
                    s, target = make_batch("heal", d, idx, False, sigma, gen, device)
                    pred = model(to_float(s), sigma.to(device).float())
                    mask = defect_atoms(d, idx, False, tol).to(device)
                    for key, err in (("", pred - target), ("none_", -target)):
                        e = err.norm(dim=-1)
                        rows[key + "rms"].append(e.pow(2).mean(-1))
                        rows[key + "worst"].append(e.amax(-1))
                        rows[key + "defect_rms"].append(e[mask].pow(2).cpu())
                for key, v in rows.items():
                    v = torch.cat([x.flatten().cpu() for x in v])
                    v = v.mean().sqrt() if key.endswith("rms") else v.mean()
                    out[f"{name}_{key}_s{sig:g}"] = round(float(v), 5)
        else:
            hits, correct, count = [], torch.zeros(N_ROLES), torch.zeros(N_ROLES)
            for i in range(0, len(d), bs):
                idx = torch.arange(i, min(i + bs, len(d)))
                s, role = make_batch("detect", d, idx, False, None, None, device)
                logits = model(to_float(s))
                hits.append(
                    detection_hits(
                        defect_probability(logits).cpu(), d.pos[idx], d.sites[idx], d.box[idx]
                    )
                )
                pred = logits.argmax(-1).cpu()
                role = role.cpu()
                for r in range(N_ROLES):
                    m = role == r
                    count[r] += m.sum()
                    correct[r] += (pred[m] == r).sum()
            out[f"{name}_hit"] = round(float(torch.cat(hits).double().mean()), 4)
            for r in range(N_ROLES):
                if count[r] > 0:
                    out[f"{name}_recall_{AtomRole(r).name.lower()}"] = round(
                        float(correct[r] / count[r]), 4
                    )
    if task == "heal":  # defect-free glasses: plain denoising, should stay as good as DM2's
        d = next(iter(val.values())).subset(slice(0, n_val))
        for sig in cfg.get("val_sigma", [1e-3]):
            gen = torch.Generator().manual_seed(1234)
            sq = []
            for i in range(0, len(d), bs):
                idx = torch.arange(i, min(i + bs, len(d)))
                sigma = torch.full((len(idx),), float(sig))
                s, target = make_batch("heal", d, idx, True, sigma, gen, device)
                pred = model(to_float(s), sigma.to(device).float())
                sq.append((pred - target).norm(dim=-1).pow(2).mean(-1))
            out[f"clean_rms_s{sig:g}"] = round(float(torch.cat(sq).mean().sqrt()), 5)
    if task == "detect":
        d = next(iter(val.values())).subset(slice(0, n_val))
        alarms = []
        for i in range(0, len(d), bs):
            idx = torch.arange(i, min(i + bs, len(d)))
            s, _ = make_batch("detect", d, idx, True, None, None, device)
            alarms.append((defect_probability(model(to_float(s))) > 0.5).any(-1).cpu())
        out["clean_false_alarm"] = round(float(torch.cat(alarms).double().mean()), 4)
    return out


def class_weights(train: dict[str, PointDefectSet], power: float) -> torch.Tensor:
    counts = torch.zeros(N_ROLES, dtype=torch.float64)
    for d in train.values():
        counts += torch.bincount(d.role.flatten(), minlength=N_ROLES).double()
    w = counts.clamp_min(1) ** -power
    return (w / w[int(AtomRole.INTACT)]).float()


def main() -> None:
    cfg = cli_config(__doc__)
    device = pick_device(cfg.get("device", "auto"))
    ckpt = None
    if cfg.get("resume"):
        run_dir = Path(cfg["resume"])
        ckpt = torch.load(run_dir / "ckpt.pt", map_location=device, weights_only=False)
        session = {k: cfg[k] for k in SESSION_KEYS if k in cfg}
        cfg = {**ckpt["cfg"], **session}
        model_cfg = ckpt["model_cfg"]
        print(f"resuming {run_dir} at step {ckpt['step']}", flush=True)
    else:
        model_cfg = (
            load_config(cfg["model"]) if isinstance(cfg["model"], str) else dict(cfg["model"])
        )
        model_cfg.update(cfg.get("model_args", {}))
        cfg["resolved_model"] = model_cfg
        run_dir = make_run_dir(cfg["name"], cfg, root=cfg.get("runs_root", "runs"))
        print(f"run directory: {run_dir}", flush=True)
    task = cfg["task"]
    if task not in ("heal", "detect"):
        raise ValueError(f"task must be heal or detect, got {task!r}")
    start_step = 0 if ckpt is None else int(ckpt["step"])
    seconds_before = 0.0 if ckpt is None else float(ckpt.get("seconds", 0.0))
    gen = seed_everything(int(cfg["seed"]) + start_step)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    data = Path(cfg["data"])
    train = load_point_defects(data / "train.npz", cfg.get("source_train"))
    val = load_point_defects(data / "val.npz", cfg.get("source_val"))
    if task == "heal" and cfg.get("frenkel_input", "relaxed") == "return":
        search = float(cfg.get("frenkel_return_search", 1.0))
        for split in (train, val):
            if "frenkel" in split:
                split["frenkel"] = return_displaced_atom(split["frenkel"], search)
        print("Frenkel repair input: displaced atom put back at its site", flush=True)
    kinds = list(cfg.get("kind_probs", {k: 1.0 for k in train}))
    probs = torch.tensor([float(cfg.get("kind_probs", {}).get(k, 1.0)) for k in kinds])
    probs = probs / probs.sum()
    p_clean = float(cfg.get("p_clean", 0.2))

    model = build_model(model_cfg).to(device)
    if ckpt is None and cfg.get("init_from"):
        init = torch.load(cfg["init_from"], map_location=device, weights_only=False)
        state = init.get("ema", init.get("model", init))
        own = model.state_dict()
        usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(usable, strict=False)
        print(f"init_from {cfg['init_from']}: {len(usable)}/{len(own)} tensors", flush=True)
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
    print(f"{task}: {model_cfg['name']}, {n_params:,} parameters on {device}", flush=True)
    print("training examples: " + ", ".join(f"{k} {len(v)}" for k, v in train.items()), flush=True)
    weights = class_weights(train, float(cfg.get("class_weight_power", 0.5))).to(device)
    noise = cfg.get("noise", {})
    defect_weight = float(cfg.get("defect_weight", 0.0))  # extra loss weight, defect atoms
    defect_tol = float(cfg.get("defect_tol", 0.1))  # moved more than this by the defect
    bs, n_updates = int(opt_cfg["batch_size"]), int(opt_cfg["n_updates"])
    ema_decay = float(opt_cfg["ema"])
    max_seconds = float(cfg.get("max_minutes", 0)) * 60 or float("inf")
    log_path = run_dir / "log.jsonl"

    def save(step: int, seconds: float) -> None:
        tmp = run_dir / "ckpt.pt.tmp"
        state = dict(
            model_cfg=model_cfg,
            model=model.state_dict(),
            ema=ema.state_dict(),
            opt=opt.state_dict(),
            step=step,
            seconds=seconds,
            cfg=cfg,
        )
        if task == "detect":  # for logit adjustment at evaluation (eval_point_defects.py)
            state["class_weights"] = weights.cpu().tolist()
        torch.save(state, tmp)
        os.replace(tmp, run_dir / "ckpt.pt")

    step, running, n_running, t0 = start_step, 0.0, 0, time.time()
    if step >= n_updates:
        print(f"run already finished ({step} >= {n_updates} updates)")
    while step < n_updates:
        kind = kinds[int(torch.multinomial(probs, 1, generator=gen))]
        d = train[kind]
        clean = bool(torch.rand(1, generator=gen) < p_clean)
        idx = torch.randint(len(d), (bs,), generator=gen)
        sigma = sample_sigma(bs, noise, gen) if task == "heal" else None
        s, target = make_batch(task, d, idx, clean, sigma, gen, device)
        if task == "heal":
            pred = model(to_float(s), sigma.float().to(device))
            sq = ((pred - target) ** 2).sum(-1)  # [B, N]
            w = 1 + defect_weight * defect_atoms(d, idx, clean, defect_tol).to(sq)
            loss = ((sq * w).sum(-1) / w.sum(-1)).mean() / 2
        else:
            logits = model(to_float(s))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.flatten(), weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
        opt.step()
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1 - ema_decay)
        step += 1
        running, n_running = running + loss.item(), n_running + 1
        session_seconds = time.time() - t0
        elapsed = seconds_before + session_seconds
        done = step >= n_updates or session_seconds > max_seconds
        if step % int(cfg.get("val_every", 1000)) == 0 or done:
            vals = validate(task, ema, val, cfg, device)
            row = dict(step=step, seconds=round(elapsed), train_loss=running / n_running, **vals)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            running, n_running = 0.0, 0
        elif step % int(cfg.get("log_every", 100)) == 0:
            print(f"step {step} {elapsed:.0f}s train {running / n_running:.5f}", flush=True)
        if step % int(cfg.get("ckpt_every", 2000)) == 0 or done:
            save(step, elapsed)
        if done:
            break
    print(f"finished {step} updates -> {run_dir / 'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    main()
