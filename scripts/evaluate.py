"""Compute all metrics with bootstrap CIs for one run, or the natural-vs-natural floor.

    python scripts/evaluate.py --config configs/eval/default.yaml run=runs/<run> data=data/ka2d_256
    python scripts/evaluate.py --config configs/eval/default.yaml floor=true defect=D2 \\
        data=data/ka2d_256 runs_root=results       # floor row: train vs test natural defects

Needs <run>/samples.npz and <run>/relaxed.npz. Writes <run>/metrics.json: for each metric
an estimate with a 95% bootstrap CI over samples ([estimate, lower, upper]).
``also_unrelaxed=true`` adds local realism before relaxation (the "ceiling" row for B2).
[Ticket E-4]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds
from glassdiff.data.dataset import load_split
from glassdiff.eval.bootstrap import bootstrap_ci
from glassdiff.eval.metrics import (
    cost_per_success,
    diversity,
    global_realism,
    local_features,
    local_realism,
    natural_windows,
    success,
)
from glassdiff.types import DefectClass, Request, Structures
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import make_run_dir


def cached_natural_windows(data_dir: Path, split_name: str, defect: int, r_loc: float):
    """natural_windows for one split, cached under <data_dir>/.cache/ (gitignored).

    The cache key includes the split file's size and modification time, so regenerated or
    re-labelled data is never served stale. Saves ~1 min per evaluation on a CPU.
    """
    path = data_dir / f"{split_name}.npz"
    st = path.stat()
    key = f"{split_name}_{int(defect)}_{r_loc:g}_{st.st_size}_{int(st.st_mtime)}"
    cache = data_dir / ".cache" / f"windows_{key}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    windows = natural_windows(load_split(path), int(defect), r_loc)
    cache.parent.mkdir(exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save(windows, tmp)
    tmp.replace(cache)
    return windows


def _jsonable(d: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def evaluate_floor(cfg: dict, data_dir: Path, r_loc: float, n_boot: int) -> dict:
    defect = DefectClass[cfg["defect"]] if isinstance(cfg["defect"], str) else cfg["defect"]
    train, test = load_split(data_dir / "train.npz"), load_split(data_dir / "test.npz")
    gen = cached_natural_windows(data_dir, "train", int(defect), r_loc)
    ref = cached_natural_windows(data_dir, "test", int(defect), r_loc)
    return {
        "kind": "floor (natural train vs natural test)",
        "defect": DefectClass(defect).name,
        "n_windows": len(gen),
        "local": _jsonable(local_realism(gen, ref, n_boot)),
        "global": global_realism(train.subset(slice(0, 100)).structures, test.structures),
    }


def main() -> None:
    cfg = cli_config(__doc__)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    data_dir = Path(cfg["data"])
    r_loc = float(cfg.get("local", {}).get("r_loc", 3.0))
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples_w1", 1000))
    if cfg.get("floor"):
        out_dir = make_run_dir(f"floor_{cfg['defect']}", cfg, root=cfg.get("runs_root", "runs"))
        metrics = evaluate_floor(cfg, data_dir, r_loc, n_boot)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
        return

    run = Path(cfg["run"])
    meta = json.loads((data_dir / "meta.json").read_text())
    th = DefectThresholds.from_dict(meta["defects"]["thresholds"])
    with np.load(run / "samples.npz") as f:
        samp = {k: f[k] for k in f.files}
    with np.load(run / "relaxed.npz") as f:
        rel = {k: f[k] for k in f.files}
    timing = json.loads((run / "timing.json").read_text())
    defect = int(samp["defect"])
    types, box = torch.from_numpy(samp["types"]).long(), torch.from_numpy(samp["box"]).double()
    pre = Structures(torch.from_numpy(samp["pos"]).double(), types, box)
    post = Structures(torch.from_numpy(rel["pos"]).double(), types, box)
    n = pre.batch_size
    request = Request(
        target=torch.from_numpy(samp["target"]).double(),
        defect=torch.full((n,), defect, dtype=torch.long),
        pin_mask=torch.from_numpy(samp["pin_mask"]),
        pin_pos=torch.from_numpy(samp["pin_pos"]).double(),
        label=torch.from_numpy(samp["label"]).long(),
    )

    ok_before = success(pre, request, th).double().numpy()
    ok_after = success(post, request, th).double().numpy()
    test = load_split(data_dir / "test.npz")  # global-realism reference
    ref = cached_natural_windows(data_dir, "test", defect, r_loc)
    gen = local_features(post, request.target, torch.from_numpy(rel["disp"]).double(), r_loc)
    seconds = float(timing.get("sampling_seconds", 0.0)) + float(timing.get("relax_seconds", 0.0))
    metrics = {
        "run": str(run),
        "defect": DefectClass(defect).name,
        "mode": str(samp["mode"]),
        "n_samples": n,
        "relax_converged": float(rel["converged"].mean()),
        "success_before_relax": list(bootstrap_ci(ok_before)),
        "success_after_relax": list(bootstrap_ci(ok_after)),
        "seconds_total": seconds,
        "seconds_per_success": cost_per_success(seconds, int(ok_after.sum())),
        "local": _jsonable(local_realism(gen, ref, n_boot)),
        "global": global_realism(post, test.structures),
        "diversity": diversity(gen, ref, cached_natural_windows(data_dir, "train", defect, r_loc)),
    }
    if cfg.get("also_unrelaxed"):
        gen_pre = local_features(pre, request.target, None, r_loc)
        metrics["local_unrelaxed"] = _jsonable(local_realism(gen_pre, ref, n_boot))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
