"""Do natural defects survive a random perturbation followed by FIRE relaxation?

    python scripts/defect_stability.py --config configs/eval/default.yaml data=data/ka2d_256 \\
        sigmas=[0,0.05,0.1,0.15] n_glasses=100

Every atom of a natural glass (``split``, default test) is displaced by Gaussian noise of
standard deviation sigma per coordinate, the whole glass is relaxed (as in relax.py), and
each natural defect site counts as surviving when the atom of its species nearest the
original site lies within ``success.r_tol`` and carries the same label (the success test of
evaluate.py). Survival near 1 at the size of the generated samples' relaxation displacement
means a lost defect is the generator's fault, not the defect's. Writes metrics.json with
bootstrap CIs over glasses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.data.dataset import load_split
from glassdiff.data.requests import DEFECT_SPECIES
from glassdiff.eval.bootstrap import bootstrap_ci_items
from glassdiff.eval.metrics import nearest_atom
from glassdiff.physics.fire import fire_minimize
from glassdiff.types import DefectClass, Structures
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import make_run_dir, pick_device, seed_everything

CLASSES = (DefectClass.D1_MINUS, DefectClass.D1_PLUS, DefectClass.D2)


def survival(
    before: Structures, labels: Tensor, after: Structures, after_labels: Tensor, r_tol: float
) -> dict[str, list[np.ndarray]]:
    """Per defect class, one bool array per glass: did each natural site survive?"""
    out = {}
    for d in CLASSES:
        per_glass = []
        for b in range(before.batch_size):
            sites = torch.nonzero(labels[b] == d).flatten()
            if len(sites) == 0:
                continue
            n = len(sites)
            sub = Structures(
                after.pos[b].expand(n, -1, -1),
                after.types[b].expand(n, -1),
                after.box[b].expand(n, -1),
            )
            species = torch.full((n,), int(DEFECT_SPECIES[d]), device=sub.types.device)
            idx, dist = nearest_atom(sub, before.pos[b, sites], species)
            ok = (dist < r_tol) & (after_labels[b, idx] == d)
            per_glass.append(ok.cpu().numpy())
        out[d.name] = per_glass
    return out


def main() -> None:
    cfg = cli_config(__doc__)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    gen = seed_everything(int(cfg.get("seed", 0)))
    device = pick_device(cfg.get("device", "auto"))
    data_dir = Path(cfg["data"])
    meta = json.loads((data_dir / "meta.json").read_text())
    th = DefectThresholds.from_dict(meta["defects"]["thresholds"])
    split = load_split(data_dir / f"{cfg.get('split', 'test')}.npz")
    split = split.subset(slice(0, int(cfg.get("n_glasses", 100))))
    s0, labels = split.structures, split.labels
    relax = cfg.get("relax", {})
    r_tol = float(cfg.get("success", {}).get("r_tol", 0.5))
    batch_size = int(cfg.get("batch_size", 64))
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples", 10_000))
    out_dir = make_run_dir("defect_stability", cfg, root=cfg.get("runs_root", "runs"))

    results = {"data": str(data_dir), "n_glasses": s0.batch_size, "r_tol": r_tol, "sigmas": {}}
    for sigma in [float(x) for x in cfg.get("sigmas", [0.0, 0.05, 0.1, 0.15])]:
        t0 = time.time()
        noise = sigma * torch.randn(s0.pos.shape, generator=gen, dtype=s0.pos.dtype)
        noisy = s0.with_pos(s0.pos + noise).wrapped()
        pos, disp = [], []
        for i in range(0, s0.batch_size, batch_size):
            chunk = Structures(
                noisy.pos[i : i + batch_size],
                noisy.types[i : i + batch_size],
                noisy.box[i : i + batch_size],
            )
            res = fire_minimize(
                chunk.to(device),
                fmax=float(relax.get("fmax", 1e-6)),
                max_steps=int(relax.get("max_steps", 100_000)),
            )
            pos.append(res.structures.pos.cpu())
            disp.append(res.displacement.norm(dim=-1).mean(dim=-1).cpu())
        after = s0.with_pos(torch.cat(pos))
        after_labels = detect_defects(after, th)
        surv = survival(s0, labels, after, after_labels, r_tol)
        entry = {
            "mean_relax_disp": float(torch.cat(disp).mean()),
            "seconds": time.time() - t0,
        }
        for name, per_glass in surv.items():
            n_sites = int(sum(len(x) for x in per_glass))
            entry[name] = {
                "n_sites": n_sites,
                "survival": list(
                    bootstrap_ci_items(
                        per_glass, lambda sel: float(np.concatenate(sel).mean()), n_boot
                    )
                )
                if n_sites
                else None,
                "per_glass_before": float((labels == DefectClass[name]).sum(1).double().mean()),
                "per_glass_after": float(
                    (after_labels == DefectClass[name]).sum(1).double().mean()
                ),
            }
        results["sigmas"][f"{sigma:g}"] = entry
        print(
            f"sigma={sigma:g}: "
            + ", ".join(
                f"{k} {entry[k]['survival'][0]:.3f} (n={entry[k]['n_sites']})"
                for k in surv
                if entry[k]["survival"]
            )
            + f"; mean relaxation displacement {entry['mean_relax_disp']:.3f}",
            flush=True,
        )
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print("->", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
