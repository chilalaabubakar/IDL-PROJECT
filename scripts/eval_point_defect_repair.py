"""Repair without a potential: how close do the repair models bring a defected glass to the
original, with no energy minimisation? (docs/POINT_DEFECTS.md)

    python scripts/eval_point_defect_repair.py --data data/ka2d_256_point_defects/test.npz \\
        --model untrained=results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt \\
        --model v1=results/kaggle_pd_20261107-2024Z/training/pd_heal/ema.pt \\
        --model v2=results/kaggle_pd_20261108-2014Z/training/pd_heal_v2/ema.pt \\
        --out results/point_defects_v2_20261109/repair_no_potential.json

The defect is given: the stored repair input, with the atom count and species already right
at the true site (Frenkel pair: the displaced atom back in the gap at its site). So this
measures repair alone, not finding. Each model is applied to the whole glass, either one step
at sigma 0.001, or 10 steps with sigma falling from 0.05 to 0.001. Scored against the
original (atom i is atom i; a global shift is removed):

    rms           RMS distance over all atoms
    defect_rms    RMS distance over the atoms the defect moved by more than 0.1 sigma
    within_0_1    fraction of atoms within 0.1 sigma of their original position
    energy_above  median energy per atom above the original (scored with the potential,
                  never used for repair; the median, because a few glasses with two atoms
                  almost on top of each other would swamp a mean)
    overlap       fraction of glasses whose closest pair is below 0.56 sigma (the dataset's
                  quality gate, 0.7 sigma_AB): physically broken structures

Reference that needs the potential: FIRE minimisation of the repair input, also reporting
the fraction restored to the original glass (identical atoms may trade places).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.point_defects import (
    heal_target,
    load_point_defects,
    min_pair_distance,
    real_defects,
    return_displaced_atom,
    same_structure,
)
from glassdiff.models.registry import load_denoiser
from glassdiff.physics.fire import fire_minimize_many
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import Structures

BATCH, DEFECT_TOL, MIN_PAIR = 10, 0.1, 0.56


@torch.no_grad()
def apply(model, pos: torch.Tensor, types, box, sigma: float) -> torch.Tensor:
    out = []
    for i in range(0, len(pos), BATCH):
        s = Structures(pos[i : i + BATCH].float(), types[i : i + BATCH], box[i : i + BATCH].float())
        out.append(model(s, torch.full((s.batch_size,), sigma)).double())
    return pos - torch.cat(out)


def score(x, orig, types, box, mask, e0) -> dict:
    d = heal_target(x, orig, box).norm(dim=-1)
    e = ka_energy(Structures(x, types, box)) / x.shape[1]
    closest = [min_pair_distance(p, b) for p, b in zip(x.numpy(), box.numpy())]
    return dict(
        rms=round(float(d.pow(2).mean().sqrt()), 4),
        defect_rms=round(float(d[mask].pow(2).mean().sqrt()), 4),
        within_0_1=round(float((d < 0.1).double().mean()), 4),
        energy_above=round(float((e - e0).median()), 4),
        overlap=round(float(np.mean(np.array(closest) < MIN_PAIR)), 4),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="data/ka2d_256_point_defects/test.npz")
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint")
    ap.add_argument("--kinds", nargs="*", help="defect kinds (default: all)")
    ap.add_argument(
        "--real-defects-only",
        action="store_true",
        help="leave out examples that relaxed back to the original glass",
    )
    ap.add_argument("--max-examples", type=int, default=200, help="per defect kind")
    ap.add_argument("--offset", type=int, default=0, help="skip this many examples per kind")
    ap.add_argument("--no-minimise", action="store_true", help="skip the minimisation reference")
    ap.add_argument("--workers", type=int, default=None, help="minimisation processes")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.time()
    models = {name: load_denoiser(path) for name, path in (m.split("=", 1) for m in args.model)}
    data = load_point_defects(args.data)
    data = {
        name: d.subset(slice(args.offset, args.offset + args.max_examples))
        for name, d in data.items()
        if not args.kinds or name in args.kinds
    }
    out = dict(data=args.data, models=dict(m.split("=", 1) for m in args.model), kinds={})
    if args.real_defects_only:
        kept = real_defects(data)
        out["left_out_already_original"] = {k: len(data[k]) - len(kept[k]) for k in data}
        data = kept
    if "frenkel" in data:
        data["frenkel"] = return_displaced_atom(data["frenkel"])
    for name, d in data.items():
        x0, orig, types, box = d.heal_pos, d.orig_pos, d.orig_types, d.box
        mask = heal_target(x0, orig, box).norm(dim=-1) > DEFECT_TOL
        e0 = ka_energy(Structures(orig, types, box)) / x0.shape[1]
        res = {"examples": len(d), "nothing": score(x0, orig, types, box, mask, e0)}
        for label, model in models.items():
            one = apply(model, x0, types, box, 1e-3)
            x = x0.clone()
            for sig in np.linspace(0.05, 1e-3, 10):
                x = apply(model, x, types, box, float(sig))
            res[f"{label}, one step"] = score(one, orig, types, box, mask, e0)
            res[f"{label}, 10 steps"] = score(x, orig, types, box, mask, e0)
        if not args.no_minimise:
            jobs = list(zip(x0.numpy(), types.numpy(), box.numpy()))
            mins = fire_minimize_many(jobs, workers=args.workers)
            xm = torch.from_numpy(np.stack([p for p, _ in mins]))
            ref = score(xm, orig, types, box, mask, e0)
            ok = [
                same_structure(p, t, o, ot, b)
                for (p, _), t, o, ot, b in zip(
                    mins, types.numpy(), orig.numpy(), types.numpy(), box.numpy()
                )
            ]
            ref["restored"] = round(float(np.mean(ok)), 4)
            res["each"] = dict(glass=d.glass.tolist(), restored=[int(v) for v in ok])
            res["minimise (needs the potential)"] = ref
        out["kinds"][name] = res
        shown = {k: v for k, v in res.items() if k != "each"}
        print(name, json.dumps(shown), f"{time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
