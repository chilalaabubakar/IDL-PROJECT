"""What two simple structural measures see of a relaxed 2D point defect, and which rule to use.

    python scripts/pilots/structural_signatures.py \
        --out results/point_defects_recheck_20261111/structural_signatures.json

The measures are those of scripts/pilots/baseline_routes.py: the Voronoi cell area and the
number of unlike first-shell neighbours (A-B bonds), each as a z-score within its species.

    defect_free   mean and spread of both measures per species, and the correlation of the
                  atom energy with the number of A-B bonds
    antisite      both measures on the two exchanged atoms after relaxation (test split)
    rules         defects found (flagged atom within 1.5 sigma) by each simple rule, on the
                  validation split: the rules of the baselines were chosen here, not on test

Only real defects are used (Frenkel pairs that are the original glass again are left out).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("routes", ROOT / "scripts/pilots/baseline_routes.py")
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)
ev = br.ev


def reference(full: dict) -> dict:
    """Both measures and the atom energy in the defect-free originals (one per glass)."""
    ref = next(iter(full.values()))
    _, first = np.unique(ref.glass.numpy(), return_index=True)
    ref = ref.subset(torch.from_numpy(first))
    pos, types, box = ref.orig_pos.numpy(), ref.orig_types.numpy(), ref.box.numpy()
    area = np.stack([br.voronoi_area(p, b) for p, b in zip(pos, box)])
    bonds = np.stack([br.unlike_neighbours(p, t, b) for p, t, b in zip(pos, types, box)])
    energy = ev.ka_energy(
        ev.Structures(ref.orig_pos, ref.orig_types, ref.box), per_atom=True
    ).numpy()
    return dict(types=types, area=area, bonds=bonds, energy=energy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/ka2d_256_point_defects")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out: dict = {"data": args.data}

    # ---- defect-free glasses (test split)
    test = ev.load_point_defects(f"{args.data}/test.npz")
    ref = reference(test)
    t = ref["types"]
    out["defect_free"] = {}
    for s, name in ((0, "A"), (1, "B")):
        out["defect_free"][name] = dict(
            cell_area=dict(
                mean=round(float(ref["area"][t == s].mean()), 3),
                sd=round(float(ref["area"][t == s].std()), 3),
            ),
            unlike_neighbours=dict(
                mean=round(float(ref["bonds"][t == s].mean()), 2),
                sd=round(float(ref["bonds"][t == s].std()), 2),
            ),
            correlation_energy_with_unlike_neighbours=round(
                float(np.corrcoef(ref["energy"][t == s], ref["bonds"][t == s])[0, 1]), 2
            ),
        )
    print(json.dumps(out["defect_free"]), flush=True)

    # ---- the two exchanged atoms of an antisite (test split)
    mu_a, sd_a = br.species_stats(ref["area"], t)
    mu_u, sd_u = br.species_stats(ref["bonds"], t)
    d = test["antisite"]
    pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
    otypes = d.orig_types.numpy()
    za = br.area_z(pos, types, box, mu_a, sd_a)
    bonds = np.stack([br.unlike_neighbours(p, ty, b) for p, ty, b in zip(pos, types, box)])
    swapped = types != otypes
    out["antisite"] = {}
    for s, name in ((0, "A atom on a B site"), (1, "B atom on an A site")):
        m = swapped & (types == s)
        out["antisite"][name] = dict(
            atoms=int(m.sum()),
            cell_area_z_median=round(float(np.median(za[m])), 2),
            cell_area_larger_than_usual=round(float((za[m] > 0).mean()), 2),
            unlike_neighbours_mean=round(float(bonds[m].mean()), 2),
            unlike_neighbours_usual=round(float(mu_u[s]), 2),
            fewer_unlike_neighbours_than_usual=round(float((bonds[m] < mu_u[s]).mean()), 2),
        )
    print(json.dumps(out["antisite"]), flush=True)

    # ---- which rule finds each defect: validation split
    val = ev.load_point_defects(f"{args.data}/val.npz")
    vref = reference(val)
    mu_a, sd_a = br.species_stats(vref["area"], vref["types"])
    mu_u, sd_u = br.species_stats(vref["bonds"], vref["types"])
    out["rules"] = dict(split="val", found={})
    for name, d in ev.real_defects(val).items():
        pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
        sites = d.sites.numpy()
        za = br.area_z(pos, types, box, mu_a, sd_a)
        zu = br.bonds_z(pos, types, box, mu_u, sd_u)
        dist = np.stack(
            [
                np.linalg.norm(br.wrap(pos - sites[:, k][:, None], box[:, None]), axis=-1)
                for k in range(2)
            ]
        ).min(0)
        rules = {
            "cell: largest": za,
            "cell: smallest": -za,
            "cell: most unusual": np.abs(za),
            "cell: B large or A small": np.where(types == 1, za, -za),
            "cell: A large or B small": np.where(types == 1, -za, za),
            "unlike neighbours: fewest": -zu,
            "unlike neighbours: most": zu,
        }
        ar = np.arange(len(pos))
        out["rules"]["found"][name] = dict(
            examples=len(pos),
            **{k: ev.frac(dist[ar, v.argmax(-1)] < ev.HIT_RADIUS) for k, v in rules.items()},
        )
        print(name, json.dumps(out["rules"]["found"][name]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
