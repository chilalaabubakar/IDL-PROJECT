"""Finding and repairing 2D point defects without a learned model: two baselines.

    geometry   Voronoi cell area, z-score within species. No potential, no learning: the
               simplest structural indicator of missing or extra free volume.
    energy     per-atom energy, z-score within species. Needs the interatomic potential.

Each baseline flags one atom per glass. "Found" means the flagged atom lies within 1.5 sigma
of the defect. "Restored" applies the repair steps that scripts/eval_point_defects.py applies
for the learned finder, then FIRE minimisation, and asks for the original glass (identical
atoms may trade places):

    vacancy       an atom of the missing species into the largest gap within 1.5 sigma of the
                  flagged atom
    interstitial  the flagged atom removed
    Frenkel pair  the flagged atom moved to one of up to 8 gaps 1-3.2 sigma away
                  (geometry: the gap whose neighbours have the largest cells; the energy
                  route, which minimises every gap, is run by scripts/eval_point_defects.py)
    antisite      the neighbouring A-B pair near the flagged atom swapped back
                  (geometry: the pair with the most misfitting cells; energy: the pair whose
                  swap gives the lowest energy)

Which atom a baseline flags follows the physics of each defect:

    geometry  vacancy: the largest cell; interstitial: the smallest cell; Frenkel pair: the
              smallest cell (the squeezed, displaced atom) for repair and the most unusual
              cell for finding; antisite: a B atom with too large a cell or an A atom with
              too small a cell
    energy    the highest energy

Reference statistics (mean and standard deviation per species) come from the defect-free
originals of the same split. Results per example are saved, with the glass each came from.

    python scripts/pilots/baseline_routes.py --real-defects-only \
        --out results/point_defects_recheck_20261111/baseline_routes.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Voronoi

from glassdiff.data.point_defects import PointDefect, candidate_gaps, largest_gap

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("eval_pd", ROOT / "scripts/eval_point_defects.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
wrap = ev.wrap


def voronoi_area(pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Area of each atom's Voronoi cell in the periodic box (the areas sum to the box area)."""
    n = len(pos)
    shifts = np.array([[i, j] for i in (0, -1, 1) for j in (0, -1, 1)]) * box  # centre first
    vor = Voronoi(np.concatenate([np.mod(pos, box) + s for s in shifts]))
    area = np.empty(n)
    for i in range(n):
        x, y = vor.vertices[vor.regions[vor.point_region[i]]].T
        area[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def area_z(pos, types, box, mu, sd) -> np.ndarray:
    return np.stack([(voronoi_area(p, b) - mu[t]) / sd[t] for p, t, b in zip(pos, types, box)])


def species_stats(values: np.ndarray, types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.array([values[types == s].mean() for s in (0, 1)])
    sd = np.array([values[types == s].std() for s in (0, 1)])
    return mu, sd


def evaluate(d, name: str, za: np.ndarray, ze: np.ndarray, workers) -> dict:
    kind = d.kind
    pos, types = d.pos.numpy(), d.types.numpy()
    box, sites = d.box.numpy(), d.sites.numpy()
    orig, otypes = d.orig_pos.numpy(), d.orig_types.numpy()
    m, ar = len(pos), np.arange(len(pos))
    misfit = np.where(types == 1, za, -za)  # B with too large a cell, A with too small a cell
    find_score = {
        "geometry": {
            PointDefect.VACANCY: za,
            PointDefect.INTERSTITIAL: -za,
            PointDefect.FRENKEL: np.abs(za),
            PointDefect.ANTISITE: misfit,
        }[kind],
        "energy": ze,
    }
    repair_score = dict(find_score)
    if kind == PointDefect.FRENKEL:
        repair_score["geometry"] = -za  # the displaced atom is the squeezed one
    dist = np.stack(
        [np.linalg.norm(wrap(pos - sites[:, k][:, None], box[:, None]), axis=-1) for k in range(2)]
    ).min(0)

    def restored(new_pos, new_types) -> np.ndarray:
        out = ev.fire_minimize_many(list(zip(new_pos, new_types, box)), workers=workers)
        return np.array(
            [
                ev.same_structure(p, t, o, ot, b)
                for (p, _), t, o, ot, b in zip(out, new_types, orig, otypes, box)
            ]
        )

    res = dict(examples=m, glass=d.glass.tolist(), chance=ev.frac((dist < ev.HIT_RADIUS).mean(-1)))
    for who in ("geometry", "energy"):
        top = find_score[who].argmax(-1)
        found = dist[ar, top] < ev.HIT_RADIUS
        f = repair_score[who].argmax(-1)
        ok = None
        if kind == PointDefect.VACANCY:
            full = [
                np.insert(pos[b], d.atom[b].item(), largest_gap(pos[b], box[b], pos[b, f[b]], 0.0, 1.5), axis=0)
                for b in range(m)
            ]  # fmt: skip
            ok = restored(np.stack(full), otypes)
        elif kind == PointDefect.INTERSTITIAL:
            keep = np.ones(pos.shape[:2], dtype=bool)
            keep[ar, f] = False
            ok = restored(pos[keep].reshape(m, -1, 2), types[keep].reshape(m, -1))
        elif kind == PointDefect.FRENKEL and who == "geometry":
            moved = []
            for b in range(m):
                gaps = candidate_gaps(pos[b], box[b], pos[b, f[b]], 1.0, 3.2, k=8, exclude=f[b])
                others = np.delete(np.arange(pos.shape[1]), f[b])
                crit = []
                for g in gaps:  # an empty site is ringed by atoms with enlarged cells
                    ring = np.linalg.norm(wrap(pos[b, others] - g, box[b]), axis=-1) < 1.3
                    crit.append(za[b, others][ring].mean() if ring.any() else -np.inf)
                new = pos[b].copy()
                new[f[b]] = gaps[int(np.argmax(crit))]
                moved.append(new)
            ok = restored(np.stack(moved), types)
        elif kind == PointDefect.ANTISITE:
            new_types = []
            for b in range(m):
                near = np.nonzero(
                    np.linalg.norm(wrap(pos[b] - pos[b, f[b]], box[b]), axis=-1) < 1.5
                )[0]
                pairs = set()
                for p in near:
                    dq = np.linalg.norm(wrap(pos[b] - pos[b, p], box[b]), axis=-1)
                    for q in np.nonzero((dq < 1.6) & (types[b] != types[b, p]))[0]:
                        pairs.add(tuple(sorted((int(p), int(q)))))
                pairs = sorted(pairs)
                if not pairs:
                    new_types.append(types[b])
                    continue
                swapped = np.repeat(types[b][None], len(pairs), 0)
                for k, (p, q) in enumerate(pairs):
                    swapped[k, p], swapped[k, q] = types[b, q], types[b, p]
                if who == "geometry":
                    crit = -np.array([misfit[b, p] + misfit[b, q] for p, q in pairs])
                else:  # total energy after the swap, positions unchanged
                    rep = len(pairs)
                    crit = ev.energy_z(
                        np.repeat(pos[b][None], rep, 0), swapped, np.repeat(box[b][None], rep, 0),
                        np.zeros(2), np.ones(2),
                    ).sum(-1)  # fmt: skip
                new_types.append(swapped[int(np.argmin(crit))])
            new_types = np.stack(new_types)
            ok = restored(pos, new_types)
        res[who] = dict(found=ev.frac(found), found_each=found.astype(int).tolist())
        if ok is not None:
            res[who].update(restored=ev.frac(ok), restored_each=ok.astype(int).tolist())
    line = {k: {a: b for a, b in v.items() if not a.endswith("_each")} for k, v in res.items()
            if isinstance(v, dict)}  # fmt: skip
    print(name, m, json.dumps(line), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/ka2d_256_point_defects/test.npz")
    ap.add_argument("--max-examples", type=int, default=200, help="per defect kind")
    ap.add_argument("--offset", type=int, default=0, help="skip this many examples per kind")
    ap.add_argument("--kinds", nargs="*", help="defect kinds (default: all)")
    ap.add_argument("--workers", type=int, default=None, help="minimisation processes")
    ap.add_argument(
        "--real-defects-only",
        action="store_true",
        help="leave out examples that relaxed back to the original glass",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    full = ev.load_point_defects(args.data)
    # reference statistics from the defect-free originals (one per glass)
    ref = next(iter(full.values()))
    _, first = np.unique(ref.glass.numpy(), return_index=True)
    ref = ref.subset(torch.from_numpy(first))
    rp, rt, rb = ref.orig_pos.numpy(), ref.orig_types.numpy(), ref.box.numpy()
    mu_a, sd_a = species_stats(np.stack([voronoi_area(p, b) for p, b in zip(rp, rb)]), rt)
    # energy statistics as in scripts/eval_point_defects.py (the originals of every example)
    first_kind = next(iter(full.values()))
    e = ev.ka_energy(
        ev.Structures(first_kind.orig_pos, first_kind.orig_types, first_kind.box), per_atom=True
    )
    mu_e, sd_e = species_stats(e.numpy(), first_kind.orig_types.numpy())
    data = {
        name: d.subset(slice(args.offset, args.offset + args.max_examples))
        for name, d in full.items()
        if not args.kinds or name in args.kinds
    }
    out = dict(
        data=args.data,
        cell_area=dict(mean=mu_a.round(4).tolist(), sd=sd_a.round(4).tolist()),
        kinds={},
    )
    if args.real_defects_only:
        kept = ev.real_defects(data)
        out["left_out_already_original"] = {k: len(data[k]) - len(kept[k]) for k in data}
        data = kept
    for name, d in data.items():
        pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
        za = area_z(pos, types, box, mu_a, sd_a)
        ze = ev.energy_z(pos, types, box, mu_e, sd_e)
        out["kinds"][name] = evaluate(d, name, za, ze, args.workers)
    out["seconds"] = round(time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
