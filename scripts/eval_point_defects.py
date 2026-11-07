"""Find and repair point defects in the relaxed defected test glasses (docs/POINT_DEFECTS.md).

    # before training: the trained unconditional denoiser, never shown a defect
    python scripts/eval_point_defects.py --data data/ka2d_256_point_defects/test.npz \\
        --healer results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt \\
        --out results/point_defects_20261107/eval_untrained.json
    # after training (scripts/train_point_defects.py)
    python scripts/eval_point_defects.py --data data/ka2d_256_point_defects/test.npz \\
        --healer runs/<heal run>/ckpt.pt --detector runs/<detect run>/ckpt.pt --out <json>

Finding. Every atom gets a score; the top atom is a hit when it lies within 1.5 sigma of the
defect (either site of a Frenkel pair or antisite). Score: with --detector, the probability of
the role the repair needs (vacancy: next to an empty site; interstitial: extra atom; Frenkel:
displaced atom; antisite: wrong species); without, the size of the healer's output at
sigma 0.05. Reference that needs the potential: the atom whose energy is highest for its
species (z-score). Chance: the fraction of atoms within 1.5 sigma of the defect.

Repair, then FIRE minimisation (glassdiff.physics.fire, all atoms free, max force 1e-6, as
every earlier evaluation) to see which minimum the glass settles in. Not LAMMPS: its FIRE
cannot start from overlapping atoms (a shaken region), and the minimum reached from such a
start depends on the minimiser, so one minimiser is used for every method.
Restored = the result is the original glass: every original atom has an atom of its species
within 0.05 sigma, one to one (identical atoms may trade places), after removing a global
shift.
    vacancy       put an atom of the missing species in the largest gap within 1.5 sigma of
                  the flagged atom; minimise, or first heal the 3-sigma region from sigma 0.2
    interstitial  remove the flagged atom; minimise, or first heal the region from sigma 0.2
    Frenkel pair  the region around both sites (3.5 sigma) is given; shake it by 0.35 and
                  minimise, or shake and heal from sigma 0.35, then minimise
    antisite      among A-B neighbour pairs (< 1.6 sigma) with an atom within 1.5 sigma of the
                  flagged atom, swap back the pair the model prefers (detector: the highest
                  summed wrong-species probability; else the smallest summed squared healer
                  output after the swap) or, with the potential, the lowest energy; minimise
"Heal" runs x <- x - model(x, sigma_k) inside the region, sigma_k from the start value to
0.001 in 30 steps (no added noise). Every random choice is seeded, so a run is repeatable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.point_defects import (
    NAMES,
    AtomRole,
    PointDefect,
    PointDefectSet,
    largest_gap,
    load_point_defects,
    wrap,
)
from glassdiff.models.registry import load_denoiser
from glassdiff.physics.fire import fire_minimize_many
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import Structures

REPAIR_ROLE = {
    PointDefect.VACANCY: AtomRole.NEAR_VACANCY,
    PointDefect.INTERSTITIAL: AtomRole.INTERSTITIAL,
    PointDefect.FRENKEL: AtomRole.DISPLACED,
    PointDefect.ANTISITE: AtomRole.WRONG_SPECIES,
}
HIT_RADIUS, SAME_TOL, BATCH = 1.5, 0.05, 10


class Models:
    def __init__(self, healer: str, detector: str | None, threads: int, device: str = "cpu"):
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.healer = load_denoiser(healer, self.device)
        self.detector = load_denoiser(detector, self.device) if detector else None

    def _batch(self, pos, types, box, i) -> Structures:
        to = dict(device=self.device)
        p = torch.as_tensor(np.asarray(pos[i : i + BATCH]), dtype=torch.float32, **to)
        t = torch.as_tensor(np.asarray(types[i : i + BATCH]), dtype=torch.long, **to)
        b = torch.as_tensor(np.asarray(box[i : i + BATCH]), dtype=torch.float32, **to)
        return Structures(p, t, b)

    @torch.no_grad()
    def heal_out(self, pos, types, box, sigma: float) -> np.ndarray:
        """Healer output [M, N, 2] (float64 numpy) for numpy inputs."""
        res = []
        for i in range(0, len(pos), BATCH):
            s = self._batch(pos, types, box, i)
            sig = torch.full((s.batch_size,), float(sigma), device=self.device)
            res.append(self.healer(s, sig).double().cpu().numpy())
        return np.concatenate(res)

    @torch.no_grad()
    def role_prob(self, pos, types, box) -> np.ndarray:
        """Detector role probabilities [M, N, 5]."""
        res = []
        for i in range(0, len(pos), BATCH):
            s = self._batch(pos, types, box, i)
            res.append(self.detector(s).softmax(-1).double().cpu().numpy())
        return np.concatenate(res)

    def heal(self, pos, types, box, region, sigma0: float, n: int = 30) -> np.ndarray:
        pos = np.array(pos, dtype=np.float64)
        for sig in np.linspace(sigma0, 1e-3, n):
            pos = pos - self.heal_out(pos, types, box, sig) * region[..., None]
        return pos


def energy_z(pos, types, box, mu, sd) -> np.ndarray:
    s = Structures(torch.as_tensor(pos), torch.as_tensor(types).long(), torch.as_tensor(box))
    e = ka_energy(s, per_atom=True).numpy()
    return (e - mu[types]) / sd[types]


def same_structure(pos, types, orig, otypes, box, tol=SAME_TOL) -> bool:
    """Is pos the original glass, allowing identical atoms to trade places?"""
    if len(pos) != len(orig) or not np.array_equal(np.bincount(types), np.bincount(otypes)):
        return False
    shift = np.zeros(2)
    for _ in range(2):  # estimate and remove a global shift, then match
        ok, match, vecs = True, [], []
        for sp in (0, 1):
            a, b = np.nonzero(otypes == sp)[0], np.nonzero(types == sp)[0]
            d = wrap(pos[b][None] - shift - orig[a][:, None], box)
            r = np.linalg.norm(d, axis=-1)
            j = r.argmin(1)
            match.append(b[j])
            vecs.append(d[np.arange(len(a)), j])
            ok &= bool((r.min(1) < tol).all())
        shift = shift + np.concatenate(vecs).mean(0)
    match = np.concatenate(match)
    return ok and len(np.unique(match)) == len(match)


def frac(x) -> float:
    return round(float(np.mean(x)), 3)


def evaluate_kind(d: PointDefectSet, models: Models, mu, sd, rng, workers: int) -> dict:
    kind = d.kind
    pos, types = d.pos.numpy(), d.types.numpy()
    box, sites = d.box.numpy(), d.sites.numpy()
    orig, otypes = d.orig_pos.numpy(), d.orig_types.numpy()
    m = len(pos)
    ar = np.arange(m)
    # --- finding
    if models.detector is not None:
        score = models.role_prob(pos, types, box)[..., int(REPAIR_ROLE[kind])]
        score_name = "detector"
    else:
        score = np.linalg.norm(models.heal_out(pos, types, box, 0.05), axis=-1)
        score_name = "healer_output"
    ez = energy_z(pos, types, box, mu, sd)

    def hits(sc):
        top = pos[ar, sc.argmax(-1)]
        dd = np.stack([np.linalg.norm(wrap(top - sites[:, k], box), axis=-1) for k in range(2)])
        return dd.min(0) < HIT_RADIUS

    d_all = np.stack(
        [np.linalg.norm(wrap(pos - sites[:, k][:, None], box[:, None]), axis=-1) for k in range(2)]
    ).min(0)
    res = dict(
        examples=m,
        find=dict(
            model=frac(hits(score)),
            energy_needs_potential=frac(hits(ez)),
            chance=frac((d_all < HIT_RADIUS).mean(-1)),
            model_score=score_name,
        ),
    )
    flagged = {"model": score.argmax(-1), "energy": ez.argmax(-1)}

    def restored(results, rtypes):
        return np.array(
            [
                same_structure(p, t, o, ot, b)
                for (p, _), t, o, ot, b in zip(results, rtypes, orig, otypes, box)
            ]  # fmt: skip
        )

    def minimise(pos_list, types_list):
        return fire_minimize_many(list(zip(pos_list, types_list, box)), workers=workers)

    rep: dict = {}
    if kind == PointDefect.VACANCY:
        for who, f in flagged.items():
            full, region = [], []
            for b in range(m):
                gap = largest_gap(pos[b], box[b], pos[b, f[b]], 0.0, 1.5)
                full.append(np.insert(pos[b], d.atom[b].item(), gap, axis=0))
                region.append(np.linalg.norm(wrap(full[-1] - gap, box[b]), axis=-1) < 3.0)
            full, region = np.stack(full), np.stack(region)
            rep[f"flag by {who}: insert, minimise"] = frac(restored(minimise(full, otypes), otypes))
            if who == "model":
                healed = models.heal(full, otypes, box, region, 0.2)
                rep["flag by model: insert, heal, minimise"] = frac(
                    restored(minimise(healed, otypes), otypes)
                )
    elif kind == PointDefect.INTERSTITIAL:
        for who, f in flagged.items():
            keep = np.ones(pos.shape[:2], dtype=bool)
            keep[ar, f] = False
            cut = pos[keep].reshape(m, -1, 2)
            ctypes = types[keep].reshape(m, -1)
            rep[f"flag by {who}: removed the extra atom"] = frac(f == pos.shape[1] - 1)
            rep[f"flag by {who}: remove, minimise"] = frac(restored(minimise(cut, ctypes), ctypes))
            if who == "model":
                region = np.linalg.norm(wrap(cut - pos[ar, f][:, None], box[:, None]), axis=-1) < 3
                healed = models.heal(cut, ctypes, box, region, 0.2)
                rep["flag by model: remove, heal, minimise"] = frac(
                    restored(minimise(healed, ctypes), ctypes)
                )
    elif kind == PointDefect.FRENKEL:
        centre = sites[:, 0] + 0.5 * wrap(sites[:, 1] - sites[:, 0], box)
        region = np.linalg.norm(wrap(pos - centre[:, None], box[:, None]), axis=-1) < 3.5
        shaken = pos + rng.normal(0, 0.35, pos.shape) * region[..., None]
        out_a = minimise(shaken, types)
        out_b = minimise(models.heal(shaken, types, box, region, 0.35), types)
        e0 = ka_energy(Structures(d.orig_pos, d.orig_types, d.box)).numpy() / pos.shape[1]
        for name, out in (("shake, minimise", out_a), ("shake, heal, minimise", out_b)):
            p = np.stack([o[0] for o in out])
            e = ka_energy(Structures(torch.as_tensor(p), d.types, d.box)).numpy() / pos.shape[1]
            dd = np.linalg.norm(wrap(p - orig, box[:, None]), axis=-1)
            rep[name] = dict(
                restored=frac(restored(out, types)),
                atoms_within_0_1=frac((dd < 0.1).mean(-1)),
                energy_above_original=round(float((e - e0).mean()), 4),
            )
    else:  # antisite
        prob = None
        if models.detector is not None:
            prob = models.role_prob(pos, types, box)[..., int(AtomRole.WRONG_SPECIES)]
        choice = {"model": [], "energy": []}
        for b in range(m):
            near = np.nonzero(np.linalg.norm(wrap(pos[b] - pos[b, flagged["model"][b]], box[b]),
                                             axis=-1) < 1.5)[0]  # fmt: skip
            pairs = set()
            for p in near:
                dq = np.linalg.norm(wrap(pos[b] - pos[b, p], box[b]), axis=-1)
                for q in np.nonzero((dq < 1.6) & (types[b] != types[b, p]))[0]:
                    pairs.add(tuple(sorted((int(p), int(q)))))
            pairs = sorted(pairs)
            if not pairs:  # nothing to swap: leave the species as they are
                choice["model"].append(types[b])
                choice["energy"].append(types[b])
                continue
            swapped = []
            for p, q in pairs:
                t = types[b].copy()
                t[p], t[q] = types[b, q], types[b, p]
                swapped.append(t)
            sw = np.stack(swapped)
            e = energy_z(np.repeat(pos[b][None], len(sw), 0), sw, np.repeat(box[b][None], len(sw), 0),
                         np.zeros(2), np.ones(2)).sum(-1)  # fmt: skip
            if prob is not None:
                crit = -np.array([prob[b, p] + prob[b, q] for p, q in pairs])
            else:
                out = models.heal_out(np.repeat(pos[b][None], len(sw), 0), sw,
                                      np.repeat(box[b][None], len(sw), 0), 0.05)  # fmt: skip
                crit = (out**2).sum((1, 2))
            choice["model"].append(sw[int(np.argmin(crit))])
            choice["energy"].append(sw[int(np.argmin(e))])
        for who in ("model", "energy"):
            new_types = np.stack(choice[who])
            right = (new_types == otypes).all(-1)
            rep[f"search near model flag, pick by {who}: right pair swapped back"] = frac(right)
            rep[f"search near model flag, pick by {who}: swap, minimise"] = frac(
                restored(minimise(pos, new_types), new_types)
            )
    res["repair"] = rep
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default="data/ka2d_256_point_defects/test.npz")
    ap.add_argument("--healer", required=True, help="denoiser checkpoint (EGNN)")
    ap.add_argument("--detector", help="classifier checkpoint (optional)")
    ap.add_argument("--max-examples", type=int, default=200, help="per defect kind")
    ap.add_argument("--kinds", nargs="*", default=list(NAMES.values()))
    ap.add_argument("--workers", type=int, default=None, help="minimisation processes")
    ap.add_argument("--threads", type=int, default=4, help="torch threads")
    ap.add_argument("--device", default="cpu", help="for the models (minimisation is on CPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    data = load_point_defects(args.data)
    models = Models(args.healer, args.detector, args.threads, args.device)
    # per-species energy statistics of the defect-free originals (z-score reference)
    any_set = next(iter(data.values()))
    e = ka_energy(Structures(any_set.orig_pos, any_set.orig_types, any_set.box), per_atom=True)
    t = any_set.orig_types
    mu = np.array([float(e[t == s].mean()) for s in (0, 1)])
    sd = np.array([float(e[t == s].std()) for s in (0, 1)])
    out = dict(data=args.data, healer=args.healer, detector=args.detector, seed=args.seed, kinds={})
    for i, name in enumerate(args.kinds):
        d = data[name].subset(slice(0, args.max_examples))
        rng = np.random.default_rng([args.seed, i])
        out["kinds"][name] = evaluate_kind(d, models, mu, sd, rng, args.workers)
        print(name, json.dumps(out["kinds"][name]), f"{time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
