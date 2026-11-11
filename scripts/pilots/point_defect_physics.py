"""What each 2D point defect is after relaxation: the physics behind the find-and-repair test.

    python scripts/pilots/point_defect_physics.py \
        --out results/point_defects_recheck_20261111/defect_physics.json

For every split it counts the examples that are the original glass again after relaxation
(glassdiff.data.point_defects.already_original): Frenkel pairs that recombined by exchange.
For the real defects of one split (default: test) it then measures:

    atoms moved        other atoms displaced by more than 0.1 sigma by the defect
    energy             E(relaxed defected glass) - E(original), for the two defects that keep
                       the atoms (Frenkel pair, antisite): the formation energy
    vacancy            the open space at the empty site (distance from the most open point
                       within 1 sigma of the site to the nearest atom), right after removal and
                       after relaxation, against the same measure around atoms of defect-free
                       glasses
    interstitial       the extra atom's nearest-neighbour distance and energy, against B atoms
                       of defect-free glasses
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from glassdiff.data.point_defects import (
    PointDefect,
    already_original,
    largest_gap,
    load_point_defects,
    wrap,
)
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import Structures


def quartiles(x) -> dict:
    q = np.percentile(np.asarray(x, dtype=float), [25, 50, 75])
    return dict(median=round(float(q[1]), 3), q25=round(float(q[0]), 3), q75=round(float(q[2]), 3))


def clearance(pos: np.ndarray, box: np.ndarray, point: np.ndarray) -> float:
    return float(np.linalg.norm(wrap(pos - point, box), axis=-1).min())


def nearest(pos: np.ndarray, box: np.ndarray, i: int) -> float:
    d = np.linalg.norm(wrap(pos - pos[i], box), axis=-1)
    d[i] = np.inf
    return float(d.min())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/ka2d_256_point_defects")
    ap.add_argument("--split", default="test", help="split for the measurements")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = Path(args.data)
    out: dict = dict(data=str(root), already_original={}, split=args.split, kinds={})

    for split in ("train", "val", "test"):
        path = root / f"{split}.npz"
        if not path.exists():
            continue
        data = load_point_defects(path)
        out["already_original"][split] = {
            name: dict(examples=len(d), already_original=int(already_original(d).sum()))
            for name, d in data.items()
        }
        print(split, json.dumps(out["already_original"][split]), flush=True)

    data = load_point_defects(root / f"{args.split}.npz")
    rng = np.random.default_rng(0)
    # reference: open space around atoms of the defect-free glasses (4 atoms per glass)
    ref = next(iter(data.values()))
    _, first = np.unique(ref.glass.numpy(), return_index=True)
    ref_gap = []
    for b in first:
        p, box = ref.orig_pos[b].numpy(), ref.box[b].numpy()
        for i in rng.choice(len(p), 4, replace=False):
            ref_gap.append(clearance(p, box, largest_gap(p, box, p[i], 0.0, 1.0)))
    ref_gap = np.array(ref_gap)
    out["open_space_in_defect_free_glass"] = dict(
        samples=len(ref_gap), **quartiles(ref_gap), p95=round(float(np.percentile(ref_gap, 95)), 3)
    )

    for name, d in data.items():
        d = d.subset(~already_original(d))
        pos, box = d.pos.numpy(), d.box.numpy()
        orig, otypes = d.orig_pos.numpy(), d.orig_types.numpy()
        m, atom = len(d), d.atom.numpy()
        res: dict = dict(real_defects=m)
        moved, largest = [], []
        for b in range(m):
            if d.kind == PointDefect.VACANCY:
                before, after = np.delete(orig[b], atom[b], 0), pos[b]
            elif d.kind == PointDefect.INTERSTITIAL:
                before, after = orig[b], pos[b][:-1]
            else:
                before, after = orig[b], pos[b]
            disp = np.linalg.norm(wrap(after - before, box[b]), axis=-1)
            if d.kind == PointDefect.FRENKEL:
                disp = np.delete(disp, atom[b])  # the moved atom itself is not counted
            moved.append(int((disp > 0.1).sum()))
            largest.append(float(disp.max()))
        res["other_atoms_moved_over_0.1"] = quartiles(moved)
        res["largest_move_of_another_atom"] = quartiles(largest)
        if d.kind in (PointDefect.FRENKEL, PointDefect.ANTISITE):
            de = (
                ka_energy(Structures(d.pos, d.types, d.box))
                - ka_energy(Structures(d.orig_pos, d.orig_types, d.box))
            ).numpy()
            res["energy_above_original"] = dict(
                **quartiles(de),
                fraction_above=round(float((de > 1e-3).mean()), 3),
                fraction_below=round(float((de < -1e-3).mean()), 3),
            )
        if d.kind == PointDefect.VACANCY:
            removed = [nearest(orig[b], box[b], atom[b]) for b in range(m)]
            relaxed = np.array(
                [
                    clearance(pos[b], box[b], largest_gap(pos[b], box[b], d.sites[b, 0].numpy(), 0.0, 1.0))
                    for b in range(m)
                ]
            )  # fmt: skip
            res["open_space_right_after_removal"] = quartiles(removed)
            res["open_space_after_relaxation"] = quartiles(relaxed)
            res["fraction_more_open_than_95pct_of_defect_free"] = round(
                float((relaxed > np.percentile(ref_gap, 95)).mean()), 3
            )
            res["removed_atom_is_A"] = round(float((otypes[np.arange(m), atom] == 0).mean()), 3)
        if d.kind == PointDefect.INTERSTITIAL:
            e_def = ka_energy(Structures(d.pos, d.types, d.box), per_atom=True).numpy()
            e_ref = ka_energy(Structures(d.orig_pos, d.orig_types, d.box), per_atom=True).numpy()
            b_ref = [
                nearest(orig[b], box[b], i) for b in range(0, m, 4) for i in np.nonzero(otypes[b] == 1)[0][:10]
            ]  # fmt: skip
            res["extra_atom_nearest_neighbour"] = quartiles(
                [nearest(pos[b], box[b], -1) for b in range(m)]
            )
            res["B_atom_nearest_neighbour_defect_free"] = quartiles(b_ref)
            res["extra_atom_energy"] = quartiles(e_def[:, -1])
            res["B_atom_energy_defect_free"] = quartiles(e_ref[otypes == 1])
        if d.kind == PointDefect.FRENKEL:
            hop = np.linalg.norm(wrap((d.sites[:, 1] - d.sites[:, 0]).numpy(), box), axis=-1)
            res["distance_site_to_moved_atom"] = quartiles(hop)
        out["kinds"][name] = res
        print(name, json.dumps(res), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
