"""What each relaxed defect is, judged by the sites of its original glass (definition C).

    python scripts/pilots/defect_occupancy.py \
        --out results/point_defects_recheck_20261111/defect_occupancy.json

The Frenkel pair is made by an operation (move one atom 1.2-3 sigma into a gap), and the
relaxed result is not always a Frenkel pair. Here every relaxed defected glass is judged
against its original glass (glassdiff.data.point_defects.site_occupancy): each atom is
assigned to the nearest atom of the original glass, as in a Wigner-Seitz analysis of a
crystal, and the empty and doubly occupied sites are counted. A Frenkel pair exists only if
exactly one site is empty and one is doubly occupied, more than one neighbour shell
(1.4 sigma_ab) apart.

    validation   the rule applied to the defects whose answer is known: vacancies (should
                 give one empty site), interstitials (one doubly occupied site), antisites
                 (no empty or doubly occupied site, the species changed) and the Frenkel
                 pairs that undid themselves (no defect at all); validation and test splits
    frenkel      the class of every test Frenkel pair, in the order of the dataset, with
                 whether it is one of the real defects of the report (already_original)
    vacancy      the species of the removed atom of every real test vacancy, in order
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from glassdiff.data.point_defects import (
    OCCUPANCY_CLASSES,
    already_original,
    load_point_defects,
    real_defects,
    site_occupancy,
)


def classify(d) -> list[dict]:
    pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
    orig, otypes = d.orig_pos.numpy(), d.orig_types.numpy()
    return [site_occupancy(pos[e], types[e], orig[e], otypes[e], box[e]) for e in range(len(d))]


def counts(classes) -> dict:
    c = collections.Counter(classes)
    return {k: c[k] for k in OCCUPANCY_CLASSES if c[k]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data/ka2d_256_point_defects")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out: dict = {"data": args.data, "classes": list(OCCUPANCY_CLASSES), "validation": {}}
    for split in ("val", "test"):
        data = load_point_defects(f"{args.data}/{split}.npz")
        undone = already_original(data["frenkel"]).numpy()
        res: dict = {}
        for name in ("vacancy", "interstitial", "antisite"):
            occ = classify(data[name])
            d = data[name]
            res[name] = dict(examples=len(d), classes=counts(o["class"] for o in occ))
            if name == "vacancy":  # is the empty site the site of the removed atom?
                res[name]["empty_is_removed_atom"] = sum(
                    o["class"] == "vacancy" and o["empty"][0] == int(a) for o, a in zip(occ, d.atom)
                )
            if name == "antisite":  # are the two changed sites the two exchanged atoms?
                res[name]["changed_are_the_exchanged"] = sum(
                    o["class"] == "antisite-like" and set(o["wrong"]) == {int(a), int(b)}
                    for o, a, b in zip(occ, d.atom, d.atom2)
                )
        occ = classify(data["frenkel"])
        res["frenkel_undone"] = dict(
            examples=int(undone.sum()),
            classes=counts(o["class"] for o, u in zip(occ, undone) if u),
        )
        res["frenkel_real"] = dict(
            examples=int((~undone).sum()),
            classes=counts(o["class"] for o, u in zip(occ, undone) if not u),
            pair_without_species_change=sum(
                o["class"] == "pair" and not o["wrong"] for o, u in zip(occ, undone) if not u
            ),
        )
        out["validation"][split] = res
        print(split, json.dumps(res), flush=True)
        if split == "test":
            d = data["frenkel"]
            out["frenkel"] = dict(
                glass=d.glass.tolist(),
                real=(~undone).tolist(),
                cls=[o["class"] for o in occ],
                species_changed=[len(o["wrong"]) for o in occ],
                gap=[o["gap"] for o in occ],
            )
            v = real_defects({"vacancy": data["vacancy"]})["vacancy"]
            removed = v.orig_types.numpy()[np.arange(len(v)), v.atom.numpy()]
            out["vacancy"] = dict(glass=v.glass.tolist(), removed_species=removed.tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
