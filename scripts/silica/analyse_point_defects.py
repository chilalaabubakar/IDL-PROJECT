"""Silica point defects, stage 1: what the study's summary does not show.

    python scripts/silica/analyse_point_defects.py results/silica_point_defects_<date>

Reads the saved steps of scripts/silica/point_defects_study.py (no minimisation) and writes
analysis.json next to them:

    natural      off-ideal bonding states of each defect-free original glass
    find         finding rate of each finder at hit radii 2.5 (the study's), 3.0 and 3.5 A,
                 over all defects and per glass, and the median distance of the flagged atom;
                 also, as a diagnostic, the two coordination scores with only off-ideal atoms
                 allowed to be flagged ("off-ideal atoms only")
    relaxed      what each defect became when relaxed: for an oxygen vacancy the Si-Si
                 distance and the coordination of its two Si; for an interstitial the
                 bonding of the inserted O and whether the extra O is still that atom; for a
                 Frenkel pair the bonding of the moved O; for a silicon vacancy how many of
                 its four O stay non-bridging

The finding rates at 2.5 A are recomputed exactly as in the study and checked against it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from glassdiff.silica.defects import OXYGEN, SILICON
from glassdiff.silica.point_defects import (
    IDEAL_CN,
    SilicaDefectConfig,
    coordination,
    coordination_score,
    coordination_states,
    rarity_table,
    wrap,
)

KINDS = ("o_vacancy", "o_interstitial", "o_frenkel", "si_vacancy")
FINDERS = ("coordination", "coordination, rare states first", "energy")
OFF_ONLY = (
    "coordination, off-ideal atoms only",
    "coordination, rare states first, off-ideal atoms only",
)
RADII = (2.5, 3.0, 3.5)


def label(species: int, cn: int, peroxy: int = 0) -> str:
    return f"{'Si' if species == SILICON else 'O'}{cn}" + (" peroxy" if peroxy else "")


def scores(kind, pos, types, pe, box, cfg, rarity, stats, seed) -> dict[str, np.ndarray]:
    """The study's finder scores (scripts/silica/point_defects_study.py)."""
    tie = 1e-6 * np.random.default_rng(seed).random(len(types))
    out = {
        "coordination": coordination_score(pos, types, box, cfg) + tie,
        "coordination, rare states first": coordination_score(pos, types, box, cfg, rarity=rarity)
        + tie,
        "energy": np.where(
            types == OXYGEN,
            (pe - stats[OXYGEN][0]) / stats[OXYGEN][1],
            (pe - stats[SILICON][0]) / stats[SILICON][1],
        ),
    }
    cn, _, peroxy = coordination(pos, types, box, cfg)
    off = (cn != np.where(types == SILICON, IDEAL_CN[SILICON], IDEAL_CN[OXYGEN])) | (peroxy > 0)
    for name, base in zip(OFF_ONLY, FINDERS[:2]):  # diagnostic, not one of the study's finders
        out[name] = np.where(off, out[base], -np.inf)
    if kind in ("o_interstitial", "o_frenkel"):  # the extra or moved atom is an O
        out = {f: np.where(types == OXYGEN, s, -np.inf) for f, s in out.items()}
    return out


def relaxed_state(kind, e, d, orig, cfg) -> dict:
    """What one relaxed defect became (see the module docstring)."""
    pos, types, g = d["pos"][e], d["types"][e], int(d["glass"][e])
    box, po, to = orig["box"][g], orig["pos"][g], orig["types"][g]
    a = int(d["atom"][e])
    cn, _, peroxy = coordination(pos, types, box, cfg)
    if kind in ("o_vacancy", "si_vacancy"):
        r0 = np.linalg.norm(wrap(po - po[a], box), axis=1)
        other = SILICON if kind == "o_vacancy" else OXYGEN
        shell0 = np.nonzero((to == other) & (r0 < cfg.si_o_cutoff))[0]
        shell = shell0 - (shell0 > a)  # indices in the defected glass
        if kind == "si_vacancy":
            return {"non-bridging O left (of 4)": int((cn[shell] == 1).sum())}
        out = {"Si coordination": "+".join(str(c) for c in sorted(cn[shell]))}
        if len(shell) == 2:
            out["Si-Si before"] = float(np.linalg.norm(wrap(po[shell0[0]] - po[shell0[1]], box)))
            out["Si-Si after"] = float(np.linalg.norm(wrap(pos[shell[0]] - pos[shell[1]], box)))
        return out
    if kind == "o_interstitial":
        # the extra O: the O farthest from every O site of the original glass
        tree = cKDTree(np.mod(po[to == OXYGEN], box), boxsize=box)
        oxygens = np.nonzero(types == OXYGEN)[0]
        dist, _ = tree.query(np.mod(pos[oxygens], box))
        extra = int(oxygens[np.argmax(dist)])
        return {
            "inserted O": label(OXYGEN, cn[a], peroxy[a]),
            "extra O is the inserted O": extra == a,
        }
    return {"moved O": label(OXYGEN, cn[a], peroxy[a])}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    cfg = SilicaDefectConfig()
    with np.load(run / "originals.npz") as f:
        orig = {k: f[k] for k in f.files}
    rarity = rarity_table(
        [
            coordination_states(p, t, b, cfg)
            for p, t, b in zip(orig["pos"], orig["types"], orig["box"])
        ]
    )
    stats = {
        sp: (orig["pe"][orig["types"] == sp].mean(), orig["pe"][orig["types"] == sp].std())
        for sp in (OXYGEN, SILICON)
    }
    rates = orig["cooling_rate"]
    by_rate = [int(g) for g in np.argsort(rates)]
    out = {"natural": {}, "kinds": {}}
    for g in by_rate:
        p, t, b = orig["pos"][g], orig["types"][g], orig["box"][g]
        cn, _, peroxy = coordination(p, t, b, cfg)
        ideal = np.where(t == SILICON, IDEAL_CN[SILICON], IDEAL_CN[OXYGEN])
        off = Counter(label(t[i], cn[i]) for i in np.nonzero((cn != ideal) | (peroxy > 0))[0])
        out["natural"][f"{rates[g]:g} K/ps"] = dict(sorted(off.items()))

    for kind in KINDS:
        path = run / f"defects_{kind}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        keep = np.nonzero(d["valid"])[0]
        dist_top = {f: [] for f in FINDERS + OFF_ONLY}
        states = []
        for e in keep:
            pos, types, g = d["pos"][e], d["types"][e], int(d["glass"][e])
            box = orig["box"][g]
            sc = scores(kind, pos, types, d["pe"][e], box, cfg, rarity, stats, int(e))
            dist = np.min(
                [np.linalg.norm(wrap(pos - d["sites"][e, k], box), axis=1) for k in range(2)],
                axis=0,
            )
            for f in FINDERS + OFF_ONLY:
                dist_top[f].append(float(dist[int(np.argmax(sc[f]))]))
            states.append(relaxed_state(kind, int(e), d, orig, cfg))
        glass = d["glass"][keep]
        find = {}
        for f in FINDERS + OFF_ONLY:
            v = np.array(dist_top[f])
            find[f] = {f"{r} A": round(float(np.mean(v < r)), 4) for r in RADII}
            find[f]["median distance (A)"] = round(float(np.median(v)), 2)
            find[f]["per glass"] = {
                f"{rates[g]:g} K/ps": {
                    f"{r} A": round(float(np.mean(v[glass == g] < r)), 2) for r in RADII[:2]
                }
                for g in by_rate
            }
        study = run / f"eval_{kind}.json"
        if study.exists():  # the recomputed rates are the study's
            reported = json.loads(study.read_text())["find"]
            assert all(abs(find[f]["2.5 A"] - reported[f]) < 1e-3 for f in FINDERS), kind
        relaxed = {}
        for key in states[0]:
            values = [s[key] for s in states if key in s]
            if isinstance(values[0], float):
                relaxed[key] = {
                    "mean": round(float(np.mean(values)), 2),
                    "min": round(float(np.min(values)), 2),
                    "below 2.6 A": int(np.sum(np.array(values) < 2.6)),
                }
            else:
                relaxed[key] = dict(sorted(Counter(str(v) for v in values).items()))
        out["kinds"][kind] = dict(
            defects=len(d["valid"]), valid=len(keep), find=find, relaxed=relaxed
        )
        print(kind, json.dumps(out["kinds"][kind]["relaxed"]), flush=True)
    (run / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print("->", run / "analysis.json")


if __name__ == "__main__":
    main()
