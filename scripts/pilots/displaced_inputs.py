"""Every method against the size of a random displacement of the atoms (docs/NPJ_EXTENSIONS.md, item 3).

    python scripts/pilots/displaced_inputs.py --out results/displaced_inputs_<date> \\
        --finder main=results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt \\
        --finder displaced_s0=results/kaggle_displaced_<date>/training/pd_detect_displaced_s0/ema.pt \\
        --trees tuned=results/descriptor_classifier_20261124/models_tuned.pkl \\
        --trees displaced=results/descriptor_classifier_20261124/models_displaced.pkl

The validation glasses by default (--data); the fresh test set only once, at the end. Every
atom of each relaxed defected glass is shifted by Gaussian noise of standard deviation s
(in sigma), with the same shifts for every method, for s = 0, 0.01, 0.02, 0.05 and 0.1. A
defect is found when the flagged atom (at its undisplaced position) lies within 1.5 sigma of
the defect. Finders are calibrated (logit adjustment, scripts/eval_point_defects.py); trees
(scripts/pilots/descriptor_classifier.py) rank atoms by the log-odds of the role against
"intact", which was best for every tree model on the held-out and validation glasses; the
single measures use the manuscript's rules and the statistics of the defect-free training
glasses. About 2 minutes per displacement size on 4 CPU cores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("descriptor_classifier", ROOT / "scripts/pilots/descriptor_classifier.py")  # fmt: skip
dc = importlib.util.module_from_spec(_spec)
sys.modules["descriptor_classifier"] = dc  # the pool workers find glass_inputs by module name
_spec.loader.exec_module(dc)
ev = dc.ev
SIZES = (0.0, 0.01, 0.02, 0.05, 0.1)


def named_paths(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {item!r}")
        out[name] = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    return out


def summary(res: dict, out: Path) -> None:
    sizes = res["sizes"]
    methods = res["methods"]
    lines = [
        "# Defects found against the size of a random displacement",
        "",
        f"Real defects of `{res['data']}`, found (%) over all four types, when every atom is",
        "shifted by Gaussian noise of the given standard deviation (sigma). Made by",
        "`scripts/pilots/displaced_inputs.py`.",
        "",
        "| Method | " + " | ".join(f"{s:g}" for s in sizes) + " |",
        "|---|" + "---|" * len(sizes),
    ]
    for m in methods:
        cells = [f"{100 * res['found'][str(s)][m]['all']:.1f}" for s in sizes]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    ref = 0.02 if 0.02 in sizes else sizes[min(1, len(sizes) - 1)]
    lines += ["", f"By defect type at {ref:g} sigma:", ""]
    lines += ["| Method | " + " | ".join(dc.KINDS) + " |", "|---|" + "---|" * len(dc.KINDS)]
    for m in methods:
        cells = [f"{100 * res['found'][str(ref)][m][k]:.1f}" for k in dc.KINDS]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if res.get("paired"):
        lines += ["", "McNemar p over all defects, by displacement in sigma (exact, not corrected for multiple tests):", ""]  # fmt: skip
        for pair, per in res["paired"].items():
            lines.append(f"- {pair}: " + ", ".join(f"at {s}: p = {p:.3g}" if p >= 0.001 else f"at {s}: p < 0.001" for s, p in per.items()))  # fmt: skip
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=dc.DATA / "val.npz")
    ap.add_argument("--finder", action="append", default=[], help="name=checkpoint (repeat)")
    ap.add_argument("--trees", action="append", default=[], help="name=models.pkl (repeat)")
    ap.add_argument("--versions", nargs="*", default=["D", "E"], help="tree versions to score")
    ap.add_argument("--sizes", nargs="*", type=float, default=list(SIZES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="for the finders (cuda on a GPU)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = dc.reference_stats()
    data = dc.real_defects(dc.load_point_defects(args.data))
    finders = {}
    for name, path in named_paths(args.finder).items():
        log_prior = ev.finder_log_prior(str(path), dc.DATA)[0]
        finders[name] = ev.Models(str(path), str(path), threads=4, device=args.device,
                                  log_prior=log_prior, do_heal=False)  # fmt: skip
    trees = {}
    for name, path in named_paths(args.trees).items():
        with open(path, "rb") as fh:
            trees[name] = pickle.load(fh)
    data_name = str(args.data.resolve().relative_to(ROOT))
    res: dict = {"data": data_name, "sizes": args.sizes, "seed": args.seed, "found": {},
                 "each": {}, "finders": {k: str(v) for k, v in named_paths(args.finder).items()},
                 "trees": {k: str(v) for k, v in named_paths(args.trees).items()}}  # fmt: skip
    for size in args.sizes:
        rng = np.random.default_rng([args.seed, int(round(size * 1000))])
        each: dict[str, dict] = {}
        for kind in dc.KINDS:
            d = data[kind]
            dd = dc.displaced(d, size, rng) if size > 0 else d
            role = int(ev.REPAIR_ROLE[dc.PointDefect[kind.upper()]])
            x = dc.set_inputs(dd, stats, args.workers)
            pos, types, box = dd.pos.numpy(), dd.types.numpy(), dd.box.numpy()
            for name, m in finders.items():
                score = m.role_prob(pos, types, box)[..., role]
                each.setdefault(f"finder {name}", {})[kind] = dc.found_each(score, d)
            for name, models in trees.items():
                for v in args.versions:
                    score = dc.role_scores(models[f"trees {v}"], x[v], role)["odds"]
                    each.setdefault(f"trees {v} {name}", {})[kind] = dc.found_each(score, d)
            for name, score in dc.single_measures(dd, x, kind).items():
                each.setdefault(name, {})[kind] = dc.found_each(score, d)
        res["methods"] = list(each)
        res["found"][str(size)] = {
            m: {**{k: round(float(v.mean()), 4) for k, v in per.items()},
                "all": round(float(np.concatenate(list(per.values())).mean()), 4)}
            for m, per in each.items()
        }  # fmt: skip
        res["each"][str(size)] = {m: {k: v.tolist() for k, v in per.items()} for m, per in each.items()}  # fmt: skip
        shown = {m: v["all"] for m, v in res["found"][str(size)].items()}
        print(f"{size:g}: {json.dumps(shown)} ({time.time() - t0:.0f}s)", flush=True)
        (args.out / "found.json").write_text(json.dumps(res, indent=1) + "\n")
    # each finder against each tree model, over all defects, at every size
    res["paired"] = {}
    for f in [m for m in res["methods"] if m.startswith("finder")]:
        for t in [m for m in res["methods"] if m.startswith("trees")]:
            res["paired"][f"{f} vs {t}"] = {
                str(s): round(dc.mcnemar(
                    np.concatenate([res["each"][str(s)][f][k] for k in dc.KINDS]),
                    np.concatenate([res["each"][str(s)][t][k] for k in dc.KINDS])), 4)
                for s in args.sizes
            }  # fmt: skip
    res["seconds"] = round(time.time() - t0)
    (args.out / "found.json").write_text(json.dumps(res, indent=1) + "\n")
    summary(res, args.out)


if __name__ == "__main__":
    main()
