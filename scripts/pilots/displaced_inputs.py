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

    python scripts/pilots/displaced_inputs.py --out results/thermal_snapshots_<date> \
        --thermal data/ka2d_256_point_defects_thermal/val.npz --minimised --finder ... --trees ...

Thermal mode: the inputs are the thermal snapshots of scripts/pilots/thermal_snapshots.py
(neighbouring atoms move together), at each temperature, for the defects that survived; with
--minimised also the minimised snapshots (a reference that needs the potential).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

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


def shift_conditions(data: dict, sizes, seed: int):
    """(label, {kind: (reference set, input set)}, info): Gaussian shifts of every atom."""
    for size in sizes:
        rng = np.random.default_rng([seed, int(round(size * 1000))])
        sets = {k: (data[k], dc.displaced(data[k], size, rng) if size > 0 else data[k]) for k in dc.KINDS}  # fmt: skip
        yield str(size), sets, {}


def thermal_conditions(path: Path, full: dict, temperatures, minimised: bool):
    """(label, {kind: (reference set, input set)}, info): thermal snapshots of surviving defects."""
    z = np.load(path)
    for temp in temperatures:
        for source in ("snapshot", "minimised") if minimised else ("snapshot",):
            sets, info = {}, {}
            for kind in dc.KINDS:
                key = f"{kind}_T{temp:g}"
                keep = z[f"{key}_survived"]
                d = full[kind].subset(torch.from_numpy(z[f"{key}_example"][keep]))
                box = d.box.numpy().astype(np.float64)[:, None, :]
                pos = np.mod(z[f"{key}_{source}"][keep].astype(np.float64), box)
                sets[kind] = (d, dataclasses.replace(d, pos=torch.from_numpy(pos)))
                info[kind] = dict(scored=int(keep.sum()), of=int(len(keep)),
                                  rms_median=round(float(np.median(z[f"{key}_rms"][keep])), 4))  # fmt: skip
            yield f"T{temp:g}" + (" minimised" if source == "minimised" else ""), sets, info


def summary(res: dict, out: Path) -> None:
    labels = res.get("conditions") or [str(s) for s in res["sizes"]]
    methods = res["methods"]
    thermal = bool(res.get("thermal"))
    head = (
        ["# Defects found in thermal snapshots", "",
         f"Real defects of `{res['data']}` that survived the molecular dynamics (same defect at",
         "the same sites after minimising the snapshot), found (%) over all four types, at each",
         "temperature T; \"minimised\": the minimised snapshot (a reference that needs the",
         "potential). Made by `scripts/pilots/displaced_inputs.py --thermal`."]
        if thermal else
        ["# Defects found against the size of a random displacement", "",
         f"Real defects of `{res['data']}`, found (%) over all four types, when every atom is",
         "shifted by Gaussian noise of the given standard deviation (sigma). Made by",
         "`scripts/pilots/displaced_inputs.py`."]
    )  # fmt: skip
    lines = [*head, "", "| Method | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for m in methods:
        cells = [f"{100 * res['found'][c][m]['all']:.1f}" for c in labels]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if thermal:
        lines += [
            "",
            "Defects scored and median per-coordinate RMS displacement of the snapshot:",
            "",
        ]
        lines += ["| Condition | " + " | ".join(dc.KINDS) + " |", "|---|" + "---|" * len(dc.KINDS)]
        for c in labels:
            if not c.endswith("minimised"):
                cells = [f"{v['scored']}/{v['of']}, {v['rms_median']:.3f}σ" for v in res["info"][c].values()]  # fmt: skip
                lines.append(f"| {c} | " + " | ".join(cells) + " |")
    ref = next((c for c in labels if c in ("0.02", "T0.05")), labels[min(1, len(labels) - 1)])
    lines += ["", f"By defect type at {ref}:", ""]
    lines += ["| Method | " + " | ".join(dc.KINDS) + " |", "|---|" + "---|" * len(dc.KINDS)]
    for m in methods:
        cells = [f"{100 * res['found'][ref][m][k]:.1f}" for k in dc.KINDS]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if res.get("paired"):
        lines += ["", "McNemar p over all defects, by condition (exact, not corrected for multiple tests):", ""]  # fmt: skip
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
    ap.add_argument(
        "--thermal", type=Path, help="thermal snapshots (scripts/pilots/thermal_snapshots.py)"
    )
    ap.add_argument("--temperatures", nargs="*", type=float, default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument(
        "--minimised", action="store_true", help="thermal: also the minimised snapshots"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = dc.reference_stats()
    full = dc.load_point_defects(args.data)
    data = dc.real_defects(full)
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
                 "trees": {k: str(v) for k, v in named_paths(args.trees).items()},
                 "conditions": [], "info": {}}  # fmt: skip
    if args.thermal:
        res["thermal"] = str(args.thermal.resolve().relative_to(ROOT))
        res["temperatures"], res["sizes"] = args.temperatures, []
        conditions = thermal_conditions(args.thermal, full, args.temperatures, args.minimised)
    else:
        conditions = shift_conditions(data, args.sizes, args.seed)
    for label, sets, info in conditions:
        res["conditions"].append(label)
        res["info"][label] = info
        each: dict[str, dict] = {}
        for kind in dc.KINDS:
            d, dd = sets[kind]
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
        res["found"][label] = {
            m: {**{k: round(float(v.mean()), 4) for k, v in per.items()},
                "all": round(float(np.concatenate(list(per.values())).mean()), 4)}
            for m, per in each.items()
        }  # fmt: skip
        res["each"][label] = {m: {k: v.tolist() for k, v in per.items()} for m, per in each.items()}  # fmt: skip
        shown = {m: v["all"] for m, v in res["found"][label].items()}
        print(f"{label}: {json.dumps(shown)} ({time.time() - t0:.0f}s)", flush=True)
        (args.out / "found.json").write_text(json.dumps(res, indent=1) + "\n")
    # each finder against each tree model, over all defects, in every condition
    res["paired"] = {}
    for f in [m for m in res["methods"] if m.startswith("finder")]:
        for t in [m for m in res["methods"] if m.startswith("trees")]:
            res["paired"][f"{f} vs {t}"] = {
                c: round(dc.mcnemar(
                    np.concatenate([res["each"][c][f][k] for k in dc.KINDS]),
                    np.concatenate([res["each"][c][t][k] for k in dc.KINDS])), 4)
                for c in res["conditions"]
            }  # fmt: skip
    res["seconds"] = round(time.time() - t0)
    (args.out / "found.json").write_text(json.dumps(res, indent=1) + "\n")
    summary(res, args.out)


if __name__ == "__main__":
    main()
