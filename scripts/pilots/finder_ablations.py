"""Experiments on the point-defect finder: every trained variant on the same test defects.

    bash scripts/pilots/finder_ablations.sh       # trains the six short ablation runs
    python scripts/pilots/finder_ablations.py     # this evaluation

Two groups of finders are scored on the real test defects (the flagged atom within 1.5 sigma
of the defect, as scripts/eval_point_defects.py), and compared in pairs on the same defects
(McNemar's exact test):

    trained   the finders of the report: after 20,000 updates and after 60,000, and the
              second with and without calibration (logit adjustment)
    ablation  six runs of 2,000 updates that change one setting each
              (scripts/pilots/finder_ablations.sh): the starting weights, the number of
              layers, the class weights, and the cleaned training set

The ablation runs are scored with their trained weights, not with the moving average of the
weights that the long runs use: with a decay of 0.999, an average over 2,000 updates is still
13% starting weights, which would count against the runs that start from random weights.
The average is scored as well and stored (key "moving_average").

Intervals are 95% and resample whole test glasses (10,000 bootstrap samples).
Writes <out>/ablations.json, <out>/summary.md and, for each finished ablation run,
<out>/<name>.pt (model settings and trained weights, readable by load_denoiser).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("eval_pd", ROOT / "scripts/eval_point_defects.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

TRAINED = {
    "20,000 updates": "results/kaggle_pd_20261107-2024Z/training/pd_detect/ema.pt",
    "60,000 updates": "results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt",
}
ABLATIONS = {  # run name -> what it is
    "reference": "starts from the denoiser, 4 layers, class weights",
    "scratch": "random starting weights",
    "layers2": "random starting weights, 2 layers",
    "layers1": "random starting weights, 1 layer",
    "no_weights": "no class weights",
    "real_only": "trained without the annihilated Frenkel pairs",
}
COMPARE = {  # each ablation against the run that differs from it in one setting
    "scratch": "reference",
    "layers2": "scratch",
    "layers1": "scratch",
    "no_weights": "reference",
    "real_only": "reference",
}


def found(path, data: dict, calibrate: bool = True) -> dict[str, np.ndarray]:
    """Per defect kind, whether the atom the finder flags lies within 1.5 sigma of the defect."""
    log_prior = ev.finder_log_prior(str(path), ROOT / "data/ka2d_256_point_defects")[0]
    models = ev.Models(
        str(path),
        str(path),  # the healer is required by Models but not used for finding
        threads=2,
        device="cpu",
        log_prior=log_prior if calibrate else None,
        do_heal=False,
    )
    out = {}
    for name, d in data.items():
        pos, types, box, sites = d.pos.numpy(), d.types.numpy(), d.box.numpy(), d.sites.numpy()
        score = models.role_prob(pos, types, box)[..., int(ev.REPAIR_ROLE[d.kind])]
        top = pos[np.arange(len(pos)), score.argmax(-1)]
        dist = np.stack(
            [np.linalg.norm(ev.wrap(top - sites[:, k], box), axis=-1) for k in range(2)]
        ).min(0)
        out[name] = dist < ev.HIT_RADIUS
    return out


def interval(each: np.ndarray, glass: np.ndarray, rng) -> tuple[float, float]:
    ids = np.unique(glass)
    total = np.array([each[glass == g].sum() for g in ids], dtype=float)
    count = np.array([(glass == g).sum() for g in ids], dtype=float)
    draw = rng.integers(0, len(ids), size=(10_000, len(ids)))
    lo, hi = np.percentile(total[draw].sum(1) / count[draw].sum(1), [2.5, 97.5])
    return float(lo), float(hi)


def rates(hits: dict[str, np.ndarray], glass: dict[str, np.ndarray], rng) -> dict:
    out = {}
    for k, h in hits.items():
        lo, hi = interval(h, glass[k], rng)
        out[k] = dict(rate=float(h.mean()), lo=lo, hi=hi)
    everything = np.concatenate(list(hits.values()))
    # a glass holds defects of every kind: resample the glasses, keep all their defects
    lo, hi = interval(everything, np.concatenate([glass[k] for k in hits]), rng)
    out["all"] = dict(rate=float(everything.mean()), lo=lo, hi=hi)
    return out


def paired(base: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict:
    """A finder against its base: defects only one of the two finds, McNemar's exact p."""

    def one(x: np.ndarray, y: np.ndarray) -> dict:
        only_base, only_new = int((x & ~y).sum()), int((~x & y).sum())
        p = binomtest(only_new, only_base + only_new).pvalue if only_base + only_new else 1.0
        return dict(only_new=only_new, only_base=only_base, p=float(p))

    out = {k: one(base[k], new[k]) for k in base}
    out["all"] = one(np.concatenate(list(base.values())), np.concatenate(list(new.values())))
    return out


def slim(run: Path, out: Path) -> tuple[Path, Path, dict] | None:
    """Trained weights and their moving average of a finished run, as small checkpoints."""
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    wanted = int(ckpt["cfg"]["optim"]["n_updates"])
    if int(ckpt["step"]) < wanted:
        return None
    keep = {k: ckpt[k] for k in ("model_cfg", "step", "cfg", "class_weights")}
    # load_denoiser reads the key "ema"; here it holds the trained weights themselves
    trained = out / f"{run.name.split('_', 1)[1]}.pt"
    torch.save(dict(keep, ema=ckpt["model"], weights="trained weights"), trained)
    average = out / "runs" / run.name / "average.pt"
    torch.save(dict(keep, ema=ckpt["ema"], weights="moving average"), average)
    log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    info = dict(
        updates=int(ckpt["step"]),
        seconds=round(float(ckpt.get("seconds", 0))),
        parameters=sum(v.numel() for v in ckpt["model"].values()),
        train_loss=round(log[-1]["train_loss"], 4) if log else None,
    )
    return trained, average, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results/point_defects_ablations_20261112")
    ap.add_argument("--data", default=str(ROOT / "data/ka2d_256_point_defects/test.npz"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    data = ev.real_defects(ev.load_point_defects(args.data))
    glass = {k: d.glass.numpy() for k, d in data.items()}
    out: dict = dict(data=args.data, n={k: len(d) for k, d in data.items()})

    # ---- the finders of the report
    hits = {name: found(ROOT / path, data) for name, path in TRAINED.items()}
    hits["60,000 updates, not calibrated"] = found(
        ROOT / TRAINED["60,000 updates"], data, calibrate=False
    )
    out["trained"] = {name: rates(h, glass, rng) for name, h in hits.items()}
    out["trained_tests"] = {
        "60,000 against 20,000 updates": paired(hits["20,000 updates"], hits["60,000 updates"]),
        "calibrated against not calibrated": paired(
            hits["60,000 updates, not calibrated"], hits["60,000 updates"]
        ),
    }
    for name, r in out["trained"].items():
        print(name, {k: round(v["rate"], 3) for k, v in r.items()}, flush=True)

    # ---- the ablation runs
    out["ablations"], out["ablation_tests"], out["moving_average"] = {}, {}, {}
    ab_hits = {}
    for name, what in ABLATIONS.items():
        runs = sorted((args.out / "runs").glob(f"*_{name}"))
        files = slim(runs[-1], args.out) if runs else None
        if files is None:
            print(name, "not finished", flush=True)
            continue
        trained, average, info = files
        ab_hits[name] = found(trained, data)
        out["ablations"][name] = dict(what=what, **info, found=rates(ab_hits[name], glass, rng))
        out["moving_average"][name] = rates(found(average, data), glass, rng)
        print(name, {k: round(v["rate"], 3) for k, v in out["ablations"][name]["found"].items()})
    for name, other in COMPARE.items():
        if name in ab_hits and other in ab_hits:
            out["ablation_tests"][f"{name} against {other}"] = paired(ab_hits[other], ab_hits[name])
    (args.out / "ablations.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---- Markdown
    kinds = [*data, "all"]
    label = dict(vacancy="Vacancy", interstitial="Interstitial", frenkel="Frenkel pair",
                 antisite="Antisite", all="All")  # fmt: skip

    def row(name: str, r: dict) -> str:
        cells = [f"{100 * r[k]['rate']:.1f}% [{100 * r[k]['lo']:.0f}, {100 * r[k]['hi']:.0f}]"
                 for k in kinds]  # fmt: skip
        return f"| {name} | " + " | ".join(cells) + " |"

    head = "| Finder | " + " | ".join(label[k] for k in kinds) + " |\n" + "|---" * 6 + "|"
    md = ["<!-- written by scripts/pilots/finder_ablations.py -->", "",
          "Found on the real test defects (" + ", ".join(f"{label[k]} {n}" for k, n in out["n"].items())
          + "). Brackets: 95% intervals from resampling test glasses.", "",
          "The finders of the report:", "", head]  # fmt: skip
    md += [row(name, r) for name, r in out["trained"].items()]
    if out["ablations"]:
        md += ["", "Ablation runs (2,000 updates each, trained weights):", "", head]
        md += [row(f"{name}: {a['what']}", a["found"]) for name, a in out["ablations"].items()]
    md += ["", "Paired comparisons on the same defects (McNemar's exact test; counts over all "
           "defects):", "", "| Comparison | Only the first | Only the second | p (all) | p per type |",
           "|---|---|---|---|---|"]  # fmt: skip
    for tests in (out["trained_tests"], out["ablation_tests"]):
        for name, t in tests.items():
            first, _, second = name.partition(" against ")
            per = ", ".join(f"{label[k]} {t[k]['p']:.3f}" for k in data)
            md.append(
                f"| {name} | {t['all']['only_new']} ({first}) | {t['all']['only_base']} "
                f"({second}) | {t['all']['p']:.4f} | {per} |"
            )
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
