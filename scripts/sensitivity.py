"""Sensitivity of the evaluation to its own settings (docs/PROJECT_PLAN.md §5).

    python scripts/sensitivity.py --config configs/eval/default.yaml run=runs/<run> data=data/ka2d_256
    python scripts/sensitivity.py --config configs/eval/default.yaml floor=true defect=D2 \\
        data=data/ka2d_256 out=results/sensitivity/floor_D2.json
    python scripts/sensitivity.py --config configs/eval/default.yaml summary=true \\
        runs="[results/sensitivity]" out=results/sensitivity/summary

Re-scores one finished run (samples.npz + relaxed.npz, as for evaluate.py) with each setting
moved on its own, all else at the frozen values:

  - the defect cutoff of the requested class by -0.05 / +0.05: ``cn_factor`` for D1- and D1+
    (the A-A coordination cutoff moves by 0.05 sigma, A-B by 0.04), ``d2_rc`` for D2;
  - the success radius ``r_tol`` by -0.1 / +0.1;
  - the local-realism window ``r_loc`` by -0.5 / +0.5.

Natural reference defects are relabelled with the same shifted cutoffs, so the success test
and the reference always use one definition. Each variant reports success after relaxation
(when the variant can change it) and the local W1 of the per-atom energy (when the variant
can change it), with 95% bootstrap CIs over samples; the ``default`` variant equals
evaluate.py's numbers. Writes <run>/sensitivity.json.

``floor=true``: the same variants for natural vs natural (train windows against test
windows), the reference level of the local W1. ``summary=true``: one table of every
sensitivity.json and floor file found under ``runs``.  The generation-side radius R_out
(``request.host_r_out``) needs new samples; see notebooks/kaggle_sensitivity.ipynb.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.data.dataset import load_split
from glassdiff.eval.bootstrap import bootstrap_ci
from glassdiff.eval.metrics import local_features, local_realism, natural_windows, success
from glassdiff.types import DefectClass, Request, Structures
from glassdiff.utils.config import cli_config

CUTOFF_STEP, R_TOL_STEP, R_LOC_STEP = 0.05, 0.1, 0.5


class Variant(NamedTuple):
    name: str
    th: DefectThresholds
    r_tol: float
    r_loc: float
    scores_success: bool
    scores_local: bool


def variants(th: DefectThresholds, r_tol: float, r_loc: float, defect: int) -> list[Variant]:
    key = "d2_rc" if defect == DefectClass.D2 else "cn_factor"
    out = [Variant("default", th, r_tol, r_loc, True, True)]
    for sign in (-1, 1):
        value = round(getattr(th, key) + sign * CUTOFF_STEP, 4)
        out.append(
            Variant(
                f"{key}={value:g}",
                dataclasses.replace(th, **{key: value}),
                r_tol,
                r_loc,
                True,
                True,
            )
        )
    for sign in (-1, 1):
        value = round(r_tol + sign * R_TOL_STEP, 4)
        out.append(Variant(f"r_tol={value:g}", th, value, r_loc, True, False))
    for sign in (-1, 1):
        value = round(r_loc + sign * R_LOC_STEP, 4)
        out.append(Variant(f"r_loc={value:g}", th, r_tol, value, False, True))
    return out


def windows(data_dir: Path, split_name: str, defect: int, v: Variant):
    """Natural windows of one class, labelled with the variant's cutoffs (cached on disk)."""
    path = data_dir / f"{split_name}.npz"
    st = path.stat()
    key = f"{split_name}_{defect}_{v.r_loc:g}_{v.th.digest()}_{st.st_size}_{int(st.st_mtime)}"
    cache = data_dir / ".cache" / f"sens_windows_{key}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    split = load_split(path)
    split = dataclasses.replace(split, labels=detect_defects(split.structures, v.th))
    feats = natural_windows(split, defect, v.r_loc)
    cache.parent.mkdir(exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save(feats, tmp)
    tmp.replace(cache)
    return feats


def frozen(cfg: dict, data_dir: Path) -> tuple[DefectThresholds, float, float]:
    meta = json.loads((data_dir / "meta.json").read_text())
    th = DefectThresholds.from_dict(meta["defects"]["thresholds"])
    r_tol = float(cfg.get("success", {}).get("r_tol", 0.5))
    r_loc = float(cfg.get("local", {}).get("r_loc", 3.0))
    return th, r_tol, r_loc


def describe(v: Variant) -> dict:
    return {"cn_factor": v.th.cn_factor, "d2_rc": v.th.d2_rc, "r_tol": v.r_tol, "r_loc": v.r_loc}


def score_run(cfg: dict, run: Path, data_dir: Path, n_boot: int) -> dict:
    with np.load(run / "samples.npz") as f:
        samp = {k: f[k] for k in f.files}
    with np.load(run / "relaxed.npz") as f:
        rel = {k: f[k] for k in f.files}
    defect = int(samp["defect"])
    types, box = torch.from_numpy(samp["types"]).long(), torch.from_numpy(samp["box"]).double()
    post = Structures(torch.from_numpy(rel["pos"]).double(), types, box)
    n = post.batch_size
    request = Request(
        target=torch.from_numpy(samp["target"]).double(),
        defect=torch.full((n,), defect, dtype=torch.long),
        pin_mask=torch.from_numpy(samp["pin_mask"]),
        pin_pos=torch.from_numpy(samp["pin_pos"]).double(),
        label=torch.from_numpy(samp["label"]).long(),
    )
    disp = torch.from_numpy(rel["disp"]).double()
    out = {
        "run": run.name,
        "defect": DefectClass(defect).name,
        "mode": str(samp["mode"]),
        "n_samples": n,
        "variants": {},
    }
    for v in variants(*frozen(cfg, data_dir), defect):
        row = describe(v)
        if v.scores_success:
            ok = success(post, request, v.th, v.r_tol).double().numpy()
            row["success_after_relax"] = list(bootstrap_ci(ok))
        if v.scores_local:
            gen = local_features(post, request.target, disp, v.r_loc)
            ref = windows(data_dir, "test", defect, v)
            row["local_w1_pe_atom"] = list(local_realism(gen, ref, n_boot)["w1_pe_atom"])
        out["variants"][v.name] = row
        print(v.name, json.dumps(row), flush=True)
    return out


def score_floor(cfg: dict, defect: int, data_dir: Path, n_boot: int) -> dict:
    out = {
        "run": f"floor_{DefectClass(defect).name}",
        "defect": DefectClass(defect).name,
        "mode": "floor (natural train vs natural test)",
        "variants": {},
    }
    for v in variants(*frozen(cfg, data_dir), defect):
        if not v.scores_local:
            continue
        row = describe(v)
        gen, ref = windows(data_dir, "train", defect, v), windows(data_dir, "test", defect, v)
        row["n_windows"] = len(gen)
        row["local_w1_pe_atom"] = list(local_realism(gen, ref, n_boot)["w1_pe_atom"])
        out["variants"][v.name] = row
        print(v.name, json.dumps(row), flush=True)
    return out


def summarize(roots: list[str], out: Path) -> None:
    """One row per (run, variant): success and local W1 with CIs, and the change from default."""
    files = sorted(
        {
            p
            for r in roots
            for p in Path(r).rglob("*.json")
            if p.name == "sensitivity.json" or p.name.startswith("floor_")
        }
    )
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        if "variants" not in d:
            continue
        name = p.parent.name if p.name == "sensitivity.json" else p.stem
        base = d["variants"].get("default", {})
        for vname, v in d["variants"].items():
            row = {"run": name, "defect": d["defect"], "variant": vname}
            for metric in ("success_after_relax", "local_w1_pe_atom"):
                if metric in v:
                    e, lo, hi = v[metric]
                    row[metric] = f"{e:.3f} [{lo:.3f}, {hi:.3f}]"
                    if metric in base and vname != "default":
                        row[f"{metric}_change"] = f"{e - base[metric][0]:+.3f}"
            rows.append(row)
    cols = [
        "run",
        "defect",
        "variant",
        "success_after_relax",
        "success_after_relax_change",
        "local_w1_pe_atom",
        "local_w1_pe_atom_change",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f'"{r.get(c, "")}"' for c in cols) + "\n")
    with open(out.with_suffix(".md"), "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(r.get(c, "") for c in cols) + " |\n")
    print(f"{len(rows)} rows from {len(files)} files -> {out.with_suffix('.md')}")


def main() -> None:
    cfg = cli_config(__doc__)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    if cfg.get("summary"):
        roots = cfg["runs"] if isinstance(cfg["runs"], list) else [cfg["runs"]]
        summarize(roots, Path(cfg["out"]))
        return
    data_dir = Path(cfg["data"])
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples_w1", 1000))
    if cfg.get("floor"):
        defect = DefectClass[cfg["defect"]] if isinstance(cfg["defect"], str) else cfg["defect"]
        result = score_floor(cfg, int(defect), data_dir, n_boot)
        out = Path(cfg["out"])
    else:
        run = Path(cfg["run"])
        result = score_run(cfg, run, data_dir, n_boot)
        out = run / "sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
