"""Make the point-defect dataset: vacancy, interstitial, Frenkel pair and antisite examples in
the 2D Kob-Andersen glasses (docs/POINT_DEFECTS.md).

    python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml
    python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml \\
        splits=[test] n_glasses=10 out_dir=data/pd_smoke          # quick check

For every glass of each split and every defect kind, ``sites_per_glass`` defects are made at
random sites (one per example) and the defected glass is minimised with LAMMPS (FIRE, all
atoms free, as the dataset's inherent structures). Output, per split: <out_dir>/<split>.npz
with arrays ``<kind>_<field>`` (fields: glassdiff.data.point_defects) and an ``info`` record
that pins the source split by SHA256; plus <out_dir>/meta.json (settings, counts, checks,
timing). Glasses keep their split, so test defects only ever come from test glasses.
Deterministic for a given config: each (split, kind) has its own random stream and each
minimisation runs in one single-threaded LAMMPS instance. About 0.05 s per example on 4 cores.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from glassdiff.data.dataset import load_split
from glassdiff.data.point_defects import (
    NAMES,
    DefectConfig,
    PointDefect,
    file_sha256,
    generate,
    save_point_defects,
)
from glassdiff.md.minimize import minimize_many
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import git_state

SPLITS = ("train", "val", "test")


def main() -> None:
    cfg = cli_config(__doc__)
    source = Path(cfg["source"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg = DefectConfig(
        **{k: cfg[k] for k in DefectConfig.__dataclass_fields__ if k in cfg},
    )
    dcfg.frenkel_hop = tuple(dcfg.frenkel_hop)
    dcfg.kinds = tuple(dcfg.kinds)
    kinds = [next(k for k, v in NAMES.items() if v == name) for name in dcfg.kinds]
    relax = partial(minimize_many, workers=cfg.get("workers"), ftol=dcfg.ftol)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(config=cfg, git=git_state(), splits=meta.get("splits", {}))

    for split_id, split in enumerate(SPLITS):
        if split not in cfg.get("splits", SPLITS):
            continue
        src = source / f"{split}.npz"
        glasses = load_split(src)
        if cfg.get("n_glasses"):
            glasses = glasses.subset(slice(0, int(cfg["n_glasses"])))
        s = glasses.structures
        pos, types, box = s.pos.numpy(), s.types.numpy(), s.box.numpy()
        examples, summary = {}, {}
        for kind in kinds:
            t0 = time.time()
            rng = np.random.default_rng([int(dcfg.seed), split_id, int(kind)])
            n = int(dcfg.sites_per_glass[split])
            res = generate(PointDefect(kind), pos, types, box, n, rng, dcfg, relax)
            name = NAMES[kind]
            examples[name] = res
            summary[name] = dict(
                examples=int(len(res["glass"])),
                valid=int(res["valid"].sum()),
                atoms=int(res["pos"].shape[1]),
                max_force=float(res["fmax"].max()),
                seconds=round(time.time() - t0, 1),
            )
            print(f"{split} {name}: {summary[name]}", flush=True)
        path = out_dir / f"{split}.npz"
        info = save_point_defects(
            path, examples, src, n_glasses=len(pos), kinds=list(dcfg.kinds), split=split
        )
        meta["splits"][split] = dict(info, file_sha256=file_sha256(path), kinds_summary=summary)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
