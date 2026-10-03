"""Freeze defect thresholds on the TRAIN split and label every split.  [Ticket E-2]

    python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256
    python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_1024 \
        thresholds_from=data/ka2d_256      # sets without a train split reuse frozen values

Fits the D3 void radius on train, writes the thresholds, their hash and per-split defect
rates into <data>/meta.json, and adds a ``labels`` array to every split file. Refuses to
change thresholds that are already frozen unless ``force=true``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from glassdiff.analysis.defects import DefectThresholds, defect_rates, detect_defects
from glassdiff.analysis.descriptors import delaunay_voids
from glassdiff.data.dataset import load_split, save_split
from glassdiff.utils.config import cli_config
from glassdiff.utils.runs import git_state

SPLITS = ("train", "val", "test")


def label_split(split, th: DefectThresholds, chunk: int = 100) -> torch.Tensor:
    s = split.structures
    parts = []
    for i in range(0, s.batch_size, chunk):
        part = split.subset(slice(i, i + chunk)).structures
        parts.append(detect_defects(part, th))
    return torch.cat(parts)


def main() -> None:
    cfg = cli_config(__doc__)
    data_dir = Path(cfg["data"])
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text())

    if cfg.get("thresholds_from"):
        src = Path(cfg["thresholds_from"])
        th = DefectThresholds.from_dict(
            json.loads((src / "meta.json").read_text())["defects"]["thresholds"]
        )
        fitted_on = f"{src}/train"
    else:
        th = DefectThresholds.from_dict(cfg)
        train = load_split(data_dir / "train.npz")
        if cfg.get("d3_quantile") is not None:
            radii = torch.cat([r for _, r in delaunay_voids(train.structures)])
            th = replace(th, d3_min_radius=float(torch.quantile(radii, cfg["d3_quantile"])))
        fitted_on = f"{data_dir}/train"

    old = meta.get("defects")
    if old and old["digest"] != th.digest() and not cfg.get("force", False):
        raise SystemExit(
            f"thresholds already frozen with digest {old['digest']}; new digest {th.digest()}. "
            "Pass force=true only if no model has been trained on these labels."
        )

    rates = {}
    for name in SPLITS:
        path = data_dir / f"{name}.npz"
        if not path.exists():
            continue
        split = load_split(path)
        split.labels = label_split(split, th)
        save_split(path, split)
        rates[name] = defect_rates(split.labels, split.structures.types)
        r = rates[name]
        print(
            f"{name}: D1- {r['D1_MINUS_of_species']:.4f} of A, D1+ {r['D1_PLUS_of_species']:.4f} "
            f"of A, D2 {r['D2_of_species']:.4f} of B | per glass: "
            f"D1- {r['D1_MINUS_per_glass']:.2f}, D1+ {r['D1_PLUS_per_glass']:.2f}, "
            f"D2 {r['D2_per_glass']:.2f}"
        )

    meta["defects"] = {
        "thresholds": th.to_dict(),
        "digest": th.digest(),
        "fitted_on": fitted_on,
        "config": {k: v for k, v in cfg.items() if k != "data"},
        "rates": rates,
        "git": git_state(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"frozen thresholds {th.digest()}: {th.to_dict()}")


if __name__ == "__main__":
    main()
