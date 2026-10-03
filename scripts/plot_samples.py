"""Plot configurations: dataset glasses or a run's samples (before or after relaxation).

    python scripts/plot_samples.py --config configs/eval/default.yaml data=data/ka2d_256 \\
        split=test n=8 out=results/figures/test_glasses.png
    python scripts/plot_samples.py --config configs/eval/default.yaml data=data/ka2d_256 \\
        run=runs/<run> stage=relaxed n=8

Defect atoms are detected with the dataset's frozen thresholds and ringed; for runs the
requested location and pinned atoms are shown, and each panel says whether it succeeded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.data.dataset import load_split
from glassdiff.eval.metrics import success
from glassdiff.types import DefectClass, Request, Structures
from glassdiff.utils.config import cli_config
from glassdiff.viz import plot_structures


def main() -> None:
    cfg = cli_config(__doc__)
    data_dir = Path(cfg["data"])
    meta = json.loads((data_dir / "meta.json").read_text())
    th = DefectThresholds.from_dict(meta["defects"]["thresholds"])
    n = int(cfg.get("n", 8))
    if cfg.get("run"):
        run = Path(cfg["run"])
        stage = cfg.get("stage", "relaxed")
        with np.load(run / "samples.npz") as f:
            samp = {k: f[k] for k in f.files}
        pos = samp["pos"]
        if stage == "relaxed":
            with np.load(run / "relaxed.npz") as f:
                pos = f["pos"]
        pos, types, box = pos[:n], samp["types"][:n], samp["box"][:n]
        s = Structures(
            torch.from_numpy(pos).double(),
            torch.from_numpy(types).long(),
            torch.from_numpy(box).double(),
        )
        defect = int(samp["defect"])
        req = Request(
            torch.from_numpy(samp["target"][:n]).double(),
            torch.full((len(pos),), defect),
            torch.from_numpy(samp["pin_mask"][:n]),
            torch.from_numpy(samp["pin_pos"][:n]).double(),
            torch.from_numpy(samp["label"][:n]).long(),
        )
        ok = success(s, req, th).tolist()
        titles = [f"{DefectClass(defect).name} · {'success' if o else 'miss'}" for o in ok]
        out = cfg.get("out") or run / f"{stage}.png"
        path = plot_structures(
            pos,
            types,
            box,
            out,
            labels=detect_defects(s, th).numpy(),
            pin=samp["pin_mask"][:n],
            targets=samp["target"][:n],
            titles=titles,
        )
    else:
        split = load_split(data_dir / f"{cfg.get('split', 'test')}.npz")
        s = split.subset(slice(0, n)).structures
        out = cfg.get("out") or f"results/figures/{data_dir.name}_{cfg.get('split', 'test')}.png"
        path = plot_structures(
            s.pos.numpy(),
            s.types.numpy(),
            s.box.numpy(),
            out,
            labels=detect_defects(s, th).numpy(),
            titles=[f"seed {int(x)}" for x in split.seed[:n]],
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
