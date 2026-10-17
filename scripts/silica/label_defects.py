"""Label every glass of a silica dataset with the frozen defect detectors and check the gate.

    python scripts/silica/label_defects.py --config configs/silica/defects.yaml \\
        data=data/sio2_576 [strict=true]

Rewrites <data>/{train,val,test,...}.npz with a ``labels`` array, stores the thresholds and
their digest in <data>/meta.json (train.py refuses labels made with other thresholds) and
prints defects per glass for each split and cooling rate. The gate (docs/SILICA_3D_PLAN.md
2.3): every primary group needs ``gate.min_train_count`` training instances; with
strict=true a short group is an error, otherwise a warning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from glassdiff.silica.data import load_split, save_split
from glassdiff.silica.defects import (
    SilicaThresholds,
    coordination_summary,
    gate_counts,
    label_glasses,
)
from glassdiff.utils.config import cli_config, load_config
from glassdiff.utils.runs import git_state


def label_dataset(data_dir: Path, defects_cfg: dict) -> dict:
    """Label all splits in data_dir; return the 'labels' block written to meta.json."""
    th = SilicaThresholds(**defects_cfg["thresholds"])
    gate = defects_cfg["gate"]
    report: dict = {
        "thresholds": th.to_dict(),
        "digest": th.digest(),
        "git": git_state(),
        "splits": {},
    }
    for path in sorted(data_dir.glob("*.npz")):
        split = load_split(path)
        labels = label_glasses(split.pos.numpy(), split.types.numpy(), split.box.numpy(), th)
        split.labels = torch.from_numpy(labels)
        save_split(path, split)
        types = split.types.numpy()
        rates = split.cooling_rate.numpy()
        per_rate = {}
        for r in np.unique(rates):
            m = rates == r
            per_rate[f"{r:g}"] = {
                "glasses": int(m.sum()),
                **{
                    k: round(v, 4)
                    for k, v in coordination_summary(labels[m], types[m]).items()
                    if k.endswith("_per_glass")
                },
            }
        report["splits"][path.stem] = {
            "glasses": len(split),
            "atoms": split.n_atoms,
            "gate_counts": gate_counts(labels, gate["groups"] | gate.get("stretch", {})),
            "by_rate": per_rate,
        }
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["labels"] = report
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return report


def check_gate(report: dict, gate: dict) -> list[str]:
    counts = report["splits"].get("train", {}).get("gate_counts", {})
    need = int(gate["min_train_count"])
    return [f"{g}: {counts.get(g, 0)} < {need}" for g in gate["groups"] if counts.get(g, 0) < need]


def print_report(report: dict) -> None:
    print(f"thresholds {report['thresholds']} (digest {report['digest']})")
    for name, s in report["splits"].items():
        print(f"{name}: {s['glasses']} glasses of {s['atoms']} atoms; instances {s['gate_counts']}")
        for rate, row in s["by_rate"].items():
            per = ", ".join(
                f"{k.removesuffix('_per_glass')} {v:g}" for k, v in row.items() if k != "glasses"
            )
            print(f"    {rate:>5} K/ps ({row['glasses']} glasses), per glass: {per}")


def main() -> None:
    cfg = cli_config(__doc__)
    defects_cfg = load_config(cfg["defects"]) if "defects" in cfg else cfg
    data_dir = Path(cfg["data"])
    report = label_dataset(data_dir, defects_cfg)
    print_report(report)
    short = check_gate(report, defects_cfg["gate"])
    if short:
        msg = "gate not met (merge or drop these groups, or add 100 K/ps glasses): " + "; ".join(
            short
        )
        if cfg.get("strict"):
            sys.exit(msg)
        print("WARNING " + msg)
    else:
        print("gate met: every primary group has enough training instances")


if __name__ == "__main__":
    main()
