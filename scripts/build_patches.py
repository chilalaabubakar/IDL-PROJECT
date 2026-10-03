"""Build the reference-defect patch library from the TRAIN split.  [Ticket E-3]

    python scripts/build_patches.py --config configs/defects/v0.yaml data=data/ka2d_256

Needs labels from scripts/freeze_thresholds.py. Writes <data>/patches_train.npz.
"""

from __future__ import annotations

from pathlib import Path

from glassdiff.data.dataset import load_split
from glassdiff.data.patches import PatchLibrary
from glassdiff.utils.config import cli_config


def main() -> None:
    cfg = cli_config(__doc__)
    data_dir = Path(cfg["data"])
    train = load_split(data_dir / "train.npz")
    if train.labels is None:
        raise SystemExit("train split has no labels; run scripts/freeze_thresholds.py first")
    lib = PatchLibrary.build(train.structures, train.labels, radius=cfg.get("patch_radius", 1.5))
    out = data_dir / "patches_train.npz"
    lib.save(out)
    sizes = [len(p.types) for p in lib.patches]
    print(f"{len(lib)} patches {lib.counts()}, size {min(sizes)}-{max(sizes)} atoms -> {out}")


if __name__ == "__main__":
    main()
