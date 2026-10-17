"""Import DM2's 24 published silica glasses as a first training set (data/sio2_dm2).

    python scripts/silica/import_dm2.py --config configs/silica/data_dm2.yaml \\
        [dm2_dir=/path/to/dm2/demo/demo_training/simu_data]

DM2 (github.com/digital-synthesis-lab/dm2, MIT licence) publishes 3,000-atom a-SiO2 glasses
made with the same potential we use, 6 at each of 0.1, 1, 10 and 100 K/ps. We split the
samples of every rate 4 / 1 / 1 into train / val / test, wrap positions into the box, label
defects with the frozen detectors (configs/silica/defects.yaml) and write
<out_dir>/{train,val,test}.npz and meta.json. This lets training start now; our own 576-atom
glasses (scripts/silica/make_glasses.py) replace it as the main dataset.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_defects import check_gate, label_dataset, print_report  # noqa: E402

from glassdiff.silica.data import read_lammps_data, save_split, stack  # noqa: E402
from glassdiff.utils.config import cli_config, load_config  # noqa: E402
from glassdiff.utils.runs import git_state  # noqa: E402

PATTERN = re.compile(r"sio2_(\d+)_glass_(.+)_sample(\d+)\.dat$")


def find_dm2_dir(cfg: dict) -> Path:
    if cfg.get("dm2_dir"):
        return Path(cfg["dm2_dir"])
    if os.environ.get("DM2_DIR"):
        return Path(os.environ["DM2_DIR"]) / "demo" / "demo_training" / "simu_data"
    sys.exit("set dm2_dir=... or the DM2_DIR environment variable (a clone of DM2)")


def main() -> None:
    cfg = cli_config(__doc__)
    src = find_dm2_dir(cfg)
    files = sorted(src.glob("sio2_*_glass_*_sample*.dat"))
    if not files:
        sys.exit(f"no DM2 glasses in {src}")
    rates = {str(k): float(v) for k, v in cfg["rates"].items()}
    sample_split = {int(i): name for name, ids in cfg["split"].items() for i in ids}
    groups: dict[str, list] = {name: [] for name in cfg["split"]}
    for path in files:
        m = PATTERN.search(path.name)
        if m is None or m.group(2) not in rates:
            print(f"skipping {path.name}")
            continue
        split = sample_split.get(int(m.group(3)))
        if split is None:
            continue
        pos, types, box = read_lammps_data(path)
        groups[split].append((pos, types, box, rates[m.group(2)], path.name))

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, glasses in groups.items():
        data = stack(glasses)
        save_split(out / f"{name}.npz", data)
        dens = data.density()
        print(
            f"{name}: {len(data)} glasses of {data.n_atoms} atoms, density "
            f"{dens.mean():.3f} +- {dens.std(correction=0):.3f} g/cm3"
        )
    meta = {
        "name": cfg["name"],
        "source": "DM2 published glasses (github.com/digital-synthesis-lab/dm2, MIT licence)",
        "potential": "Bouhadja-Jakse-Pasturel BMH, DSF Coulomb 0.25 1/A, 8 A cutoffs",
        "species": {"0": "O", "1": "Si"},
        "units": {"length": "Angstrom", "cooling_rate": "K/ps"},
        "rates": rates,
        "split": {k: [int(i) for i in v] for k, v in cfg["split"].items()},
        "files": {k: [g[4] for g in v] for k, v in groups.items()},
        "git": git_state(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    defects_cfg = load_config(cfg["defects"])
    report = label_dataset(out, defects_cfg)
    print_report(report)
    for line in check_gate(report, defects_cfg["gate"]):
        print(f"note: below the 576-atom dataset's gate (expected for 16 glasses): {line}")


if __name__ == "__main__":
    main()
