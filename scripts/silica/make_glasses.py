"""Make our silica glasses with LAMMPS (melt-quench; src/glassdiff/silica/md.py).

    python scripts/silica/make_glasses.py --config configs/silica/md.yaml workers=64
    python scripts/silica/make_glasses.py --config configs/silica/md.yaml dry=true workers=4
    python scripts/silica/make_glasses.py --config configs/silica/md.yaml only=train limit=8
    python scripts/silica/make_glasses.py --config configs/silica/md.yaml assemble=true

Every glass is written to <out_dir>/glasses/<split>_r<rate>_s<seed>.npz as soon as it is done,
so the script can be stopped and restarted (a SLURM time limit, a lost node) without losing
work: finished seeds are skipped. A glass that fails the gates (density, Si 4-fold fraction,
closest pair) is kept for the record and replaced with the next seed. When a split is
complete, <out_dir>/<split>.npz is assembled and every split is labelled with the frozen
defect detectors (scripts/silica/label_defects.py).

``limit=k`` makes at most k new glasses this session (e.g. to fill a job's time limit);
``assemble=true`` only assembles and labels what exists.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_defects import check_gate, label_dataset, print_report  # noqa: E402

from glassdiff.silica.data import save_split, stack  # noqa: E402
from glassdiff.silica.md import (  # noqa: E402
    SilicaGates,
    SilicaProtocol,
    make_glass,
    passes_gates,
    structure_stats,
)
from glassdiff.utils.config import cli_config, load_config  # noqa: E402
from glassdiff.utils.runs import git_state  # noqa: E402

MAX_ATTEMPTS = 3  # glasses tried per accepted glass before giving up on a group


def glass_path(root: Path, split: str, rate: float, seed: int) -> Path:
    return root / "glasses" / f"{split}_r{rate:g}_s{seed}.npz"


def _work(job: tuple[str, str, dict, dict, int]) -> tuple[str, bool, dict]:
    path, split, protocol, gates, seed = job
    p = SilicaProtocol(**protocol)
    g = make_glass(seed, p)
    stats = {**g.stats, **structure_stats(g.pos, g.types, g.box)}
    ok = passes_gates(stats, SilicaGates(**gates))
    np.savez_compressed(
        path,
        pos=g.pos.astype(np.float32),
        types=g.types,
        box=g.box.astype(np.float32),
        rate=np.float32(p.rate),
        seed=seed,
        ok=ok,
        stats=json.dumps(stats),
    )
    return path, ok, stats


def read_glass(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def plan_jobs(cfg: dict, root: Path, splits: dict, limit: int | None) -> tuple[list, dict]:
    """New jobs to run, and per group the accepted glass files found so far."""
    jobs, accepted = [], {}
    for split, groups in splits.items():
        for grp in groups:
            protocol = {**cfg["protocol"], "rate": float(grp["rate"])}
            if "n_sio2" in grp:
                protocol["n_sio2"] = int(grp["n_sio2"])
            key = (split, float(grp["rate"]))
            n, seed0 = int(grp["n"]), int(grp["seed0"])
            good, missing = [], []
            for seed in range(seed0, seed0 + MAX_ATTEMPTS * n):
                path = glass_path(root, split, grp["rate"], seed)
                if path.exists():
                    if bool(read_glass(path)["ok"]):
                        good.append(path)
                elif len(good) + len(missing) < n:
                    missing.append(seed)
                if len(good) >= n:
                    break
            accepted[key] = good[:n]
            for seed in missing[: max(0, n - len(good))]:
                path = glass_path(root, split, grp["rate"], seed)
                jobs.append((str(path), split, protocol, cfg["gates"], seed))
    if limit is not None:
        jobs = jobs[:limit]
    return jobs, accepted


def assemble(cfg: dict, root: Path, splits: dict, accepted: dict) -> list[str]:
    done = []
    for split, groups in splits.items():
        files = []
        for grp in groups:
            got = accepted[(split, float(grp["rate"]))]
            if len(got) < int(grp["n"]):
                print(f"{split} @ {grp['rate']:g} K/ps: {len(got)} / {grp['n']} glasses so far")
                break
            files += got
        else:
            glasses = []
            for path in files:
                g = read_glass(path)
                glasses.append((g["pos"], g["types"], g["box"], float(g["rate"]), path.name))
            data = stack(glasses)
            save_split(root / f"{split}.npz", data)
            dens = data.density()
            print(
                f"{split}: {len(data)} glasses of {data.n_atoms} atoms, density "
                f"{dens.mean():.3f} +- {dens.std(correction=0):.3f} g/cm3 -> {root / f'{split}.npz'}"
            )
            done.append(split)
    return done


def main() -> None:
    cfg = cli_config(__doc__)
    splits = {**cfg["splits"], **cfg.get("extra_splits", {})} if cfg.get("only") else cfg["splits"]
    if cfg.get("only"):
        splits = {cfg["only"]: splits[cfg["only"]]}
    root = Path(cfg["out_dir"])
    if cfg.get("dry"):
        dry = dict(cfg["dry_protocol"])
        rate = float(dry.pop("rate"))
        cfg["protocol"] = {**cfg["protocol"], **dry}
        root = Path(str(root) + "_dry")
        splits = {
            "train": [{"rate": rate, "n": 2, "seed0": 1}],
            "val": [{"rate": rate, "n": 1, "seed0": 101}],
            "test": [{"rate": rate, "n": 1, "seed0": 201}],
        }
        cfg["gates"] = {**cfg["gates"], "density_range": [1.5, 3.0], "min_si_cn4": 0.5}
    (root / "glasses").mkdir(parents=True, exist_ok=True)
    workers = int(cfg.get("workers", 1))
    limit = int(cfg["limit"]) if cfg.get("limit") is not None else None

    if not cfg.get("assemble"):
        jobs, _ = plan_jobs(cfg, root, splits, limit)
        p0 = SilicaProtocol(**cfg["protocol"])
        est = sum(replace(p0, **{"rate": j[2]["rate"]}).total_steps() for j in jobs)
        print(
            f"{len(jobs)} glasses to make with {workers} workers "
            f"(~{est * 3 * p0.n_sio2 / 2.0e5 / 3600:.0f} core-hours at 200k atom-steps/s)",
            flush=True,
        )
        t0, n_ok = time.time(), 0
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for i, (path, ok, stats) in enumerate(pool.imap_unordered(_work, jobs), 1):
                n_ok += ok
                print(
                    f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {Path(path).name} "
                    f"{'ok' if ok else 'REJECTED'} density {stats['density']:.3f} "
                    f"Si4 {stats['si_cn4']:.3f} {stats['seconds']:.0f}s",
                    flush=True,
                )
        print(f"{n_ok} / {len(jobs)} accepted this session")

    _, accepted = plan_jobs(cfg, root, splits, None)
    done = assemble(cfg, root, splits, accepted)
    if not done:
        return
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "name": cfg["name"] + ("_dry" if cfg.get("dry") else ""),
            "source": "our LAMMPS melt-quench (src/glassdiff/silica/md.py)",
            "protocol": cfg["protocol"],
            "gates": cfg["gates"],
            "splits": {k: v for k, v in splits.items() if k in done},
            "species": {"0": "O", "1": "Si"},
            "units": {"length": "Angstrom", "cooling_rate": "K/ps"},
            "git": git_state(),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    defects_cfg = load_config(cfg["defects"])
    report = label_dataset(root, defects_cfg)
    print_report(report)
    for line in check_gate(report, defects_cfg["gate"]):
        print(f"WARNING gate not met: {line}")


if __name__ == "__main__":
    main()
