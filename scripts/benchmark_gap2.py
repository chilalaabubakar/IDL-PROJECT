"""Generate one independent host per generalization condition and measure CPU cost.

This is a feasibility benchmark, not a performance/generalization evaluation.
python scripts/benchmark_gap2.py --out results/gap2_feasibility --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from glassdiff.md.lammps_quench import (
    QualityGates,
    QuenchProtocol,
    make_glass,
    passes_quality_gates,
)


def worker(job):
    config, out = job
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path(config).read_text())
    folder = Path(out) / cfg["name"]
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    status = folder / "benchmark.json"
    if status.exists():
        result = json.loads(status.read_text())
        if result["config_sha256"] != digest:
            raise ValueError("Benchmark resume configuration changed")
        return result
    protocol = QuenchProtocol(**cfg["protocol"])
    # Keep benchmark seeds separate from all preregistered evaluation seed ranges.
    seed = int(cfg["splits"]["test"]["seed0"]) + 10000
    start = time.monotonic()
    glass = make_glass(cfg["n_atoms"], seed, protocol)
    ok, stats = passes_quality_gates(glass, protocol.frac_a, QualityGates())
    elapsed = time.monotonic() - start
    np.savez_compressed(
        folder / "host.npz",
        pos=glass.pos,
        types=glass.types,
        box=glass.box,
        pe_atom=glass.pe_atom,
        seed=seed,
    )
    result = dict(
        condition=cfg["name"],
        config_sha256=digest,
        n_atoms=cfg["n_atoms"],
        rate=protocol.rate,
        seed=seed,
        seconds=elapsed,
        quality_pass=ok,
        stats=stats,
        purpose="single-host feasibility only; not a generalization result",
    )
    status.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    configs = sorted(Path("configs/data/gap2").glob("*.yaml"))
    assert len(configs) == 6
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, [(str(c), str(args.out)) for c in configs]))
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Gap 2 host-generation feasibility benchmark\n",
        "One independently generated host per condition; no trained-model evaluation has been run. These are not generalization performance estimates. Benchmark seeds are excluded from the planned evaluation draws. Wall times include generation and quality checks under concurrent CPU load.\n",
        "| Atoms | Cooling rate | Seconds | Quality pass | Energy/atom |",
        "|---:|---:|---:|---|---:|",
    ]
    for r in sorted(results, key=lambda x: (x["n_atoms"], x["rate"])):
        lines.append(
            f"| {r['n_atoms']} | {r['rate']} | {r['seconds']:.1f} | {r['quality_pass']} | {r['stats']['pe_mean']:.6f} |"
        )
    lines.append(
        "\nSee `docs/GAP2_GENERALIZATION.md` and `configs/eval/gap2_generalization.json` for the fixed study design. Per-condition raw hosts, force residuals, order and pair-distance diagnostics are retained here. One host per cell cannot establish a quench-history trend or support a reliable full-study runtime estimate.\n"
    )
    (args.out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
