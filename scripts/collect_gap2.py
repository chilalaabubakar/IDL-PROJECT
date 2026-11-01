"""Collect completed disjoint PSC/Kaggle conditions without overwriting conflicting runs.

python scripts/collect_gap2.py --source results/kaggle_gap2_... /path/to/psc_root \
    --out runs/gap2_combined

Only complete conditions are imported. Source data remain unchanged. The original
reporter declares the full study complete only when all six conditions are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

METHODS = ("seed0", "seed1", "seed2", "sham_seed0", "hand_insert")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(path):
    return {
        str(f.relative_to(path)): digest(f)
        for f in path.rglob("*")
        if f.is_file() and f.suffix in (".npz", ".json") and not f.name.startswith("psc_")
    }


def validate(folder, condition, manifest, manifest_sha):
    requests = folder / "requests.npz"
    if not requests.exists() or not (folder / "hosts/protocol.json").exists():
        raise ValueError(f"Missing host/request provenance: {folder}")
    with np.load(requests, allow_pickle=False) as f:
        req = {k: f[k] for k in f.files}
    if len(req["host_ids"]) != manifest["n_requests"]:
        raise ValueError(f"Unexpected request count: {folder}")
    if len(np.unique(req["host_ids"])) != manifest["n_hosts"]:
        raise ValueError(f"Unexpected host count: {folder}")
    host_protocol = json.loads((folder / "hosts/protocol.json").read_text())
    cfg = Path("configs/data/gap2") / (condition + ".yaml")
    if host_protocol["config"]["config_sha256"] != digest(cfg):
        raise ValueError(f"Host preparation differs from frozen config: {folder}")
    for method in METHODS:
        base = folder / method
        done = json.loads((base / "complete.json").read_text())
        config = json.loads((base / "config.json").read_text())
        if done["n_requests"] != manifest["n_requests"] or config["batch_size"] != 4:
            raise ValueError(f"Wrong method configuration: {base}")
        if (
            config["requests_sha256"] != digest(requests)
            or config["manifest_sha256"] != manifest_sha
        ):
            raise ValueError(f"Request/manifest fingerprint mismatch: {base}")
        expected = (
            None
            if method == "hand_insert"
            else manifest["checkpoints"]["seed0" if method == "sham_seed0" else method]
        )
        if config["checkpoint"] != expected:
            raise ValueError(f"Frozen checkpoint mismatch: {base}")
        if (
            method == "hand_insert"
            and config["patch_library_sha256"] != manifest["training_patch_library"]["sha256"]
        ):
            raise ValueError(f"Frozen baseline library mismatch: {base}")
        with np.load(base / "samples.npz", allow_pickle=False) as d:
            if len(d["success"]) != manifest["n_requests"]:
                raise ValueError(f"Incomplete samples: {base}")
            for key in ("host_ids", "host_seeds", "target", "original_success"):
                if not np.array_equal(d[key], req[key]):
                    raise ValueError(f"Method requests do not match: {base}/{key}")
            for saved, original in (
                ("types", "types"),
                ("box", "box"),
                ("pin_mask", "pin_mask"),
                ("host_pos", "pin_pos"),
            ):
                if not np.array_equal(d[saved], req[original]):
                    raise ValueError(f"Method host/context disagreement: {base}/{saved}")
            if np.any(d["success"] & (~d["finite"] | ~d["converged"])):
                raise ValueError(f"Failed numerical output counted as success: {base}")
            if np.any(d["converged"] & ~(d["max_force"] <= 1e-6)):
                raise ValueError(f"Converged force exceeds tolerance: {base}")
            if int(d["success"].sum()) != done["successes"]:
                raise ValueError(f"Completion count mismatch: {base}")


def collect(sources, out):
    manifest_file = Path("configs/eval/gap2_generalization.json")
    manifest = json.loads(manifest_file.read_text())
    manifest_sha = digest(manifest_file)
    conditions = [Path(c).stem for c in manifest["dataset_configs"]]
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance_file = out / "collection_sources.json"
    provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
    imported = 0
    for source in sources:
        source = source.resolve()
        root = source / "study" if (source / "study").is_dir() else source
        if root == out:
            raise ValueError("Source and collection output must differ")
        for condition in conditions:
            folder = root / condition
            if not all((folder / m / "complete.json").exists() for m in METHODS):
                continue
            if out.is_relative_to(folder):
                raise ValueError("Collection output cannot be inside a source condition")
            validate(folder, condition, manifest, manifest_sha)
            target = out / condition
            if target.exists():
                if tree_digest(folder) != tree_digest(target):
                    raise ValueError(
                        f"Conflicting completed condition; choose a source explicitly: {condition}"
                    )
            else:
                temp = Path(tempfile.mkdtemp(prefix=".gap2-import-", dir=out))
                try:
                    shutil.copytree(
                        folder,
                        temp / condition,
                        ignore=shutil.ignore_patterns("*.tmp", "*.log", ".psc.lock"),
                    )
                    (temp / condition).rename(target)
                finally:
                    shutil.rmtree(temp)
            provenance.setdefault(condition, [])
            origin = str(folder)
            if origin not in provenance[condition]:
                provenance[condition].append(origin)
            imported += 1
            print("Validated complete condition:", condition, flush=True)
    if not imported:
        raise ValueError("No complete conditions were found in the selected sources")
    provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run([sys.executable, "scripts/run_gap2.py", "report", "--out", str(out)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.source, args.out)


if __name__ == "__main__":
    main()
