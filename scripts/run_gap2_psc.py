"""Run one reserved Bridges-2 condition with exclusive ownership and unchanged science.

Preparation uses CPU nodes. Evaluation uses two scheduler-visible GPUs; it does not
change CUDA_VISIBLE_DEVICES, retrain models, or write into Kaggle checkpoint folders.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CONDITIONS = ("ka2d_1024_standard", "ka2d_1024_fast")
METHODS = ("seed0", "seed1", "seed2", "sham_seed0", "hand_insert")


def _installed_version(package: str) -> str | None:
    """Version of an installed package, or None when it is not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def command(stage, root, condition, workers, method=None, device=None):
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_gap2.py",
        stage,
        "--out",
        str(root),
        "--condition",
        condition,
        "--workers",
        str(workers),
        "--batch-size",
        "4",
    ]
    if method:
        cmd += ["--model", method]
    if device:
        cmd += ["--device", device]
    return cmd


def run_jobs(jobs, folder):
    """Fail promptly and terminate only this launcher's child process groups."""
    processes, handles = [], []
    try:
        for name, cmd in jobs:
            handle = (folder / f"{name}.log").open("a")
            handles.append(handle)
            print("Starting", name, flush=True)
            processes.append(
                subprocess.Popen(
                    cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
                )
            )
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"A stage failed; inspect {folder}/*.log")
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f"A stage failed; inspect {folder}/*.log")
    finally:
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in processes:
            if p.poll() is None:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
        for handle in handles:
            handle.close()


def stage_jobs(stage, root, condition, workers):
    if stage == "all":
        return stage_jobs("prepare", root, condition, workers) + stage_jobs(
            "evaluate", root, condition, workers
        )
    if stage == "prepare":
        return [
            [("hosts", command("hosts", root, condition, workers))],
            [("requests", command("requests", root, condition, workers))],
        ]
    return [
        [
            ("seed0", command("model", root, condition, workers, "seed0", "cuda:0")),
            ("seed1", command("model", root, condition, workers, "seed1", "cuda:1")),
            ("hand_insert", command("insertion", root, condition, workers)),
        ],
        [
            ("seed2", command("model", root, condition, workers, "seed2", "cuda:0")),
            ("sham_seed0", command("model", root, condition, workers, "sham_seed0", "cuda:1")),
        ],
    ]


def save_status(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def interrupted(signum, frame):
    raise KeyboardInterrupt(
        f"Scheduler signal {signum}; completed atomic checkpoints are resumable"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "evaluate", "all"])
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    root = args.out.resolve()
    # PSC owns a dedicated result root, never the notebook's public checkpoint root.
    if "kaggle_gap2" in str(root) or root == Path("results").resolve():
        p.error("Use a dedicated PSC output root, outside Kaggle result folders")
    folder = root / args.condition
    folder.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, interrupted)
    with (folder / ".psc.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another job already owns {folder}") from e
        import torch

        torch.set_num_threads(1)
        provenance = dict(
            condition=args.condition,
            stage=args.stage,
            batch_size=4,
            git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            run_gap2_sha256=hashlib.sha256(Path("scripts/run_gap2.py").read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(
                Path("configs/eval/gap2_generalization.json").read_bytes()
            ).hexdigest(),
            python=sys.version,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            numpy=version("numpy"),
            lammps=_installed_version("lammps"),  # None where absent; prepare checks it
            job_id=os.environ.get("SLURM_JOB_ID"),
            array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
            scope="PSC standard/fast conditions only; no overall study completion implied",
        )
        if args.stage == "evaluate":
            if (
                not (folder / "hosts/protocol.json").exists()
                or not (folder / "requests.npz").exists()
            ):
                raise RuntimeError("CPU preparation must finish before evaluation")
        if args.stage in ("evaluate", "all"):
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError("Evaluation requires two scheduler-visible GPUs")
            provenance["gpus"] = [torch.cuda.get_device_name(i) for i in range(2)]
            for i in range(2):
                # Exercise an actual CUDA kernel before spending time on hosts.
                if (torch.ones(1, device=f"cuda:{i}") + 1).item() != 2:
                    raise RuntimeError("CUDA kernel preflight failed")
        if args.stage in ("prepare", "all"):
            from lammps import lammps

            lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
            lmp.close()
        statusfile = folder / f"psc_{args.stage}.json"
        provenance["status"] = "running"
        save_status(statusfile, provenance)
        try:
            for jobs in stage_jobs(args.stage, root, args.condition, args.workers):
                run_jobs(jobs, folder)
            if args.stage in ("evaluate", "all"):
                for method in METHODS:
                    result = json.loads((folder / method / "complete.json").read_text())
                    if result["n_requests"] != 256:
                        raise RuntimeError(f"Incomplete request count for {method}")
            provenance["status"] = "complete"
        except BaseException:
            provenance["status"] = "incomplete"
            raise
        finally:
            save_status(statusfile, provenance)
        print(f"{args.condition}: {args.stage} complete. Outputs: {folder}", flush=True)


if __name__ == "__main__":
    main()
