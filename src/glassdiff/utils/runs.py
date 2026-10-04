"""Run directories: every result must be reproducible from (config, git commit, seed)."""

from __future__ import annotations

import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> torch.Generator:
    """Seed python, numpy and torch; return a torch Generator for explicit sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def pick_device(name: str | None = "auto") -> torch.device:
    """ "auto" -> CUDA when available, else CPU. On CUDA, allow TF32 matmuls (A100/L4 speed-up;
    no effect on T4)."""
    if name in (None, "", "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def git_state() -> str:
    """Current commit hash, with ' (dirty)' appended when there are uncommitted changes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + (" (dirty)" if dirty else "")


def make_run_dir(name: str, cfg: dict[str, Any], root: str | Path = "runs") -> Path:
    """Create runs/<UTC timestamp>_<name>/ holding the resolved config and git state."""
    run_dir = Path(root) / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%SZ}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "git.txt").write_text(git_state() + "\n")
    return run_dir
