"""YAML configs with dotted command-line overrides, e.g. ``train.lr=1e-4 model.hidden=64``."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(value)
    return cfg


def _parse_value(text: str) -> Any:
    """YAML scalar parsing, plus floats like 1e-4 that YAML 1.1 would leave as strings."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def cli_config(description: str) -> dict[str, Any]:
    """Parse ``--config path [key.sub=value ...]`` from the command line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. optim.lr=1e-4")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)
