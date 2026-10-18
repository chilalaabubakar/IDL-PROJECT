"""Denominators and protocol validation for extended credibility evaluation."""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gap1", ROOT / "scripts/complete_gap1.py")
gap1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gap1)


def test_joint_yield_weights_requests_and_keeps_initial_failures():
    # Two requests share one host: a host-balanced average would incorrectly give .25.
    estimate = gap1.host_ratio_ci(
        np.array([1.0, 0.0, 0.0]), np.array([4, 4, 9]), np.array([True, True, True])
    )
    assert estimate[0] == pytest.approx(1 / 3)
    eligible = gap1.host_ratio_ci(
        np.array([1.0, 0.0, 0.0]), np.array([4, 4, 9]), np.array([False, True, True])
    )
    assert eligible[0] == 0


def test_md_rejects_reversed_observation_times_before_execution():
    with pytest.raises(ValueError, match="strictly increase"):
        gap1.p.md_worker(("test", 0, 0.01, 101, (), "unused", {"steps": [0, 10, 5]}))


def test_kaggle_notebook_has_executable_python_and_no_saved_outputs():
    notebook = json.loads((ROOT / "notebooks/kaggle_gap1_completion.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
