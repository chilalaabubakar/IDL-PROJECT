"""Small statistical checks for successful-only analysis, without expensive geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "success_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_successful_defects.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_empty_and_single_sample_have_no_invented_uncertainty():
    reference = [np.array([1.0, 2.0])]
    assert analysis.w1([], reference, 20, 0) == [None, None, None]
    assert analysis.w1(reference, reference, 20, 0) == [0.0, None, None]
    assert analysis.w1(reference, [np.array([])], 20, 0) == [None, None, None]


def test_distribution_translation_has_known_w1():
    reference = [np.linspace(-1.0, 1.0, 20)]
    generated = [reference[0] + 0.75 for _ in range(4)]
    result = analysis.w1(generated, reference, 30, 0)
    assert result == pytest.approx([0.75, 0.75, 0.75], abs=0.03)


def test_request_bootstrap_is_repeatable_and_retains_point_estimate():
    result = analysis.ci([0.0, 1.0, 1.0, 0.0], np.mean, 100, 7)
    assert result == analysis.ci([0.0, 1.0, 1.0, 0.0], np.mean, 100, 7)
    assert result[0] == 0.5
    assert 0 <= result[1] <= result[2] <= 1
