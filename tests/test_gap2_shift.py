"""Cross-condition shift analysis for Gap 2 (scripts/analyze_gap2_shift.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gap2_shift", ROOT / "scripts/analyze_gap2_shift.py")
shift = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shift)


def _condition(root: Path, name: str, rates: dict, make_random, n_hosts=6, per_host=4):
    """A fake completed condition: requests, five methods and saved hosts."""
    folder = root / name
    ids = np.repeat(np.arange(n_hosts), per_host)
    initially = np.zeros(len(ids), dtype=bool)
    initially[0] = True
    folder.mkdir(parents=True)
    np.savez(folder / "requests.npz", host_ids=ids, original_success=initially)
    rng = np.random.default_rng(len(name))
    for m in shift.METHODS:
        (folder / m).mkdir()
        success = rng.random(len(ids)) < rates[m]
        np.savez(folder / m / "samples.npz", host_ids=ids, success=success)
        (folder / m / "complete.json").write_text("{}")
    s = make_random(n_hosts, 64, seed=3)
    (folder / "hosts").mkdir()
    np.savez(
        folder / "hosts" / "samples.npz",
        pos=s.pos.numpy(),
        types=s.types.numpy().astype(np.int8),
        box=s.box.numpy(),
        pe_atom=np.full((n_hosts, 64), -3.7),
    )
    return folder


def test_shift_is_zero_against_itself_and_signed(tmp_path, make_random):
    rates = dict(seed0=0.5, seed1=0.4, seed2=0.45, sham_seed0=0.1, hand_insert=0.3)
    study = tmp_path / "study"
    _condition(study, shift.REFERENCE, rates, make_random)
    _condition(study, "ka2d_1024_slow", {**rates, "seed0": 0.9}, make_random)
    folders = shift.complete_conditions([study])
    assert set(folders) == {shift.REFERENCE, "ka2d_1024_slow"}
    data = {c: shift.load(f) for c, f in folders.items()}
    rows = shift.shift_table(data, "initially_nonD2")
    by = {r["quantity"]: r for r in rows}
    assert by["seed0"]["condition_value"] > by["seed0"]["reference_value"]
    assert by["seed0"]["lo"] <= by["seed0"]["shift"] <= by["seed0"]["hi"]
    assert by["seed_mean"]["shift"] == pytest.approx(
        np.mean([by[s]["condition_value"] for s in shift.SEEDS])
        - np.mean([by[s]["reference_value"] for s in shift.SEEDS]),
        abs=1e-3,
    )
    # the request that was already D2 is excluded from the primary denominator
    assert by["seed0"]["n_hosts_condition"] == 6
    same = shift.shift_table(
        {shift.REFERENCE: data[shift.REFERENCE], "copy": data[shift.REFERENCE]}, "all_requests"
    )
    assert all(r["shift"] == 0 for r in same)


def test_incomplete_condition_is_ignored(tmp_path, make_random):
    rates = dict(seed0=0.5, seed1=0.4, seed2=0.45, sham_seed0=0.1, hand_insert=0.3)
    study = tmp_path / "study"
    _condition(study, shift.REFERENCE, rates, make_random)
    partial = _condition(study, "ka2d_256_slow", rates, make_random)
    (partial / "seed2" / "complete.json").unlink()
    assert set(shift.complete_conditions([study])) == {shift.REFERENCE}


def test_script_writes_tables(tmp_path, make_random, monkeypatch):
    rates = dict(seed0=0.5, seed1=0.4, seed2=0.45, sham_seed0=0.1, hand_insert=0.3)
    study = tmp_path / "study"
    _condition(study, shift.REFERENCE, rates, make_random)
    _condition(study, "ka2d_1024_slow", rates, make_random)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--study",
            str(study),
            "--out",
            str(out),
            "--thresholds",
            str(ROOT / "data/ka2d_256/meta.json"),
        ],
    )
    shift.main()
    assert (out / "hosts.csv").read_text().count("\n") == 3
    assert "seed_mean_minus_hand" in (out / "shift.csv").read_text()
    assert json.loads((out / "metadata.json").read_text())["reference"] == shift.REFERENCE
