"""Silica point defects (glassdiff.silica.point_defects, scripts/silica/point_defects_study.py).

The fixture is an ideal beta-cristobalite network (Si on a diamond lattice, O midway between
bonded Si): every Si has 4 O and every O 2 Si, so each defect operation has an exact outcome.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from glassdiff.data.point_defects import AtomRole
from glassdiff.silica.defects import OXYGEN, SILICON
from glassdiff.silica.point_defects import (
    SilicaDefectConfig,
    SilicaPointDefect,
    coordination,
    coordination_score,
    coordination_states,
    finish_defect,
    gaps,
    make_defect,
    rarity_table,
    same_glass,
    wrap,
)

ROOT = Path(__file__).resolve().parents[1]
SI_O = 1.6  # A


def cristobalite(reps: int = 2):
    a = 4 * 2 * SI_O / np.sqrt(3)  # Si-Si = 2 Si-O (straight bridges)
    base = np.array(
        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
         [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]]
    )  # fmt: skip
    cells = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
    si = ((base[None] + cells[:, None]) * a).reshape(-1, 3)
    box = np.full(3, reps * a)
    d = wrap(si[None] - si[:, None], box)
    r = np.linalg.norm(d, axis=-1)
    i, j = np.nonzero(np.triu(np.isclose(r, 2 * SI_O, atol=1e-6)))
    o = np.mod(si[i] + 0.5 * d[i, j], box)
    pos = np.vstack([o, si])
    types = np.array([OXYGEN] * len(o) + [SILICON] * len(si))
    return pos, types, box


@pytest.fixture
def net():
    return cristobalite()


def test_fixture_is_an_ideal_network(net):
    pos, types, box = net
    cn, _, peroxy = coordination(pos, types, box, SilicaDefectConfig())
    assert (types == SILICON).sum() == 64 and (types == OXYGEN).sum() == 128
    assert (cn[types == SILICON] == 4).all() and (cn[types == OXYGEN] == 2).all()
    assert (peroxy == 0).all()
    assert (coordination_score(pos, types, box, SilicaDefectConfig()) == 0).all()


@pytest.mark.parametrize("kind", list(SilicaPointDefect))
def test_defect_operations(kind, net):
    pos, types, box = net
    cfg = SilicaDefectConfig()
    made = make_defect(kind, pos, types, box, np.random.default_rng(3), cfg)
    n = len(pos)
    change = {0: -1, 1: 1, 2: 0, 3: -1}[int(kind)]
    assert len(made["pos"]) == len(made["types"]) == n + change
    cn, _, _ = coordination(made["pos"], made["types"], box, cfg)
    fin = finish_defect(kind, made, made["pos"], pos, box, cfg)
    role = fin["role"]
    if kind == SilicaPointDefect.O_VACANCY:
        assert types[made["atom"]] == OXYGEN and len(made["shell"]) == 2
        assert ((made["types"] == SILICON) & (cn == 3)).sum() == 2  # two 3-fold Si
        assert (role == AtomRole.NEAR_VACANCY).sum() == 2
    elif kind == SilicaPointDefect.SI_VACANCY:
        assert types[made["atom"]] == SILICON and len(made["shell"]) == 4
        assert ((made["types"] == OXYGEN) & (cn == 1)).sum() == 4  # four non-bridging O
        assert (role == AtomRole.NEAR_VACANCY).sum() == 4
    elif kind == SilicaPointDefect.O_INTERSTITIAL:
        assert made["types"][-1] == OXYGEN and role[-1] == AtomRole.INTERSTITIAL
        others = np.linalg.norm(wrap(made["pos"][:-1] - made["pos"][-1], box), axis=1)
        assert others.min() > 1.5  # put in a gap
    else:
        hop = np.linalg.norm(wrap(made["pos"][made["atom"]] - pos[made["atom"]], box))
        assert cfg.frenkel_hop[0] - 1e-9 <= hop <= cfg.frenkel_hop[1] + 1e-9
        assert role[made["atom"]] == AtomRole.DISPLACED and fin["valid"]
    # the coordination analysis finds the unrelaxed defect
    score = coordination_score(made["pos"], made["types"], box, cfg)
    top = int(np.argmax(score))
    dist = min(np.linalg.norm(wrap(made["pos"][top] - s, box)) for s in fin["sites"])
    assert score.max() > 0 and dist < 2.5


def test_rarity_ranks_rare_bonding_states_first(net):
    pos, types, box = net
    cfg = SilicaDefectConfig()
    ideal = coordination_states(pos, types, box, cfg)
    # defect-free reference: mostly ideal, some non-bridging O (state 10), one 3-fold Si (130)
    table = rarity_table([ideal, np.full(40, 10), np.array([130])])
    assert table[130] > table[10] > table[20] and table[-1] > table[130]
    made = make_defect(SilicaPointDefect.O_VACANCY, pos, types, box, np.random.default_rng(1), cfg)
    score = coordination_score(made["pos"], made["types"], box, cfg, rarity=table)
    cn, _, _ = coordination(made["pos"], made["types"], box, cfg)
    top = int(np.argmax(score))
    assert made["types"][top] == SILICON and cn[top] == 3
    assert (score[(cn == np.where(made["types"] == SILICON, 4, 2))] >= 0).all()
    assert coordination_score(pos, types, box, cfg, rarity=table).max() == 0


def test_gaps_are_distinct_open_and_in_the_shell(net):
    pos, _, box = net
    pts, clearance = gaps(pos, box, pos[0], 1.0, 4.0, k=5, min_sep=1.0, exclude=0)
    assert 1 <= len(pts) <= 5 and (np.diff(clearance) <= 1e-12).all()
    r = np.linalg.norm(wrap(pts - pos[0], box), axis=1)
    assert ((r >= 1.0 - 1e-9) & (r <= 4.0 + 1e-9)).all()
    sep = np.linalg.norm(wrap(pts[:, None] - pts[None], box), axis=-1)
    assert (sep[~np.eye(len(pts), dtype=bool)] >= 1.0 - 1e-9).all()


def test_same_glass(net):
    pos, types, box = net
    assert same_glass(pos + 0.02, types, pos, types, box)
    o = np.nonzero(types == OXYGEN)[0][:2]
    perm = np.arange(len(pos))
    perm[o] = o[::-1]
    assert same_glass(pos[perm], types[perm], pos, types, box)
    moved = pos.copy()
    moved[5] += 0.3
    assert not same_glass(moved, types, pos, types, box)
    assert not same_glass(pos[1:], types[1:], pos, types, box)


@pytest.mark.lammps
def test_study_script_end_to_end(tmp_path, monkeypatch, net):
    pytest.importorskip("lammps")
    from glassdiff.silica.data import SilicaSplit, save_split

    pos, types, box = net
    split = SilicaSplit(
        torch.tensor(pos)[None].float(),
        torch.tensor(types)[None],
        torch.tensor(box)[None].float(),
        torch.tensor([1.0]),
    )
    save_split(tmp_path / "glass.npz", split)
    spec = importlib.util.spec_from_file_location(
        "silica_pd", ROOT / "scripts/silica/point_defects_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    args = ["x", "--config", str(ROOT / "configs/silica/point_defects.yaml"),
            f"data={tmp_path / 'glass.npz'}", f"out_dir={out}", "n_per_glass=1", "workers=1",
            "kinds=[o_vacancy,si_vacancy]"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", args)
    try:
        module.main()
    except OSError as exc:  # the LAMMPS library cannot be loaded here
        pytest.skip(str(exc))
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["kinds"]) == {"o_vacancy", "si_vacancy"}
    for r in summary["kinds"].values():
        finders = {"coordination", "coordination, rare states first", "energy"}
        assert set(r["find"]) == finders | {"chance"}
        assert "defect given (ceiling)" in r["repair"]
    # a rerun continues from the saved steps instead of redoing them
    module.main()
    assert json.loads((out / "summary.json").read_text())["kinds"] == summary["kinds"]
