"""Tests for Ticket E-3 (reference-defect patch library)."""

import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.data.patches import PatchLibrary
from glassdiff.types import DefectClass, Structures


def _vacancy_lattice(make_triangular):
    s = make_triangular(8, 8, a=1.1)
    keep = torch.tensor([k for k in range(64) if k != 4 * 8 + 3])
    return Structures(s.pos[:, keep], s.types[:, keep], s.box)


def test_build_save_load_sample(make_triangular, tmp_path):
    s = _vacancy_lattice(make_triangular)
    labels = detect_defects(s, DefectThresholds())
    lib = PatchLibrary.build(s, labels, radius=1.5)
    assert lib.counts() == {"D1_MINUS": 6}
    for p in lib.patches:
        assert torch.equal(p.rel_pos[0], torch.zeros(2, dtype=p.rel_pos.dtype))
        assert len(p.types) == 6  # centre + 5 remaining first-shell neighbours
        assert torch.allclose(
            p.rel_pos[1:].norm(dim=-1), torch.full((5,), 1.1, dtype=torch.float64)
        )

    lib.save(tmp_path / "p.npz")
    lib2 = PatchLibrary.load(tmp_path / "p.npz")
    assert len(lib2) == 6
    assert all(torch.allclose(a.rel_pos, b.rel_pos) for a, b in zip(lib.patches, lib2.patches))

    g = torch.Generator().manual_seed(0)
    p = lib2.sample(DefectClass.D1_MINUS, generator=g)
    d_before = torch.cdist(lib.patches[0].rel_pos, lib.patches[0].rel_pos)
    d_after = torch.cdist(p.rel_pos, p.rel_pos)
    assert torch.allclose(torch.sort(d_before.flatten())[0], torch.sort(d_after.flatten())[0])
