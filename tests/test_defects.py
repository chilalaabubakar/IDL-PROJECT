"""Acceptance tests for Ticket E-2 (defect detectors), on hand-built configurations."""

import pytest
import torch

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.types import DefectClass, Structures

pytestmark = pytest.mark.skip(reason="Ticket E-2: delete this line when implementing")

NX = NY = 8


def site(i, j):
    return (j % NY) * NX + (i % NX)


def test_perfect_lattice_has_no_defects(make_triangular):
    s = make_triangular(NX, NY, a=1.1)
    assert (detect_defects(s, DefectThresholds()) == DefectClass.NONE).all()


def test_vacancy_makes_six_under_coordinated_neighbours(make_triangular):
    s = make_triangular(NX, NY, a=1.1)
    removed = site(3, 4)  # j even: neighbours (i+-1, j), (i-1, j+-1), (i, j+-1)
    neighbours = {site(2, 4), site(4, 4), site(2, 3), site(3, 3), site(2, 5), site(3, 5)}
    keep = torch.tensor([k for k in range(NX * NY) if k != removed])
    s2 = Structures(s.pos[:, keep], s.types[:, keep], s.box)
    labels = detect_defects(s2, DefectThresholds())[0]
    for new_k, old_k in enumerate(keep.tolist()):
        expect = DefectClass.D1_MINUS if old_k in neighbours else DefectClass.NONE
        assert labels[new_k] == expect


def test_b_cluster_is_d2(make_triangular):
    s = make_triangular(NX, NY, a=1.1)  # a = 1.1 < 1.4 * 0.8, so A-B pairs still count
    cluster = [site(0, 0), site(1, 0), site(0, 1)]  # mutually adjacent triangle
    s.types[0, cluster] = 1
    labels = detect_defects(s, DefectThresholds())[0]
    for k in range(NX * NY):
        assert labels[k] == (DefectClass.D2 if k in cluster else DefectClass.NONE)
