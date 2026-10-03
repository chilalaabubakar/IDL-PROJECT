import pytest
import torch

from glassdiff.types import NUM_LABEL_TOKENS, DefectClass, LabelToken, Structures, defect_token


def test_structures_validates_shapes():
    pos = torch.zeros(2, 5, 2)
    types = torch.zeros(2, 5, dtype=torch.long)
    box = torch.ones(2, 2)
    s = Structures(pos, types, box)
    assert (s.batch_size, s.n_atoms) == (2, 5)
    with pytest.raises(ValueError):
        Structures(torch.zeros(2, 5, 3), types, box)
    with pytest.raises(ValueError):
        Structures(pos, torch.zeros(2, 4, dtype=torch.long), box)
    with pytest.raises(ValueError):
        Structures(pos, types, torch.ones(1, 2))


def test_wrapped_puts_positions_in_box():
    box = torch.tensor([[3.0, 4.0]])
    pos = torch.tensor([[[-0.5, 4.5], [3.0, 1.0]]])
    s = Structures(pos, torch.zeros(1, 2, dtype=torch.long), box).wrapped()
    assert torch.allclose(s.pos, torch.tensor([[[2.5, 0.5], [0.0, 1.0]]]))


def test_label_tokens_do_not_collide():
    tokens = {int(LabelToken.UNLABELLED), int(LabelToken.NULL)}
    tokens |= {defect_token(int(c)) for c in DefectClass}
    assert len(tokens) == NUM_LABEL_TOKENS == 2 + len(DefectClass)
    assert max(tokens) == NUM_LABEL_TOKENS - 1
