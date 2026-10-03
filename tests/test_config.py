import pytest

from glassdiff.utils.config import load_config


def test_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("optim:\n  lr: 0.1\n  batch_size: 16\nname: x\n")
    cfg = load_config(p, ["optim.lr=1e-4", "model.hidden=64", "name=y", "flag=true"])
    assert cfg["optim"] == {"lr": 1e-4, "batch_size": 16}
    assert cfg["model"]["hidden"] == 64
    assert cfg["name"] == "y" and cfg["flag"] is True
    with pytest.raises(ValueError):
        load_config(p, ["no_equals_sign"])
