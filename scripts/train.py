"""Train a denoiser (unconditional or conditional).  [Tickets M-3, M-4]

    python scripts/train.py --config configs/train/uncond.yaml
    python scripts/train.py --config configs/train/cond.yaml model=configs/model/mpnn.yaml

Loop: batch of clean glasses -> (conditional: RequestSampler) -> sample_sigma -> add_noise
-> model -> masked_displacement_loss -> AdamW step -> EMA update. Writes everything into a
run directory from glassdiff.utils.runs.make_run_dir (config, git state, losses, checkpoints).
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Tickets M-3 / M-4")
