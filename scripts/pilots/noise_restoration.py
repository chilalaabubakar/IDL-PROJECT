"""Pilot: what the trained unconditional denoiser does with shaken glasses (before point defects).

    python scripts/pilots/noise_restoration.py results/point_defects_20261107/noise_restoration.json

Uses the unconditional EGNN (egnn_uncond_smax1, never shown a defect) on the first 50 test
glasses of data/ka2d_256, CPU, 4 threads, fixed seeds (about 45 minutes).

A. What the network computes: on test glasses with Gaussian noise of size sigma, how much of
   the added shift it predicts (R^2) and how well its correction -d lines up with the
   physical Kob-Andersen force (cosine), against the straight line back to the original.
B. Restoration: shake every atom (Gaussian, size delta) or take the stored T = 0.01 thermal
   snapshot; recover the original inherent structure with FIRE minimisation, the denoiser
   (one step, or 30 steps from delta down to 0.001), or the denoiser then FIRE.
   same_basin = every atom back within 0.05 of the original.
C. Local damage: shake only the atoms within 2.5 of a random site by 0.3; restore as in B;
   then: does the per-atom size of the denoiser output pick out the damaged atoms (AUC)?
   References that need the potential: per-atom force and energy change.
Run as written for the results in results/point_defects_20261107/ (no changes since).
"""

import json
import sys
import time

import torch

from glassdiff.data.dataset import load_split
from glassdiff.geometry import minimum_image
from glassdiff.models.registry import load_denoiser
from glassdiff.physics.fire import fire_minimize
from glassdiff.physics.ka_potential import ka_energy, ka_forces
from glassdiff.types import Structures

torch.set_num_threads(4)
M = 50
CKPT = "results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt"
split = load_split("data/ka2d_256/test.npz").subset(slice(0, M))
s0 = split.structures
types, box = s0.types, s0.box
model = load_denoiser(CKPT)
E0 = ka_energy(s0) / s0.n_atoms
out = {}


def S(pos):
    return Structures(pos, types, box)


@torch.no_grad()
def den(pos, sigma):
    res = []
    for i in range(0, M, 10):
        s = Structures(pos[i : i + 10].float(), types[i : i + 10], box[i : i + 10].float())
        res.append(model(s, torch.full((len(s.pos),), float(sigma))).double())
    return torch.cat(res)


def disp(pos):
    d = minimum_image(pos - s0.pos, box)
    return d - d.mean(1, keepdim=True)  # remove any global drift


def fire(pos):
    return fire_minimize(S(pos), fmax=1e-6, max_steps=100_000).structures.pos


def iterate(pos, delta, n=30):
    for sig in torch.linspace(delta, 1e-3, n).tolist():
        pos = pos - den(pos, sig)
    return pos


def score(pos, mask=None):
    d = disp(pos).norm(dim=-1)
    if mask is not None:
        d = d[mask]
    e = ka_energy(S(pos)) / s0.n_atoms - E0
    return dict(
        rmsd=round(float(d.pow(2).mean().sqrt()), 4),
        frac_back=round(float((d < 0.1).double().mean()), 3),
        dE_atom=round(float(e.mean()), 4),
    )


def same_basin(pos):
    return round(float((disp(pos).norm(dim=-1).amax(-1) < 0.05).double().mean()), 3)


def auc(score_, label):
    """P(score of a damaged atom > score of an intact atom)."""
    s, y = score_.flatten(), label.flatten()
    pos_, neg = s[y], s[~y]
    ranks = torch.cat([pos_, neg]).argsort().argsort().double() + 1
    r = ranks[: len(pos_)].sum()
    return round(float((r - len(pos_) * (len(pos_) + 1) / 2) / (len(pos_) * len(neg))), 3)


g = torch.Generator().manual_seed(0)
t0 = time.time()

# A. what the network computes
rows = []
for sigma in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
    eps = torch.randn(s0.pos.shape, generator=g, dtype=torch.float64)
    xs = s0.pos + sigma * eps
    d = den(xs, sigma)
    target = sigma * eps
    r2 = 1 - ((d - target) ** 2).sum() / (target**2).sum()
    f = ka_forces(S(xs))
    cos = torch.nn.functional.cosine_similarity(-d, f, dim=-1)
    cos_true = torch.nn.functional.cosine_similarity(-target, f, dim=-1)
    rows.append(
        dict(
            sigma=sigma,
            R2_shift_explained=round(float(r2), 3),
            cos_minus_d_force=round(float(cos.mean()), 3),
            cos_true_restoring_force=round(float(cos_true.mean()), 3),
        )
    )
    print(rows[-1], flush=True)
out["A_what_it_learns"] = rows

# B. restoration from whole-glass perturbations and thermal snapshots
rows = []
cases = [
    (f"gauss {d}", d, s0.pos + d * torch.randn(s0.pos.shape, generator=g, dtype=torch.float64))
    for d in [0.05, 0.1, 0.2, 0.3]
]
th = split.thermal_pos
th_rms = float(disp(th).norm(dim=-1).pow(2).mean().sqrt())
cases.append((f"thermal T=0.01 (rms {th_rms:.3f})", th_rms, th))
for name, delta, xp in cases:
    r = dict(case=name, perturbed=score(xp))
    xf = fire(xp)
    r["FIRE"] = score(xf) | dict(same_basin=same_basin(xf))
    x1 = xp - den(xp, delta)
    r["denoiser_one_shot"] = score(x1)
    xi = iterate(xp, delta)
    r["denoiser_iterative"] = score(xi)
    xif = fire(xi)
    r["denoiser_then_FIRE"] = score(xif) | dict(same_basin=same_basin(xif))
    rows.append(r)
    print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
out["B_restore_whole_glass"] = rows

# C. local damage, restoration and detection
centre = torch.randint(0, 256, (M,), generator=g)
c_pos = s0.pos[torch.arange(M), centre]
dist = minimum_image(s0.pos - c_pos[:, None], box).norm(dim=-1)
damaged = dist < 2.5
xp = s0.pos + 0.3 * torch.randn(s0.pos.shape, generator=g, dtype=torch.float64) * damaged[..., None]
r = dict(n_damaged_per_glass=round(float(damaged.sum(1).double().mean()), 1))
r["perturbed_damaged"] = score(xp, damaged)
xf = fire(xp)
r["FIRE_damaged"] = score(xf, damaged)
r["FIRE_same_basin"] = same_basin(xf)
xi = iterate(xp, 0.3)
r["denoiser_damaged"] = score(xi, damaged)
r["denoiser_intact_rmsd"] = round(float(disp(xi).norm(dim=-1)[~damaged].pow(2).mean().sqrt()), 4)
xif = fire(xi)
r["denoiser_then_FIRE_damaged"] = score(xif, damaged)
r["denoiser_then_FIRE_same_basin"] = same_basin(xif)
for sig in [0.05, 0.1, 0.3]:
    r[f"AUC_denoiser_|d|_sigma{sig}"] = auc(den(xp, sig).norm(dim=-1), damaged)
r["AUC_force_|F|_(needs_potential)"] = auc(ka_forces(S(xp)).norm(dim=-1), damaged)
r["AUC_atom_energy_change_(needs_potential_and_original)"] = auc(
    (ka_energy(S(xp), per_atom=True) - ka_energy(s0, per_atom=True)), damaged
)
print(json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
out["C_local_damage"] = r
out["setup"] = dict(
    glasses=M, split="ka2d_256 test", checkpoint=CKPT, seconds=round(time.time() - t0)
)
json.dump(out, open(sys.argv[1], "w"), indent=1)
