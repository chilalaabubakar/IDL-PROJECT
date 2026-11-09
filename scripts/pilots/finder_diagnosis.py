"""Diagnose the trained finder on the validation split (never the test split).

    python scripts/pilots/finder_diagnosis.py > results/point_defects_finder_20261109/val_diagnosis.log

For each defect kind: what the finder predicts for each atom role (confusion counts, rows =
true role, columns = predicted intact / near vacancy / interstitial / displaced / wrong
species), how often the top-scored atom lies within 1.5 sigma of the defect for several
scores, and how often it is exactly the defect atom. Then, over defect-free glasses (the
originals of the validation examples) and defected glasses: the fraction with any atom above
probability 0.5 or 0.9 of being defective, and the glass-level AUC. Each with raw
probabilities and with the logit adjustment (subtracting the log training class weights,
Menon et al., ICLR 2021). Uses the trained finder of results/kaggle_pd_20261107-2024Z/.
"""

import importlib.util
import json

import numpy as np
import torch

from glassdiff.data.point_defects import AtomRole, load_point_defects, wrap
from glassdiff.models.registry import load_denoiser
from glassdiff.types import Structures

spec = importlib.util.spec_from_file_location("t", "scripts/train_point_defects.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
torch.set_num_threads(4)
finder = load_denoiser("results/kaggle_pd_20261107-2024Z/training/pd_detect/ema.pt")
train = load_point_defects("data/ka2d_256_point_defects/train.npz")
w = t.class_weights(train, 0.5)
print("class weights", [round(float(x), 1) for x in w])
val = load_point_defects("data/ka2d_256_point_defects/val.npz")
R = [r.name.lower() for r in AtomRole]


@torch.no_grad()
def logits_of(pos, types, box):
    out = []
    for i in range(0, len(pos), 20):
        out.append(
            finder(Structures(pos[i : i + 20].float(), types[i : i + 20], box[i : i + 20].float()))
        )
    return torch.cat(out).double()


def hit(score, d):
    top = d.pos[torch.arange(len(d)), score.argmax(-1)]
    dd = np.stack([wrap((top - d.sites[:, k]).numpy(), d.box.numpy()) for k in range(2)])
    return float((np.linalg.norm(dd, axis=-1).min(0) < 1.5).mean())


res = {}
for name, d in val.items():
    lg = logits_of(d.pos, d.types, d.box)
    for label, L in (("raw", lg), ("adjusted", lg - torch.log(w))):
        p = L.softmax(-1)
        pred = p.argmax(-1)
        conf = torch.zeros(5, 5, dtype=torch.long)
        for a in range(5):
            m = d.role == a
            conf[a] = torch.bincount(pred[m], minlength=5)
        need = {"vacancy": 1, "interstitial": 2, "frenkel": 3, "antisite": 4}[name]
        scores = {"needed role": p[..., need], "any defect": 1 - p[..., 0]}
        if name == "frenkel":
            scores["displaced+interstitial"] = p[..., 2] + p[..., 3]
        hits = {k: round(hit(v, d), 3) for k, v in scores.items()}
        exact = None
        if name in ("frenkel", "interstitial"):
            for k, v in scores.items():
                exact = exact or {}
                exact[k] = round(float((v.argmax(-1) == d.atom).double().mean()), 3)
        res[f"{name} {label}"] = dict(
            hits=hits,
            exact_atom=exact,
            confusion={R[a]: conf[a].tolist() for a in range(5) if conf[a].sum() > 0},
        )
    print(
        name,
        json.dumps(res[f"{name} raw"]),
        "\n   adjusted:",
        json.dumps(res[f"{name} adjusted"]["hits"]),
        res[f"{name} adjusted"]["exact_atom"],
        flush=True,
    )

# defect-free glasses (the originals of the validation examples, each glass once)
d = val["vacancy"]
_, first = np.unique(d.glass.numpy(), return_index=True)
clean = d.subset(torch.from_numpy(first))
lg_clean = logits_of(clean.orig_pos, clean.orig_types, clean.box)
for label, adj in (("raw", 0), ("adjusted", torch.log(w))):
    pc = (lg_clean - adj).softmax(-1)
    mx_clean = (1 - pc[..., 0]).amax(-1)
    defected = []
    for name, dd in val.items():
        p = (logits_of(dd.pos, dd.types, dd.box) - adj).softmax(-1)
        defected.append((1 - p[..., 0]).amax(-1))
    mx_def = torch.cat(defected)
    # glass-level AUC: is this glass defective?
    s = torch.cat([mx_def, mx_clean])
    y = torch.cat([torch.ones(len(mx_def)), torch.zeros(len(mx_clean))]).bool()
    ranks = s.argsort().argsort().double() + 1
    auc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    fa = {th: round(float((mx_clean > th).double().mean()), 3) for th in (0.5, 0.9)}
    det = {th: round(float((mx_def > th).double().mean()), 3) for th in (0.5, 0.9)}
    print(
        f"{label}: defect-free glasses {len(mx_clean)}: false alarm {fa}; defected flagged {det}; glass-level AUC {float(auc):.3f}"
    )
