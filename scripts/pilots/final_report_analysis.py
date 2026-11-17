"""Analyses for the final course report (paper/final_report.tex), from trained weights only.

    python scripts/pilots/final_report_analysis.py shapes        # ~10 s
    python scripts/pilots/final_report_analysis.py splits        # ~3 min
    python scripts/pilots/final_report_analysis.py features      # ~3 min
    python scripts/pilots/final_report_analysis.py sensitivity   # ~25 min on 4 CPU cores

Each writes one JSON (features: also an .npz) to results/final_report_20261117/:

    shapes       every block of the four networks: output shape for a batch of 16 glasses of
                 256 atoms, and its number of parameters
    splits       the finder of the main results on training, validation and test defects:
                 class-weighted cross-entropy (the training loss), plain cross-entropy, and
                 defects found (calibrated, as in the report). Training: 200 defects of each
                 type drawn at random from the training split; validation and test: all real
                 defects
    features     the 128 numbers each atom carries after the last layer (its features), for
                 the denoiser (before any defect labels) and for the finder, embedded in two
                 dimensions with t-SNE; and how well the five roles can be told apart from
                 them (balanced accuracy of a 5-nearest-neighbour vote, half the atoms
                 predicting the other half)
    sensitivity  defects found when every atom of the test glasses is shifted at random
                 (Gaussian, standard deviation 0 to 0.1 sigma, as thermal vibration would),
                 for the finders and the three measures without learning; the flagged
                 atom is judged at its position before the shift
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from glassdiff.data.point_defects import AtomRole, load_point_defects, real_defects, wrap
from glassdiff.models.registry import build_model, load_denoiser
from glassdiff.types import Structures

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/final_report_20261117"
DATA = ROOT / "data/ka2d_256_point_defects"
FINDER = ROOT / "results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt"
DENOISER = ROOT / "results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt"
OTHERS = {  # seed 0 of the comparison of the three kinds of finder
    "transformer": ROOT / "results/kaggle_arch_20261115-0124Z/training/attention_s0/ema.pt",
    "cnn": ROOT / "results/kaggle_arch_20261115-1235Z/training/cnn_s0/ema.pt",
    "egnn_scratch": ROOT / "results/kaggle_arch_20261115-0124Z/training/egnn_scratch_s0/ema.pt",
}


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


br = load("routes", "scripts/pilots/baseline_routes.py")
ev = br.ev


def models_for(path: Path):
    log_prior = ev.finder_log_prior(str(path), DATA)[0]
    return ev.Models(str(path), str(path), threads=4, device="cpu", log_prior=log_prior, do_heal=False)  # fmt: skip


def site_distance(d, pos: np.ndarray | None = None) -> np.ndarray:
    """[E, N]: distance of every atom to the nearer of the two defect sites."""
    pos = d.pos.numpy() if pos is None else pos
    box, sites = d.box.numpy(), d.sites.numpy()
    return np.stack(
        [np.linalg.norm(wrap(pos - sites[:, k][:, None], box[:, None]), axis=-1) for k in range(2)]
    ).min(0)


def found(score: np.ndarray, d) -> np.ndarray:
    dist = site_distance(d)
    return dist[np.arange(len(dist)), score.argmax(-1)] < ev.HIT_RADIUS


# ----------------------------------------------------------------------------- shapes


def shapes() -> dict:
    """Output shape and parameter count of every block, for 16 glasses of 256 atoms."""
    torch.manual_seed(0)
    test = load_point_defects(DATA / "test.npz")["antisite"]  # 256 atoms each
    s = Structures(test.pos[:16].float(), test.types[:16], test.box[:16].float())
    cfgs = {
        "finder (EGNN)": torch.load(FINDER, map_location="cpu", weights_only=False)["model_cfg"],
        "denoiser (EGNN)": torch.load(DENOISER, map_location="cpu", weights_only=False)[
            "model_cfg"
        ],
        "transformer": yaml.safe_load((ROOT / "configs/train/detect_attention.yaml").read_text())[
            "model"
        ],
        "cnn": yaml.safe_load((ROOT / "configs/train/detect_cnn.yaml").read_text())["model"],
    }
    out = {}
    for name, cfg in cfgs.items():
        model = build_model(cfg).eval()
        if hasattr(model, "tta"):
            model.tta = False  # one pass, so that the shapes are those of a single image
        blocks, hooks = [], []

        def record(label):
            def hook(_m, _inp, output):
                tensors = output if isinstance(output, (tuple, list)) else (output,)
                shape = [list(t.shape) for t in tensors if torch.is_tensor(t)]
                blocks.append(dict(block=label, shape=shape))

            return hook

        params = {}
        for child_name, child in model.named_children():
            items = (
                [(f"{child_name}.{i}", m) for i, m in enumerate(child)]
                if isinstance(child, torch.nn.ModuleList)
                else [(child_name, child)]
            )
            for label, m in items:
                params[label] = sum(p.numel() for p in m.parameters())
                hooks.append(m.register_forward_hook(record(label)))
        with torch.no_grad():
            sigma = torch.full((16,), 1e-3)
            final = model(s, sigma) if getattr(model, "sigma_aware", False) else model(s)
        for h in hooks:
            h.remove()
        seen, rows = set(), []
        for b in blocks:  # first call of each block
            if b["block"] not in seen:
                seen.add(b["block"])
                rows.append(dict(**b, parameters=params[b["block"]]))
        for label, n in params.items():  # blocks used without a forward call of their own
            if label not in seen:
                rows.append(dict(block=label, shape=None, parameters=n))
        out[name] = dict(
            config=cfg,
            total=sum(p.numel() for p in model.parameters()),
            output=list(final.shape),
            blocks=rows,
        )
        print(name, out[name]["total"], out[name]["output"], flush=True)
    return out


# ----------------------------------------------------------------------------- splits


def splits() -> dict:
    model = load_denoiser(FINDER, "cpu")
    weights = torch.tensor(
        torch.load(FINDER, map_location="cpu", weights_only=False)["class_weights"]
    )
    models = models_for(FINDER)
    rng = np.random.default_rng(0)
    out = {}
    for split in ("train", "val", "test"):
        data = real_defects(load_point_defects(DATA / f"{split}.npz"))
        if split == "train":
            data = {
                k: d.subset(torch.from_numpy(np.sort(rng.choice(len(d), 200, replace=False))))
                for k, d in data.items()
            }
        res, all_hits = {}, []
        for kind, d in data.items():
            weighted, plain, n_atoms = 0.0, 0.0, 0
            with torch.no_grad():
                for i in range(0, len(d), 20):
                    s = Structures(d.pos[i : i + 20].float(), d.types[i : i + 20], d.box[i : i + 20].float())  # fmt: skip
                    logits = model(s).reshape(-1, 5)
                    target = d.role[i : i + 20].flatten()
                    w = weights[target]
                    ce = F.cross_entropy(logits, target, reduction="none")
                    weighted += float((ce * w).sum() / w.sum()) * len(target)
                    plain += float(ce.sum())
                    n_atoms += len(target)
            pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
            role = int(ev.REPAIR_ROLE[d.kind])
            hit = found(models.role_prob(pos, types, box)[..., role], d)
            all_hits.append(hit)
            res[kind] = dict(
                examples=len(d),
                weighted_ce=round(weighted / n_atoms, 4),
                ce=round(plain / n_atoms, 4),
                found=round(float(hit.mean()), 4),
            )
        res["all"] = dict(found=round(float(np.concatenate(all_hits).mean()), 4))
        out[split] = res
        print(split, json.dumps(res), flush=True)
    return out


# ----------------------------------------------------------------------------- features


def trunk_features(model, d, sigma: float | None) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in range(0, len(d), 20):
            s = Structures(d.pos[i : i + 20].float(), d.types[i : i + 20], d.box[i : i + 20].float())  # fmt: skip
            sig = torch.full((s.batch_size,), sigma if sigma else 1.0)
            h, _ = model.trunk(s, sig)
            feats.append(h.numpy())
    return np.concatenate(feats)


def features() -> dict:
    from sklearn.manifold import TSNE
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    data = real_defects(load_point_defects(DATA / "test.npz"))
    data = {k: d.subset(slice(0, 60)) for k, d in data.items()}
    nets = {"denoiser": (load_denoiser(DENOISER, "cpu"), 1e-3), "finder": (load_denoiser(FINDER, "cpu"), None)}  # fmt: skip
    rng = np.random.default_rng(0)
    roles, species, kinds, feats = [], [], [], {n: [] for n in nets}
    for kind, d in data.items():
        role, types = d.role.numpy().reshape(-1), d.types.numpy().reshape(-1)
        keep = np.nonzero(role != int(AtomRole.INTACT))[0]
        intact = np.nonzero(role == int(AtomRole.INTACT))[0]
        keep = np.sort(np.concatenate([keep, rng.choice(intact, 800, replace=False)]))
        roles.append(role[keep])
        species.append(types[keep])
        kinds += [kind] * len(keep)
        for name, (net, sigma) in nets.items():
            feats[name].append(trunk_features(net, d, sigma).reshape(-1, 128)[keep])
    roles, species = np.concatenate(roles), np.concatenate(species)
    out = dict(atoms=int(len(roles)), roles={int(r): int((roles == r).sum()) for r in np.unique(roles)})  # fmt: skip
    arrays = dict(role=roles, species=species, kind=np.array(kinds))
    half = rng.permutation(len(roles)) < len(roles) // 2
    for name in nets:
        x = np.concatenate(feats[name])
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        knn = KNeighborsClassifier(5).fit(x[half], roles[half])
        out[f"{name}_knn_balanced_accuracy"] = round(
            float(balanced_accuracy_score(roles[~half], knn.predict(x[~half]))), 3
        )
        arrays[name] = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(x)
        print(name, out[f"{name}_knn_balanced_accuracy"], flush=True)
    np.savez_compressed(OUT / "features.npz", **arrays)
    return out


# ----------------------------------------------------------------------------- sensitivity


def sensitivity(levels=(0.0, 0.02, 0.05, 0.1)) -> dict:
    full = load_point_defects(DATA / "test.npz")
    data = real_defects(full)
    # reference statistics of the measures, from the defect-free test glasses (as the report)
    ref = next(iter(full.values()))
    _, first = np.unique(ref.glass.numpy(), return_index=True)
    r = ref.subset(torch.from_numpy(first))
    rp, rt, rb = r.orig_pos.numpy(), r.orig_types.numpy(), r.box.numpy()
    mu_a, sd_a = br.species_stats(np.stack([br.voronoi_area(p, b) for p, b in zip(rp, rb)]), rt)
    mu_u, sd_u = br.species_stats(
        np.stack([br.unlike_neighbours(p, t, b) for p, t, b in zip(rp, rt, rb)]), rt
    )
    k0 = next(iter(full.values()))
    e = ev.ka_energy(Structures(k0.orig_pos, k0.orig_types, k0.box), per_atom=True).numpy()
    mu_e, sd_e = br.species_stats(e, k0.orig_types.numpy())
    finders = {"finder": FINDER, **OTHERS}
    loaded = {name: models_for(path) for name, path in finders.items()}
    out: dict = {"levels": list(levels), "found": {}}
    for level in levels:
        rng = np.random.default_rng(int(level * 1000) + 1)
        res: dict = {}
        for kind, d in data.items():
            pos, types, box = d.pos.numpy(), d.types.numpy(), d.box.numpy()
            noisy = np.mod(pos + level * rng.standard_normal(pos.shape), box[:, None, :])
            role = int(ev.REPAIR_ROLE[d.kind])
            scores = {n: m.role_prob(noisy, types, box)[..., role] for n, m in loaded.items()}
            za = br.area_z(noisy, types, box, mu_a, sd_a)
            scores["cell size"] = {
                "vacancy": za, "interstitial": -za, "frenkel": np.abs(za), "antisite": np.abs(za)
            }[kind]  # fmt: skip
            scores["unlike neighbours"] = -br.bonds_z(noisy, types, box, mu_u, sd_u)
            scores["atom energy"] = ev.energy_z(noisy, types, box, mu_e, sd_e)
            res[kind] = {n: found(sc, d).astype(int).tolist() for n, sc in scores.items()}
        names = list(res["vacancy"])
        out["found"][str(level)] = {
            n: dict(
                {k: round(float(np.mean(res[k][n])), 4) for k in res},
                all=round(float(np.mean(np.concatenate([res[k][n] for k in res]))), 4),
            )
            for n in names
        }
        print(level, json.dumps({n: v["all"] for n, v in out["found"][str(level)].items()}), flush=True)  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["shapes", "splits", "features", "sensitivity"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res = globals()[args.what]()
    (OUT / f"{args.what}.json").write_text(json.dumps(res, indent=1) + "\n")
    print("->", OUT / f"{args.what}.json")


if __name__ == "__main__":
    main()
