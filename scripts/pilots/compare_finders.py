"""Paired comparison of two point-defect finders on the same test defects.

Both finders are calibrated (logit adjustment, as `eval_point_defects.py --calibrate`) and
scored as there: a defect is found when the top-scored atom lies within 1.5 sigma of it.
Because both see the same defects, the comparison is paired: McNemar's exact test on the
defects that only one of the two finds.

    python scripts/pilots/compare_finders.py \
        --a results/kaggle_pd_20261107-2024Z/training/pd_detect/ema.pt \
        --b results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt \
        --out results/point_defects_finder_v2_20261110/paired_find.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("eval_pd", ROOT / "scripts/eval_point_defects.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def found(path: str, data: dict, data_dir: Path, n: int, device: str) -> dict[str, np.ndarray]:
    log_prior, _ = ev.finder_log_prior(path, data_dir)
    # the healer is required by Models but not used for finding
    models = ev.Models(path, path, threads=4, device=device, log_prior=log_prior, do_heal=False)
    out = {}
    for name, d in data.items():
        d = d.subset(slice(0, n))
        pos, types, box, sites = d.pos.numpy(), d.types.numpy(), d.box.numpy(), d.sites.numpy()
        score = models.role_prob(pos, types, box)[..., int(ev.REPAIR_ROLE[d.kind])]
        top = pos[np.arange(len(pos)), score.argmax(-1)]
        dist = np.stack(
            [np.linalg.norm(ev.wrap(top - sites[:, k], box), axis=-1) for k in range(2)]
        ).min(0)
        out[name] = dist < ev.HIT_RADIUS
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    only_b, only_a = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(only_b, only_a + only_b).pvalue if only_a + only_b else 1.0
    return dict(
        examples=len(a),
        a=round(float(a.mean()), 4),
        b=round(float(b.mean()), 4),
        only_a=only_a,
        only_b=only_b,
        mcnemar_p=round(float(p), 4),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="first finder checkpoint")
    ap.add_argument("--b", required=True, help="second finder checkpoint")
    ap.add_argument("--data", default="data/ka2d_256_point_defects/test.npz")
    ap.add_argument("--max-examples", type=int, default=200, help="per defect kind")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--real-defects-only",
        action="store_true",
        help="leave out examples that relaxed back to the original glass",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = ev.load_point_defects(args.data)
    data = {name: d.subset(slice(0, args.max_examples)) for name, d in data.items()}
    if args.real_defects_only:
        data = ev.real_defects(data)
    data_dir = Path(args.data).parent
    hits_a = found(args.a, data, data_dir, args.max_examples, args.device)
    hits_b = found(args.b, data, data_dir, args.max_examples, args.device)
    out = dict(a=args.a, b=args.b, data=args.data, kinds={})
    for name in data:
        out["kinds"][name] = paired(hits_a[name], hits_b[name])
        print(name, out["kinds"][name])
    out["all"] = paired(
        np.concatenate(list(hits_a.values())), np.concatenate(list(hits_b.values()))
    )
    print("all", out["all"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
