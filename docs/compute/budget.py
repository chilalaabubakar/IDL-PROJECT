"""Compute budget for the rest of the project, in the ACCESS-proposal style:

    GPU-hours = number of variants (ablations) x epochs x epoch time

    python docs/compute/budget.py            # prints the tables used in README.md

Every number below is either measured (marked "measured") or an assumption that the team can
change. Credits use the PSC Explore guide: 200,000 credits ~ 3,700 GPU-hours, 10,000 credits
~ 1 TB of storage, and about 1 credit per CPU core-hour (to confirm in the ACCESS exchange
calculator).
"""

from __future__ import annotations

CREDITS_PER_GPU_HOUR = 200_000 / 3_700
CREDITS_PER_TB = 10_000
CREDITS_PER_CORE_HOUR = 1.0  # assumption: check allocations.access-ci.org/exchange_calculator

# ---------------------------------------------------------------- measured anchors
# 2D (our runs on Kaggle/Colab T4 GPUs): 0.27-0.37 s per update, batch 16 x 256 atoms;
# one 256-sample evaluation row (200 sampler steps + relaxation) takes 15 min (median).
T4_EPOCH_2D = 50 * 0.32  # s; 800 training glasses / batch 16 = 50 updates per epoch
T4_ROW_2D = 0.25  # GPU-h per evaluation row
# 3D silica, 576-atom cells, batch 16 (V100/A6000-class GPU). DM2 trains 30,000 updates on
# 3,000-atom cells in ~40 h on an A6000 (4.8 s per update for 48,000 atoms), i.e. 0.1 ms per
# atom per update; our CPU benchmark (bench_models.py) gives NequIP 2.9x our EGNN per atom,
# which predicts about half that. We plan with the slower DM2 figure:
UPDATE_NEQUIP = 1.2  # s per update (16 x 576 atoms)
UPDATE_EGNN3D = 0.35  # s per update (EGNN measured in 2D x 3D neighbour count)
FORWARD_FRACTION = 0.21  # measured: forward pass / training step, per atom (NequIP, CPU)
MD_S_PER_ATOM_STEP = 1 / 200_000  # measured: LAMMPS, silica, 8 A Born + DSF Coulomb, 1 core
N_ATOMS = 576
TRAIN_GLASSES, BATCH = 600, 16
UPDATES_PER_EPOCH = TRAIN_GLASSES / BATCH  # 37.5
FULL_RUN_EPOCHS = round(50_000 / UPDATES_PER_EPOCH)  # 1,333 (as in 2D: 50,000 updates)
EPOCH_H = {
    "NequIP": UPDATES_PER_EPOCH * UPDATE_NEQUIP / 3600,
    "EGNN-3D": UPDATES_PER_EPOCH * UPDATE_EGNN3D / 3600,
}


def training(model: str, stages: list[tuple[str, int, int]]) -> list[tuple]:
    """stages: (description, variants, epochs each) -> rows with GPU-hours."""
    return [(d, v, e, v * e, v * e * EPOCH_H[model]) for d, v, e in stages]


def nequip_stages(screen_variants: int) -> list[tuple[str, int, int]]:
    return [
        (f"Screening ({screen_variants} variants)", screen_variants, 133),
        ("Top 6 continue", 6, 267),
        ("Top 2 continue to full length", 2, 933),
        ("Best variant, 2 more seeds", 2, FULL_RUN_EPOCHS),
        ("Model without labels (for baselines)", 1, FULL_RUN_EPOCHS),
    ]


EGNN_STAGES = [
    ("Screening (4 variants)", 4, 133),
    ("Best variant, 3 seeds", 3, FULL_RUN_EPOCHS),
]


def sampling_rows(steps: int) -> list[tuple]:
    """Evaluation rows of 256 samples; CFG rows run the network twice per step."""
    per_atom_fwd = {
        m: FORWARD_FRACTION * u / (BATCH * N_ATOMS)
        for m, u in (("NequIP", UPDATE_NEQUIP), ("EGNN-3D", UPDATE_EGNN3D))
    }

    def hours(rows, passes, model, samples=256, n_steps=steps):
        return rows * samples * n_steps * passes * N_ATOMS * per_atom_fwd[model] / 3600

    return [
        ("Main table: 3 defect types x w = 2, 6 x 3 seeds", 18, hours(18, 2, "NequIP")),
        ("Guidance sweep: 3 types x w = 0, 1, 4", 9, hours(9, 2, "NequIP")),
        ("Region-size sweep: 3 types x 2 extra sizes", 6, hours(6, 2, "NequIP")),
        ("Baselines without labels (chance, clamp, RePaint)", 9, hours(9, 1, "NequIP")),
        ("EGNN-3D rows: 3 types x 3 seeds", 9, hours(9, 2, "EGNN-3D")),
        (
            "Step-count calibration (100-3,000 steps, 64 samples)",
            4,
            hours(1, 1, "NequIP", samples=64, n_steps=100 + 300 + 1000 + 3000),
        ),
    ]


def glass_core_hours(rate_k_per_ps: float, n_atoms: int = N_ATOMS) -> float:
    """DM2 protocol: 50 ps NVT + 50 ps NPT + 50 ps melt + quench 5000 -> 300 K + 100 ps."""
    ps = 250 + 4700 / rate_k_per_ps
    return ps * 1000 * MD_S_PER_ATOM_STEP * n_atoms / 3600


def cpu_rows() -> list[tuple]:
    md_step_h = MD_S_PER_ATOM_STEP * N_ATOMS / 3600
    lmq_attempt = ((50 + 470) * 1000 + 2000) * md_step_h  # melt, quench at 10 K/ps, minimise
    data = (
        550 * glass_core_hours(10)
        + 200 * glass_core_hours(100)
        + 50 * glass_core_hours(1)
        + 10 * glass_core_hours(1, 1536)
    )
    return [
        (
            "MD dataset: 550 glasses at 10 K/ps, 200 at 100 K/ps, 50 at 1 K/ps, "
            "10 large at 1 K/ps (+10% reruns)",
            data * 1.1,
        ),
        ("Local melt-quench baseline: 3 types x 2 sizes x 256 attempts", 1536 * lmq_attempt),
        ("Relaxing generated samples in LAMMPS: ~60 rows x 256", 60 * 256 * 2000 * md_step_h),
        ("Hand-built baseline and energy-barrier (NEB) application", 150),
    ]


GPU_CONTINGENCY = 1.2  # failed runs, reruns, surprises
CPU_CONTINGENCY = 1.2
DFT_CORE_H = 5_000  # optional DFT check, upper end of 2,000-5,000
STORAGE_TB = 1


def gpu_3d(screen_variants: int, steps: int) -> dict[str, list]:
    other = [
        ("Physics checks: MACE re-relaxation of ~2,300 structures, vibrations", 20.0),
        ("Setup, benchmarks and debugging", 15.0),
    ]
    return {
        "nequip": training("NequIP", nequip_stages(screen_variants)),
        "egnn": training("EGNN-3D", EGNN_STAGES),
        "sampling": sampling_rows(steps),
        "other": other,
    }


def gpu_3d_hours(parts: dict[str, list]) -> float:
    return (
        sum(r[4] for r in parts["nequip"])
        + sum(r[4] for r in parts["egnn"])
        + sum(r[2] for r in parts["sampling"])
        + sum(h for _, h in parts["other"])
    )


def rows_2d() -> list[tuple]:
    return [
        ("Train with a range of region sizes (D2 and D1- recipes)", 2, 1000 * T4_EPOCH_2D / 3600),
        ("Size test on 1,024-atom glasses (5 rows, 4x atoms)", 5, 4 * T4_ROW_2D),
        ("Slowly cooled hosts (5 rows)", 5, T4_ROW_2D),
        ("Choose w per model on validation (6 models x 3 weights)", 18, T4_ROW_2D),
        ("Evaluate the region-range models (2 x 3 sizes)", 6, T4_ROW_2D),
        ("Larger region for D1+ (3 rows)", 3, T4_ROW_2D),
    ]


def phases(parts: dict[str, list], gpu_2d: float, cpu: list[tuple]) -> list[tuple]:
    """(weeks, milestone, GPU-h, core-h), contingency included."""
    nq = {r[0].split(" (")[0]: r[4] for r in parts["nequip"]}
    eg = {r[0].split(" (")[0]: r[4] for r in parts["egnn"]}
    smp = {r[0]: r[2] for r in parts["sampling"]}
    other = dict(parts["other"])
    calib = smp.pop("Step-count calibration (100-3,000 steps, 64 samples)")
    md, lmq, relax, neb = (h for _, h in cpu)
    g, c = GPU_CONTINGENCY, CPU_CONTINGENCY
    return [
        ("1-3", "Finish the 2D paper (optional runs)", gpu_2d * g, 0.0),
        (
            "1-2",
            "M0 Setup: allocation, potential check, benchmarks",
            other["Setup, benchmarks and debugging"] * g,
            0.0,
        ),
        ("2-5", "M1 Data: silica glasses, detectors, stability checks", 0.0, md * c),
        (
            "4-8",
            "M2 Model: screening, top 6, step calibration",
            (nq["Screening"] + nq["Top 6 continue"] + eg["Screening"] + calib) * g,
            0.0,
        ),
        (
            "7-11",
            "M3 Main results: full runs, seeds, all sampling rows, baselines",
            (
                nq["Top 2 continue to full length"]
                + nq["Best variant, 2 more seeds"]
                + nq["Model without labels"]
                + eg["Best variant, 3 seeds"]
                + sum(smp.values())
            )
            * g,
            (lmq + relax) * c,
        ),
        (
            "10-13",
            "M4 Physics checks, energy barriers, DFT",
            other["Physics checks: MACE re-relaxation of ~2,300 structures, vibrations"] * g,
            neb * c + DFT_CORE_H,
        ),
        ("12-16", "M5 Paper: figures, release", 0.0, 0.0),
    ]


def print_parts(parts: dict[str, list]) -> None:
    for title, key in (("NequIP", "nequip"), ("EGNN-3D", "egnn")):
        print(f"| {title} stage | variants | epochs each | epochs | GPU-h |\n|---|---|---|---|---|")
        for d, v, e, te, h in parts[key]:
            print(f"| {d} | {v} | {e:,} | {te:,} | {h:.1f} |")
        print()
    print("| Sampling and checks | rows | GPU-h |\n|---|---|---|")
    for d, r, h in parts["sampling"]:
        print(f"| {d} | {r} | {h:.1f} |")
    for d, h in parts["other"]:
        print(f"| {d} | | {h:.0f} |")


def main() -> None:
    print(
        f"Epoch (600 glasses, batch 16 = {UPDATES_PER_EPOCH} updates): NequIP "
        f"{EPOCH_H['NequIP'] * 3600:.0f} s, EGNN-3D {EPOCH_H['EGNN-3D'] * 3600:.0f} s; "
        f"full run {FULL_RUN_EPOCHS:,} epochs = {FULL_RUN_EPOCHS * EPOCH_H['NequIP']:.1f} "
        f"GPU-h (NequIP)"
    )
    print(
        f"MD per glass (576 atoms, 1 core): 10 K/ps {glass_core_hours(10):.2f} h, "
        f"100 K/ps {glass_core_hours(100):.2f} h, 1 K/ps {glass_core_hours(1):.2f} h"
    )

    # ---- plan of record: the upper plan (64 screening variants, 3,000 sampler steps)
    plan = gpu_3d(64, 3000)
    gpu3d = gpu_3d_hours(plan)
    gpu2d = sum(n * h for _, n, h in rows_2d())
    cpu = cpu_rows()
    cpu_sub = sum(h for _, h in cpu)
    gpu_total = (gpu3d + gpu2d) * GPU_CONTINGENCY
    cpu_total = cpu_sub * CPU_CONTINGENCY + DFT_CORE_H

    print("\n# Plan of record: upper plan (64 screening variants, 3,000 sampler steps)")
    print("\n## 2D work left (T4 GPU-hours)\n\n| Item | runs/rows | GPU-h |\n|---|---|---|")
    for d, n, h in rows_2d():
        print(f"| {d} | {n} | {n * h:.1f} |")
    print(f"\n2D subtotal {gpu2d:.1f} GPU-h\n\n## 3D GPU work\n")
    print_parts(plan)
    print(f"\n3D subtotal {gpu3d:.1f} GPU-h")
    print("\n## CPU (core-hours)\n\n| Item | core-h |\n|---|---|")
    for d, h in cpu:
        print(f"| {d} | {h:.0f} |")
    print(f"| Optional DFT check (upper end) | {DFT_CORE_H:,} |")

    g_cr = gpu_total * CREDITS_PER_GPU_HOUR
    c_cr = cpu_total * CREDITS_PER_CORE_HOUR
    s_cr = STORAGE_TB * CREDITS_PER_TB
    print("\n## Totals (20% contingency on GPU and on non-DFT CPU)\n")
    print("| Resource | Amount | ACCESS credits |\n|---|---|---|")
    print(f"| GPU | ({gpu3d:.0f} + {gpu2d:.0f}) x 1.2 = {gpu_total:.0f} GPU-h | {g_cr:,.0f} |")
    print(
        f"| CPU | {cpu_sub:,.0f} x 1.2 + {DFT_CORE_H:,} = {cpu_total:,.0f} core-h | {c_cr:,.0f} |"
    )
    print(f"| Storage | {STORAGE_TB} TB | {s_cr:,.0f} |")
    print(f"| Total | | {g_cr + c_cr + s_cr:,.0f} ({(g_cr + c_cr + s_cr) / 2000:.0f}% of 200K) |")

    print(
        "\n## By phase\n\n| Weeks | Milestone | GPU-h | core-h | credits |\n|---|---|---|---|---|"
    )
    tg = tc = 0.0
    for weeks, name, gh, ch in phases(plan, gpu2d, cpu):
        tg, tc = tg + gh, tc + ch
        print(
            f"| {weeks} | {name} | {gh:.0f} | {ch:,.0f} | {gh * CREDITS_PER_GPU_HOUR + ch * CREDITS_PER_CORE_HOUR:,.0f} |"
        )
    print(
        f"| | Total (+ storage {s_cr:,.0f}) | {tg:.0f} | {tc:,.0f} | {tg * CREDITS_PER_GPU_HOUR + tc * CREDITS_PER_CORE_HOUR + s_cr:,.0f} |"
    )

    base = gpu_3d_hours(gpu_3d(32, 1000))
    print(
        f"\nLower bound for reference (32 screening variants, 1,000 sampler steps): "
        f"{(base + gpu2d) * GPU_CONTINGENCY:.0f} GPU-h"
    )


if __name__ == "__main__":
    main()
