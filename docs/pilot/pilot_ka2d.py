"""Pilot: 2D Kob-Andersen 65:35 glasses (rho=1.2) with LAMMPS.

Melt at T=2.0, linear quench to T=0.01 at a given rate, FIRE-minimize to the
inherent structure. Reports timing, energy, crystallinity (psi6) and the
statistics behind the proposal's defect definitions D1/D2/D3.

Usage:  python pilot_ka2d.py <cooling_rate_per_tau> <n_glasses> [N=256]
        (N must be a perfect square; LAMMPS: `pip install lammps mpich` and put
        the directory holding libmpi.so.12 on LD_LIBRARY_PATH)
This is a feasibility pilot, not the dataset generator.
"""
import sys, time
import numpy as np
from scipy.spatial import Delaunay
from lammps import lammps

N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
RHO, FRAC_A = 1.2, 0.65
M = int(round(np.sqrt(N)))
L = np.sqrt(N / RHO)
DT = 0.005


def make_glass(seed, rate):
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    n_a = int(round(FRAC_A * N))
    t_hi, t_lo = 2.0, 0.01
    quench_steps = int((t_hi - t_lo) / rate / DT)
    cmds = f"""
units lj
dimension 2
atom_style atomic
boundary p p p
lattice sq {RHO}
region box block 0 {M} 0 {M} -0.5 0.5
create_box 2 box
create_atoms 1 box
set type 1 type/subset 2 {N - n_a} {seed + 1}
mass * 1.0
pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 1.0 1.0  2.5
pair_coeff 1 2 1.5 0.8  2.0
pair_coeff 2 2 0.5 0.88 2.2
neighbor 0.3 bin
fix e2d all enforce2d
min_style fire
minimize 0 1e-6 10000 100000
timestep {DT}
velocity all create {t_hi} {seed + 2} dist gaussian
fix nvt all nvt temp {t_hi} {t_hi} 0.1
run 20000
unfix nvt
fix nvt all nvt temp {t_hi} {t_lo} 0.1
run {quench_steps}
unfix nvt
minimize 0 1e-8 100000 1000000
"""
    for c in cmds.strip().splitlines():
        lmp.command(c)
    x = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)[:, :2] % L
    typ = np.array(lmp.gather_atoms("type", 0, 1))
    pe = lmp.get_thermo("pe")
    lmp.close()
    return x, typ, pe, quench_steps


def periodic_images(x):
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)]) * L
    return np.concatenate([x + s for s in shifts]), len(shifts)


def voronoi_neighbors(x):
    """Voronoi (Delaunay) neighbor lists under periodic boundaries."""
    xi, _ = periodic_images(x)
    tri = Delaunay(xi)
    nb = [set() for _ in range(N)]
    for simp in tri.simplices:
        for a in simp:
            for b in simp:
                if a != b and a % N != b % N and xi[a].min() >= 0 and xi[a].max() < L:
                    nb[a % N].add(b % N)
    return nb, tri, xi


def psi6(x, nb):
    out = np.zeros(N)
    for i in range(N):
        d = np.array([x[j] - x[i] for j in nb[i]])
        d -= L * np.round(d / L)
        th = np.arctan2(d[:, 1], d[:, 0])
        out[i] = np.abs(np.exp(6j * th).mean())
    return out


def largest_empty_circle(tri, xi):
    """Max Delaunay circumradius over triangles centred inside the box (D3 proxy)."""
    p = xi[tri.simplices]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    ux = ((a**2).sum(1) * (b[:, 1] - c[:, 1]) + (b**2).sum(1) * (c[:, 1] - a[:, 1]) + (c**2).sum(1) * (a[:, 1] - b[:, 1])) / d
    uy = ((a**2).sum(1) * (c[:, 0] - b[:, 0]) + (b**2).sum(1) * (a[:, 0] - c[:, 0]) + (c**2).sum(1) * (b[:, 0] - a[:, 0])) / d
    r = np.hypot(a[:, 0] - ux, a[:, 1] - uy)
    inside = (ux >= 0) & (ux < L) & (uy >= 0) & (uy < L)
    return r[inside]


def cutoff_cn(x, typ):
    """Coordination within species-pair cutoffs 1.4*sigma_ab (approx. first g(r) minimum)."""
    sig = np.array([[1.0, 0.8], [0.8, 0.88]])
    d = x[:, None] - x[None]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=-1)
    t = typ - 1
    rc = 1.4 * sig[t[:, None], t[None]]
    return ((r < rc) & (r > 0)).sum(1)


def global_psi6(x, nb):
    tot = []
    for i in range(N):
        d = np.array([x[j] - x[i] for j in nb[i]])
        d -= L * np.round(d / L)
        tot.append(np.exp(6j * np.arctan2(d[:, 1], d[:, 0])).mean())
    return np.abs(np.mean(tot))


def cutoff_bb_neighbors(x, typ, rc=1.2):
    """Number of B neighbours of each B atom within rc (first-shell cut for BB)."""
    xb = x[typ == 2]
    d = xb[:, None] - xb[None]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=-1)
    return ((r < rc) & (r > 0)).sum(1)


if __name__ == "__main__":
    rate = float(sys.argv[1])
    n_glass = int(sys.argv[2])
    coord, bb, psi, pes, circ, times, cn, ta, gpsi = [], [], [], [], [], [], [], [], []
    for s in range(n_glass):
        t0 = time.time()
        x, typ, pe, qs = make_glass(1000 + 10 * s, rate)
        times.append(time.time() - t0)
        nb, tri, xi = voronoi_neighbors(x)
        coord.append([len(n) for n in nb])
        bb.append(cutoff_bb_neighbors(x, typ))
        psi.append(psi6(x, nb)[typ == 1])
        pes.append(pe)
        cn.append(cutoff_cn(x, typ)); ta.append(typ); gpsi.append(global_psi6(x, nb))
        circ.append(largest_empty_circle(tri, xi))
    coord = np.concatenate(coord); bb = np.concatenate(bb)
    psi = np.concatenate(psi); circ = np.concatenate(circ)
    print(f"rate={rate}/tau  quench_steps={qs}  glasses={n_glass}  L={L:.3f}")
    print(f"wall time per glass: {np.mean(times):.1f} s (1 core)")
    print(f"PE/atom (inherent structure): {np.mean(pes):.4f} +- {np.std(pes):.4f}")
    print(f"psi6 (A atoms) mean={psi.mean():.3f}  frac>0.7={np.mean(psi > 0.7):.3f}")
    vals, cnts = np.unique(coord, return_counts=True)
    print("Voronoi coordination:", {int(v): round(c / len(coord), 4) for v, c in zip(vals, cnts)})
    vals, cnts = np.unique(bb, return_counts=True)
    print("B atoms: # B neighbours (r<1.2):", {int(v): round(c / len(bb), 4) for v, c in zip(vals, cnts)})
    cn = np.concatenate(cn); ta = np.concatenate(ta)
    print(f"global |Psi6| per glass: {np.round(gpsi, 3)}  frac psi6>0.9 (A): {np.mean(psi > 0.9):.3f}")
    for t, name in ((1, "A"), (2, "B")):
        vals, cnts = np.unique(cn[ta == t], return_counts=True)
        print(f"cutoff CN ({name}):", {int(v): round(c / (ta == t).sum(), 4) for v, c in zip(vals, cnts)})
    q = np.quantile(circ, [0.5, 0.95, 0.99, 0.999])
    print("Delaunay circumradius quantiles 50/95/99/99.9%:", np.round(q, 3))
