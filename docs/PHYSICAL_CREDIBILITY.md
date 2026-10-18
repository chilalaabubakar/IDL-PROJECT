# Physical credibility follow-up

These checks investigate credibility; they cannot force generated samples to match natural defects. Report any observed distortions and stability losses. The primary setting is the existing wider-noise D2 model sampled with guidance 2 and regeneration radius 6.

## Host-aware structural comparison

```bash
python scripts/check_physical_credibility.py structural \
  --out results/credibility_followup_20261018 --bootstrap 500
```

The retained runs are declared in the script. Host identities are recovered from the original sequential-host sampling convention and validated using saved host coordinates for model runs. For hand insertion, saved target sites, species and boxes validate the convention; its `pin_pos` contains inserted positions and is never mistaken for the original host.

The natural test reference selects one uniformly chosen D2 site per host. Successful generated requests are compared against this reference using local energy, bond-angle and species-resolved pair-distance W1. Bootstrap draws resample entire host groups on both sides. The sample-count-matched natural comparison repeatedly selects as many natural training windows as the run has successful requests. Its band describes subsampling/reference variability, not a confidence interval on a universal floor. Its host multiplicity pattern is not matched to the generated ensemble.

Host disturbance is measured against original coordinates in shells around the requested site. Host-balanced means and host-bootstrap intervals are reported. Whole-box relaxation is permitted; these are not pinned-boundary displacement guarantees.

## Finite-temperature tests

```bash
# If needed for the pip LAMMPS/MPICH installation:
export LD_LIBRARY_PATH="$(python -c 'import sys; print(sys.prefix)')/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/check_physical_credibility.py cohorts --out results/credibility_followup_20261018
python scripts/check_physical_credibility.py md --out results/credibility_followup_20261018 --workers 2
```

Four cohorts have 32 distinct host glasses each: successful generated D2 placements, successful hand insertions, natural D2 sites, and unedited original hosts paired with the generated cohort. Hosts are selected before selecting one successful request per host. These cohorts measure conditional credibility and are not unbiased estimates of placement success.

The protocol uses the repository's shifted Kob–Andersen potential, freely evolving whole boxes at fixed density, NVT, dt=0.005 and thermostat damping 0.1. Temperatures are 0.01 (the dataset's final temperature) and 0.1 (a 10x-temperature stress test, without a claim of calibrated application relevance). Three velocity seeds are tested per structure. Observations are at 0, 5, 20 and 50 LJ time units: 768 trajectories total. Initial LAMMPS energies must agree with the PyTorch potential within 1e-6 total LJ energy.

Track the initial target B atom and report its defect identity in thermal snapshots and in separately minimized copies. The separate conjugate-gradient minimization engine never changes the running trajectory. Snapshot minimization must converge at force tolerance 1e-6. If conjugate-gradient minimization stops above tolerance, the repository FIRE implementation refines that snapshot; its convergence and refinement steps are recorded. LAMMPS reports the largest Cartesian force component; FIRE reports the per-atom force norm, a stricter bound on that component. Both are required to be at most 1e-6. Previously completed CG snapshots already below tolerance remain valid. Also report whether the same atom remains a D2 within 0.5 of the original site. This is pointwise occupancy, not proof of continuous survival between observations, long-term stability, or equivalence to real materials.

Velocity realizations are grouped by host. Wilson intervals on the proportion of hosts retaining the defect in all three velocity realizations remain nonzero in width when no losses are observed. Raw per-trajectory observations are retained. Unedited controls may not start with a target defect; their occupancy must not be called retention of a defect that was never present.

## Repeat trained models and add a sham control

CPU pilot (eight matched requests per model):

```bash
python scripts/repeat_credibility.py --out runs/credibility_repeat_pilot \
  --n 8 --workers 4 --batch-size 2
```

Larger GPU evaluation, in a separate output directory:

```bash
python scripts/repeat_credibility.py --out runs/credibility_repeat_full \
  --n 256 --workers 1 --batch-size 64 --device cuda
```

The three retained trained models and the seed-0 null-label sham share original hosts, sites, pins, initial noise and sampling-noise seeds within an invocation. The sham replaces the requested class token with NULL and uses the bare denoiser. A zero guidance weight with the defect label still present would not be a sham. All output structures are relaxed with every atom free, matching the earlier experiments. Save original host coordinates and IDs for disturbance analysis.

Results are checkpointed by batch. Resume only with the same configuration. Changing sample count or batch size requires a new output directory. The eight-request pilot is an implementation/replication diagnostic with wide uncertainty; it does not replace the 256-request, three-seed evaluation needed for a publication claim.
