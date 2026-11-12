#!/usr/bin/env bash
# The test evaluation of the point-defect report, on real defects only, in small pieces.
#
#   bash scripts/pilots/recheck_real_defects.sh [output directory]
#
# Every piece (one defect kind, 50 test examples) writes its own file under <output>/chunks,
# so a run that is interrupted continues where it stopped when the command is given again.
# paper/make_report_numbers.py joins the pieces. About 75 minutes on 4 CPU cores.
set -euo pipefail
OUT=${1:-results/point_defects_recheck_20261111}
HEALER=results/kaggle_pd_20261108-2014Z/training/pd_heal_v2/ema.pt
HEALER_V1=results/kaggle_pd_20261107-2024Z/training/pd_heal/ema.pt
FINDER=results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt
mkdir -p "$OUT/chunks"

run() { # run <output file> <command ...>: nothing to do when the output exists
  local out=$1
  shift
  if [ -s "$out" ]; then return 0; fi
  echo "$(date -u +%H:%M:%S) $out" >> "$OUT/recheck.log"
  "$@" --out "$out.tmp" >> "$OUT/recheck.log" 2>&1
  mv "$out.tmp" "$out"
}

# is a glass defective at all (finder only)
run "$OUT/glass_level.json" python scripts/eval_point_defects.py --kinds --calibrate --no-heal \
  --real-defects-only --healer "$HEALER" --detector "$FINDER"
# the repair networks without a potential (no minimisation)
run "$OUT/repair_networks.json" python scripts/eval_point_defect_repair.py --real-defects-only \
  --no-minimise --model "v1=$HEALER_V1" --model "v2=$HEALER"

for kind in vacancy interstitial antisite frenkel; do
  for offset in 0 50 100 150; do
    piece="--kinds $kind --offset $offset --max-examples 50 --real-defects-only"
    # cell size and atom energy
    run "$OUT/chunks/baseline_${kind}_${offset}.json" python scripts/pilots/baseline_routes.py $piece \
      --methods geometry energy
    # number of unlike neighbours
    run "$OUT/chunks/bonds_${kind}_${offset}.json" python scripts/pilots/baseline_routes.py $piece \
      --methods bonds
    # the true location given
    run "$OUT/chunks/ceiling_${kind}_${offset}.json" python scripts/eval_point_defect_repair.py $piece
    # the learned finder (and, for Frenkel pairs, the energy search)
    run "$OUT/chunks/finder_${kind}_${offset}.json" python scripts/eval_point_defects.py $piece \
      --calibrate --no-heal --energy-search-flags energy --healer "$HEALER" --detector "$FINDER"
  done
done
echo "$(date -u +%H:%M:%S) all pieces done" >> "$OUT/recheck.log"
