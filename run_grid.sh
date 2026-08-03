#!/usr/bin/env bash
# Preprocessing x seed sweep driver for the mobile-transformer benchmark.
#
# Runs the notebook headless once per (preprocess, seed) combination. Each run is
# fully tagged (RUN_TAG = <prep>_s<seed>), so nothing overwrites anything and the
# aggregator can pool seeds per (model, preprocess). The split is keyed by seed, so
# all four preprocess variants at a given seed share the identical train/val/test
# partition -> a clean PAIRED comparison.
#
# Convergence: EPOCHS cap is high (80) and PATIENCE early-stops at the real plateau.
#
# Usage:
#   ./run_grid.sh                      # default: 5 seeds x 4 preprocess
#   SEEDS="42 43 44" PREPS="baseline clahe" ./run_grid.sh
#
# Notes:
#  - Uses the TRAINING venv/kernel (not the aiedge one); the notebook itself shells
#    out to the aiedge venv for MobileViT TFLite conversion.
#  - Serial by design (one GPU). To parallelise across GPUs, launch subsets with
#    CUDA_VISIBLE_DEVICES set and disjoint SEEDS.

set -uo pipefail

NOTEBOOK="${NOTEBOOK:-$HOME/research/une-remote/02-mobile-transformers.ipynb}"
SEEDS="${SEEDS:-42 43 44 45 46}"
PREPS="${PREPS:-baseline clahe sobel clahe_sobel}"
EPOCHS="${EPOCHS:-80}"
LOG_DIR="${LOG_DIR:-/scratch/ttran72/logs/grid}"
OUT_DIR="${OUT_DIR:-/scratch/ttran72/logs/executed_nb}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "grid: seeds=[$SEEDS] preps=[$PREPS] epochs_cap=$EPOCHS"
echo "notebook: $NOTEBOOK"
total=0; ok=0; fail=0
start_all=$(date +%s)

for seed in $SEEDS; do
  for prep in $PREPS; do
    total=$((total+1))
    tag="${prep}_s${seed}"
    log="$LOG_DIR/${tag}.log"
    out_nb="$OUT_DIR/${tag}.ipynb"
    echo "===================================================================="
    echo "[$(date '+%F %T')] RUN $tag   (log: $log)"
    echo "===================================================================="
    t0=$(date +%s)
    PREP="$prep" SEED="$seed" EPOCHS="$EPOCHS" \
      jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --output "$out_nb" "$NOTEBOOK" > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
      ok=$((ok+1));   echo "[$(date '+%F %T')] OK   $tag  (${dt}s)"
    else
      fail=$((fail+1)); echo "[$(date '+%F %T')] FAIL $tag  (rc=$rc, ${dt}s) -- see $log"
    fi
  done
done

echo "===================================================================="
echo "done: $ok ok / $fail fail / $total total   (wall $(( ($(date +%s)-start_all)/60 )) min)"
echo "aggregate with:  python aggregate_preprocessing.py"
