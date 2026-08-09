#!/usr/bin/env bash
# Preprocessing x seed sweep driver for the mobile-transformer benchmark.
#
# Runs the notebook headless once per (preprocess, seed). Each run is fully tagged
# (RUN_TAG=<prep>_s<seed>) so nothing overwrites anything. Split is keyed by seed, so
# all preprocess variants at a given seed share the identical split (paired design).
# Convergence: EPOCHS cap high (80); PATIENCE early-stops at the real plateau.
#
# WATCHDOG: each run gets a hard wall-clock limit (RUN_TIMEOUT). A healthy run is
# ~13-18h; the default 30h kills only a genuinely hung run (the earlier failure mode
# was nbconvert waiting forever on a dead kernel with timeout=-1). On timeout the run
# is killed, a FAILED email is sent, and the grid CONTINUES to the next one instead of
# stalling for days.
#
# Emails via notify.py: one per run (metrics + log attached) plus a final summary.
# Mail failures never abort the grid.
#
# Usage:
#   ./run_grid.sh
#   NUM_WORKERS=24 SEEDS="43 44" ./run_grid.sh
#   RUN_TIMEOUT=36h ./run_grid.sh

set -uo pipefail

NOTEBOOK="${NOTEBOOK:-$HOME/research/une-remote/02-mobile-transformers.ipynb}"
NOTIFY="${NOTIFY:-$HOME/research/une-remote/notify.py}"
CKPT_DIR="${CKPT_DIR:-/scratch/ttran72/checkpoints}"
SEEDS="${SEEDS:-42 43 44 45 46}"
PREPS="${PREPS:-baseline clahe sobel clahe_sobel}"
EPOCHS="${EPOCHS:-80}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-30h}"          # hard per-run wall-clock cap (kills hangs)
LOG_DIR="${LOG_DIR:-/scratch/ttran72/logs/grid}"
OUT_DIR="${OUT_DIR:-/scratch/ttran72/logs/executed_nb}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

notify() {  # notify <status> <log> <results_json>   (never fails the grid)
  [ -f "$NOTIFY" ] && python "$NOTIFY" "$1" "$2" "$3" || true
}

summary="$LOG_DIR/_grid_summary.log"
: > "$summary"
echo "grid: seeds=[$SEEDS] preps=[$PREPS] epochs=$EPOCHS workers=$NUM_WORKERS timeout=$RUN_TIMEOUT" | tee -a "$summary"
total=0; ok=0; fail=0; timedout=0
start_all=$(date +%s)

for seed in $SEEDS; do
  for prep in $PREPS; do
    total=$((total+1))
    tag="${prep}_s${seed}"
    log="$LOG_DIR/${tag}.log"
    out_nb="$OUT_DIR/${tag}.ipynb"
    res="$CKPT_DIR/results_${tag}.json"
    echo "===================================================================="
    echo "[$(date '+%F %T')] RUN $tag   (log: $log, timeout: $RUN_TIMEOUT)"
    echo "===================================================================="
    t0=$(date +%s)
    # --signal=KILL after --kill-after grace; SIGTERM first so Python can clean up.
    timeout --kill-after=120 "$RUN_TIMEOUT" \
      env PREP="$prep" SEED="$seed" EPOCHS="$EPOCHS" NUM_WORKERS="$NUM_WORKERS" \
      jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --output "$out_nb" "$NOTEBOOK" > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      timedout=$((timedout+1)); fail=$((fail+1))
      msg="TIMEOUT $tag  (killed after ${RUN_TIMEOUT}, ${dt}s) -- likely hung, see $log"
      notify "TIMEOUT ${tag} (hung, killed)" "$log" "$res"
    elif [ $rc -eq 0 ] && [ -f "$res" ]; then
      ok=$((ok+1))
      msg="OK   $tag  (${dt}s)"
      notify "SUCCESS ${tag}" "$log" "$res"
    else
      fail=$((fail+1))
      msg="FAIL $tag  (rc=$rc, ${dt}s) -- see $log"
      notify "FAILED ${tag} (exit $rc)" "$log" "$res"
    fi
    echo "[$(date '+%F %T')] $msg" | tee -a "$summary"
  done
done

wall=$(( ($(date +%s)-start_all)/60 ))
echo "====================================================================" | tee -a "$summary"
echo "grid done: $ok ok / $fail fail ($timedout timed out) / $total total   (wall ${wall} min)" | tee -a "$summary"
echo "aggregate with:  python aggregate_preprocessing.py" | tee -a "$summary"
notify "GRID DONE ($ok ok / $fail fail, $timedout hung)" "$summary" "$CKPT_DIR/results_$(echo $PREPS | awk '{print $1}')_s$(echo $SEEDS | awk '{print $1}').json"
