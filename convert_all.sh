#!/usr/bin/env bash
# Bulk TFLite conversion from saved .pt weights. Runs AFTER training, decoupled.
# Discovers every completed run from results_<tag>.json and converts all 3 models:
#   - mobilenetv2_100, efficientformerv2_s0  via onnx2tf   (training venv)
#   - mobilevit_xxs                          via litert    (aiedge venv, torch 2.6)
# Sizes are merged back into each results_<tag>.json (fixes the None/None).
#
# Usage:
#   ./convert_all.sh                      # all tags found
#   TAGS="baseline_s42 clahe_s42" ./convert_all.sh
#
# Safe to re-run: conversion overwrites its own outputs and skips missing weights.

set -uo pipefail
CKPT="${CKPT_DIR:-/scratch/ttran72/checkpoints}"
SCRIPTS="${SCRIPTS:-$HOME/research/une-remote}"
TRAIN_PY="${TRAIN_PY:-python}"                                  # onnx2tf path
AIEDGE_PY="${AIEDGE_PY:-/scratch/ttran72/venvs/aiedge/bin/python}"
NCLS="${NCLS:-2721}"
NOTIFY="${NOTIFY:-$SCRIPTS/notify.py}"

# tags: explicit list, else every results_<tag>.json in CKPT
if [ -n "${TAGS:-}" ]; then
  tags="$TAGS"
else
  tags=$(ls "$CKPT"/results_*_s*.json 2>/dev/null \
         | sed -E 's#.*/results_(.*)\.json#\1#' | sort -u)
fi
[ -z "$tags" ] && { echo "no results_<tag>.json found in $CKPT"; exit 1; }
echo "converting tags: $tags"

fail=0
for tag in $tags; do
  echo "===================================================================="
  echo "convert $tag"
  echo "===================================================================="
  # onnx2tf models (training venv) -- offline calibration fix is inside the script
  "$TRAIN_PY" "$SCRIPTS/export_tflite.py" --run-tag "$tag" --num-classes "$NCLS" \
      --models mobilenetv2_100 efficientformerv2_s0 || { echo "  onnx2tf FAILED for $tag"; fail=$((fail+1)); }
  # mobilevit (aiedge venv)
  if [ -x "$AIEDGE_PY" ]; then
    "$AIEDGE_PY" "$SCRIPTS/export_mobilevit_litert.py" --run-tag "$tag" --num-classes "$NCLS" \
        --models mobilevit_xxs || { echo "  litert FAILED for $tag"; fail=$((fail+1)); }
  else
    echo "  aiedge venv not found at $AIEDGE_PY -> mobilevit skipped for $tag"; fail=$((fail+1))
  fi
done

echo "===================================================================="
echo "conversion done ($fail failures). tflite files now present:"
ls "$CKPT"/*_s*_tf/*.tflite "$CKPT"/mobilevit_xxs_*_float*.tflite 2>/dev/null | wc -l
first=$(echo $tags | awk '{print $1}')
[ -f "$NOTIFY" ] && python "$NOTIFY" "CONVERSION DONE ($fail fail)" /dev/null "$CKPT/results_${first}.json" || true
