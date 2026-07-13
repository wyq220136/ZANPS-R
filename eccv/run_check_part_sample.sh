#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

ROOT="${ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train/test}"
SAMPLE_NUM="${SAMPLE_NUM:-20}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
OBJECTS="${OBJECTS:-}"
START="${START:-0}"
END="${END:-}"

PRED_MASK_SUBDIR="${PRED_MASK_SUBDIR:-pred_mask}"
OUT_SUBDIR="${OUT_SUBDIR:-chosen_part_check_sample}"
SELECTED_JSON_NAME="${SELECTED_JSON_NAME:-selected_parts.json}"
CHECK_PART_SAMPLE_DIR="${CHECK_PART_SAMPLE_DIR:-$SCRIPT_DIR/check_part_samples}"
SAMPLE_JSON="${SAMPLE_JSON:-$CHECK_PART_SAMPLE_DIR/sampled_objects.json}"
SUMMARY_JSON="${SUMMARY_JSON:-$CHECK_PART_SAMPLE_DIR/check_part_summary.json}"

REFERENCE_POLICY="${REFERENCE_POLICY:-first}"
MIN_VISIBLE_PIXELS="${MIN_VISIBLE_PIXELS:-64}"
MAX_CANDIDATES_PER_REF="${MAX_CANDIDATES_PER_REF:-30}"
MAX_SELECTED_PER_REF="${MAX_SELECTED_PER_REF:-0}"
VLM_WORKERS="${VLM_WORKERS:-1}"
UNIQUE_REFERENCE_PARTS="${UNIQUE_REFERENCE_PARTS:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="$SCRIPT_DIR/log"
mkdir -p "$LOG_DIR"
LOG_TS="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/${LOG_TS}_check_part_sample.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging terminal output to: $LOG_FILE"
echo "ROOT=$ROOT"
echo "SAMPLE_NUM=$SAMPLE_NUM SAMPLE_SEED=$SAMPLE_SEED"
echo "PRED_MASK_SUBDIR=$PRED_MASK_SUBDIR OUT_SUBDIR=$OUT_SUBDIR"
echo "SAMPLE_JSON=$SAMPLE_JSON"
echo "SUMMARY_JSON=$SUMMARY_JSON"

cmd=(
  "$PYTHON_BIN" eccv/segmentation/check_part.py
  --root "$ROOT"
  --sample-num "$SAMPLE_NUM"
  --seed "$SAMPLE_SEED"
  --start "$START"
  --pred-mask-subdir "$PRED_MASK_SUBDIR"
  --out-subdir "$OUT_SUBDIR"
  --selected-json-name "$SELECTED_JSON_NAME"
  --sample-json "$SAMPLE_JSON"
  --summary-json "$SUMMARY_JSON"
  --reference-policy "$REFERENCE_POLICY"
  --min-visible-pixels "$MIN_VISIBLE_PIXELS"
  --max-candidates-per-ref "$MAX_CANDIDATES_PER_REF"
  --max-selected-per-ref "$MAX_SELECTED_PER_REF"
  --vlm-workers "$VLM_WORKERS"
)

if [[ -n "$END" ]]; then
  cmd+=(--end "$END")
fi
if [[ -n "$OBJECTS" ]]; then
  cmd+=(--objects "$OBJECTS")
fi
if [[ "$UNIQUE_REFERENCE_PARTS" == "1" ]]; then
  cmd+=(--unique-reference-parts)
else
  cmd+=(--no-unique-reference-parts)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"
