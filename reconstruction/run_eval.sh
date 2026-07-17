#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train}"
WORK_ROOT="${WORK_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_eval}"
SPLIT="${SPLIT:-test}"
OBJECTS="${OBJECTS:-}"
METHODS="${METHODS:-sam3d,sam3d_tsdf,sam3d_dmesh,sam3d_tsdf_dmesh,hunyuan3d,hunyuan3d_tsdf,hunyuan3d_dmesh,hunyuan3d_tsdf_dmesh}"
SAMPLES="${SAMPLES:-50000}"
FSCORE_THRESH="${FSCORE_THRESH:-0.01}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
POSE_CONVENTION="${POSE_CONVENTION:-sapien}"

cd "$REPO_ROOT"

cmd=(
  python reconstruction/eval/eval_all_methods.py
  --data-root "$DATA_ROOT"
  --split "$SPLIT"
  --work-root "$WORK_ROOT"
  --methods "$METHODS"
  --output-root "$OUTPUT_ROOT"
  --samples "$SAMPLES"
  --fscore-thresh "$FSCORE_THRESH"
  --max-eval-frames "$MAX_EVAL_FRAMES"
  --frame-stride "$FRAME_STRIDE"
  --pose-convention "$POSE_CONVENTION"
)

if [[ -n "$OBJECTS" ]]; then
  cmd+=(--objects "$OBJECTS")
fi

echo "[eval] data_root=$DATA_ROOT"
echo "[eval] work_root=$WORK_ROOT"
echo "[eval] output_root=$OUTPUT_ROOT"
echo "[eval] split=$SPLIT objects=${OBJECTS:-all}"
echo "[eval] methods=$METHODS"
echo "[eval] command: ${cmd[*]}"

"${cmd[@]}"
