#!/usr/bin/env bash

# Usage:
#   bash reconstruction/run_pose_est_eval.sh pose sam3d
#   bash reconstruction/run_pose_est_eval.sh eval sam3d
#   bash reconstruction/run_pose_est_eval.sh both sam3d --overwrite
#   bash reconstruction/run_pose_est_eval.sh both hunyuan
#   bash reconstruction/run_pose_est_eval.sh pose all
#   bash reconstruction/run_pose_est_eval.sh eval all
#   bash reconstruction/run_pose_est_eval.sh both hunyuan --overwrite
#
# Outputs:
#   Pose estimates: sibling of WORK_ROOT, default <WORK_ROOT>/../reconstruction_pose_est/<method>
#   Eval JSONs:     sibling of WORK_ROOT, default <WORK_ROOT>/../reconstruction_pose_eval/<method>.json

set -euo pipefail

# =========================
# Parameter Settings
# =========================
# All parameters can be edited here or overridden from the command line as
# environment variables, e.g.:
#   FRAME_STRIDE=5 USE_NVDIFFRAST=0 bash reconstruction/run_pose_est_eval.sh both sam3d

# Root of the dataset. It should contain the selected split directory, e.g.
#   $DATA_ROOT/test/<object>/{rgb,depth,masks,cam_params,models,K.txt}
DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train}"

# Dataset split to evaluate, usually test/val/train.
SPLIT="${SPLIT:-test}"

# Root of reconstruction outputs. Each reconstruction pipeline is expected at:
#   $WORK_ROOT/<pipeline>/$SPLIT/<object>/pose_ready_models/view_0/<part>/model.obj
WORK_ROOT="${WORK_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_runs}"

# Pose-estimation output root. Each method writes to:
#   $POSE_ROOT/<pipeline>/
POSE_ROOT="${POSE_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_pose_runs}"

# Evaluation output root. Each method writes one JSON file:
#   $EVAL_ROOT/<pipeline>.json
EVAL_ROOT="${EVAL_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_eval}"

# BOP toolkit root. The eval script imports bop_toolkit_lib.pose_error from here
# and uses its ADD/ADI/Re/Te/MSSD/MSPD implementations.
BOP_TOOLKIT_ROOT="${BOP_TOOLKIT_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/related_works/bop_toolkit}"

# Optional comma-separated object names. Empty means all objects under
# $DATA_ROOT/$SPLIT.
OBJECTS="${OBJECTS:-}"

# Pose-estimation frame stride. 1 evaluates every frame with a part mask;
# larger values subsample frames for faster debugging.
FRAME_STRIDE="${FRAME_STRIDE:-1}"

# Maximum number of frames per part for pose estimation. 0 means no limit.
MAX_FRAMES_PER_PART="${MAX_FRAMES_PER_PART:-0}"

# Number of FoundationPose refinement iterations per frame.
REFINE_ITERATIONS="${REFINE_ITERATIONS:-8}"

# GPUs used by pose-est workers, e.g. "0,1,2,3". Empty means do not modify
# CUDA_VISIBLE_DEVICES and run in the current shell GPU environment.
POSE_GPU_IDS="${POSE_GPU_IDS:-0,1}"

# Number of pose-est worker processes per GPU. Used when POSE_NUM_WORKERS is 0.
# Example: POSE_GPU_IDS="0,1" POSE_WORKERS_PER_GPU=3 gives 6 workers.
POSE_WORKERS_PER_GPU="${POSE_WORKERS_PER_GPU:-5}"

# Total pose-est worker processes. 1 runs serially. 0 means infer from
# POSE_GPU_IDS * POSE_WORKERS_PER_GPU. If set to a positive value, it overrides
# POSE_WORKERS_PER_GPU.
POSE_NUM_WORKERS="${POSE_NUM_WORKERS:-0}"

# Whether to use nvdiffrast rendering/scoring inside FoundationPose.
# 1 gives the normal FoundationPose path; 0 uses the coarse fallback path and is
# mainly useful for dependency/debug checks.
USE_NVDIFFRAST="${USE_NVDIFFRAST:-1}"

# Whether to initialize FoundationPose from GT pose. 0 is the normal evaluation
# setting; 1 is useful for oracle/debug runs.
USE_GT_INIT="${USE_GT_INIT:-0}"

# Whether to save pose visualization images during inference. Each image is
# saved as $POSE_ROOT/<pipeline>/<object>/vis/<frame>.png and contains all
# successfully estimated parts in that frame.
SAVE_VIS="${SAVE_VIS:-1}"

# Maximum projected mesh vertices drawn per part in visualization. The 3D bbox
# visualization now follows FoundationPose's original xyz-axis + bbox style.
VIS_MAX_POINTS="${VIS_MAX_POINTS:-1500}"

# Suppress FoundationPose's verbose internal stdout/stderr/logging INFO output.
# Warnings/errors from this wrapper are still printed.
QUIET_FOUNDATIONPOSE="${QUIET_FOUNDATIONPOSE:-1}"

# Show tqdm progress bars for pose inference.
POSE_PROGRESS="${POSE_PROGRESS:-1}"

# Number of GT mesh surface samples used by the eval script for ADD/ADDS/MSSD.
MODEL_SAMPLES="${MODEL_SAMPLES:-10000}"

# Whether to overwrite old outputs. This can also be enabled by passing
# --overwrite as the third command-line argument. When enabled, pose outputs for
# the selected pipeline are removed before pose inference, and eval JSON files
# for the selected pipeline are removed before evaluation.
OVERWRITE="${OVERWRITE:-1}"

# The twelve supported reconstruction pipelines. Use the second positional
# argument to select one of these, or use "all".
ALL_METHODS=(
  sam3d
  sam3d_tsdf
  sam3d_tsdf_dmesh
  sam3d_partcut_tsdf_dmesh
  hunyuan3d
  hunyuan3d_tsdf
  hunyuan3d_tsdf_dmesh
  hunyuan3d_partcut_tsdf_dmesh
  instantmesh
  instantmesh_tsdf
  instantmesh_tsdf_dmesh
  instantmesh_partcut_tsdf_dmesh
)

# Convenience group for the three original Hunyuan3D pipelines.
HUNYUAN_METHODS=(
  hunyuan3d
  hunyuan3d_tsdf
  hunyuan3d_tsdf_dmesh
)

cmd="${1:-both}"
method_arg="${2:-all}"
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

for extra_arg in "$@"; do
  case "$extra_arg" in
    --overwrite) OVERWRITE=1 ;;
    --no-overwrite) OVERWRITE=0 ;;
    *)
      echo "Unknown extra argument: $extra_arg" >&2
      echo "Supported extra arguments: --overwrite, --no-overwrite" >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_ROOT="${WORK_ROOT%/}/pose_est_eval_log"
mkdir -p "$LOG_ROOT"
LOG_TS="$(date +'%Y%m%d_%H%M%S')"
LOG_SAFE="${cmd}_${method_arg}"
LOG_SAFE="${LOG_SAFE//[^A-Za-z0-9_.-]/_}"
LOG_FILE="$LOG_ROOT/${LOG_TS}_${LOG_SAFE}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging terminal output to: $LOG_FILE"
echo "Command: bash reconstruction/run_pose_est_eval.sh $cmd $method_arg"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPLIT=$SPLIT"
echo "WORK_ROOT=$WORK_ROOT"
echo "POSE_ROOT=${POSE_ROOT:-<work-root-sibling/reconstruction_pose_est>}"
echo "EVAL_ROOT=${EVAL_ROOT:-<work-root-sibling/reconstruction_pose_eval>}"
echo "BOP_TOOLKIT_ROOT=$BOP_TOOLKIT_ROOT"
echo "POSE_GPU_IDS=${POSE_GPU_IDS:-<current environment>}"
echo "POSE_WORKERS_PER_GPU=$POSE_WORKERS_PER_GPU"
echo "POSE_NUM_WORKERS=$POSE_NUM_WORKERS"
echo "SAVE_VIS=$SAVE_VIS"
echo "VIS_MAX_POINTS=$VIS_MAX_POINTS"
echo "QUIET_FOUNDATIONPOSE=$QUIET_FOUNDATIONPOSE"
echo "POSE_PROGRESS=$POSE_PROGRESS"
echo "OVERWRITE=$OVERWRITE"

bool_flag() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) echo "--$2" ;;
    *) echo "--no-$2" ;;
  esac
}

methods_to_run=()
if [[ "$method_arg" == "all" ]]; then
  methods_to_run=("${ALL_METHODS[@]}")
elif [[ "$method_arg" == "hunyuan" || "$method_arg" == "hunyuan3d_all" ]]; then
  methods_to_run=("${HUNYUAN_METHODS[@]}")
else
  methods_to_run=("$method_arg")
fi

run_pose_one() {
  local method="$1"
  local script="reconstruction/pose_est/pose_est_${method}.py"
  if [[ ! -f "$script" ]]; then
    echo "Unknown or unsupported pose-est method: $method" >&2
    exit 2
  fi
  if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    local pose_method_dir="${POSE_ROOT%/}/$method"
    if [[ -n "$POSE_ROOT" && -d "$pose_method_dir" ]]; then
      echo "[OVERWRITE] remove old pose output: $pose_method_dir"
      rm -rf "$pose_method_dir"
    fi
  fi
  local pose_args=(
    --data-root "$DATA_ROOT"
    --split "$SPLIT"
    --work-root "$WORK_ROOT"
    --frame-stride "$FRAME_STRIDE"
    --max-frames-per-part "$MAX_FRAMES_PER_PART"
    --refine-iterations "$REFINE_ITERATIONS"
    --gpu-ids "$POSE_GPU_IDS"
    --workers-per-gpu "$POSE_WORKERS_PER_GPU"
    --num-workers "$POSE_NUM_WORKERS"
    "$(bool_flag "$USE_NVDIFFRAST" "use-nvdiffrast")"
    "$(bool_flag "$USE_GT_INIT" "use-gt-init")"
    "$(bool_flag "$SAVE_VIS" "save-vis")"
    --vis-max-points "$VIS_MAX_POINTS"
    "$(bool_flag "$QUIET_FOUNDATIONPOSE" "quiet-foundationpose")"
    "$(bool_flag "$POSE_PROGRESS" "progress")"
  )
  if [[ -n "$POSE_ROOT" ]]; then
    pose_args+=(--pose-root "$POSE_ROOT")
  fi
  if [[ -n "$OBJECTS" ]]; then
    pose_args+=(--objects "$OBJECTS")
  fi
  echo "[POSE] $method"
  python "$script" "${pose_args[@]}"
}

run_eval() {
  local methods_csv="$1"
  if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && -n "$EVAL_ROOT" ]]; then
    local eval_method
    local old_ifs="$IFS"
    IFS=,
    for eval_method in $methods_csv; do
      local eval_json="${EVAL_ROOT%/}/${eval_method}.json"
      if [[ -f "$eval_json" ]]; then
        echo "[OVERWRITE] remove old eval output: $eval_json"
        rm -f "$eval_json"
      fi
    done
    IFS="$old_ifs"
  fi
  local eval_args=(
    --data-root "$DATA_ROOT"
    --split "$SPLIT"
    --work-root "$WORK_ROOT"
    --bop-toolkit-root "$BOP_TOOLKIT_ROOT"
    --methods "$methods_csv"
    --model-samples "$MODEL_SAMPLES"
  )
  if [[ -n "$POSE_ROOT" ]]; then
    eval_args+=(--pose-root "$POSE_ROOT")
  fi
  if [[ -n "$EVAL_ROOT" ]]; then
    eval_args+=(--eval-root "$EVAL_ROOT")
  fi
  if [[ -n "$OBJECTS" ]]; then
    eval_args+=(--objects "$OBJECTS")
  fi
  echo "[EVAL] $methods_csv"
  python reconstruction/eval/eval_pose_est.py "${eval_args[@]}"
}

case "$cmd" in
  pose)
    for m in "${methods_to_run[@]}"; do
      run_pose_one "$m"
    done
    ;;
  eval)
    methods_csv="$(IFS=,; echo "${methods_to_run[*]}")"
    run_eval "$methods_csv"
    ;;
  both)
    for m in "${methods_to_run[@]}"; do
      run_pose_one "$m"
    done
    methods_csv="$(IFS=,; echo "${methods_to_run[*]}")"
    run_eval "$methods_csv"
    ;;
  *)
    echo "Unknown command: $cmd. Use pose, eval, or both." >&2
    exit 2
    ;;
esac
