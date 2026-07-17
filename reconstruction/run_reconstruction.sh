#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash reconstruction/run_reconstruction.sh
#   bash reconstruction/run_reconstruction.sh sequence
#
# The "sequence" entry runs, in order:
#   1. instantmesh_tsdf_dmesh
#   2. instantmesh_partcut_tsdf_dmesh
#   3. sam3d_partcut_tsdf_dmesh
#   4. hunyuan3d_partcut_tsdf_dmesh
#   5. instantmesh_dmesh
#   6. sam3d_dmesh
#   7. hunyuan3d_dmesh

DATA_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
WORK_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_runs"
SPLIT="${SPLIT:-test}"
OBJECT_SOURCE="${OBJECT_SOURCE:-all}"

# Conda environments used by each reconstruction family.
SAM3D_CONDA_ENV="${SAM3D_CONDA_ENV:-sam3d}"
HUNYUAN_CONDA_ENV="${HUNYUAN_CONDA_ENV:-hunyuan}"
INSTANTMESH_CONDA_ENV="${INSTANTMESH_CONDA_ENV:-instantmesh}"
CONDA_SH="${CONDA_SH:-}"

# Per-method multi-GPU / multi-process settings for the sequence entry.
#
# *_WORKERS is the old total worker count and is still used when
# *_WORKERS_PER_GPU is empty.
#
# *_WORKERS_PER_GPU optionally overrides *_WORKERS with either one count for
# every listed GPU or comma-separated counts aligned with *_GPUS.
# Examples:
#   INSTANTMESH_PARTCUT_GPUS=0,1 INSTANTMESH_PARTCUT_WORKERS_PER_GPU=5,8
#   SAM3D_DMESH_GPUS=0,1,2,3 SAM3D_DMESH_WORKERS_PER_GPU=4
RECON_GPU_IDS="${RECON_GPU_IDS:-0,1}"

INSTANTMESH_TSDF_DMESH_GPUS="${INSTANTMESH_TSDF_DMESH_GPUS:-$RECON_GPU_IDS}"
INSTANTMESH_TSDF_DMESH_WORKERS="${INSTANTMESH_TSDF_DMESH_WORKERS:-40}"
INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU="${INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU:-20,20}"

INSTANTMESH_PARTCUT_GPUS="${INSTANTMESH_PARTCUT_GPUS:-$RECON_GPU_IDS}"
INSTANTMESH_PARTCUT_WORKERS="${INSTANTMESH_PARTCUT_WORKERS:-40}"
INSTANTMESH_PARTCUT_WORKERS_PER_GPU="${INSTANTMESH_PARTCUT_WORKERS_PER_GPU:-20,20}"
SAM3D_PARTCUT_GPUS="${SAM3D_PARTCUT_GPUS:-$RECON_GPU_IDS}"
SAM3D_PARTCUT_WORKERS="${SAM3D_PARTCUT_WORKERS:-40}"
SAM3D_PARTCUT_WORKERS_PER_GPU="${SAM3D_PARTCUT_WORKERS_PER_GPU:-20,20}"
HUNYUAN_PARTCUT_GPUS="${HUNYUAN_PARTCUT_GPUS:-$RECON_GPU_IDS}"
HUNYUAN_PARTCUT_WORKERS="${HUNYUAN_PARTCUT_WORKERS:-40}"
HUNYUAN_PARTCUT_WORKERS_PER_GPU="${HUNYUAN_PARTCUT_WORKERS_PER_GPU:-20,20}"

INSTANTMESH_DMESH_GPUS="${INSTANTMESH_DMESH_GPUS:-$RECON_GPU_IDS}"
INSTANTMESH_DMESH_WORKERS="${INSTANTMESH_DMESH_WORKERS:-40}"
INSTANTMESH_DMESH_WORKERS_PER_GPU="${INSTANTMESH_DMESH_WORKERS_PER_GPU:-20,20}"
SAM3D_DMESH_GPUS="${SAM3D_DMESH_GPUS:-$RECON_GPU_IDS}"
SAM3D_DMESH_WORKERS="${SAM3D_DMESH_WORKERS:-40}"
SAM3D_DMESH_WORKERS_PER_GPU="${SAM3D_DMESH_WORKERS_PER_GPU:-20,20}"
HUNYUAN_DMESH_GPUS="${HUNYUAN_DMESH_GPUS:-$RECON_GPU_IDS}"
HUNYUAN_DMESH_WORKERS="${HUNYUAN_DMESH_WORKERS:-40}"
HUNYUAN_DMESH_WORKERS_PER_GPU="${HUNYUAN_DMESH_WORKERS_PER_GPU:-20,20}"

cmd="${1:-instantmesh_tsdf_dmesh}"
RUN_MODEL_NAME="${RUN_MODEL_NAME:-$cmd}"
LOG_DIR="$WORK_ROOT/log"
mkdir -p "$LOG_DIR"
LOG_TS="$(date +'%Y%m%d_%H%M%S')"
LOG_SAFE_MODEL="${RUN_MODEL_NAME//[^A-Za-z0-9_.-]/_}"
LOG_FILE="$LOG_DIR/${LOG_TS}_${LOG_SAFE_MODEL}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging terminal output to: $LOG_FILE"
echo "Timestamp: $LOG_TS"
echo "Model: $RUN_MODEL_NAME"
echo "Command: bash reconstruction/run_reconstruction.sh $cmd"
echo "Data root: $DATA_ROOT"
echo "Work root: $WORK_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# SAM3D offline loading paths. These mirror
# ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d.
SAM3D_PROJECT_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects"
SAM3D_CONFIG_PATH="$SAM3D_PROJECT_ROOT/checkpoints/hf/pipeline.yaml"
SAM3D_DINO_REPO_OR_DIR="$SAM3D_PROJECT_ROOT/SAM3D_DINO/dinov2"
# DINOv2 hub callable name. The corresponding checkpoint file is named
# dinov2_vitl14_reg4_pretrain.pth under $TORCH_HOME/hub/checkpoints.
SAM3D_DINO_MODEL="dinov2_vitl14_reg"
SAM3D_DINO_SOURCE="local"
TORCH_HOME_DIR="$SAM3D_PROJECT_ROOT/SAM3D_DINO/torch_cache"

# Exported for sam-3d-objects/notebook/inference.py and the DINO wrapper used by
# SAM3D. Without these, SAM3D may fall back to facebookresearch/dinov2 through
# torch.hub and try to access the network.
export SAM3D_DINO_REPO_OR_DIR
export SAM3D_DINO_MODEL
export SAM3D_DINO_SOURCE
export SAM3D_PROJECT_ROOT
export SAM3D_CONFIG_PATH
export TORCH_HOME="$TORCH_HOME_DIR"

INSTANTMESH_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/baseline/Any6D/instantmesh"
INSTANTMESH_CONFIG_PATH="$INSTANTMESH_ROOT/configs/instant-mesh-large.yaml"
ANY6D_ZERO123PLUS_MODEL="$INSTANTMESH_ROOT/zero123_ckpts"
ANY6D_DINO_MODEL="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/baseline/Any6D/dino_vitb16"
ANY6D_INSTANTMESH_UNET="$INSTANTMESH_ROOT/ckpts/diffusion_pytorch_model.bin"
ANY6D_INSTANTMESH_MODEL="$INSTANTMESH_ROOT/ckpts/instant_mesh_large.ckpt"
INSTANTMESH_ARGS=(
  --instantmesh-root "$INSTANTMESH_ROOT"
  --instantmesh-config-path "$INSTANTMESH_CONFIG_PATH"
  --instantmesh-diffusion-model "$ANY6D_ZERO123PLUS_MODEL"
  --instantmesh-dino-model "$ANY6D_DINO_MODEL"
  --instantmesh-unet-path "$ANY6D_INSTANTMESH_UNET"
  --instantmesh-model-path "$ANY6D_INSTANTMESH_MODEL"
)

init_conda() {
  if [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return
  fi
  if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
    return
  fi
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    return
  fi
  echo "Could not initialize conda. Set CONDA_SH=/path/to/conda.sh." >&2
  exit 1
}

activate_recon_env() {
  local env_name="$1"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]]; then
    echo
    echo "========== conda activate $env_name =========="
    conda activate "$env_name"
  fi
}

common_recon_args() {
  printf '%s\n' \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --work-root "$WORK_ROOT" \
    --object-source "$OBJECT_SOURCE" \
    --reset-coord
}

run_step() {
  local env_name="$1"
  local label="$2"
  shift 2
  activate_recon_env "$env_name"
  echo
  echo "========== START $label =========="
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
  "$@"
  echo "========== DONE $label =========="
}

run_sequence() {
  init_conda

  local common_args=()
  while IFS= read -r line; do
    common_args+=("$line")
  done < <(common_recon_args)

  run_step "$INSTANTMESH_CONDA_ENV" "instantmesh_tsdf_dmesh" \
    python run/recon_instantmesh_tsdf_dmesh.py \
    "${common_args[@]}" \
    --gpus "$INSTANTMESH_TSDF_DMESH_GPUS" \
    --num-workers "$INSTANTMESH_TSDF_DMESH_WORKERS" \
    --workers-per-gpu "$INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU" \
    --build-base-if-missing \
    "${INSTANTMESH_ARGS[@]}"

  run_step "$INSTANTMESH_CONDA_ENV" "instantmesh_partcut_tsdf_dmesh" \
    python run/recon_instantmesh_partcut_tsdf_dmesh.py \
    "${common_args[@]}" \
    --gpus "$INSTANTMESH_PARTCUT_GPUS" \
    --num-workers "$INSTANTMESH_PARTCUT_WORKERS" \
    --workers-per-gpu "$INSTANTMESH_PARTCUT_WORKERS_PER_GPU" \
    "${INSTANTMESH_ARGS[@]}"

  run_step "$SAM3D_CONDA_ENV" "sam3d_partcut_tsdf_dmesh" \
    python run/recon_sam3d_partcut_tsdf_dmesh.py \
    "${common_args[@]}" \
    --gpus "$SAM3D_PARTCUT_GPUS" \
    --num-workers "$SAM3D_PARTCUT_WORKERS" \
    --workers-per-gpu "$SAM3D_PARTCUT_WORKERS_PER_GPU"

  run_step "$HUNYUAN_CONDA_ENV" "hunyuan3d_partcut_tsdf_dmesh" \
    python run/recon_hunyuan3d_partcut_tsdf_dmesh.py \
    "${common_args[@]}" \
    --gpus "$HUNYUAN_PARTCUT_GPUS" \
    --num-workers "$HUNYUAN_PARTCUT_WORKERS" \
    --workers-per-gpu "$HUNYUAN_PARTCUT_WORKERS_PER_GPU"

  run_step "$INSTANTMESH_CONDA_ENV" "instantmesh_dmesh" \
    python run/recon_instantmesh_dmesh.py \
    "${common_args[@]}" \
    --gpus "$INSTANTMESH_DMESH_GPUS" \
    --num-workers "$INSTANTMESH_DMESH_WORKERS" \
    --workers-per-gpu "$INSTANTMESH_DMESH_WORKERS_PER_GPU" \
    --build-base-if-missing \
    "${INSTANTMESH_ARGS[@]}"

  run_step "$SAM3D_CONDA_ENV" "sam3d_dmesh" \
    python run/recon_sam3d_dmesh.py \
    "${common_args[@]}" \
    --gpus "$SAM3D_DMESH_GPUS" \
    --num-workers "$SAM3D_DMESH_WORKERS" \
    --workers-per-gpu "$SAM3D_DMESH_WORKERS_PER_GPU" \
    --build-base-if-missing

  run_step "$HUNYUAN_CONDA_ENV" "hunyuan3d_dmesh" \
    python run/recon_hunyuan3d_dmesh.py \
    "${common_args[@]}" \
    --gpus "$HUNYUAN_DMESH_GPUS" \
    --num-workers "$HUNYUAN_DMESH_WORKERS" \
    --workers-per-gpu "$HUNYUAN_DMESH_WORKERS_PER_GPU" \
    --build-base-if-missing
}

case "$cmd" in
  sequence|special|all-sequence)
    run_sequence
    exit 0
    ;;
esac

# python reconstruction/run/recon_sam3d.py --data-root dataset_train --split test --work-root reconstruction_runs --objects bottle_3517 --num-workers 1
# python reconstruction/run/recon_sam3d_tsdf.py --data-root dataset_train --split test --work-root reconstruction_runs --objects bottle_3517 --num-workers 4
python run/recon_sam3d_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 1 --num-workers 10 --build-base-if-missing --reset-coord
# python run/recon_sam3d_tsdf_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 0,1 --num-workers 16 --build-base-if-missing --reset-coord
# New reference-only part-cut TSDF+DLMesh pipeline. It reads existing
# <work-root>/sam3d and writes separate sam3d_partcut* result folders.
# python run/recon_sam3d_partcut_tsdf_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 0,1 --num-workers 12 --build-base-if-missing --reset-coord

# python reconstruction/run/recon_hunyuan3d.py --data-root dataset_train --split test --work-root reconstruction_runs --objects bottle_3517 --num-workers 1
# python reconstruction/run/recon_hunyuan3d_tsdf.py --data-root dataset_train --split test --work-root reconstruction_runs --objects bottle_3517 --num-workers 4
# python run/recon_hunyuan3d_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 0,1 --num-workers 20 --build-base-if-missing --reset-coord
# python run/recon_hunyuan3d_tsdf_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 0,1 --num-workers 30 --build-base-if-missing --reset-coord
# python run/recon_hunyuan3d_partcut_tsdf_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 0,1 --num-workers 12 --build-base-if-missing --reset-coord --no-fail-on-empty-tsdf

# python reconstruction/run/recon_instantmesh.py --data-root dataset_train --split test --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1 "${INSTANTMESH_ARGS[@]}"
# python reconstruction/run/recon_instantmesh_tsdf.py --data-root dataset_train --split test --work-root reconstruction_runs --objects bottle_3517 --num-workers 4 --build-base-if-missing "${INSTANTMESH_ARGS[@]}"
# python run/recon_instantmesh_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 1 --num-workers 10 --build-base-if-missing --reset-coord "${INSTANTMESH_ARGS[@]}"
# python run/recon_instantmesh_tsdf_dmesh.py --data-root "$DATA_ROOT" --split "$SPLIT" --work-root "$WORK_ROOT" --object-source "$OBJECT_SOURCE" --gpus 0,1 --num-workers 6 --build-base-if-missing --reset-coord "${INSTANTMESH_ARGS[@]}" --overwrite
# python reconstruction/run/recon_instantmesh_partcut_tsdf_dmesh.py --data-root $DATA_ROOT --split test --work-root $WORK_ROOT --object-source all --gpus 0 --num-workers 1 --reset-coord "${INSTANTMESH_ARGS[@]}"

# python reconstruction/reconstruct_instantmesh.py --help
# python reconstruction/reconstruct.py --help
# python reconstruction/point_reconstruct.py --help
