#!/usr/bin/env bash

# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d-tsdf
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan-tsdf
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh-tsdf
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d-tsdf-dmesh
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan-tsdf-dmesh
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh-tsdf-dmesh
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-all
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan-instantmesh
# bash ref_pose/learning/training/run_train_ddp_workflow.sh dataset-index
# bash ref_pose/learning/training/run_train_ddp_workflow.sh train

set -euo pipefail

# =========================
# Parameter Settings
# =========================
# Edit this block before running the commands above.

# Paths.
CONFIG="ref_pose/learning/training/configs/train_ddp.yaml"
TRAIN_SCRIPT="ref_pose/learning/training/train_ddp.py"
RECON_PREBUILD_SCRIPT="ref_pose/learning/training/prebuild_recon_cache.py"
DATASET_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
RECON_CACHE_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_recon_cache"
SAM3D_CONFIG_PATH="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects/checkpoints/hf/pipeline.yaml"
TRAIN_SUBDIR="train"
VAL_SUBDIR="val"

# Distributed launch for train-only mode.
TRAIN_NPROC_PER_NODE=1
MASTER_PORT=29500

# Reconstruction cache prebuild.
# Backward-compatible defaults used when per-model settings are left empty.
RECON_GPU_IDS="0,1"
# Either a single integer used by every selected GPU, or a comma-separated list
# aligned with *_GPU_IDS, e.g. "1,2" means gpu0 gets 1 worker and gpu1 gets 2.
RECON_WORKERS_PER_GPU=3

# SAM3D reconstruction resources.
# Base reconstruction and TSDF/TSDF+DLMesh use separate resource groups so they
# can be run independently and in parallel.
# TSDF group is shared by sam3d_tsdf, sam3d_tsdf_dmesh, and
# sam3d_partcut_tsdf_dmesh.
SAM3D_BASE_GPU_IDS="${SAM3D_BASE_GPU_IDS:-$RECON_GPU_IDS}"
SAM3D_BASE_WORKERS_PER_GPU="${SAM3D_BASE_WORKERS_PER_GPU:-2}"
SAM3D_TSDF_DMESH_GPU_IDS="${SAM3D_TSDF_DMESH_GPU_IDS:-$RECON_GPU_IDS}"
SAM3D_TSDF_DMESH_WORKERS_PER_GPU="${SAM3D_TSDF_DMESH_WORKERS_PER_GPU:-4}"

# Hunyuan3D reconstruction resources.
# TSDF group is shared by hunyuan3d_tsdf, hunyuan3d_tsdf_dmesh, and
# hunyuan3d_partcut_tsdf_dmesh.
HUNYUAN_BASE_GPU_IDS="${HUNYUAN_BASE_GPU_IDS:-$RECON_GPU_IDS}"
HUNYUAN_BASE_WORKERS_PER_GPU="${HUNYUAN_BASE_WORKERS_PER_GPU:-12}"
HUNYUAN_TSDF_DMESH_GPU_IDS="${HUNYUAN_TSDF_DMESH_GPU_IDS:-$RECON_GPU_IDS}"
HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU="${HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU:-12}"

# InstantMesh reconstruction resources.
# TSDF group is shared by instantmesh_tsdf, instantmesh_tsdf_dmesh, and
# instantmesh_partcut_tsdf_dmesh.
INSTANTMESH_BASE_GPU_IDS="${INSTANTMESH_BASE_GPU_IDS:-$RECON_GPU_IDS}"
INSTANTMESH_BASE_WORKERS_PER_GPU="${INSTANTMESH_BASE_WORKERS_PER_GPU:-5}"
INSTANTMESH_TSDF_DMESH_GPU_IDS="${INSTANTMESH_TSDF_DMESH_GPU_IDS:-$RECON_GPU_IDS}"
INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU="${INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU:-5}"

# Legacy compatibility knobs. If the new per-stage settings above are left
# empty, these old values still provide a usable fallback.
SAM3D_RECON_GPU_IDS="${SAM3D_RECON_GPU_IDS:-$SAM3D_BASE_GPU_IDS}"
SAM3D_RECON_NUM_WORKERS="${SAM3D_RECON_NUM_WORKERS:-4}"
SAM3D_RECON_WORKERS_PER_GPU="${SAM3D_RECON_WORKERS_PER_GPU:-$SAM3D_BASE_WORKERS_PER_GPU}"
HUNYUAN_RECON_GPU_IDS="${HUNYUAN_RECON_GPU_IDS:-$HUNYUAN_BASE_GPU_IDS}"
HUNYUAN_RECON_NUM_WORKERS="${HUNYUAN_RECON_NUM_WORKERS:-24}"
HUNYUAN_RECON_WORKERS_PER_GPU="${HUNYUAN_RECON_WORKERS_PER_GPU:-$HUNYUAN_BASE_WORKERS_PER_GPU}"
INSTANTMESH_RECON_GPU_IDS="${INSTANTMESH_RECON_GPU_IDS:-$INSTANTMESH_BASE_GPU_IDS}"
INSTANTMESH_RECON_NUM_WORKERS="${INSTANTMESH_RECON_NUM_WORKERS:-10}"
INSTANTMESH_RECON_WORKERS_PER_GPU="${INSTANTMESH_RECON_WORKERS_PER_GPU:-$INSTANTMESH_BASE_WORKERS_PER_GPU}"

FORCE_REBUILD_RECON=0
FORCE_RESAMPLE_RECON=0
DATASET_INDEX_SPLIT="both"
FORCE_REBUILD_DATASET_INDEX=0

# Conda environments for sequential reconstruction.
SAM3D_CONDA_ENV="sam3d"
HUNYUAN_CONDA_ENV="hunyuan"
INSTANTMESH_CONDA_ENV="instantmesh"
CONDA_SH=""

# SAM3D DINO backbone. Leave empty to use SAM3D defaults
# (facebookresearch/dinov2 via torch.hub). For offline use, set
# SAM3D_DINO_REPO_OR_DIR to a local dinov2 repo/cache directory that contains
# hubconf.py, set SAM3D_DINO_SOURCE="local", and make sure TORCH_HOME points to
# the cache containing checkpoints/dinov2_vitb14_pretrain.pth if needed.
SAM3D_DINO_REPO_OR_DIR="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects/SAM3D_DINO/dinov2"
SAM3D_DINO_MODEL="dinov2_vitl14_reg"
SAM3D_DINO_SOURCE="local"
TORCH_HOME_DIR="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects/SAM3D_DINO/torch_cache"

# InstantMesh reconstruction paths.
INSTANTMESH_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/baseline/Any6D/instantmesh"
INSTANTMESH_CONFIG_PATH="$INSTANTMESH_ROOT/configs/instant-mesh-large.yaml"
ANY6D_ZERO123PLUS_MODEL="$INSTANTMESH_ROOT/zero123_ckpts"
ANY6D_DINO_MODEL="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/baseline/Any6D/dino_vitb16"
ANY6D_INSTANTMESH_UNET="$INSTANTMESH_ROOT/ckpts/diffusion_pytorch_model.bin"
ANY6D_INSTANTMESH_MODEL="$INSTANTMESH_ROOT/ckpts/instant_mesh_large.ckpt"
INSTANTMESH_DIFFUSION_MODEL="$ANY6D_ZERO123PLUS_MODEL"
INSTANTMESH_DINO_MODEL="$ANY6D_DINO_MODEL"
INSTANTMESH_UNET_PATH="$ANY6D_INSTANTMESH_UNET"
INSTANTMESH_MODEL_PATH="$ANY6D_INSTANTMESH_MODEL"

# Train-only run.
TRAIN_RECON_MODEL="${TRAIN_RECON_MODEL:-all}"
VM_WEIGHT=0.75
OUT_DIR="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/ref_pose/learning/weights/refine_validmask_ddp_custom"
FINAL_CKPT_NAME="my_best_vmw075.pth"
EPOCHS=20
LR=0.0001
WEIGHT_DECAY=0.0001

# Dataloader. Leave per-rank strings empty to use the same value on every rank.
BATCH_SIZE=1
BATCH_SIZE_PER_RANK=""
NUM_WORKERS=0
NUM_WORKERS_PER_RANK=""

# =========================
# Runtime
# =========================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

cmd="${1:-}"
LOG_DIR="$ROOT_DIR/ref_pose/learning/training/log"
mkdir -p "$LOG_DIR"
log_cmd="${cmd:-usage}"
log_cmd="${log_cmd//[^A-Za-z0-9_.-]/_}"
LOG_FILE="$LOG_DIR/$(date +'%Y%m%d_%H%M%S')_${log_cmd}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging terminal output to: $LOG_FILE"
echo "RECON_GPU_IDS=$RECON_GPU_IDS RECON_WORKERS_PER_GPU=$RECON_WORKERS_PER_GPU"
echo "SAM3D_BASE GPU_IDS=$SAM3D_BASE_GPU_IDS WORKERS_PER_GPU=$SAM3D_BASE_WORKERS_PER_GPU"
echo "SAM3D_TSDF_DMESH GPU_IDS=$SAM3D_TSDF_DMESH_GPU_IDS WORKERS_PER_GPU=$SAM3D_TSDF_DMESH_WORKERS_PER_GPU"
echo "HUNYUAN_BASE GPU_IDS=$HUNYUAN_BASE_GPU_IDS WORKERS_PER_GPU=$HUNYUAN_BASE_WORKERS_PER_GPU"
echo "HUNYUAN_TSDF_DMESH GPU_IDS=$HUNYUAN_TSDF_DMESH_GPU_IDS WORKERS_PER_GPU=$HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU"
echo "INSTANTMESH_BASE GPU_IDS=$INSTANTMESH_BASE_GPU_IDS WORKERS_PER_GPU=$INSTANTMESH_BASE_WORKERS_PER_GPU"
echo "INSTANTMESH_TSDF_DMESH GPU_IDS=$INSTANTMESH_TSDF_DMESH_GPU_IDS WORKERS_PER_GPU=$INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU"
echo "TRAIN_RECON_MODEL=$TRAIN_RECON_MODEL"

if [[ -n "$SAM3D_DINO_REPO_OR_DIR" ]]; then
  export SAM3D_DINO_REPO_OR_DIR
fi
if [[ -n "$SAM3D_DINO_MODEL" ]]; then
  export SAM3D_DINO_MODEL
fi
if [[ -n "$SAM3D_DINO_SOURCE" ]]; then
  export SAM3D_DINO_SOURCE
fi
if [[ -n "$TORCH_HOME_DIR" ]]; then
  export TORCH_HOME="$TORCH_HOME_DIR"
fi

common_args=(
  --train-config "$CONFIG"
  --dataset-root "$DATASET_ROOT"
  --train-subdir "$TRAIN_SUBDIR"
  --val-subdir "$VAL_SUBDIR"
  --recon-cache-root "$RECON_CACHE_ROOT"
)

if [[ -n "$SAM3D_CONFIG_PATH" ]]; then
  common_args+=(--sam3d-config-path "$SAM3D_CONFIG_PATH")
fi
common_args+=(
  --instantmesh-root "$INSTANTMESH_ROOT"
  --instantmesh-config-path "$INSTANTMESH_CONFIG_PATH"
  --instantmesh-diffusion-model "$INSTANTMESH_DIFFUSION_MODEL"
  --instantmesh-dino-model "$INSTANTMESH_DINO_MODEL"
  --instantmesh-unet-path "$INSTANTMESH_UNET_PATH"
  --instantmesh-model-path "$INSTANTMESH_MODEL_PATH"
)

bool_flag() {
  local value="$1"
  local enabled="$2"
  local disabled="$3"
  case "$value" in
    1|true|TRUE|yes|YES|on|ON) printf '%s\n' "$enabled" ;;
    *) printf '%s\n' "$disabled" ;;
  esac
}

count_csv_items() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    printf '0\n'
    return
  fi
  python -c "print(len([x.strip() for x in '''$raw'''.split(',') if x.strip()]))"
}

torchrun_args_for_nproc() {
  local nproc="$1"
  printf '%s\n' "--nproc_per_node=$nproc" "--master_port=$MASTER_PORT"
}

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  python -c "import math; print(int(math.ceil(float($numerator) / max(1.0, float($denominator)))))"
}

expand_workers_per_gpu() {
  local raw="$1"
  local nproc="$2"
  python -c "
import sys
raw = sys.argv[1].strip()
n = max(1, int(sys.argv[2]))
if not raw:
    print('')
    raise SystemExit
vals = [x.strip() for x in raw.split(',') if x.strip()]
if len(vals) == 1:
    vals = vals * n
elif len(vals) != n:
    raise SystemExit(f'workers-per-gpu list length mismatch: expected {n}, got {len(vals)} in {raw!r}')
ints = [max(1, int(v)) for v in vals]
print(','.join(str(v) for v in ints))
" "$raw" "$nproc"
}

sum_csv_ints() {
  local raw="$1"
  python -c "print(sum(int(x.strip()) for x in '''$raw'''.split(',') if x.strip()))"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Required $label not found: $path" >&2
    exit 1
  fi
}

check_instantmesh_paths() {
  require_path "$INSTANTMESH_ROOT" "InstantMesh root"
  require_path "$INSTANTMESH_CONFIG_PATH" "InstantMesh config"
  require_path "$INSTANTMESH_ROOT/zero123plus" "InstantMesh Zero123++ custom pipeline"
  require_path "$INSTANTMESH_DIFFUSION_MODEL" "InstantMesh Zero123++ model"
  require_path "$INSTANTMESH_DINO_MODEL" "InstantMesh DINO model"
  require_path "$INSTANTMESH_UNET_PATH" "InstantMesh UNet checkpoint"
  require_path "$INSTANTMESH_MODEL_PATH" "InstantMesh reconstruction checkpoint"
}

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

run_prebuild() {
  local recon_model="$1"
  local recon_gpu_ids="${2:-$RECON_GPU_IDS}"
  local recon_num_workers="${3:-}"
  local recon_workers_per_gpu="${4:-}"
  local force_rebuild_flag
  local force_resample_flag
  local recon_worker_args
  local recon_nproc
  local workers_per_gpu
  local torchrun_args
  if [[ "$recon_model" == "instantmesh" || "$recon_model" == "all" ]]; then
    check_instantmesh_paths
  fi
  recon_nproc="$(count_csv_items "$recon_gpu_ids")"
  if [[ "$recon_nproc" -le 0 ]]; then
    recon_nproc=1
  fi
  workers_per_gpu=""
  if [[ -n "$recon_workers_per_gpu" ]]; then
    workers_per_gpu="$(expand_workers_per_gpu "$recon_workers_per_gpu" "$recon_nproc")"
  elif [[ -z "$recon_num_workers" && -n "$RECON_WORKERS_PER_GPU" ]]; then
    workers_per_gpu="$(expand_workers_per_gpu "$RECON_WORKERS_PER_GPU" "$recon_nproc")"
  fi
  if [[ -n "$workers_per_gpu" ]]; then
    recon_num_workers="$(sum_csv_ints "$workers_per_gpu")"
    recon_worker_args=(--recon-num-workers-per-rank "$workers_per_gpu")
  else
    if [[ -z "$recon_num_workers" ]]; then
      recon_num_workers=$(( recon_nproc * RECON_WORKERS_PER_GPU ))
    fi
    local workers_per_rank
    workers_per_rank="$(ceil_div "$recon_num_workers" "$recon_nproc")"
    recon_worker_args=(--recon-num-workers "$workers_per_rank")
    workers_per_gpu="$workers_per_rank"
  fi
  force_rebuild_flag="$(bool_flag "$FORCE_REBUILD_RECON" "--force-rebuild-recon" "--no-force-rebuild-recon")"
  force_resample_flag="$(bool_flag "$FORCE_RESAMPLE_RECON" "--force-resample-recon" "--no-force-resample-recon")"
  torchrun_args=()
  while IFS= read -r line; do
    torchrun_args+=("$line")
  done < <(torchrun_args_for_nproc "$recon_nproc")
  echo "[recon] model=$recon_model gpu_ids=${recon_gpu_ids:-<current CUDA_VISIBLE_DEVICES>} nproc_per_node=$recon_nproc total_workers=$recon_num_workers workers_per_gpu=$workers_per_gpu"
  if [[ -n "$recon_gpu_ids" ]]; then
    export CUDA_VISIBLE_DEVICES="$recon_gpu_ids"
  fi

  torchrun "${torchrun_args[@]}" "$RECON_PREBUILD_SCRIPT" \
    "${common_args[@]}" \
    --prebuild-recon \
    --prebuild-only \
    --prebuild-split both \
    --recon-model "$recon_model" \
    "${recon_worker_args[@]}" \
    "$force_rebuild_flag" \
    "$force_resample_flag"
}

run_recon_all() {
  init_conda

  activate_recon_env "$SAM3D_CONDA_ENV"
  run_prebuild sam3d "$SAM3D_BASE_GPU_IDS" "$SAM3D_RECON_NUM_WORKERS" "$SAM3D_BASE_WORKERS_PER_GPU"

  activate_recon_env "$HUNYUAN_CONDA_ENV"
  run_prebuild hunyuan3d "$HUNYUAN_BASE_GPU_IDS" "$HUNYUAN_RECON_NUM_WORKERS" "$HUNYUAN_BASE_WORKERS_PER_GPU"

  activate_recon_env "$INSTANTMESH_CONDA_ENV"
  run_prebuild instantmesh "$INSTANTMESH_BASE_GPU_IDS" "$INSTANTMESH_RECON_NUM_WORKERS" "$INSTANTMESH_BASE_WORKERS_PER_GPU"
}

run_recon_hunyuan_instantmesh() {
  init_conda

  activate_recon_env "$HUNYUAN_CONDA_ENV"
  run_prebuild hunyuan3d "$HUNYUAN_BASE_GPU_IDS" "$HUNYUAN_RECON_NUM_WORKERS" "$HUNYUAN_BASE_WORKERS_PER_GPU"

  activate_recon_env "$INSTANTMESH_CONDA_ENV"
  run_prebuild instantmesh "$INSTANTMESH_BASE_GPU_IDS" "$INSTANTMESH_RECON_NUM_WORKERS" "$INSTANTMESH_BASE_WORKERS_PER_GPU"
}

run_dataset_index() {
  local force_index_flag
  force_index_flag="$(bool_flag "$FORCE_REBUILD_DATASET_INDEX" "--force-rebuild-dataset-index" "--no-force-rebuild-dataset-index")"

  CUDA_VISIBLE_DEVICES="" python "$TRAIN_SCRIPT" \
    "${common_args[@]}" \
    --build-dataset-index-only \
    --dataset-index-split "$DATASET_INDEX_SPLIT" \
    --recon-model instantmesh \
    "$force_index_flag"
}

run_train_only() {
  local loader_args
  local torchrun_args
  loader_args=(--batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
  if [[ -n "$BATCH_SIZE_PER_RANK" ]]; then
    loader_args+=(--batch-size-per-rank "$BATCH_SIZE_PER_RANK")
  fi
  if [[ -n "$NUM_WORKERS_PER_RANK" ]]; then
    loader_args+=(--num-workers-per-rank "$NUM_WORKERS_PER_RANK")
  fi
  torchrun_args=()
  while IFS= read -r line; do
    torchrun_args+=("$line")
  done < <(torchrun_args_for_nproc "$TRAIN_NPROC_PER_NODE")

  torchrun "${torchrun_args[@]}" "$TRAIN_SCRIPT" \
    "${common_args[@]}" \
    --no-prebuild-recon \
    --recon-model "$TRAIN_RECON_MODEL" \
    --vm-weight "$VM_WEIGHT" \
    --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    "${loader_args[@]}" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY"

  local vm_tag
  local actual_out_dir
  local best_ckpt
  vm_tag="$(python -c "v=max(0.0,min(1.0,float('$VM_WEIGHT'))); print(f'vmw{v:.3f}')")"
  actual_out_dir="$OUT_DIR"
  if [[ "$(basename "$OUT_DIR")" != *"$vm_tag"* ]]; then
    actual_out_dir="$OUT_DIR/$vm_tag"
  fi
  best_ckpt="$actual_out_dir/train_best_by_val_${vm_tag}.pth"
  if [[ ! -f "$best_ckpt" ]]; then
    echo "Best checkpoint not found: $best_ckpt" >&2
    exit 1
  fi
  cp "$best_ckpt" "$actual_out_dir/$FINAL_CKPT_NAME"
  echo "Copied best checkpoint to: $actual_out_dir/$FINAL_CKPT_NAME"
}

usage() {
  cat <<'USAGE'
Usage:
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d-tsdf
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan-tsdf
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh-tsdf
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d-tsdf-dmesh
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan-tsdf-dmesh
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh-tsdf-dmesh
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-all
  bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan-instantmesh
  bash ref_pose/learning/training/run_train_ddp_workflow.sh dataset-index
  bash ref_pose/learning/training/run_train_ddp_workflow.sh train

Edit the "Parameter Settings" block at the top of this file before running.
Recon commands use ref_pose/learning/training/prebuild_recon_cache.py, which
does not import FoundationPose/Utils/mycpp. Train mode still uses train_ddp.py.
Common parameters:
  DATASET_ROOT, RECON_CACHE_ROOT, TRAIN_NPROC_PER_NODE, MASTER_PORT
  TRAIN_RECON_MODEL, VM_WEIGHT, OUT_DIR, FINAL_CKPT_NAME
  EPOCHS, BATCH_SIZE, BATCH_SIZE_PER_RANK, NUM_WORKERS, NUM_WORKERS_PER_RANK
  RECON_GPU_IDS, RECON_WORKERS_PER_GPU, FORCE_REBUILD_RECON
  SAM3D_BASE_GPU_IDS, SAM3D_BASE_WORKERS_PER_GPU
  SAM3D_TSDF_DMESH_GPU_IDS, SAM3D_TSDF_DMESH_WORKERS_PER_GPU
  HUNYUAN_BASE_GPU_IDS, HUNYUAN_BASE_WORKERS_PER_GPU
  HUNYUAN_TSDF_DMESH_GPU_IDS, HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU
  INSTANTMESH_BASE_GPU_IDS, INSTANTMESH_BASE_WORKERS_PER_GPU
  INSTANTMESH_TSDF_DMESH_GPU_IDS, INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU
  Legacy compatibility vars: SAM3D_RECON_*, HUNYUAN_RECON_*, INSTANTMESH_RECON_*
  DATASET_INDEX_SPLIT, FORCE_REBUILD_DATASET_INDEX
  SAM3D_CONDA_ENV, HUNYUAN_CONDA_ENV, INSTANTMESH_CONDA_ENV, CONDA_SH
  SAM3D_DINO_REPO_OR_DIR, SAM3D_DINO_MODEL, SAM3D_DINO_SOURCE, TORCH_HOME_DIR
USAGE
}

case "$cmd" in
  dataset-index|index) run_dataset_index ;;
  recon-instantmesh) run_prebuild instantmesh "$INSTANTMESH_BASE_GPU_IDS" "$INSTANTMESH_RECON_NUM_WORKERS" "$INSTANTMESH_BASE_WORKERS_PER_GPU" ;;
  recon-hunyuan) run_prebuild hunyuan3d "$HUNYUAN_BASE_GPU_IDS" "$HUNYUAN_RECON_NUM_WORKERS" "$HUNYUAN_BASE_WORKERS_PER_GPU" ;;
  recon-sam3d) run_prebuild sam3d "$SAM3D_BASE_GPU_IDS" "$SAM3D_RECON_NUM_WORKERS" "$SAM3D_BASE_WORKERS_PER_GPU" ;;
  recon-sam3d-tsdf) run_prebuild sam3d_tsdf "$SAM3D_TSDF_DMESH_GPU_IDS" "$SAM3D_RECON_NUM_WORKERS" "$SAM3D_TSDF_DMESH_WORKERS_PER_GPU" ;;
  recon-hunyuan-tsdf) run_prebuild hunyuan3d_tsdf "$HUNYUAN_TSDF_DMESH_GPU_IDS" "$HUNYUAN_RECON_NUM_WORKERS" "$HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU" ;;
  recon-instantmesh-tsdf) run_prebuild instantmesh_tsdf "$INSTANTMESH_TSDF_DMESH_GPU_IDS" "$INSTANTMESH_RECON_NUM_WORKERS" "$INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU" ;;
  recon-sam3d-tsdf-dmesh) run_prebuild sam3d_tsdf_dmesh "$SAM3D_TSDF_DMESH_GPU_IDS" "$SAM3D_RECON_NUM_WORKERS" "$SAM3D_TSDF_DMESH_WORKERS_PER_GPU" ;;
  recon-hunyuan-tsdf-dmesh) run_prebuild hunyuan3d_tsdf_dmesh "$HUNYUAN_TSDF_DMESH_GPU_IDS" "$HUNYUAN_RECON_NUM_WORKERS" "$HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU" ;;
  recon-instantmesh-tsdf-dmesh) run_prebuild instantmesh_tsdf_dmesh "$INSTANTMESH_TSDF_DMESH_GPU_IDS" "$INSTANTMESH_RECON_NUM_WORKERS" "$INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU" ;;
  recon-all|recon-sequence) run_recon_all ;;
  recon-hunyuan-instantmesh|recon-hi) run_recon_hunyuan_instantmesh ;;
  train) run_train_only ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
