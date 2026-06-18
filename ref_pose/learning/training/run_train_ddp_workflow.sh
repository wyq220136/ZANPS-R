#!/usr/bin/env bash

# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-instantmesh
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-hunyuan
# bash ref_pose/learning/training/run_train_ddp_workflow.sh recon-sam3d
# bash ref_pose/learning/training/run_train_ddp_workflow.sh train

set -euo pipefail

# =========================
# Parameter Settings
# =========================
# Edit this block before running the commands above.

# Paths.
CONFIG="ref_pose/learning/training/configs/train_ddp.yaml"
TRAIN_SCRIPT="ref_pose/learning/training/train_ddp.py"
DATASET_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
RECON_CACHE_ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_recon_cache"
TRAIN_SUBDIR="train"
VAL_SUBDIR="val"

# Distributed launch.
NPROC_PER_NODE=8
MASTER_PORT=29500

# Reconstruction cache prebuild.
RECON_NUM_WORKERS=0
RECON_NUM_WORKERS_PER_RANK=""
FORCE_REBUILD_RECON=0
FORCE_RESAMPLE_RECON=0

# Train-only run.
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

common_args=(
  --train-config "$CONFIG"
  --dataset-root "$DATASET_ROOT"
  --train-subdir "$TRAIN_SUBDIR"
  --val-subdir "$VAL_SUBDIR"
  --recon-cache-root "$RECON_CACHE_ROOT"
)

torchrun_args=(
  "--nproc_per_node=$NPROC_PER_NODE"
  "--master_port=$MASTER_PORT"
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

run_prebuild() {
  local recon_model="$1"
  local force_rebuild_flag
  local force_resample_flag
  local recon_worker_args
  recon_worker_args=(--recon-num-workers "$RECON_NUM_WORKERS")
  force_rebuild_flag="$(bool_flag "$FORCE_REBUILD_RECON" "--force-rebuild-recon" "--no-force-rebuild-recon")"
  force_resample_flag="$(bool_flag "$FORCE_RESAMPLE_RECON" "--force-resample-recon" "--no-force-resample-recon")"
  if [[ -n "$RECON_NUM_WORKERS_PER_RANK" ]]; then
    recon_worker_args+=(--recon-num-workers-per-rank "$RECON_NUM_WORKERS_PER_RANK")
  fi

  torchrun "${torchrun_args[@]}" "$TRAIN_SCRIPT" \
    "${common_args[@]}" \
    --prebuild-recon \
    --prebuild-only \
    --prebuild-split both \
    --recon-model "$recon_model" \
    "${recon_worker_args[@]}" \
    "$force_rebuild_flag" \
    "$force_resample_flag"
}

run_train_only() {
  local loader_args
  loader_args=(--batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
  if [[ -n "$BATCH_SIZE_PER_RANK" ]]; then
    loader_args+=(--batch-size-per-rank "$BATCH_SIZE_PER_RANK")
  fi
  if [[ -n "$NUM_WORKERS_PER_RANK" ]]; then
    loader_args+=(--num-workers-per-rank "$NUM_WORKERS_PER_RANK")
  fi

  torchrun "${torchrun_args[@]}" "$TRAIN_SCRIPT" \
    "${common_args[@]}" \
    --no-prebuild-recon \
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
  bash ref_pose/learning/training/run_train_ddp_workflow.sh train

Edit the "Parameter Settings" block at the top of this file before running.
Common parameters:
  DATASET_ROOT, RECON_CACHE_ROOT, NPROC_PER_NODE, MASTER_PORT
  VM_WEIGHT, OUT_DIR, FINAL_CKPT_NAME
  EPOCHS, BATCH_SIZE, BATCH_SIZE_PER_RANK, NUM_WORKERS, NUM_WORKERS_PER_RANK
  RECON_NUM_WORKERS, RECON_NUM_WORKERS_PER_RANK, FORCE_REBUILD_RECON
USAGE
}

cmd="${1:-}"
case "$cmd" in
  recon-instantmesh) run_prebuild instantmesh ;;
  recon-hunyuan) run_prebuild hunyuan3d ;;
  recon-sam3d) run_prebuild sam3d ;;
  train) run_train_only ;;
  -h|--help|help|"") usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
