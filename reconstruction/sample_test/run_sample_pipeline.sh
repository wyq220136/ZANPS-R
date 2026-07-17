#!/usr/bin/env bash

set -eo pipefail

# Run reconstruction, reconstruction-quality eval, pose estimation, and pose eval
# for objects listed in the ECCV sampled_objects.json. All outputs stay under this folder.
#
# Example:
#   bash reconstruction/sample_test/run_sample_pipeline.sh
#   SAMPLE_JSON=reconstruction/sample_test/my_sample.json bash reconstruction/sample_test/run_sample_pipeline.sh
#
# Per-GPU worker examples:
#   SAM3D_BASE_GPU_IDS=0 SAM3D_BASE_WORKERS_PER_GPU=1 \
#   SAM3D_TSDF_DMESH_GPU_IDS=0,1 SAM3D_TSDF_DMESH_WORKERS_PER_GPU=2,4 \
#   POSE_GPU_IDS=0,1 POSE_WORKERS_PER_GPU=4 bash ...
#
# Per-method output-folder override examples:
#   SAM3D_OUT_SUBDIR=my_sam3d HUNYUAN3D_TSDF_OUT_SUBDIR=my_hy_tsdf bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Keep imported project modules read-only outside sample_test by preventing
# Python from creating __pycache__ files next to them.
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train}"
SPLIT="${SPLIT:-test}"
ECCV_SAMPLE_JSON="${ECCV_SAMPLE_JSON:-$REPO_ROOT/eccv/check_part_samples/sampled_objects.json}"
SAMPLE_JSON="${SAMPLE_JSON:-$ECCV_SAMPLE_JSON}"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/default}"
RECON_SUBDIR="${RECON_SUBDIR:-reconstruction_runs}"
REUSE_EXISTING_RECON="${REUSE_EXISTING_RECON:-1}"
REUSE_RECON_ROOT="${REUSE_RECON_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_runs}"
POSE_SUBDIR="${POSE_SUBDIR:-pose_runs}"
POSE_SHARD_SUBDIR="${POSE_SHARD_SUBDIR:-pose_shards}"
RECON_EVAL_SUBDIR="${RECON_EVAL_SUBDIR:-reconstruction_quality_eval}"
POSE_EVAL_SUBDIR="${POSE_EVAL_SUBDIR:-pose_eval}"
FINAL_SUBDIR="${FINAL_SUBDIR:-final_json}"
LOG_SUBDIR="${LOG_SUBDIR:-log}"

WORK_ROOT="$RUN_ROOT/$RECON_SUBDIR"
POSE_ROOT="$RUN_ROOT/$POSE_SUBDIR"
POSE_SHARD_ROOT="$RUN_ROOT/$POSE_SHARD_SUBDIR"
RECON_EVAL_ROOT="$RUN_ROOT/$RECON_EVAL_SUBDIR"
POSE_EVAL_ROOT="$RUN_ROOT/$POSE_EVAL_SUBDIR"
FINAL_ROOT="$RUN_ROOT/$FINAL_SUBDIR"
LOG_ROOT="$RUN_ROOT/$LOG_SUBDIR"
mkdir -p "$WORK_ROOT" "$POSE_ROOT" "$POSE_SHARD_ROOT" "$RECON_EVAL_ROOT" "$POSE_EVAL_ROOT" "$FINAL_ROOT" "$LOG_ROOT"

RECON_GPU_IDS="${RECON_GPU_IDS:-0,1}"
RECON_WORKERS_PER_GPU="${RECON_WORKERS_PER_GPU:-1}"

# Reconstruction resources are split by base model and stage.
# For each base model:
#   *_BASE_* controls the raw base reconstruction method.
#   *_TSDF_DMESH_* controls tsdf, tsdf_dmesh, and partcut_tsdf_dmesh.
SAM3D_BASE_GPU_IDS="${SAM3D_BASE_GPU_IDS:-$RECON_GPU_IDS}"
SAM3D_BASE_WORKERS_PER_GPU="${SAM3D_BASE_WORKERS_PER_GPU:-2}"
SAM3D_TSDF_DMESH_GPU_IDS="${SAM3D_TSDF_DMESH_GPU_IDS:-$RECON_GPU_IDS}"
SAM3D_TSDF_DMESH_WORKERS_PER_GPU="${SAM3D_TSDF_DMESH_WORKERS_PER_GPU:-12}"

HUNYUAN_BASE_GPU_IDS="${HUNYUAN_BASE_GPU_IDS:-$RECON_GPU_IDS}"
HUNYUAN_BASE_WORKERS_PER_GPU="${HUNYUAN_BASE_WORKERS_PER_GPU:-5}"
HUNYUAN_TSDF_DMESH_GPU_IDS="${HUNYUAN_TSDF_DMESH_GPU_IDS:-$RECON_GPU_IDS}"
HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU="${HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU:-12}"

INSTANTMESH_BASE_GPU_IDS="${INSTANTMESH_BASE_GPU_IDS:-$RECON_GPU_IDS}"
INSTANTMESH_BASE_WORKERS_PER_GPU="${INSTANTMESH_BASE_WORKERS_PER_GPU:-1}"
INSTANTMESH_TSDF_DMESH_GPU_IDS="${INSTANTMESH_TSDF_DMESH_GPU_IDS:-$RECON_GPU_IDS}"
INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU="${INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU:-12}"

POSE_GPU_IDS="${POSE_GPU_IDS:-$RECON_GPU_IDS}"
POSE_WORKERS_PER_GPU="${POSE_WORKERS_PER_GPU:-3}"

FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES_PER_PART="${MAX_FRAMES_PER_PART:-0}"
REFINE_ITERATIONS="${REFINE_ITERATIONS:-8}"
SAVE_VIS="${SAVE_VIS:-0}"
USE_NVDIFFRAST="${USE_NVDIFFRAST:-1}"
USE_GT_INIT="${USE_GT_INIT:-0}"
QUIET_FOUNDATIONPOSE="${QUIET_FOUNDATIONPOSE:-1}"
POSE_PROGRESS="${POSE_PROGRESS:-1}"
SIMPLIFY_MESH="${SIMPLIFY_MESH:-1}"
MESH_TARGET_FACES="${MESH_TARGET_FACES:-8000}"
MESH_SIMPLIFY_MIN_FACES="${MESH_SIMPLIFY_MIN_FACES:-12000}"
MESH_SIMPLIFY_BOUNDARY_WEIGHT="${MESH_SIMPLIFY_BOUNDARY_WEIGHT:-1.0}"
MESH_SIMPLIFY_LOG="${MESH_SIMPLIFY_LOG:-0}"
MODEL_SAMPLES="${MODEL_SAMPLES:-10000}"
RECON_EVAL_SAMPLES="${RECON_EVAL_SAMPLES:-50000}"
RECON_EVAL_FRAME_STRIDE="${RECON_EVAL_FRAME_STRIDE:-1}"
RECON_EVAL_MAX_FRAMES="${RECON_EVAL_MAX_FRAMES:-0}"
OVERWRITE="${OVERWRITE:-0}"

SAM3D_CONDA_ENV="${SAM3D_CONDA_ENV:-sam3d}"
HUNYUAN_CONDA_ENV="${HUNYUAN_CONDA_ENV:-hunyuan}"
INSTANTMESH_CONDA_ENV="${INSTANTMESH_CONDA_ENV:-instantmesh}"
POSE_CONDA_ENV="${POSE_CONDA_ENV:-$SAM3D_CONDA_ENV}"
CONDA_SH="${CONDA_SH:-}"

BOP_TOOLKIT_ROOT="${BOP_TOOLKIT_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/related_works/bop_toolkit}"

SAM3D_PROJECT_ROOT="${SAM3D_PROJECT_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects}"
SAM3D_CONFIG_PATH="${SAM3D_CONFIG_PATH:-$SAM3D_PROJECT_ROOT/checkpoints/hf/pipeline.yaml}"
SAM3D_DINO_REPO_OR_DIR="${SAM3D_DINO_REPO_OR_DIR:-$SAM3D_PROJECT_ROOT/SAM3D_DINO/dinov2}"
SAM3D_DINO_MODEL="${SAM3D_DINO_MODEL:-dinov2_vitl14_reg}"
SAM3D_DINO_SOURCE="${SAM3D_DINO_SOURCE:-local}"
TORCH_HOME_DIR="${TORCH_HOME_DIR:-$SAM3D_PROJECT_ROOT/SAM3D_DINO/torch_cache}"
export SAM3D_PROJECT_ROOT SAM3D_CONFIG_PATH SAM3D_DINO_REPO_OR_DIR SAM3D_DINO_MODEL SAM3D_DINO_SOURCE
export TORCH_HOME="$TORCH_HOME_DIR"

INSTANTMESH_ROOT="${INSTANTMESH_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/baseline/Any6D/instantmesh}"
INSTANTMESH_CONFIG_PATH="${INSTANTMESH_CONFIG_PATH:-$INSTANTMESH_ROOT/configs/instant-mesh-large.yaml}"
ANY6D_ZERO123PLUS_MODEL="${ANY6D_ZERO123PLUS_MODEL:-$INSTANTMESH_ROOT/zero123_ckpts}"
ANY6D_DINO_MODEL="${ANY6D_DINO_MODEL:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/baseline/Any6D/dino_vitb16}"
ANY6D_INSTANTMESH_UNET="${ANY6D_INSTANTMESH_UNET:-$INSTANTMESH_ROOT/ckpts/diffusion_pytorch_model.bin}"
ANY6D_INSTANTMESH_MODEL="${ANY6D_INSTANTMESH_MODEL:-$INSTANTMESH_ROOT/ckpts/instant_mesh_large.ckpt}"

HUNYUAN_MODEL_PATH="${HUNYUAN_MODEL_PATH:-$REPO_ROOT/Hunyuan3D-2.1/ckpts}"

METHODS=(
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
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"
ACTIVE_METHODS=()
ACTIVE_METHODS_CSV=""
RECON_STATUS_TSV="$RUN_ROOT/reconstruction_status.tsv"
declare -A RECON_MISSING_OBJECTS=()

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

activate_env() {
  local env_name="$1"
  if [[ -n "$env_name" && "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]]; then
    echo
    echo "========== conda activate $env_name =========="
    conda activate "$env_name"
  fi
}

bool_flag() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) echo "--$2" ;;
    *) echo "--no-$2" ;;
  esac
}

sum_workers_per_gpu() {
  python - "$1" "$2" <<'PY'
import sys
gpus = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
raw = sys.argv[2].strip()
vals = [x.strip() for x in raw.split(",") if x.strip()]
if not gpus:
    print(max(1, int(vals[0]) if vals else 1))
elif not vals:
    print(len(gpus))
elif len(vals) == 1:
    print(len(gpus) * max(1, int(vals[0])))
elif len(vals) == len(gpus):
    print(sum(max(1, int(v)) for v in vals))
else:
    raise SystemExit(f"workers-per-gpu length mismatch: gpus={gpus} workers={raw}")
PY
}

objects_csv() {
  python - "$SAMPLE_JSON" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
objects = data.get("objects", data if isinstance(data, list) else [])
if not objects:
    raise SystemExit(f"no objects found in {sys.argv[1]}")
print(",".join(str(x) for x in objects))
PY
}

method_env() {
  case "$1" in
    sam3d*) echo "$SAM3D_CONDA_ENV" ;;
    hunyuan3d*) echo "$HUNYUAN_CONDA_ENV" ;;
    instantmesh*) echo "$INSTANTMESH_CONDA_ENV" ;;
    *) echo "" ;;
  esac
}

method_script() {
  echo "reconstruction/run/recon_$1.py"
}

recon_gpu_ids_for_method() {
  case "$1" in
    sam3d) echo "$SAM3D_BASE_GPU_IDS" ;;
    sam3d_tsdf|sam3d_tsdf_dmesh|sam3d_partcut_tsdf_dmesh) echo "$SAM3D_TSDF_DMESH_GPU_IDS" ;;
    hunyuan3d) echo "$HUNYUAN_BASE_GPU_IDS" ;;
    hunyuan3d_tsdf|hunyuan3d_tsdf_dmesh|hunyuan3d_partcut_tsdf_dmesh) echo "$HUNYUAN_TSDF_DMESH_GPU_IDS" ;;
    instantmesh) echo "$INSTANTMESH_BASE_GPU_IDS" ;;
    instantmesh_tsdf|instantmesh_tsdf_dmesh|instantmesh_partcut_tsdf_dmesh) echo "$INSTANTMESH_TSDF_DMESH_GPU_IDS" ;;
    *) echo "$RECON_GPU_IDS" ;;
  esac
}

recon_workers_per_gpu_for_method() {
  case "$1" in
    sam3d) echo "$SAM3D_BASE_WORKERS_PER_GPU" ;;
    sam3d_tsdf|sam3d_tsdf_dmesh|sam3d_partcut_tsdf_dmesh) echo "$SAM3D_TSDF_DMESH_WORKERS_PER_GPU" ;;
    hunyuan3d) echo "$HUNYUAN_BASE_WORKERS_PER_GPU" ;;
    hunyuan3d_tsdf|hunyuan3d_tsdf_dmesh|hunyuan3d_partcut_tsdf_dmesh) echo "$HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU" ;;
    instantmesh) echo "$INSTANTMESH_BASE_WORKERS_PER_GPU" ;;
    instantmesh_tsdf|instantmesh_tsdf_dmesh|instantmesh_partcut_tsdf_dmesh) echo "$INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU" ;;
    *) echo "$RECON_WORKERS_PER_GPU" ;;
  esac
}

method_out_subdir() {
  local method="$1"
  local var
  var="$(echo "${method}_OUT_SUBDIR" | tr '[:lower:]' '[:upper:]')"
  printf '%s\n' "${!var:-$method}"
}

recon_object_complete_at_root() {
  local root="$1"
  local method="$2"
  local object="$3"
  python - "$root" "$method" "$object" "$DATA_ROOT" "$SPLIT" <<'PY'
import re
import sys
from pathlib import Path

root, method, object_name, data_root, split = sys.argv[1:6]

def natural_sort_key(value):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(value))]

def part_model_name(part_name, fallback_idx=0):
    match = re.search(r"(\d+)", str(part_name))
    part_id = int(match.group(1)) if match else int(fallback_idx)
    return f"model_{part_id:04d}"

obj_dir = Path(root) / method / split / object_name
if not obj_dir.is_dir():
    raise SystemExit(1)

masks_dir = Path(data_root) / split / object_name / "masks"
if not masks_dir.is_dir():
    raise SystemExit(1)
parts = sorted([p.name for p in masks_dir.iterdir() if p.is_dir()], key=natural_sort_key)
expected = [part_model_name(part, idx) for idx, part in enumerate(parts)]

if not expected:
    raise SystemExit(1)

candidate_roots = [
    obj_dir / "pose_ready_models" / "view_0",
    obj_dir / "models" / "view_0",
]
for model in expected:
    if not any((base / model / "model.obj").is_file() for base in candidate_roots):
        raise SystemExit(1)
PY
}

copy_recon_object_from_reuse_root() {
  local method="$1"
  local object="$2"
  local src="$REUSE_RECON_ROOT/$method/$SPLIT/$object"
  local dst="$WORK_ROOT/$method/$SPLIT/$object"
  local dst_parent="$WORK_ROOT/$method/$SPLIT"
  if [[ "$REUSE_EXISTING_RECON" =~ ^(0|false|FALSE|no|NO|off|OFF)$ ]]; then
    return 1
  fi
  if [[ "$(cd "$REUSE_RECON_ROOT" 2>/dev/null && pwd -P || true)" == "$(cd "$WORK_ROOT" 2>/dev/null && pwd -P || true)" ]]; then
    return 1
  fi
  if ! recon_object_complete_at_root "$REUSE_RECON_ROOT" "$method" "$object"; then
    return 1
  fi
  mkdir -p "$dst_parent"
  if [[ -e "$dst" ]]; then
    if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      rm -rf "$dst"
    else
      return 1
    fi
  fi
  cp -a "$src" "$dst_parent/"
  recon_object_complete_at_root "$WORK_ROOT" "$method" "$object"
}

filter_missing_recon_objects() {
  local method="$1"
  local objects_csv="$2"
  local missing=()
  local object
  local old_ifs="$IFS"
  IFS=,
  for object in $objects_csv; do
    object="$(echo "$object" | xargs)"
    [[ -z "$object" ]] && continue
    if recon_object_complete_at_root "$WORK_ROOT" "$method" "$object"; then
      echo "[reuse-local] method=$method object=$object" >&2
      continue
    fi
    if copy_recon_object_from_reuse_root "$method" "$object"; then
      echo "[reuse-copy] method=$method object=$object src=$REUSE_RECON_ROOT/$method/$SPLIT/$object" >&2
      continue
    fi
    echo "[reuse-miss] method=$method object=$object" >&2
    missing+=("$object")
  done
  IFS="$old_ifs"
  local joined=""
  if (( ${#missing[@]} > 0 )); then
    joined="$(IFS=,; echo "${missing[*]}")"
  fi
  printf '%s\n' "$joined"
}

preflight_reconstruction_reuse() {
  local method
  local missing
  echo
  echo "========== RECON PRECHECK AND REUSE =========="
  for method in "${METHODS[@]}"; do
    missing="$(filter_missing_recon_objects "$method" "$OBJECTS")"
    RECON_MISSING_OBJECTS["$method"]="$missing"
    if [[ -z "$missing" ]]; then
      echo "[precheck] method=$method missing=<none>"
    else
      echo "[precheck] method=$method missing=$missing"
    fi
  done
  echo "========== DONE RECON PRECHECK =========="
}

ensure_method_link() {
  local method="$1"
  local out_name="$2"
  local canonical="$WORK_ROOT/$method"
  local custom="$WORK_ROOT/$out_name"
  if [[ "$out_name" == "$method" ]]; then
    return
  fi
  if [[ -d "$canonical" && ! -L "$canonical" && ! -e "$custom" ]]; then
    mv "$canonical" "$custom"
  fi
  if [[ -e "$custom" && ! -e "$canonical" ]]; then
    ln -s "$out_name" "$canonical"
  fi
}

run_recon_method() {
  local method="$1"
  local script
  local env_name
  local workers
  local out_name
  local recon_gpu_ids
  local recon_workers_per_gpu
  script="$(method_script "$method")"
  env_name="$(method_env "$method")"
  out_name="$(method_out_subdir "$method")"
  recon_gpu_ids="$(recon_gpu_ids_for_method "$method")"
  recon_workers_per_gpu="$(recon_workers_per_gpu_for_method "$method")"
  workers="$(sum_workers_per_gpu "$recon_gpu_ids" "$recon_workers_per_gpu")"
  local objects_to_run
  objects_to_run="${RECON_MISSING_OBJECTS[$method]-}"
  if [[ -z "$objects_to_run" ]]; then
    echo
    echo "========== RECON $method skipped: all sampled objects reused =========="
    ensure_method_link "$method" "$out_name"
    ACTIVE_METHODS+=("$method")
    printf '%s\t%s\t%s\t%s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$method" "reused" "0" >> "$RECON_STATUS_TSV"
    return 0
  fi
  if [[ ! -f "$script" ]]; then
    echo "Missing reconstruction script for method=$method script=$script" >&2
    exit 2
  fi
  activate_env "$env_name"
  echo
  echo "========== RECON $method -> $out_name =========="
  echo "[recon-resource] method=$method gpu_ids=${recon_gpu_ids:-default} workers_per_gpu=$recon_workers_per_gpu total_workers=$workers"
  local args=(
    --data-root "$DATA_ROOT"
    --split "$SPLIT"
    --work-root "$WORK_ROOT"
    --objects "$objects_to_run"
    --gpus "$recon_gpu_ids"
    --num-workers "$workers"
    --workers-per-gpu "$recon_workers_per_gpu"
    --reset-coord
  )
  case "$method" in
    sam3d_tsdf|sam3d_tsdf_dmesh|sam3d_dmesh)
      args+=(--build-base-if-missing)
      ;;
    hunyuan3d_tsdf|hunyuan3d_tsdf_dmesh|hunyuan3d_dmesh)
      args+=(--build-base-if-missing)
      ;;
    instantmesh|instantmesh_tsdf|instantmesh_tsdf_dmesh|instantmesh_dmesh)
      args+=(
        --build-base-if-missing
        --instantmesh-root "$INSTANTMESH_ROOT"
        --instantmesh-config-path "$INSTANTMESH_CONFIG_PATH"
        --instantmesh-diffusion-model "$ANY6D_ZERO123PLUS_MODEL"
        --instantmesh-dino-model "$ANY6D_DINO_MODEL"
        --instantmesh-unet-path "$ANY6D_INSTANTMESH_UNET"
        --instantmesh-model-path "$ANY6D_INSTANTMESH_MODEL"
      )
      ;;
    instantmesh_partcut_tsdf_dmesh)
      args+=(
        --instantmesh-root "$INSTANTMESH_ROOT"
        --instantmesh-config-path "$INSTANTMESH_CONFIG_PATH"
        --instantmesh-diffusion-model "$ANY6D_ZERO123PLUS_MODEL"
        --instantmesh-dino-model "$ANY6D_DINO_MODEL"
        --instantmesh-unet-path "$ANY6D_INSTANTMESH_UNET"
        --instantmesh-model-path "$ANY6D_INSTANTMESH_MODEL"
      )
      ;;
    hunyuan3d|hunyuan3d_tsdf|hunyuan3d_tsdf_dmesh)
      args+=(--model-path "$HUNYUAN_MODEL_PATH")
      ;;
  esac
  if python "$script" "${args[@]}"; then
    ensure_method_link "$method" "$out_name"
    ACTIVE_METHODS+=("$method")
    printf '%s\t%s\t%s\t%s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$method" "success" "0" >> "$RECON_STATUS_TSV"
    echo "========== DONE RECON $method =========="
  else
    local exit_code="$?"
    printf '%s\t%s\t%s\t%s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$method" "failed" "$exit_code" >> "$RECON_STATUS_TSV"
    echo "[WARN] reconstruction failed but sample pipeline will continue: method=$method exit_code=$exit_code" >&2
    return 0
  fi
}

run_recon_eval() {
  if [[ -z "$ACTIVE_METHODS_CSV" ]]; then
    echo "[WARN] skip reconstruction eval: no successful reconstruction methods." >&2
    return 0
  fi
  echo
  echo "========== RECON QUALITY EVAL =========="
  python reconstruction/eval/eval_all_methods.py \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --work-root "$WORK_ROOT" \
    --methods "$ACTIVE_METHODS_CSV" \
    --objects "$OBJECTS" \
    --output-root "$RECON_EVAL_ROOT" \
    --samples "$RECON_EVAL_SAMPLES" \
    --max-eval-frames "$RECON_EVAL_MAX_FRAMES" \
    --frame-stride "$RECON_EVAL_FRAME_STRIDE"
}

run_pose_and_eval() {
  if [[ -z "$ACTIVE_METHODS_CSV" ]]; then
    echo "[WARN] skip pose estimation/eval: no successful reconstruction methods." >&2
    return 0
  fi
  echo
  echo "========== POSE EST =========="
  if [[ -n "$POSE_CONDA_ENV" ]]; then
    activate_env "$POSE_CONDA_ENV"
  fi
  local shard_json="$RUN_ROOT/pose_shards.json"
  python "$SCRIPT_DIR/make_pose_shards.py" \
    --objects "$OBJECTS" \
    --gpu-ids "$POSE_GPU_IDS" \
    --workers-per-gpu "$POSE_WORKERS_PER_GPU" \
    --output "$shard_json"
  for method in "${ACTIVE_METHODS[@]}"; do
    echo
    echo "========== POSE $method =========="
    local script="reconstruction/pose_est/pose_est_${method}.py"
    if [[ ! -f "$script" ]]; then
      echo "Missing pose-est script for method=$method script=$script" >&2
      exit 2
    fi
    if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      rm -rf "$POSE_ROOT/$method" "$POSE_SHARD_ROOT/$method"
    fi
    local shard_tsv="$RUN_ROOT/pose_shards_${method}.tsv"
    python - "$shard_json" > "$shard_tsv" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
for i, shard in enumerate(data["shards"]):
    print(f"{i}\t{shard['gpu_id']}\t{shard['workers']}\t{shard['objects_csv']}")
PY
    while IFS=$'\t' read -r shard_id gpu_id workers shard_objects; do
      (
        shard_pose_root="$POSE_SHARD_ROOT/$method/shard_${shard_id}"
        mkdir -p "$shard_pose_root"
        python "$script" \
          --data-root "$DATA_ROOT" \
          --split "$SPLIT" \
          --work-root "$WORK_ROOT" \
          --pose-root "$shard_pose_root" \
          --objects "$shard_objects" \
          --frame-stride "$FRAME_STRIDE" \
          --max-frames-per-part "$MAX_FRAMES_PER_PART" \
          --refine-iterations "$REFINE_ITERATIONS" \
          --gpu-ids "$gpu_id" \
          --workers-per-gpu 1 \
          --num-workers "$workers" \
          "$(bool_flag "$USE_NVDIFFRAST" "use-nvdiffrast")" \
          "$(bool_flag "$USE_GT_INIT" "use-gt-init")" \
          "$(bool_flag "$SAVE_VIS" "save-vis")" \
          "$(bool_flag "$QUIET_FOUNDATIONPOSE" "quiet-foundationpose")" \
          "$(bool_flag "$POSE_PROGRESS" "progress")" \
          "$(bool_flag "$SIMPLIFY_MESH" "simplify-mesh")" \
          --mesh-target-faces "$MESH_TARGET_FACES" \
          --mesh-simplify-min-faces "$MESH_SIMPLIFY_MIN_FACES" \
          --mesh-simplify-boundary-weight "$MESH_SIMPLIFY_BOUNDARY_WEIGHT" \
          "$(bool_flag "$MESH_SIMPLIFY_LOG" "mesh-simplify-log")" \
          --debug-dir "$RUN_ROOT/foundationpose_debug/$method/shard_${shard_id}"
      ) &
    done < "$shard_tsv"
    wait
    python "$SCRIPT_DIR/merge_pose_shards.py" \
      --method "$method" \
      --shard-root "$POSE_SHARD_ROOT" \
      --pose-root "$POSE_ROOT"
  done

  echo
  echo "========== POSE EVAL =========="
  python reconstruction/eval/eval_pose_est.py \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --work-root "$WORK_ROOT" \
    --pose-root "$POSE_ROOT" \
    --eval-root "$POSE_EVAL_ROOT" \
    --bop-toolkit-root "$BOP_TOOLKIT_ROOT" \
    --methods "$ACTIVE_METHODS_CSV" \
    --objects "$OBJECTS" \
    --model-samples "$MODEL_SAMPLES"
}

collect_final_jsons() {
  python "$SCRIPT_DIR/collect_eval_jsons.py" \
    --methods "$ACTIVE_METHODS_CSV" \
    --objects-json "$SAMPLE_JSON" \
    --recon-eval-root "$RECON_EVAL_ROOT" \
    --pose-eval-root "$POSE_EVAL_ROOT" \
    --recon-output "$FINAL_ROOT/reconstruction_quality_eval.json" \
    --pose-output "$FINAL_ROOT/pose_est_eval.json"
}

ensure_sample_json() {
  if [[ -f "$SAMPLE_JSON" ]]; then
    return
  fi
  echo "Sample JSON not found: $SAMPLE_JSON" >&2
  echo "sample_test now reuses the ECCV sampling result instead of sampling objects locally." >&2
  echo "Run the ECCV sampling pipeline first, or pass SAMPLE_JSON=/path/to/sampled_objects.json." >&2
  echo "Expected default ECCV sample path: $ECCV_SAMPLE_JSON" >&2
  exit 1
}

ensure_sample_json

OBJECTS="$(objects_csv)"
LOG_FILE="$LOG_ROOT/$(date +'%Y%m%d_%H%M%S')_sample_pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging terminal output to: $LOG_FILE"
echo "SAMPLE_JSON=$SAMPLE_JSON"
echo "ECCV_SAMPLE_JSON=$ECCV_SAMPLE_JSON"
echo "OBJECTS=$OBJECTS"
echo "RUN_ROOT=$RUN_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "POSE_ROOT=$POSE_ROOT"
echo "POSE_SHARD_ROOT=$POSE_SHARD_ROOT"
echo "RECON_EVAL_ROOT=$RECON_EVAL_ROOT"
echo "POSE_EVAL_ROOT=$POSE_EVAL_ROOT"
echo "RECON_DEFAULT GPU_IDS=$RECON_GPU_IDS WORKERS_PER_GPU=$RECON_WORKERS_PER_GPU"
echo "SAM3D_BASE GPU_IDS=$SAM3D_BASE_GPU_IDS WORKERS_PER_GPU=$SAM3D_BASE_WORKERS_PER_GPU"
echo "SAM3D_TSDF_DMESH GPU_IDS=$SAM3D_TSDF_DMESH_GPU_IDS WORKERS_PER_GPU=$SAM3D_TSDF_DMESH_WORKERS_PER_GPU"
echo "HUNYUAN_BASE GPU_IDS=$HUNYUAN_BASE_GPU_IDS WORKERS_PER_GPU=$HUNYUAN_BASE_WORKERS_PER_GPU"
echo "HUNYUAN_TSDF_DMESH GPU_IDS=$HUNYUAN_TSDF_DMESH_GPU_IDS WORKERS_PER_GPU=$HUNYUAN_TSDF_DMESH_WORKERS_PER_GPU"
echo "INSTANTMESH_BASE GPU_IDS=$INSTANTMESH_BASE_GPU_IDS WORKERS_PER_GPU=$INSTANTMESH_BASE_WORKERS_PER_GPU"
echo "INSTANTMESH_TSDF_DMESH GPU_IDS=$INSTANTMESH_TSDF_DMESH_GPU_IDS WORKERS_PER_GPU=$INSTANTMESH_TSDF_DMESH_WORKERS_PER_GPU"
echo "POSE_GPU_IDS=$POSE_GPU_IDS POSE_WORKERS_PER_GPU=$POSE_WORKERS_PER_GPU"
echo "SIMPLIFY_MESH=$SIMPLIFY_MESH MESH_TARGET_FACES=$MESH_TARGET_FACES MESH_SIMPLIFY_MIN_FACES=$MESH_SIMPLIFY_MIN_FACES"
echo -e "time\tmethod\tstatus\texit_code" > "$RECON_STATUS_TSV"

init_conda
preflight_reconstruction_reuse
for method in "${METHODS[@]}"; do
  run_recon_method "$method"
done
ACTIVE_METHODS_CSV="$(IFS=,; echo "${ACTIVE_METHODS[*]}")"
echo "Successful reconstruction methods: ${ACTIVE_METHODS_CSV:-<none>}"
echo "Reconstruction status table: $RECON_STATUS_TSV"
run_recon_eval
run_pose_and_eval
collect_final_jsons
echo "[DONE] final jsons:"
echo "  $FINAL_ROOT/reconstruction_quality_eval.json"
echo "  $FINAL_ROOT/pose_est_eval.json"
