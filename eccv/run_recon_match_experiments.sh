#!/usr/bin/env bash

# Shared ECCV experiment runner.
#
# This is the top-level script for the mask-mesh matching experiments. It runs
# directly on the dataset under ROOT and does not copy data to another workdir.
#
# Full pipeline:
#   1. Shared preprocess, reused by every branch:
#      - PartNet raw-layout validation:
#          read masks/<part>/<frame>.png and object_masks/<frame>.png directly.
#      - Reference frame selection.
#      - SAM candidate segmentation on every RGB frame into:
#          <object>/pred_mask/<frame>/mask_*.png
#      - VLM filtering of reference candidate masks into:
#          <object>/chosen_part/<reference_frame>/mask_*.png
#      - SAM3D reconstruction of selected reference parts into:
#          <object>/selected_mesh/view_<view_id>/<part_model_dir>/model.obj
#      - Reference manifest:
#          <object>/chosen_part/selected_parts.json
#   2. Branch-specific matching:
#      - sam6d: final masks in <object>/sam6d
#      - cnos: final masks in <object>/cnos
#      - ours: final masks in <object>/full
#      - ours-no-complement: final masks in <object>/wo_dual
#      - ours-no-render: final masks in <object>/wo_render
#
# Reference selector:
#   sam-vlm: default experimental path. It runs SAM on selected reference frames
#            and filters candidate part masks with the VLM checker.
#   gt-mask: debug/upper-bound path. It uses dataset part masks directly.
#
# Usage examples:
#   bash eccv/run_recon_match_experiments.sh preprocess partnet
#   bash eccv/run_recon_match_experiments.sh sam6d partnet
#   bash eccv/run_recon_match_experiments.sh cnos partnet
#   bash eccv/run_recon_match_experiments.sh ours partnet
#   bash eccv/run_recon_match_experiments.sh ours-no-complement partnet
#   bash eccv/run_recon_match_experiments.sh ours-no-render partnet
#   bash eccv/run_recon_match_experiments.sh all partnet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-help}"
DATASET_KIND="${2:-partnet}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# =========================
# Logging
# =========================
# Every run writes the full terminal output to eccv/log while still printing it
# to the console. The file name records the timestamp, mode, dataset kind, and
# command arguments for easier experiment tracking.
LOG_DIR="$SCRIPT_DIR/log"
mkdir -p "$LOG_DIR"
LOG_TS="$(date +'%Y%m%d_%H%M%S')"
LOG_MODE="${MODE:-help}"
LOG_DATASET="${DATASET_KIND:-partnet}"
LOG_ARGS="$*"
LOG_SAFE_MODE="${LOG_MODE//[^A-Za-z0-9_.-]/_}"
LOG_SAFE_DATASET="${LOG_DATASET//[^A-Za-z0-9_.-]/_}"
LOG_FILE="$LOG_DIR/${LOG_TS}_${LOG_SAFE_MODE}_${LOG_SAFE_DATASET}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging terminal output to: $LOG_FILE"
echo "Timestamp: $LOG_TS"
echo "Command: bash eccv/run_recon_match_experiments.sh ${LOG_ARGS}"
echo "Working directory: $REPO_ROOT"

# =========================
# Parallel Settings
# =========================
# Edit this block when changing GPU/process parallelism.
#
# SAM segmentation:
#   SAM_GPU_IDS="0,1" uses the listed GPUs.
#   SAM_PROCS_PER_GPU controls SAM worker processes per listed GPU.
#   SAM_WORKERS_PER_GPU optionally overrides SAM_PROCS_PER_GPU with either one
#   count for every GPU or comma-separated counts aligned with SAM_GPU_IDS.
#   SAM_WORKERS is used only when SAM_GPU_IDS is empty.
SAM_GPU_IDS="${SAM_GPU_IDS:-0,1}"
SAM_PROCS_PER_GPU="${SAM_PROCS_PER_GPU:-1}"
SAM_WORKERS_PER_GPU="${SAM_WORKERS_PER_GPU:-7,4}"
SAM_WORKERS="${SAM_WORKERS:-1}"
TASK_CHUNKSIZE="${TASK_CHUNKSIZE:-1}"

# Reference-mask VLM filtering and SAM3D reconstruction.
VLM_WORKERS="${VLM_WORKERS:-1}"
SAM3D_WORKERS="${SAM3D_WORKERS:-1}"

# Matching workers. MATCH_WORKERS is the shared fallback; each pipeline can
# override it independently.
MATCH_WORKERS="${MATCH_WORKERS:-6}"
SAM6D_MATCH_WORKERS="${SAM6D_MATCH_WORKERS:-$MATCH_WORKERS}"
CNOS_MATCH_WORKERS="${CNOS_MATCH_WORKERS:-$MATCH_WORKERS}"
OURS_MATCH_WORKERS="${OURS_MATCH_WORKERS:-$MATCH_WORKERS}"
OURS_NO_COMPLEMENT_MATCH_WORKERS="${OURS_NO_COMPLEMENT_MATCH_WORKERS:-$OURS_MATCH_WORKERS}"
OURS_NO_RENDER_MATCH_WORKERS="${OURS_NO_RENDER_MATCH_WORKERS:-$OURS_MATCH_WORKERS}"

# =========================
# Overwrite Settings
# =========================
# OVERWRITE_ALL=1 is the one-switch rebuild mode:
#   - rebuild pred_mask, chosen_part, and selected_mesh
#   - force SAM3D reconstruction instead of reusing selected_mesh/view_*
OVERWRITE_ALL="${OVERWRITE_ALL:-1}"
OVERWRITE="${OVERWRITE:-$OVERWRITE_ALL}"

# =========================
# Dataset / object selection
# =========================
# DATASET_KIND is the second positional argument:
#   partnet: dataset_train/test-style layout with rgb, depth, masks,
#            object_masks/object_mask, and K.txt.
#   rgbd:    prepared ECCV RGB-D object layout with rgb, depth, gt_mask, mask,
#            models, and K.txt.
#
# ROOT is the dataset root. Override it like:
#   ROOT=/path/to/test bash eccv/run_recon_match_experiments.sh all partnet
#
# OBJECT_SOURCE:
#   all:    run every object folder under ROOT.
#   sample: run the built-in sample list when those folders exist.
#
# OBJECTS:
#   Optional comma-separated object names, e.g. OBJECTS=bottle_3520,laptop_1.
#
# START / END:
#   Slice the sorted object list as [START:END]. Leave END empty to run to the
#   end. Useful when manually splitting jobs across machines.
#
# GT_ROOT:
#   Optional pose/model root passed to SAM3D raw_pose_estimation for mesh
#   alignment, if your reconstruction code uses it.
if [[ -z "${ROOT:-}" ]]; then
  case "$DATASET_KIND" in
    partnet)
      ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train/test"
      ;;
    rgbd)
      ROOT="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs"
      ;;
    *)
      echo "Unknown DATASET_KIND: $DATASET_KIND" >&2
      exit 2
      ;;
  esac
fi

OBJECT_SOURCE="${OBJECT_SOURCE:-all}"
OBJECTS="${OBJECTS:-}"
START="${START:-0}"
END="${END:-}"
GT_ROOT="${GT_ROOT:-}"

# =========================
# Shared preprocess settings
# =========================
# REFERENCE_POLICY:
#   first: choose the first frame for each view. This is the default because the
#          current matching code also uses first-frame references.
#   best:  choose the frame with the most visible part masks.
#
# REFERENCE_SELECTOR:
#   sam-vlm: SAM segmentation on selected reference frame, then VLM filtering.
#   gt-mask: use dataset masks directly; useful for debugging/upper-bound runs.
#
# PRED_MASK_SUBDIR / CHOSEN_PART_SUBDIR / SELECTED_MESH_SUBDIR:
#   Canonical preprocess stage outputs. Matching branches reuse PRED_MASK_SUBDIR
#   instead of generating branch-local SAM masks.
#
# MIN_VISIBLE_PIXELS:
#   Minimum mask area used for reference visibility and CAD filtering.
#
# OVERWRITE_REFERENCE:
#   0 keeps existing chosen_part masks; 1 rebuilds them.
#
# OVERWRITE_COMPATIBILITY:
#   Legacy no-op for the current PartNet path. Raw masks/object_masks are read
#   directly and gt_mask/mask compatibility directories are not created.
#
# SKIP_EXISTING_RECON:
#   1 reuses existing selected_mesh/view_<id> outputs only when the reference
#   masks are also reused; freshly selected sam-vlm masks still trigger SAM3D reconstruction.
#   0 forces SAM3D reconstruction.
REFERENCE_POLICY="${REFERENCE_POLICY:-first}"
REFERENCE_SELECTOR="${REFERENCE_SELECTOR:-sam-vlm}"
PRED_MASK_SUBDIR="${PRED_MASK_SUBDIR:-pred_mask}"
CHOSEN_PART_SUBDIR="${CHOSEN_PART_SUBDIR:-chosen_part}"
SELECTED_MESH_SUBDIR="${SELECTED_MESH_SUBDIR:-selected_mesh}"
REFERENCE_MASK_SUBDIR="${REFERENCE_MASK_SUBDIR:-$CHOSEN_PART_SUBDIR}"
REF_SELECT_SUBDIR="${REF_SELECT_SUBDIR:-$CHOSEN_PART_SUBDIR}"
MIN_VISIBLE_PIXELS="${MIN_VISIBLE_PIXELS:-64}"
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_REFERENCE="${OVERWRITE_REFERENCE:-1}"
  OVERWRITE_COMPATIBILITY="${OVERWRITE_COMPATIBILITY:-1}"
  SKIP_EXISTING_RECON="${SKIP_EXISTING_RECON:-0}"
else
  OVERWRITE_REFERENCE="${OVERWRITE_REFERENCE:-0}"
  OVERWRITE_COMPATIBILITY="${OVERWRITE_COMPATIBILITY:-0}"
  SKIP_EXISTING_RECON="${SKIP_EXISTING_RECON:-1}"
fi
MAX_SAM_CANDIDATES_PER_REF="${MAX_SAM_CANDIDATES_PER_REF:-30}"
MAX_SELECTED_PARTS_PER_REF="${MAX_SELECTED_PARTS_PER_REF:-0}"
UNIQUE_REFERENCE_PARTS="${UNIQUE_REFERENCE_PARTS:-1}"

# =========================
# SAM candidate generation
# =========================
# These are passed to the SAM/SAM2 candidate mask generator used by all
# matching branches.
#
# MODEL_CFG:
#   SAM2 config path kept for sam_utils compatibility.
#
# SAM_CHECKPOINT:
#   Classic SAM checkpoint. Default points to the current server path.
#
# SAM2_CHECKPOINT:
#   Optional SAM2 checkpoint. Leave empty when using classic SAM.
#
# SAM_MODEL_TYPE:
#   vit_h, vit_l, or vit_b.
#
# POINTS_PER_SIDE / POINTS_PER_BATCH:
#   Automatic-mask-generator sampling density and batch size.
#
# PRED_IOU_THRESH / STABILITY_SCORE_THRESH / MIN_MASK_REGION_AREA:
#   Candidate-mask quality filters.
#
# DUPLICATE_IOU_THRESHOLD:
#   IoU threshold for duplicate SAM mask removal.
#
# SAM_WORKERS:
#   SAM worker processes. GPU-heavy runs are usually safest with 1.
#
# SAM_GPU_IDS:
#   Comma-separated GPU ids for SAM workers, e.g. "0,1,2,3".
#
# SAM_PROCS_PER_GPU:
#   Number of SAM worker processes per GPU when SAM_GPU_IDS is set.
#
# SAM_WORKERS_PER_GPU:
#   Optional per-GPU SAM/preprocess worker counts aligned with SAM_GPU_IDS.
#   Examples:
#     SAM_GPU_IDS=0,1 SAM_WORKERS_PER_GPU=12,5
#     SAM_GPU_IDS=0,1 SAM_WORKERS_PER_GPU=6
#
# OVERWRITE_SEGMENTATION:
#   0 reuses shared pred_mask from preprocess; 1 lets a matching branch rerun
#   SAM into PRED_MASK_SUBDIR. The default stays 0 because preprocess owns this
#   stage.
#
# KEEP_INTERMEDIATE:
#   0 lets SAM6D/CNOS remove branch-local candidates. Shared pred_mask is always
#   kept by this runner.
MODEL_CFG="${MODEL_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
SEGMENT_ANYTHING_ROOT="${SEGMENT_ANYTHING_ROOT:-/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/related_works/segment-anything}"
export SEGMENT_ANYTHING_ROOT
SAM_CHECKPOINT="${SAM_CHECKPOINT:-$SEGMENT_ANYTHING_ROOT/sam_vit_h_4b8939.pth}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-48}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-64}"
PRED_IOU_THRESH="${PRED_IOU_THRESH:-0.8}"
STABILITY_SCORE_THRESH="${STABILITY_SCORE_THRESH:-0.9}"
MIN_MASK_REGION_AREA="${MIN_MASK_REGION_AREA:-50}"
DUPLICATE_IOU_THRESHOLD="${DUPLICATE_IOU_THRESHOLD:-0.5}"
OVERWRITE_SEGMENTATION="${OVERWRITE_SEGMENTATION:-0}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"

# Shared preprocess concurrency.
# VLM_WORKERS controls concurrent VLM mask checks per reference frame.
# SAM3D_WORKERS controls object-level concurrent preprocess/SAM3D workers.
# When SAM_GPU_IDS is set, preprocess workers are assigned to GPU slots from
# SAM_WORKERS_PER_GPU, or uniformly from SAM_PROCS_PER_GPU when it is empty.

# =========================
# Match settings
# =========================
# MATCH_MODEL_NAME:
#   DINO backbone used by direct_match/SAM6D/CNOS.
#
# MATCH_SCORE_THRESH:
#   Minimum SAM6D-style match score threshold.
#
# MATCH_TOPK_PER_FRAME:
#   Top-K candidate masks kept per CAD part before adaptive reranking.
#
# MATCH_WORKERS:
#   Object-level matching workers. DINO/CNOS matching is GPU-heavy; 1 is safer.
#
# SAM6D_*_WEIGHT:
#   POS    = foreground/template appearance similarity.
#   NEG    = mask-complement/background similarity.
#   NORMAL = normal/geometry auxiliary similarity.
#   EDGE   = edge/boundary auxiliary similarity.
#
# ours-no-complement forces NEG to 0.0.
MATCH_MODEL_NAME="${MATCH_MODEL_NAME:-dinov2_vitl14}"
MATCH_SCORE_THRESH="${MATCH_SCORE_THRESH:-0.25}"
MATCH_TOPK_PER_FRAME="${MATCH_TOPK_PER_FRAME:-3}"
SAM6D_POS_WEIGHT="${SAM6D_POS_WEIGHT:-0.25}"
SAM6D_NEG_WEIGHT="${SAM6D_NEG_WEIGHT:-0.25}"
SAM6D_NORMAL_WEIGHT="${SAM6D_NORMAL_WEIGHT:-0.25}"
SAM6D_EDGE_WEIGHT="${SAM6D_EDGE_WEIGHT:-0.25}"

# Adaptive score weights for our method.
#
# Only used by ours / ours-no-*:
#   ADAPTIVE_SAM6D_WEIGHT      initial SAM6D-style match score weight.
#   ADAPTIVE_RENDER_WEIGHT     reconstructed mesh multi-view render score.
#
# ours-no-render forces ADAPTIVE_RENDER_WEIGHT to 0.0.
ADAPTIVE_SAM6D_WEIGHT="${ADAPTIVE_SAM6D_WEIGHT:-0.5}"
ADAPTIVE_RENDER_WEIGHT="${ADAPTIVE_RENDER_WEIGHT:-0.15}"

object_args=(
  --object-source "$OBJECT_SOURCE"
  --start "$START"
)
if [[ -n "$END" ]]; then
  object_args+=(--end "$END")
fi
if [[ -n "$OBJECTS" ]]; then
  object_args+=(--objects "$OBJECTS")
fi

# Common SAM arguments passed to all segmentation/matching branches.
sam_args=(
  --model-cfg "$MODEL_CFG"
  --sam-checkpoint "$SAM_CHECKPOINT"
  --sam2-checkpoint "$SAM2_CHECKPOINT"
  --sam-model-type "$SAM_MODEL_TYPE"
  --points-per-side "$POINTS_PER_SIDE"
  --points-per-batch "$POINTS_PER_BATCH"
  --pred-iou-thresh "$PRED_IOU_THRESH"
  --stability-score-thresh "$STABILITY_SCORE_THRESH"
  --min-mask-region-area "$MIN_MASK_REGION_AREA"
  --duplicate-iou-threshold "$DUPLICATE_IOU_THRESHOLD"
  --num-workers "$SAM_WORKERS"
  --sam-gpu-ids "$SAM_GPU_IDS"
  --sam-procs-per-gpu "$SAM_PROCS_PER_GPU"
  --sam-workers-per-gpu "$SAM_WORKERS_PER_GPU"
  --task-chunksize "$TASK_CHUNKSIZE"
)

# Common DINO/SAM6D-style matching arguments.
match_weight_args=(
  --match-model-name "$MATCH_MODEL_NAME"
  --match-score-thresh "$MATCH_SCORE_THRESH"
  --match-topk-per-frame "$MATCH_TOPK_PER_FRAME"
  --sam6d-pos-weight "$SAM6D_POS_WEIGHT"
  --sam6d-normal-weight "$SAM6D_NORMAL_WEIGHT"
  --sam6d-edge-weight "$SAM6D_EDGE_WEIGHT"
  --min-visible-pixels "$MIN_VISIBLE_PIXELS"
)

run_preprocess() {
  # Shared preprocess is called before every non-preprocess mode. With
  # SKIP_EXISTING_RECON=1, existing SAM3D meshes are reused when the matching
  # reference masks are unchanged, so this is cheap after the first run and
  # protects each branch from missing chosen_part/selected_mesh.
  local cmd=(
    "$PYTHON_BIN" eccv/run_shared_preprocess.py
    --dataset-kind "$DATASET_KIND"
    --root "$ROOT"
    "${object_args[@]}"
    --reference-policy "$REFERENCE_POLICY"
    --reference-selector "$REFERENCE_SELECTOR"
    --pred-mask-subdir "$PRED_MASK_SUBDIR"
    --chosen-part-subdir "$CHOSEN_PART_SUBDIR"
    --selected-mesh-subdir "$SELECTED_MESH_SUBDIR"
    --reference-mask-subdir "$REFERENCE_MASK_SUBDIR"
    --ref-select-subdir "$REF_SELECT_SUBDIR"
    --min-visible-pixels "$MIN_VISIBLE_PIXELS"
    --model-cfg "$MODEL_CFG"
    --sam-checkpoint "$SAM_CHECKPOINT"
    --sam2-checkpoint "$SAM2_CHECKPOINT"
    --sam-model-type "$SAM_MODEL_TYPE"
    --points-per-side "$POINTS_PER_SIDE"
    --points-per-batch "$POINTS_PER_BATCH"
    --pred-iou-thresh "$PRED_IOU_THRESH"
    --stability-score-thresh "$STABILITY_SCORE_THRESH"
    --min-mask-region-area "$MIN_MASK_REGION_AREA"
    --duplicate-iou-threshold "$DUPLICATE_IOU_THRESHOLD"
    --max-sam-candidates-per-ref "$MAX_SAM_CANDIDATES_PER_REF"
    --max-selected-parts-per-ref "$MAX_SELECTED_PARTS_PER_REF"
    --sam-gpu-ids "$SAM_GPU_IDS"
    --sam-procs-per-gpu "$SAM_PROCS_PER_GPU"
    --sam-workers-per-gpu "$SAM_WORKERS_PER_GPU"
    --vlm-workers "$VLM_WORKERS"
    --sam3d-workers "$SAM3D_WORKERS"
  )
  if [[ -n "$GT_ROOT" ]]; then
    cmd+=(--gt-root "$GT_ROOT")
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ "$OVERWRITE_REFERENCE" == "1" ]]; then
    cmd+=(--overwrite-reference)
  fi
  if [[ "$OVERWRITE_COMPATIBILITY" == "1" ]]; then
    cmd+=(--overwrite-compatibility)
  fi
  if [[ "$SKIP_EXISTING_RECON" == "1" ]]; then
    cmd+=(--skip-existing-recon)
  else
    cmd+=(--no-skip-existing-recon)
  fi
  if [[ "$UNIQUE_REFERENCE_PARTS" == "1" ]]; then
    cmd+=(--unique-reference-parts)
  else
    cmd+=(--no-unique-reference-parts)
  fi
  echo "[RUN] preprocess dataset=$DATASET_KIND root=$ROOT pred=$PRED_MASK_SUBDIR chosen=$CHOSEN_PART_SUBDIR mesh=$SELECTED_MESH_SUBDIR"
  "${cmd[@]}"
}

run_sam6d() {
  # Baseline 1:
  #   SAM candidate masks + SAM6D-style direct DINO matching.
  #
  # Per-object outputs:
  #   pred_mask/     shared SAM candidates from preprocess
  #   sam6d/         final selected part masks and _meta JSON/debug files
  local cmd=(
    "$PYTHON_BIN" eccv/segmentation/direct_match_sam6d.py
    --data-root "$ROOT"
    "${object_args[@]}"
    "${sam_args[@]}"
    "${match_weight_args[@]}"
    --sam6d-neg-weight "$SAM6D_NEG_WEIGHT"
    --match-workers "$SAM6D_MATCH_WORKERS"
    --pred-mask-subdir "$PRED_MASK_SUBDIR"
    --match-out-subdir sam6d/_meta
    --matched-mask-subdir sam6d
    --output-json-name match_results_sam6d_style.json
    --no-use-gt-mask-for-match
    --no-partnet-layout
    --keep-intermediate
  )
  if [[ "$OVERWRITE_SEGMENTATION" == "1" ]]; then
    cmd+=(--overwrite-segmentation)
  fi
  echo "[RUN] SAM6D baseline"
  "${cmd[@]}"
}

run_cnos() {
  # Baseline 2:
  #   SAM candidate masks + CNOS-style DINO proposal/template matching.
  #
  # Per-object outputs:
  #   pred_mask/  shared SAM candidates from preprocess
  #   cnos/       final selected part masks and _meta JSON/debug files
  local cmd=(
    "$PYTHON_BIN" eccv/segmentation/direct_match_cnos.py
    --data-root "$ROOT"
    "${object_args[@]}"
    "${sam_args[@]}"
    --match-model-name "$MATCH_MODEL_NAME"
    --match-workers "$CNOS_MATCH_WORKERS"
    --pred-mask-subdir "$PRED_MASK_SUBDIR"
    --match-out-subdir cnos/_meta
    --matched-mask-subdir cnos
    --output-json-name match_results_cnos_style.json
    --no-use-gt-mask-for-match
    --no-partnet-layout
    --keep-intermediate
  )
  if [[ "$OVERWRITE_SEGMENTATION" == "1" ]]; then
    cmd+=(--overwrite-segmentation)
  fi
  echo "[RUN] CNOS baseline"
  "${cmd[@]}"
}

run_ours_variant() {
  # Our method and ablations:
  #   direct_match.py first writes SAM6D-style top-k candidate masks, then runs
  #   adaptive reranking with optional reconstructed-mesh render comparison.
  #
  # Arguments:
  #   name          run label for logs.
  #   out_subdir    canonical output folder.
  #   neg_weight    mask-complement/background similarity weight.
  #   render_weight reconstructed mesh render score weight.
  #
  # Per-object outputs:
  #   pred_mask/                    shared SAM candidates from preprocess
  #   <out_subdir>/_meta/           match/rerank JSON files
  #   <out_subdir>/_sam6d_candidates/ top-k pre-rerank masks
  #   <out_subdir>/<frame>/         final reranked masks
  local name="$1"
  local out_subdir="$2"
  local neg_weight="$3"
  local render_weight="$4"
  local match_workers="$5"
  local cmd=(
    "$PYTHON_BIN" eccv/segmentation/direct_match.py
    --data-root "$ROOT"
    "${object_args[@]}"
    "${sam_args[@]}"
    "${match_weight_args[@]}"
    --sam6d-neg-weight "$neg_weight"
    --pred-mask-subdir "$PRED_MASK_SUBDIR"
    --match-out-subdir "${out_subdir}/_meta"
    --matched-mask-subdir "${out_subdir}/_sam6d_candidates"
    --adaptive-reranked-mask-subdir "$out_subdir"
    --adaptive-match-result-json-name match_results_sam6d_style.json
    --adaptive-reranked-json-name match_results_adaptive_weight.json
    --adaptive-sam6d-weight "$ADAPTIVE_SAM6D_WEIGHT"
    --adaptive-render-weight "$render_weight"
    --match-workers "$match_workers"
  )
  if [[ "$OVERWRITE_SEGMENTATION" == "1" ]]; then
    cmd+=(--overwrite-segmentation)
  fi
  echo "[RUN] ours variant=$name out=$out_subdir neg_weight=$neg_weight render_weight=$render_weight match_workers=$match_workers"
  "${cmd[@]}"
}

usage() {
  cat <<'USAGE'
Usage:
  bash eccv/run_recon_match_experiments.sh preprocess [partnet|rgbd]
  bash eccv/run_recon_match_experiments.sh sam6d [partnet|rgbd]
  bash eccv/run_recon_match_experiments.sh cnos [partnet|rgbd]
  bash eccv/run_recon_match_experiments.sh ours [partnet|rgbd]
  bash eccv/run_recon_match_experiments.sh ours-no-complement [partnet|rgbd]
  bash eccv/run_recon_match_experiments.sh ours-no-render [partnet|rgbd]
  bash eccv/run_recon_match_experiments.sh all [partnet|rgbd]

Environment overrides:
  Global overwrite:
    OVERWRITE_ALL=0|1

  Dataset:
    ROOT, OBJECT_SOURCE, OBJECTS, START, END, GT_ROOT

  Shared preprocess:
    REFERENCE_SELECTOR=sam-vlm|gt-mask
    REFERENCE_POLICY=first|best
    PRED_MASK_SUBDIR, CHOSEN_PART_SUBDIR, SELECTED_MESH_SUBDIR
    REFERENCE_MASK_SUBDIR, REF_SELECT_SUBDIR
    MIN_VISIBLE_PIXELS
    OVERWRITE=0|1
    OVERWRITE_REFERENCE=0|1
    OVERWRITE_COMPATIBILITY=0|1
    SKIP_EXISTING_RECON=1|0
    MAX_SAM_CANDIDATES_PER_REF
    MAX_SELECTED_PARTS_PER_REF
    UNIQUE_REFERENCE_PARTS=1|0

  SAM candidate masks:
    MODEL_CFG, SAM_CHECKPOINT, SAM2_CHECKPOINT, SAM_MODEL_TYPE
    POINTS_PER_SIDE, POINTS_PER_BATCH
    PRED_IOU_THRESH, STABILITY_SCORE_THRESH, MIN_MASK_REGION_AREA
    DUPLICATE_IOU_THRESHOLD
    SAM_WORKERS, SAM_GPU_IDS, SAM_PROCS_PER_GPU, SAM_WORKERS_PER_GPU, TASK_CHUNKSIZE
    OVERWRITE_SEGMENTATION=0|1
    KEEP_INTERMEDIATE=0|1

  Shared preprocess concurrency:
    VLM_WORKERS
    SAM3D_WORKERS

  Matching:
    MATCH_MODEL_NAME, MATCH_SCORE_THRESH, MATCH_TOPK_PER_FRAME, MATCH_WORKERS
    SAM6D_MATCH_WORKERS, CNOS_MATCH_WORKERS
    OURS_MATCH_WORKERS, OURS_NO_COMPLEMENT_MATCH_WORKERS, OURS_NO_RENDER_MATCH_WORKERS
    SAM6D_POS_WEIGHT, SAM6D_NEG_WEIGHT, SAM6D_NORMAL_WEIGHT, SAM6D_EDGE_WEIGHT

  Our adaptive rerank:
    ADAPTIVE_SAM6D_WEIGHT, ADAPTIVE_RENDER_WEIGHT

Modes:
  preprocess:
    Only run PartNet raw-layout validation, reference SAM segmentation, VLM filtering,
    and SAM3D reconstruction.

  sam6d:
    Shared preprocess, then SAM segmentation + SAM6D-style matching.

  cnos:
    Shared preprocess, then SAM segmentation + CNOS-style matching.

  ours:
    Shared preprocess, then full direct_match with mask-complement score and
    reconstructed-mesh render score enabled.

  ours-no-complement:
    Same as ours, but sets mask-complement/background weight to 0.0.

  ours-no-render:
    Same as ours, but sets reconstructed mesh render weight to 0.0.

  all:
    Run preprocess and all five matching branches.

Examples:
  Run one object only:
    OBJECTS=bottle_3520 bash eccv/run_recon_match_experiments.sh ours partnet

  Rebuild shared pred_mask/chosen_part/selected_mesh:
    OVERWRITE=1 bash eccv/run_recon_match_experiments.sh preprocess partnet

  Force SAM3D reconstruction again:
    SKIP_EXISTING_RECON=0 bash eccv/run_recon_match_experiments.sh preprocess partnet

  Multi-GPU preprocess:
    SAM_GPU_IDS=0,1 SAM_PROCS_PER_GPU=1 SAM3D_WORKERS=2 bash eccv/run_recon_match_experiments.sh preprocess partnet

  Per-GPU preprocess/SAM workers:
    SAM_GPU_IDS=0,1 SAM_WORKERS_PER_GPU=12,5 bash eccv/run_recon_match_experiments.sh preprocess partnet

  Rebuild all preprocess and matching intermediates:
    OVERWRITE_ALL=1 bash eccv/run_recon_match_experiments.sh ours partnet
USAGE
}

case "$MODE" in
  preprocess)
    run_preprocess
    ;;
  sam6d)
    run_preprocess
    run_sam6d
    ;;
  cnos)
    run_preprocess
    run_cnos
    ;;
  ours)
    run_preprocess
    run_ours_variant ours full "$SAM6D_NEG_WEIGHT" "$ADAPTIVE_RENDER_WEIGHT" "$OURS_MATCH_WORKERS"
    ;;
  ours-no-complement)
    run_preprocess
    run_ours_variant ours_no_complement wo_dual 0.0 "$ADAPTIVE_RENDER_WEIGHT" "$OURS_NO_COMPLEMENT_MATCH_WORKERS"
    ;;
  ours-no-render)
    run_preprocess
    run_ours_variant ours_no_render wo_render "$SAM6D_NEG_WEIGHT" 0.0 "$OURS_NO_RENDER_MATCH_WORKERS"
    ;;
  all)
    run_preprocess
    run_sam6d
    run_cnos
    run_ours_variant ours full "$SAM6D_NEG_WEIGHT" "$ADAPTIVE_RENDER_WEIGHT" "$OURS_MATCH_WORKERS"
    run_ours_variant ours_no_complement wo_dual 0.0 "$ADAPTIVE_RENDER_WEIGHT" "$OURS_NO_COMPLEMENT_MATCH_WORKERS"
    run_ours_variant ours_no_render wo_render "$SAM6D_NEG_WEIGHT" 0.0 "$OURS_NO_RENDER_MATCH_WORKERS"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
