import os
import re
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import sys
sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv")
try:
    from segmentation.sam_utils import *
    # from sam2_test import create_mask_generator, generate_candidate_masks
except ImportError:
    from sam_utils import *
    # from segmentation.sam2_test import create_mask_generator, generate_candidate_masks

try:
    from segmentation.sam_parallel import run_sam_tasks
except ImportError:
    from sam_parallel import run_sam_tasks


from dino_match.new_match import run_matching_for_object
from dino_match.adaptive_weight import run_adaptive_rerank_for_object

try:
    from dino_match.new_match import sample_list as DIRECT_MATCH_SAMPLE_LIST
except Exception:
    DIRECT_MATCH_SAMPLE_LIST = [
        "Box_100189", "Bucket_100438", "CoffeeMachine_103074", "Dishwasher_12530",
        "Keyboard_12738", "Microwave_7263", "Printer_103972",
        "Remote_101028", "StorageFurniture_45134", "StorageFurniture_45779",
        "StorageFurniture_45910", "Toaster_103469", "Toilet_103234", "WashingMachine_103528",
    ]


# ==========================
# Global Config (edit here)
# ==========================
DEFAULT_DATA_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs"
DEFAULT_MODEL_CFG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SEGMENT_ANYTHING_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/related_works/segment-anything"
DEFAULT_SAM_CHECKPOINT_PATH = os.path.join(DEFAULT_SEGMENT_ANYTHING_ROOT, "sam_vit_h_4b8939.pth")
DEFAULT_SAM2_CHECKPOINT_PATH = ""
# ==========================


_WORKER_MASK_GENERATOR = None
_WORKER_CFG = None


def _build_worker_cfg(args):
    return {
        "model_cfg": args.model_cfg,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam_checkpoint": args.sam_checkpoint,
        "sam_model_type": args.sam_model_type,
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "min_mask_region_area": args.min_mask_region_area,
        "duplicate_iou_threshold": args.duplicate_iou_threshold,
    }
    

def _init_sam_worker(worker_cfg):
    global _WORKER_MASK_GENERATOR, _WORKER_CFG
    _WORKER_CFG = worker_cfg
    _WORKER_MASK_GENERATOR = create_mask_generator(
        model_cfg=worker_cfg["model_cfg"],
        sam2_checkpoint=worker_cfg["sam2_checkpoint"],
        sam_checkpoint=worker_cfg["sam_checkpoint"],
        sam_model_type=worker_cfg["sam_model_type"],
        points_per_side=worker_cfg["points_per_side"],
        points_per_batch=worker_cfg["points_per_batch"],
        pred_iou_thresh=worker_cfg["pred_iou_thresh"],
        stability_score_thresh=worker_cfg["stability_score_thresh"],
        min_mask_region_area=worker_cfg["min_mask_region_area"],
    )


def _run_sam_frame(image_path, ext_mask_path, pred_frame_dir, cfg, mask_generator):
    image_rgb, raw_masks, candidates = generate_candidate_masks(
        image_path=image_path,
        mask_path=ext_mask_path if os.path.exists(ext_mask_path) else None,
        model_cfg=cfg["model_cfg"],
        sam2_checkpoint=cfg["sam2_checkpoint"],
        sam_checkpoint=cfg["sam_checkpoint"],
        sam_model_type=cfg["sam_model_type"],
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=cfg["pred_iou_thresh"],
        stability_score_thresh=cfg["stability_score_thresh"],
        min_mask_region_area=cfg["min_mask_region_area"],
        iou_threshold=cfg["duplicate_iou_threshold"],
        mask_generator=mask_generator,
    )
    _ = image_rgb
    _ = raw_masks

    os.makedirs(pred_frame_dir, exist_ok=True)
    for f in os.listdir(pred_frame_dir):
        if f.startswith("mask_") and f.lower().endswith(".png"):
            fp = os.path.join(pred_frame_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    for i, ann in enumerate(candidates):
        save_mask(ann["segmentation"].astype(bool), os.path.join(pred_frame_dir, f"mask_{i}.png"))
    return len(candidates)


def _sam_worker(task):
    obj_name, frame_id, image_path, ext_mask_path, pred_frame_dir = task
    num_cand = _run_sam_frame(
        image_path=image_path,
        ext_mask_path=ext_mask_path,
        pred_frame_dir=pred_frame_dir,
        cfg=_WORKER_CFG,
        mask_generator=_WORKER_MASK_GENERATOR,
    )
    return obj_name, frame_id, num_cand


def _collect_objects(args):
    if args.object_source == "all":
        objects = sorted(
            [
                d for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d))
            ],
            key=natural_sort_key,
        )
    else:
        allowed = set(DIRECT_MATCH_SAMPLE_LIST)
        objects = sorted(
            [
                d for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d)) and d in allowed
            ],
            key=natural_sort_key,
        )
    if args.objects:
        keep = {x.strip() for x in args.objects.split(",") if x.strip()}
        objects = [o for o in objects if o in keep]
    end = args.end if args.end is not None else len(objects)
    return objects[args.start:end]


def _parse_part_id_from_name(name, fallback=0):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else int(fallback)


def _collect_sam_tasks(args, objects):
    tasks = []
    for obj_name in objects:
        obj_dir = os.path.join(args.data_root, obj_name)
        rgb_dir = os.path.join(obj_dir, "rgb")
        pred_root = os.path.join(obj_dir, args.pred_mask_subdir)
        if not os.path.isdir(rgb_dir):
            print(f"[SKIP] {obj_name}: rgb not found")
            continue
        rgb_files = sorted(
            [f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=natural_sort_key,
        )
        for rgb_name in rgb_files:
            frame_id = os.path.splitext(rgb_name)[0]
            pred_frame_dir = os.path.join(pred_root, frame_id)
            if (not args.overwrite_segmentation) and has_valid_pred_mask(pred_frame_dir):
                continue
            image_path = os.path.join(rgb_dir, rgb_name)
            ext_mask = ""
            for subdir in ("object_masks", "object_mask"):
                for ext in (".png", ".jpg", ".jpeg"):
                    cand = os.path.join(obj_dir, subdir, f"{frame_id}{ext}")
                    if os.path.exists(cand):
                        ext_mask = cand
                        break
                if ext_mask:
                    break
            tasks.append((obj_name, frame_id, image_path, ext_mask, pred_frame_dir))
    return tasks


def _copy_gt_masks_for_match(args, objects):
    total_frames = 0
    total_masks = 0
    for obj_name in objects:
        obj_dir = os.path.join(args.data_root, obj_name)
        masks_root = os.path.join(obj_dir, "masks")
        pred_root = os.path.join(obj_dir, args.pred_mask_subdir)
        if not os.path.isdir(masks_root):
            print(f"[SKIP-RAW-MASKS] {obj_name}: raw masks not found")
            continue
        frame_to_masks = {}
        part_dirs = [
            d for d in os.listdir(masks_root)
            if os.path.isdir(os.path.join(masks_root, d))
        ]
        for part_idx, part_name in enumerate(sorted(part_dirs, key=natural_sort_key)):
            part_dir = os.path.join(masks_root, part_name)
            part_id = _parse_part_id_from_name(part_name, part_idx)
            for name in sorted(os.listdir(part_dir), key=natural_sort_key):
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                frame_id = os.path.splitext(name)[0]
                frame_to_masks.setdefault(frame_id, []).append((part_id, os.path.join(part_dir, name)))
        for frame_id in sorted(frame_to_masks, key=natural_sort_key):
            pred_frame_dir = os.path.join(pred_root, frame_id)
            if (not args.overwrite_segmentation) and has_valid_pred_mask(pred_frame_dir):
                continue
            os.makedirs(pred_frame_dir, exist_ok=True)
            for f in os.listdir(pred_frame_dir):
                if f.startswith("mask_") and f.lower().endswith(".png"):
                    fp = os.path.join(pred_frame_dir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
            copied = 0
            for idx, (part_id, src) in enumerate(frame_to_masks[frame_id]):
                mask = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if mask is None or int(np.count_nonzero(mask > 0)) == 0:
                    continue
                dst = os.path.join(pred_frame_dir, f"mask_{part_id}.png")
                if src.lower().endswith(".png"):
                    shutil.copy2(src, dst)
                else:
                    cv2.imwrite(dst, mask)
                copied += 1
            total_frames += int(copied > 0)
            total_masks += copied
    print(f"[RAW-MASKS] copied frames={total_frames} masks={total_masks} into pred_mask_subdir={args.pred_mask_subdir}")


def _run_match_for_object(obj_dir, args):
    out_dir = os.path.join(obj_dir, args.match_out_subdir)
    run_matching_for_object(
        obj_dir=obj_dir,
        out_dir=out_dir,
        model_name=args.match_model_name,
        score_thresh=args.match_score_thresh,
        topk_per_frame=args.match_topk_per_frame,
        pred_mask_subdir=args.pred_mask_subdir,
        matched_mask_subdir=args.matched_mask_subdir,
        finalize_one_to_one=(args.skip_adaptive_weight and not args.defer_adaptive_weight),
        sam6d_pos_weight=args.sam6d_pos_weight,
        sam6d_neg_weight=args.sam6d_neg_weight,
        sam6d_normal_weight=args.sam6d_normal_weight,
        sam6d_edge_weight=args.sam6d_edge_weight,
        min_visible_pixels=args.min_visible_pixels,
    )
    if (not args.skip_adaptive_weight) and (not args.defer_adaptive_weight):
        run_adaptive_rerank_for_object(
            obj_dir=obj_dir,
            match_out_dir=out_dir,
            reranked_mask_subdir=args.adaptive_reranked_mask_subdir,
            match_result_json_name=args.adaptive_match_result_json_name,
            reranked_json_name=args.adaptive_reranked_json_name,
            topk_per_cad=args.match_topk_per_frame,
            sam6d_weight=args.adaptive_sam6d_weight,
            render_weight=args.adaptive_render_weight,
            render_model_name=args.match_model_name,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="SAM segmentation -> direct DINOv2 match (no LLM check)")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Root containing object folders")
    parser.add_argument("--object-source", type=str, default="sample", choices=["sample", "all"], help="Object selection source")
    parser.add_argument("--objects", type=str, default="", help="Optional comma-separated object names")
    parser.add_argument("--start", type=int, default=0, help="Object start index (sorted)")
    parser.add_argument("--end", type=int, default=None, help="Object end index (exclusive)")

    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG_PATH)
    parser.add_argument("--sam-checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT_PATH, help="Path to SAM checkpoint")
    parser.add_argument("--sam2-checkpoint", type=str, default=DEFAULT_SAM2_CHECKPOINT_PATH, help="Compatibility only")
    parser.add_argument("--sam-model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--points-per-side", type=int, default=48)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.9)
    parser.add_argument("--min-mask-region-area", type=int, default=50)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--pred-mask-subdir", type=str, default="pred_mask_direct_match")
    parser.add_argument("--overwrite-segmentation", action="store_true")
    parser.add_argument(
        "--use-gt-mask-for-match",
        action="store_true",
        help="Use raw masks/<part>/<frame> directly as candidate masks for CAD-mask matching, skipping SAM.",
    )
    parser.add_argument("--num-workers", type=int, default=3, help="SAM stage process workers")
    parser.add_argument("--sam-gpu-ids", type=str, default="", help="Comma-separated GPU ids for SAM workers, e.g. 0,1.")
    parser.add_argument("--sam-procs-per-gpu", type=int, default=1, help="SAM worker processes per GPU when --sam-gpu-ids is set.")
    parser.add_argument(
        "--sam-workers-per-gpu",
        type=str,
        default="",
        help="Optional comma-separated SAM worker counts aligned with --sam-gpu-ids, e.g. 12,5.",
    )
    parser.add_argument("--task-chunksize", type=int, default=1, help="SAM stage ProcessPool chunksize")

    parser.add_argument("--match-out-subdir", type=str, default="match_vis_direct_match")
    parser.add_argument("--matched-mask-subdir", type=str, default="matched_pred_mask_direct_match")
    parser.add_argument("--match-model-name", type=str, default="dinov2_vitl14")
    parser.add_argument("--match-score-thresh", type=float, default=0.25)
    parser.add_argument(
        "--match-topk-per-frame",
        type=int,
        default=3,
        help="Top-K masks per CAD from SAM6D ranking used by the adaptive rerank stage.",
    )
    parser.add_argument("--match-workers", type=int, default=6, help="Object-level match workers")
    parser.add_argument("--sam6d-pos-weight", type=float, default=0.25)
    parser.add_argument("--sam6d-neg-weight", type=float, default=0.25)
    parser.add_argument("--sam6d-normal-weight", type=float, default=0.25)
    parser.add_argument("--sam6d-edge-weight", type=float, default=0.25)
    parser.add_argument("--min-visible-pixels", type=int, default=30, help="GT visibility threshold for CAD filtering in direct_match.")
    parser.add_argument("--skip-match", action="store_true", help="Only run SAM segmentation and skip match")
    parser.add_argument("--skip-adaptive-weight", action="store_true", help="Skip adaptive rerank stage")
    parser.add_argument("--defer-adaptive-weight", action="store_true", help="Write SAM6D candidates and let the caller run adaptive rerank.")
    parser.add_argument("--adaptive-reranked-mask-subdir", type=str, default="matched_pred_mask_direct_match_adaptive")
    parser.add_argument("--adaptive-match-result-json-name", type=str, default="match_results_sam6d_style.json")
    parser.add_argument("--adaptive-reranked-json-name", type=str, default="match_results_adaptive_weight.json")
    parser.add_argument("--adaptive-sam6d-weight", type=float, default=0.5)
    parser.add_argument("--adaptive-render-weight", type=float, default=0.15)
    return parser


def main():
    args = build_parser().parse_args()
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data root not found: {args.data_root}")

    objects = _collect_objects(args)
    print(f"[INFO] objects={len(objects)}")

    # Stage 1: prepare candidate masks.
    if args.use_gt_mask_for_match:
        print("[STAGE-1/RAW-MASKS] using raw masks/<part>/<frame> as direct_match candidate masks")
        _copy_gt_masks_for_match(args, objects)
        sam_tasks = []
    else:
        sam_tasks = _collect_sam_tasks(args, objects)
        print(f"[STAGE-1/SAM] frames_to_process={len(sam_tasks)}")
    if sam_tasks:
        worker_cfg = _build_worker_cfg(args)
        for obj_name, frame_id, num_cand, gpu_id in run_sam_tasks(
            sam_tasks,
            worker_cfg=worker_cfg,
            num_workers=args.num_workers,
            task_chunksize=args.task_chunksize,
            sam_gpu_ids=args.sam_gpu_ids,
            sam_procs_per_gpu=args.sam_procs_per_gpu,
            sam_workers_per_gpu=args.sam_workers_per_gpu,
        ):
            gpu_txt = f" gpu={gpu_id}" if gpu_id else ""
            print(f"[SAM DONE] {obj_name}/{frame_id}: candidates={num_cand}{gpu_txt}")

    if args.skip_match:
        print("[DONE] SAM segmentation finished (match skipped).")
        return

    # Stage 2: DINOv2 matching (parallel by object)
    print(f"[STAGE-2/MATCH] objects={len(objects)}")
    obj_dirs = [os.path.join(args.data_root, o) for o in objects]
    if args.match_workers > 1:
        def _match_task(obj_dir):
            _run_match_for_object(obj_dir, args)
            return os.path.basename(obj_dir.rstrip("/\\"))

        with ThreadPoolExecutor(max_workers=args.match_workers) as executor:
            for obj_name in executor.map(_match_task, obj_dirs):
                print(f"[MATCH DONE] {obj_name}")
    else:
        for obj_dir in obj_dirs:
            obj_name = os.path.basename(obj_dir.rstrip("/\\"))
            _run_match_for_object(obj_dir, args)
            print(f"[MATCH DONE] {obj_name}")

    print("[DONE]")


if __name__ == "__main__":
    main()
