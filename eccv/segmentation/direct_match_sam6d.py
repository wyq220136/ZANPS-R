import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np

SEGMENTATION_ROOT = os.path.dirname(os.path.abspath(__file__))
ECCV_ROOT = os.path.dirname(SEGMENTATION_ROOT)
if ECCV_ROOT not in sys.path:
    sys.path.insert(0, ECCV_ROOT)
if SEGMENTATION_ROOT not in sys.path:
    sys.path.insert(0, SEGMENTATION_ROOT)

try:
    from segmentation.sam_utils import create_mask_generator, generate_candidate_masks, has_valid_pred_mask, save_mask
except ImportError:
    from sam_utils import create_mask_generator, generate_candidate_masks, has_valid_pred_mask, save_mask

try:
    from segmentation.sam_parallel import run_sam_tasks
except ImportError:
    from sam_parallel import run_sam_tasks

try:
    from segmentation.dino_match.new_match import run_matching_for_object
except ImportError:
    from dino_match.new_match import run_matching_for_object

try:
    from segmentation.direct_match import DIRECT_MATCH_SAMPLE_LIST
except Exception:
    DIRECT_MATCH_SAMPLE_LIST = [
        "Box_100189",
        "Bucket_100438",
        "CoffeeMachine_103074",
        "Dishwasher_12530",
        "Keyboard_12738",
        "Microwave_7263",
        "Printer_103972",
        "Remote_101028",
        "StorageFurniture_45134",
        "StorageFurniture_45779",
        "StorageFurniture_45910",
        "Toaster_103469",
        "Toilet_103234",
        "WashingMachine_103528",
    ]


DEFAULT_DATA_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train/test"
DEFAULT_MODEL_CFG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SEGMENT_ANYTHING_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/related_works/segment-anything"
DEFAULT_SAM_CHECKPOINT_PATH = os.path.join(DEFAULT_SEGMENT_ANYTHING_ROOT, "sam_vit_h_4b8939.pth")
DEFAULT_SAM2_CHECKPOINT_PATH = ""
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

_WORKER_MASK_GENERATOR = None
_WORKER_CFG = None


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def _parse_part_id_from_name(name, fallback=0):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else int(fallback)


def _find_frame_file(root, frame_id, exts=IMAGE_EXTS):
    if not os.path.isdir(root):
        return ""
    for ext in exts:
        path = os.path.join(root, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return ""


def _find_external_object_mask_path(obj_dir, frame_id):
    for subdir in ("mask", "object_masks", "object_mask"):
        path = _find_frame_file(os.path.join(obj_dir, subdir), frame_id)
        if path:
            return path
    return ""


def _partnet_part_dirs(obj_dir):
    masks_root = os.path.join(obj_dir, "masks")
    if not os.path.isdir(masks_root):
        return []
    names = [
        d
        for d in os.listdir(masks_root)
        if os.path.isdir(os.path.join(masks_root, d))
    ]
    return [(name, os.path.join(masks_root, name)) for name in sorted(names, key=natural_sort_key)]


def _copy_mask_image(src, dst, overwrite=False):
    if (not overwrite) and os.path.exists(dst):
        return False
    mask = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if mask is None or int(np.count_nonzero(mask > 0)) == 0:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.lower().endswith(".png"):
        shutil.copy2(src, dst)
    else:
        cv2.imwrite(dst, mask)
    return True


def _prepare_partnet_gt_masks_for_object(obj_dir, overwrite=False):
    part_dirs = _partnet_part_dirs(obj_dir)
    if not part_dirs:
        return {"parts": 0, "frames": 0, "masks": 0}

    gt_root = os.path.join(obj_dir, "gt_mask")
    copied = 0
    frames = set()
    for part_idx, (part_name, part_dir) in enumerate(part_dirs):
        part_id = _parse_part_id_from_name(part_name, part_idx)
        mask_files = [
            f
            for f in os.listdir(part_dir)
            if f.lower().endswith(IMAGE_EXTS)
        ]
        for mask_name in sorted(mask_files, key=natural_sort_key):
            frame_id = os.path.splitext(mask_name)[0]
            src = os.path.join(part_dir, mask_name)
            dst = os.path.join(gt_root, frame_id, f"mask_{part_id}.png")
            copied += int(_copy_mask_image(src, dst, overwrite=overwrite))
            frames.add(frame_id)
    return {"parts": len(part_dirs), "frames": len(frames), "masks": copied}


def _prepare_partnet_object_masks_for_object(obj_dir, overwrite=False):
    if os.path.isdir(os.path.join(obj_dir, "mask")) and not overwrite:
        return 0
    src_root = ""
    for name in ("object_masks", "object_mask"):
        cand = os.path.join(obj_dir, name)
        if os.path.isdir(cand):
            src_root = cand
            break
    if not src_root:
        return 0
    dst_root = os.path.join(obj_dir, "mask")
    copied = 0
    for name in sorted(os.listdir(src_root), key=natural_sort_key):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        copied += int(_copy_mask_image(os.path.join(src_root, name), os.path.join(dst_root, name), overwrite=overwrite))
    return copied


def _prepare_partnet_layout(args, objects):
    if not bool(args.partnet_layout):
        return
    total_parts = 0
    total_frames = 0
    total_masks = 0
    total_object_masks = 0
    for obj_name in objects:
        obj_dir = os.path.join(args.data_root, obj_name)
        info = _prepare_partnet_gt_masks_for_object(obj_dir, overwrite=bool(args.overwrite_partnet_layout))
        total_parts += int(info["parts"])
        total_frames += int(info["frames"])
        total_masks += int(info["masks"])
        total_object_masks += _prepare_partnet_object_masks_for_object(
            obj_dir,
            overwrite=bool(args.overwrite_partnet_layout),
        )
    if total_parts > 0:
        print(
            f"[PARTNET] prepared objects={len(objects)} parts={total_parts} "
            f"frames={total_frames} gt_masks_updated={total_masks} object_masks_updated={total_object_masks}"
        )


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
    _, _, candidates = generate_candidate_masks(
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
            [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))],
            key=natural_sort_key,
        )
    else:
        allowed = set(DIRECT_MATCH_SAMPLE_LIST)
        objects = sorted(
            [
                d
                for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d)) and d in allowed
            ],
            key=natural_sort_key,
        )
    if args.objects:
        keep = {x.strip() for x in args.objects.split(",") if x.strip()}
        objects = [o for o in objects if o in keep]
    end = args.end if args.end is not None else len(objects)
    return objects[args.start:end]


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
            ext_mask = _find_external_object_mask_path(obj_dir, frame_id)
            tasks.append((obj_name, frame_id, image_path, ext_mask, pred_frame_dir))
    return tasks


def _copy_gt_masks_for_match(args, objects):
    total_frames = 0
    total_masks = 0
    for obj_name in objects:
        obj_dir = os.path.join(args.data_root, obj_name)
        gt_root = os.path.join(obj_dir, "gt_mask")
        pred_root = os.path.join(obj_dir, args.pred_mask_subdir)
        if not os.path.isdir(gt_root):
            print(f"[SKIP-GT-MASK] {obj_name}: gt_mask not found")
            continue
        frame_dirs = sorted(
            [d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))],
            key=natural_sort_key,
        )
        for frame_id in frame_dirs:
            src_dir = os.path.join(gt_root, frame_id)
            pred_frame_dir = os.path.join(pred_root, frame_id)
            if (not args.overwrite_segmentation) and has_valid_pred_mask(pred_frame_dir):
                continue
            mask_files = sorted(
                [f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                key=natural_sort_key,
            )
            if not mask_files:
                continue
            os.makedirs(pred_frame_dir, exist_ok=True)
            for f in os.listdir(pred_frame_dir):
                if f.startswith("mask_") and f.lower().endswith(".png"):
                    fp = os.path.join(pred_frame_dir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
            copied = 0
            for idx, name in enumerate(mask_files):
                src = os.path.join(src_dir, name)
                mask = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if mask is None or int(np.count_nonzero(mask > 0)) == 0:
                    continue
                dst = os.path.join(pred_frame_dir, f"mask_{idx}.png")
                if name.lower().endswith(".png"):
                    shutil.copy2(src, dst)
                else:
                    cv2.imwrite(dst, mask)
                copied += 1
            total_frames += int(copied > 0)
            total_masks += copied
    print(f"[GT-MASK] copied frames={total_frames} masks={total_masks} into pred_mask_subdir={args.pred_mask_subdir}")


def _cleanup_match_dir(out_dir, keep_json_name):
    if not os.path.isdir(out_dir):
        return
    keep_json_name = os.path.basename(keep_json_name)
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if not os.path.isfile(p):
            continue
        if name == keep_json_name:
            continue
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".txt", ".npy", ".npz", ".log")):
            os.remove(p)


def _cleanup_object_intermediate(obj_dir, args):
    if args.keep_intermediate:
        return
    if os.path.basename(str(args.pred_mask_subdir).rstrip("/\\")) == "pred_mask":
        return
    pred_root = os.path.join(obj_dir, args.pred_mask_subdir)
    if os.path.isdir(pred_root):
        shutil.rmtree(pred_root)


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
        finalize_one_to_one=True,
        sam6d_pos_weight=args.sam6d_pos_weight,
        sam6d_neg_weight=args.sam6d_neg_weight,
        sam6d_normal_weight=args.sam6d_normal_weight,
        sam6d_edge_weight=args.sam6d_edge_weight,
        min_visible_pixels=args.min_visible_pixels,
    )
    default_json = os.path.join(out_dir, "match_results_sam6d_style.json")
    if args.output_json_name and args.output_json_name != "match_results_sam6d_style.json":
        target_json = os.path.join(out_dir, args.output_json_name)
        if os.path.exists(default_json):
            shutil.copy2(default_json, target_json)
    else:
        target_json = default_json

    _cleanup_match_dir(out_dir, target_json)
    _cleanup_object_intermediate(obj_dir, args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="SAM segmentation -> SAM6D-style matching (final one-to-one). Save final results only."
    )
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
    parser.add_argument("--pred-mask-subdir", type=str, default="pred_mask_sam6d")
    parser.add_argument("--overwrite-segmentation", action="store_true")
    parser.add_argument(
        "--partnet-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-adapt raw PartNet layout: masks/<part>/<frame>.png and object_masks/object_mask.",
    )
    parser.add_argument(
        "--overwrite-partnet-layout",
        action="store_true",
        help="Rewrite generated gt_mask/mask compatibility files from raw PartNet folders.",
    )
    parser.add_argument(
        "--use-gt-mask-for-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use obj/gt_mask frame masks directly as candidate masks and skip SAM generation.",
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

    parser.add_argument("--match-out-subdir", type=str, default="match_vis_sam6d")
    parser.add_argument("--matched-mask-subdir", type=str, default="matched_pred_mask_sam6d")
    parser.add_argument("--output-json-name", type=str, default="match_results_sam6d_style.json")
    parser.add_argument("--match-model-name", type=str, default="dinov2_vitl14")
    parser.add_argument("--match-score-thresh", type=float, default=0.25)
    parser.add_argument("--match-topk-per-frame", type=int, default=3)
    parser.add_argument("--match-workers", type=int, default=6, help="Object-level match workers")
    parser.add_argument("--sam6d-pos-weight", type=float, default=0.25)
    parser.add_argument("--sam6d-neg-weight", type=float, default=0.25)
    parser.add_argument("--sam6d-normal-weight", type=float, default=0.25)
    parser.add_argument("--sam6d-edge-weight", type=float, default=0.25)
    parser.add_argument("--min-visible-pixels", type=int, default=30)
    parser.add_argument("--skip-match", action="store_true", help="Only run SAM segmentation and skip matching")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate candidate masks (pred-mask-subdir). Default only keeps final outputs.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data root not found: {args.data_root}")

    objects = _collect_objects(args)
    print(f"[INFO] objects={len(objects)}")
    _prepare_partnet_layout(args, objects)

    if args.use_gt_mask_for_match:
        print("[STAGE-1/GT-MASK] using gt_mask as candidate masks")
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
        print("[DONE] segmentation finished (match skipped).")
        return

    print(f"[STAGE-2/SAM6D-MATCH] objects={len(objects)}")
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

# python segmentation/direct_match_sam6d.py --data-root <...>/objs --object-source all
# python segmentation/direct_match_cnos.py --data-root <...>/objs --object-source all
