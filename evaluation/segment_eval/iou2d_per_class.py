import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def parse_class_name(obj_folder_name: str) -> str:
    # dataset object folder format: <cls_name>_<id>
    if "_" not in obj_folder_name:
        return obj_folder_name
    cls, maybe_id = obj_folder_name.rsplit("_", 1)
    if maybe_id.isdigit():
        return cls
    return obj_folder_name


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_ar_metrics(matched_ious, num_gt, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)
    if num_gt == 0:
        return {"AR": 0.0, "AR50": 0.0, "AR75": 0.0}

    recalls = []
    for t in thresholds:
        true_positives = np.sum(matched_ious >= t)
        recalls.append(true_positives / num_gt)
    return {
        "AR": float(np.mean(recalls)),
        "AR50": float(recalls[0]),
        "AR75": float(recalls[5]) if len(recalls) > 5 else 0.0,
    }


def evaluate_segmentation(gt_npz_path, pred_dir):
    if not os.path.exists(gt_npz_path):
        return None

    gt_data = np.load(gt_npz_path)
    gt_masks = gt_data["instance_segmentation"]
    unique_ids = np.unique(gt_masks)
    valid_ids = [i for i in unique_ids if i >= 0]
    num_gt = len(valid_ids)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")), key=natural_sort_key)
    pred_masks = []
    for f in pred_files:
        pred_masks.append(np.array(Image.open(f).convert("L")) > 0)

    num_pred = len(pred_masks)
    if num_gt == 0 or num_pred == 0:
        return {
            "mIoU_matched": 0.0,
            "mIoU_total": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "AR": 0.0,
            "AR50": 0.0,
            "AR75": 0.0,
            "num_gt": int(num_gt),
            "num_pred": int(num_pred),
        }

    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for i, g_id in enumerate(valid_ids):
        gt_inst_mask = gt_masks == g_id
        for j in range(num_pred):
            iou_matrix[i, j] = calculate_iou(gt_inst_mask, pred_masks[j])

    row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    mean_matched_iou = float(np.mean(matched_ious))
    total_miou = float(np.sum(matched_ious) / max(num_gt, num_pred))
    threshold = 0.5
    true_positives = int(np.sum(matched_ious > threshold))
    precision = float(true_positives / num_pred)
    recall = float(true_positives / num_gt)
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-6))
    ar_metrics = calculate_ar_metrics(matched_ious, num_gt)

    return {
        "mIoU_matched": mean_matched_iou,
        "mIoU_total": total_miou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "AR": ar_metrics["AR"],
        "AR50": ar_metrics["AR50"],
        "AR75": ar_metrics["AR75"],
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
    }


def collect_frame_items(root_dir: str, pred_mask_subdir: str):
    items = []
    for obj_name in sorted(os.listdir(root_dir), key=natural_sort_key):
        obj_dir = os.path.join(root_dir, obj_name)
        if not os.path.isdir(obj_dir):
            continue

        pred_masks_base = os.path.join(obj_dir, pred_mask_subdir)
        if not os.path.isdir(pred_masks_base):
            continue

        class_name = parse_class_name(obj_name)
        frame_dirs = [
            d for d in sorted(os.listdir(pred_masks_base), key=natural_sort_key)
            if os.path.isdir(os.path.join(pred_masks_base, d))
        ]
        for frame_name in frame_dirs:
            items.append(
                {
                    "class_name": class_name,
                    "obj_name": obj_name,
                    "pred_dir": os.path.join(pred_masks_base, frame_name),
                    "frame_name": frame_name,
                }
            )
    return items


def average_metrics(results):
    if not results:
        return None
    keys = ["mIoU_matched", "mIoU_total", "precision", "recall", "f1_score", "AR", "AR50", "AR75"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def format_block(title: str, metrics: dict, frame_count: int, object_count: int | None = None):
    out = []
    out.append(f"{title}")
    out.append("-" * 60)
    if object_count is not None:
        out.append(f"Objects:                {object_count}")
    out.append(f"Frames:                 {frame_count}")
    out.append(f"Average mIoU (Matched): {metrics['mIoU_matched']:.4f}")
    out.append(f"Average mIoU (Total):   {metrics['mIoU_total']:.4f}")
    out.append(f"Average Precision:      {metrics['precision']:.4f}")
    out.append(f"Average Recall:         {metrics['recall']:.4f}")
    out.append(f"Average F1-Score:       {metrics['f1_score']:.4f}")
    out.append(f"Average AR50:           {metrics['AR50']:.4f}")
    out.append(f"Average AR75:           {metrics['AR75']:.4f}")
    out.append(f"Average AR@[.50:.95]:   {metrics['AR']:.4f}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser("2D IoU evaluation grouped by object class")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs",
        help="Root containing object instance folders named <cls_name>_<id>.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/segmentation",
        help="Ground-truth npz directory.",
    )
    parser.add_argument(
        "--pred-mask-subdir",
        type=str,
        default="matched_pred_mask_direct_match_adaptive",
        help="Subfolder inside each object dir containing per-frame predicted masks.",
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        default="intra_iou2d_per_class.txt",
        help="Path to save human-readable per-class report.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="intra_iou2d_per_class.json",
        help="Path to save machine-readable per-class metrics.",
    )
    args = parser.parse_args()

    frame_items = collect_frame_items(args.root_dir, args.pred_mask_subdir)
    print(f"Collected frame dirs: {len(frame_items)}")
    if not frame_items:
        print("No frame dirs found. Please check root-dir/pred-mask-subdir.")
        return

    per_class_results = defaultdict(list)
    per_class_objects = defaultdict(set)
    global_results = []
    missing_gt = 0

    for item in frame_items:
        gt_path = os.path.join(args.gt_dir, item["frame_name"] + ".npz")
        res = evaluate_segmentation(gt_path, item["pred_dir"])
        if res is None:
            missing_gt += 1
            continue
        per_class_results[item["class_name"]].append(res)
        per_class_objects[item["class_name"]].add(item["obj_name"])
        global_results.append(res)

    if not global_results:
        print("No valid results aggregated (all GT missing or invalid).")
        return

    class_names = sorted(per_class_results.keys(), key=natural_sort_key)
    payload = {
        "root_dir": args.root_dir,
        "gt_dir": args.gt_dir,
        "pred_mask_subdir": args.pred_mask_subdir,
        "num_total_frame_dirs": len(frame_items),
        "num_missing_gt": int(missing_gt),
        "classes": {},
        "global": {
            "num_frames": len(global_results),
            "metrics": average_metrics(global_results),
        },
    }

    lines = []
    lines.append("=" * 70)
    lines.append("2D SEGMENTATION METRICS BY CLASS")
    lines.append("=" * 70)
    lines.append(f"Root Directory:         {args.root_dir}")
    lines.append(f"GT Directory:           {args.gt_dir}")
    lines.append(f"Pred Mask Subdir:       {args.pred_mask_subdir}")
    lines.append(f"Collected Frame Dirs:   {len(frame_items)}")
    lines.append(f"Missing GT Frames:      {missing_gt}")
    lines.append("=" * 70)
    lines.append("")

    for cls_name in class_names:
        cls_results = per_class_results[cls_name]
        cls_metrics = average_metrics(cls_results)
        cls_obj_count = len(per_class_objects[cls_name])
        payload["classes"][cls_name] = {
            "num_objects": cls_obj_count,
            "num_frames": len(cls_results),
            "metrics": cls_metrics,
        }
        lines.append(format_block(f"Class: {cls_name}", cls_metrics, len(cls_results), cls_obj_count))
        lines.append("")

    global_metrics = payload["global"]["metrics"]
    lines.append("=" * 70)
    lines.append(format_block("GLOBAL (ALL CLASSES)", global_metrics, len(global_results)))
    lines.append("=" * 70)
    report = "\n".join(lines) + "\n"

    with open(args.save_txt, "w", encoding="utf-8") as f:
        f.write(report)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(report)
    print(f"Saved report txt:  {args.save_txt}")
    print(f"Saved report json: {args.save_json}")


if __name__ == "__main__":
    main()
