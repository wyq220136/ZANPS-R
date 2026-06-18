import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


DEFAULT_WORK_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_val_work"


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", str(s))]


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1.shape != mask2.shape:
        mask2 = np.asarray(Image.fromarray(mask2.astype(np.uint8)).resize((mask1.shape[1], mask1.shape[0]), Image.NEAREST)) > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def calculate_ar_metrics(matched_ious, num_gt, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)
    if num_gt == 0:
        return {"AR": 0.0, "AR50": 0.0, "AR75": 0.0}
    recalls = []
    for t in thresholds:
        recalls.append(float(np.sum(matched_ious >= t) / num_gt))
    return {
        "AR": float(np.mean(recalls)),
        "AR50": float(recalls[0]),
        "AR75": float(recalls[5]) if len(recalls) > 5 else 0.0,
    }


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def list_mask_pngs(mask_dir: Path) -> List[Path]:
    if not mask_dir.is_dir():
        return []
    return sorted(
        [p for p in mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: natural_sort_key(p.name),
    )


def evaluate_frame(gt_frame_dir: Path, pred_frame_dir: Path) -> Optional[dict]:
    if not gt_frame_dir.is_dir():
        return None
    gt_files = list_mask_pngs(gt_frame_dir)
    pred_files = list_mask_pngs(pred_frame_dir)
    gt_masks = [read_mask(p) for p in gt_files]
    pred_masks = [read_mask(p) for p in pred_files]
    num_gt = len(gt_masks)
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
            "matched_ious": [],
        }

    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = calculate_iou(gt_mask, pred_mask)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]
    true_positives = int(np.sum(matched_ious > 0.5))
    precision = float(true_positives / num_pred) if num_pred > 0 else 0.0
    recall = float(true_positives / num_gt) if num_gt > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-6))
    ar = calculate_ar_metrics(matched_ious, num_gt)
    return {
        "mIoU_matched": float(np.mean(matched_ious)) if len(matched_ious) else 0.0,
        "mIoU_total": float(np.sum(matched_ious) / max(num_gt, num_pred)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "AR": ar["AR"],
        "AR50": ar["AR50"],
        "AR75": ar["AR75"],
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
        "matched_ious": [float(x) for x in matched_ious.tolist()],
    }


def parse_class_name(obj_name: str) -> str:
    if "_" not in obj_name:
        return obj_name
    cls, maybe_id = obj_name.rsplit("_", 1)
    return cls if maybe_id.isdigit() else obj_name


def resolve_objects_root(root: Path) -> Path:
    if (root / "objs").is_dir():
        return root / "objs"
    return root


def collect_objects(objects_root: Path, objects_arg: str, start: int, end: int) -> List[str]:
    if objects_arg.strip():
        names = [x.strip() for x in objects_arg.split(",") if x.strip()]
    else:
        names = sorted(
            [
                p.name
                for p in objects_root.iterdir()
                if p.is_dir() and not p.name.startswith("_") and (p / "gt_mask").is_dir()
            ],
            key=natural_sort_key,
        )
    names = [n for n in names if (objects_root / n).is_dir()]
    start = max(0, int(start))
    end = int(end)
    return names[start:] if end < 0 else names[start:end]


def collect_frame_items(objects_root: Path, object_names: List[str], pred_mask_subdir: str) -> List[dict]:
    items = []
    for obj_name in object_names:
        obj_dir = objects_root / obj_name
        pred_root = obj_dir / pred_mask_subdir
        gt_root = obj_dir / "gt_mask"
        if not pred_root.is_dir() or not gt_root.is_dir():
            continue
        for frame_dir in sorted([p for p in pred_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name)):
            items.append(
                {
                    "object": obj_name,
                    "class_name": parse_class_name(obj_name),
                    "frame_name": frame_dir.name,
                    "pred_dir": str(frame_dir),
                    "gt_dir": str(gt_root / frame_dir.name),
                }
            )
    return items


def average_metrics(results: List[dict]) -> Optional[dict]:
    if not results:
        return None
    keys = ["mIoU_matched", "mIoU_total", "precision", "recall", "f1_score", "AR", "AR50", "AR75"]
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def format_metrics(metrics: dict, frame_count: int) -> str:
    return "\n".join(
        [
            f"FINAL GLOBAL RESULTS ({frame_count} frames)",
            "-" * 40,
            f"Average mIoU (Matched): {metrics['mIoU_matched']:.4f}",
            f"Average mIoU (Total):   {metrics['mIoU_total']:.4f}",
            f"Average Precision:      {metrics['precision']:.4f}",
            f"Average Recall:         {metrics['recall']:.4f}",
            f"Average F1-Score:       {metrics['f1_score']:.4f}",
            f"Average AR50:           {metrics['AR50']:.4f}",
            f"Average AR75:           {metrics['AR75']:.4f}",
            f"Average AR@[.50:.95]:   {metrics['AR']:.4f}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser("2D IoU evaluation for dataset_train_val direct_match outputs")
    parser.add_argument("--work-root", type=str, default=DEFAULT_WORK_ROOT)
    parser.add_argument(
        "--pred-mask-subdir",
        type=str,
        default="matched_pred_mask_direct_match_dataset_train_val",
        help="Use matched_pred_mask_direct_match_adaptive_dataset_train_val if adaptive rerank was used.",
    )
    parser.add_argument("--objects", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--save-txt", type=str, default="dataset_train_val_iou2d_direct_match.txt")
    parser.add_argument("--save-json", type=str, default="dataset_train_val_iou2d_direct_match.json")
    parser.add_argument("--snapshot-json", type=str, default="dataset_train_val_iou2d_frame_snapshot.json")
    args = parser.parse_args()

    work_root = Path(args.work_root)
    objects_root = resolve_objects_root(work_root)
    if not objects_root.is_dir():
        raise FileNotFoundError(f"objects root not found: {objects_root}")

    object_names = collect_objects(objects_root, args.objects, args.start, args.end)
    frame_items = collect_frame_items(objects_root, object_names, args.pred_mask_subdir)
    print(f"Objects: {len(object_names)}")
    print(f"Collected frame dirs: {len(frame_items)}")
    with open(args.snapshot_json, "w", encoding="utf-8") as f:
        json.dump(frame_items, f, ensure_ascii=False, indent=2)

    results = []
    records = []
    missing_gt = 0
    for item in frame_items:
        res = evaluate_frame(Path(item["gt_dir"]), Path(item["pred_dir"]))
        rec = dict(item)
        if res is None:
            missing_gt += 1
            rec["status"] = "missing_gt"
            records.append(rec)
            continue
        rec["status"] = "ok"
        rec.update(res)
        records.append(rec)
        results.append(res)

    if not results:
        print("No valid results to aggregate.")
        return

    metrics = average_metrics(results)
    report = "\n".join(
        [
            "=" * 40,
            "DATASET_TRAIN_VAL DIRECT_MATCH 2D IOU",
            "=" * 40,
            f"Work Root:        {work_root}",
            f"Objects Root:     {objects_root}",
            f"Pred Mask Subdir: {args.pred_mask_subdir}",
            f"Objects:          {len(object_names)}",
            f"Frame Dirs:       {len(frame_items)}",
            f"Missing GT:       {missing_gt}",
            "=" * 40,
            format_metrics(metrics, len(results)),
            "=" * 40,
            "",
        ]
    )
    payload = {
        "work_root": str(work_root),
        "objects_root": str(objects_root),
        "pred_mask_subdir": args.pred_mask_subdir,
        "objects": object_names,
        "num_objects": len(object_names),
        "num_frame_dirs": len(frame_items),
        "num_valid_frames": len(results),
        "num_missing_gt": missing_gt,
        "metrics": metrics,
        "records": records,
    }
    print(report)
    with open(args.save_txt, "w", encoding="utf-8") as f:
        f.write(report)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved report txt:  {args.save_txt}")
    print(f"Saved report json: {args.save_json}")


if __name__ == "__main__":
    main()
