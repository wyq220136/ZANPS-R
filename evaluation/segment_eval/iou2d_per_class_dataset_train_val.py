import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iou2d_dataset_train_val import (
    DEFAULT_WORK_ROOT,
    average_metrics,
    collect_frame_items,
    collect_objects,
    evaluate_frame,
    format_metrics,
    natural_sort_key,
    parse_class_name,
    resolve_objects_root,
)


def format_block(title: str, metrics: dict, frame_count: int, object_count: int | None = None) -> str:
    out = [title, "-" * 60]
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


def main() -> None:
    parser = argparse.ArgumentParser("2D IoU evaluation by class for dataset_train_val direct_match outputs")
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
    parser.add_argument("--save-txt", type=str, default="dataset_train_val_iou2d_per_class_direct_match.txt")
    parser.add_argument("--save-json", type=str, default="dataset_train_val_iou2d_per_class_direct_match.json")
    args = parser.parse_args()

    work_root = Path(args.work_root)
    objects_root = resolve_objects_root(work_root)
    if not objects_root.is_dir():
        raise FileNotFoundError(f"objects root not found: {objects_root}")

    object_names = collect_objects(objects_root, args.objects, args.start, args.end)
    frame_items = collect_frame_items(objects_root, object_names, args.pred_mask_subdir)
    print(f"Objects: {len(object_names)}")
    print(f"Collected frame dirs: {len(frame_items)}")
    if not frame_items:
        print("No frame dirs found. Please check --work-root and --pred-mask-subdir.")
        return

    per_class_results = defaultdict(list)
    per_class_objects = defaultdict(set)
    global_results = []
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
        per_class_results[item["class_name"]].append(res)
        per_class_objects[item["class_name"]].add(item["object"])
        global_results.append(res)

    if not global_results:
        print("No valid results aggregated.")
        return

    class_names = sorted(per_class_results.keys(), key=natural_sort_key)
    payload = {
        "work_root": str(work_root),
        "objects_root": str(objects_root),
        "pred_mask_subdir": args.pred_mask_subdir,
        "num_objects": len(object_names),
        "num_total_frame_dirs": len(frame_items),
        "num_missing_gt": missing_gt,
        "classes": {},
        "global": {
            "num_frames": len(global_results),
            "metrics": average_metrics(global_results),
        },
        "records": records,
    }

    lines = [
        "=" * 70,
        "DATASET_TRAIN_VAL DIRECT_MATCH 2D METRICS BY CLASS",
        "=" * 70,
        f"Work Root:              {work_root}",
        f"Objects Root:           {objects_root}",
        f"Pred Mask Subdir:       {args.pred_mask_subdir}",
        f"Collected Frame Dirs:   {len(frame_items)}",
        f"Missing GT Frames:      {missing_gt}",
        "=" * 70,
        "",
    ]

    for cls_name in class_names:
        cls_results = per_class_results[cls_name]
        cls_metrics = average_metrics(cls_results)
        cls_obj_count = len(per_class_objects[cls_name])
        payload["classes"][cls_name] = {
            "num_objects": cls_obj_count,
            "objects": sorted(per_class_objects[cls_name], key=natural_sort_key),
            "num_frames": len(cls_results),
            "metrics": cls_metrics,
        }
        lines.append(format_block(f"Class: {cls_name}", cls_metrics, len(cls_results), cls_obj_count))
        lines.append("")

    lines.append("=" * 70)
    lines.append(format_block("GLOBAL (ALL CLASSES)", payload["global"]["metrics"], len(global_results)))
    lines.append("=" * 70)
    report = "\n".join(lines) + "\n"
    print(report)

    with open(args.save_txt, "w", encoding="utf-8") as f:
        f.write(report)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved report txt:  {args.save_txt}")
    print(f"Saved report json: {args.save_json}")


if __name__ == "__main__":
    main()
