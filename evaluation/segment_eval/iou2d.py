import os
import json
import numpy as np
import glob
import re
from PIL import Image
from scipy.optimize import linear_sum_assignment


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def calculate_ar_metrics(matched_ious, num_gt, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.05)
    if num_gt == 0:
        return {
            "AR": 0,
            "AR50": 0,
            "AR75": 0,
        }

    recalls = []
    for t in thresholds:
        true_positives = np.sum(matched_ious >= t)
        recalls.append(true_positives / num_gt)

    return {
        "AR": float(np.mean(recalls)),
        "AR50": float(recalls[0]),
        "AR75": float(recalls[5]) if len(recalls) > 5 else 0,
    }


def evaluate_segmentation(gt_npz_path, pred_dir):
    if not os.path.exists(gt_npz_path):
        print(f"GT not found: {gt_npz_path}")
        return None

    gt_data = np.load(gt_npz_path)
    gt_masks = gt_data["instance_segmentation"]

    unique_ids = np.unique(gt_masks)
    valid_ids = [i for i in unique_ids if i >= 0]
    num_gt = len(valid_ids)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")), key=natural_sort_key)
    pred_masks = []
    for f in pred_files:
        mask = np.array(Image.open(f).convert("L")) > 0
        pred_masks.append(mask)

    num_pred = len(pred_masks)
    if num_gt == 0 or num_pred == 0:
        return {
            "mIoU_matched": 0,
            "mIoU_total": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "AR": 0,
            "AR50": 0,
            "AR75": 0,
            "num_gt": num_gt,
            "num_pred": num_pred,
        }

    iou_matrix = np.zeros((num_gt, num_pred))
    for i, g_id in enumerate(valid_ids):
        gt_inst_mask = gt_masks == g_id
        for j in range(num_pred):
            iou_matrix[i, j] = calculate_iou(gt_inst_mask, pred_masks[j])

    row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]

    mean_matched_iou = np.mean(matched_ious)
    total_miou = np.sum(matched_ious) / max(num_gt, num_pred)

    threshold = 0.5
    true_positives = np.sum(matched_ious > threshold)
    precision = true_positives / num_pred
    recall = true_positives / num_gt
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
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
        "num_gt": num_gt,
        "num_pred": num_pred,
    }


def snapshot_frame_dirs(root_dir, pred_mask_subdir):
    # obj_list = os.listdir(root_dir)
    obj_list = [
        "Box_100189", "Bucket_100438", "CoffeeMachine_103074", "Dishwasher_12530",
        "Keyboard_12738", "Microwave_7263", "Printer_103972",
        "Remote_101028", "StorageFurniture_45134", "StorageFurniture_45779",
        "StorageFurniture_45910", "Toaster_103469", "Toilet_103234", "WashingMachine_103528",
    ]
    objs = [
        os.path.join(root_dir, o)
        for o in sorted(obj_list, key=natural_sort_key)
        if os.path.isdir(os.path.join(root_dir, o))
    ]

    frozen = []
    for obj_dir in objs:
        pred_masks_base = os.path.join(obj_dir, pred_mask_subdir)
        if not os.path.exists(pred_masks_base):
            continue

        frame_dirs = [
            os.path.join(pred_masks_base, d)
            for d in sorted(os.listdir(pred_masks_base), key=natural_sort_key)
            if os.path.isdir(os.path.join(pred_masks_base, d))
        ]

        for p_dir in frame_dirs:
            frame_name = os.path.basename(p_dir)
            frozen.append(
                {
                    "obj_dir": obj_dir,
                    "pred_dir": p_dir,
                    "frame_name": frame_name,
                }
            )
    return frozen


def main(
    root_dir,
    gt_dir,
    pred_mask_subdir="matched_pred_mask_direct_match_adaptive",
    save_txt="intra_iou2d_single_snapshot.txt",
    snapshot_json="iou2d_frame_snapshot.json",
):
    all_results = []

    frozen_frames = snapshot_frame_dirs(root_dir, pred_mask_subdir)
    print(f"Frozen frame count: {len(frozen_frames)}")
    with open(snapshot_json, "w", encoding="utf-8") as f:
        json.dump(frozen_frames, f, ensure_ascii=False, indent=2)
    print(f"Saved frozen frame list to: {snapshot_json}")

    for item in frozen_frames:
        p_dir = item["pred_dir"]
        frame_name = item["frame_name"]
        gt_path = os.path.join(gt_dir, frame_name + ".npz")

        res = evaluate_segmentation(gt_path, p_dir)
        if res is None:
            print(f"{p_dir} not found.")

        if res:
            all_results.append(res)

    if not all_results:
        print("No valid results to aggregate.")
        return

    avg_metrics = {
        "mIoU_matched": np.mean([r["mIoU_matched"] for r in all_results]),
        "mIoU_total": np.mean([r["mIoU_total"] for r in all_results]),
        "precision": np.mean([r["precision"] for r in all_results]),
        "recall": np.mean([r["recall"] for r in all_results]),
        "f1_score": np.mean([r["f1_score"] for r in all_results]),
        "AR": np.mean([r["AR"] for r in all_results]),
        "AR50": np.mean([r["AR50"] for r in all_results]),
        "AR75": np.mean([r["AR75"] for r in all_results]),
    }

    output_str = "\n" + "=" * 40 + "\n"
    output_str += "DATASET INFO:\n"
    output_str += "Root Directory: test_intra\n"
    output_str += f"Full Path:      {root_dir}\n"
    output_str += "=" * 40 + "\n"
    output_str += f"FINAL GLOBAL RESULTS ({len(all_results)} frames)\n"
    output_str += "-" * 40 + "\n"
    output_str += f"Average mIoU (Matched): {avg_metrics['mIoU_matched']:.4f}\n"
    output_str += f"Average mIoU (Total):   {avg_metrics['mIoU_total']:.4f}\n"
    output_str += f"Average Precision:      {avg_metrics['precision']:.4f}\n"
    output_str += f"Average Recall:         {avg_metrics['recall']:.4f}\n"
    output_str += f"Average F1-Score:       {avg_metrics['f1_score']:.4f}\n"
    output_str += f"Average AR50:           {avg_metrics['AR50']:.4f}\n"
    output_str += f"Average AR75:           {avg_metrics['AR75']:.4f}\n"
    output_str += f"Average AR@[.50:.95]:   {avg_metrics['AR']:.4f}\n"
    output_str += "=" * 40 + "\n"

    print("\n" + "=" * 30)
    print(f"FINAL GLOBAL RESULTS ({len(all_results)} frames)")
    print("=" * 30)
    print(f"Average mIoU (Matched): {avg_metrics['mIoU_matched']:.4f}")
    print(f"Average mIoU (Total):   {avg_metrics['mIoU_total']:.4f}")
    print(f"Average Precision:      {avg_metrics['precision']:.4f}")
    print(f"Average Recall:         {avg_metrics['recall']:.4f}")
    print(f"Average F1-Score:       {avg_metrics['f1_score']:.4f}")
    print(f"Average AR50:           {avg_metrics['AR50']:.4f}")
    print(f"Average AR75:           {avg_metrics['AR75']:.4f}")
    print(f"Average AR@[.50:.95]:   {avg_metrics['AR']:.4f}")
    print("=" * 30)

    with open(save_txt, "w", encoding="utf-8") as f:
        f.write(output_str)


if __name__ == "__main__":
    ROOT_DIR = "/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs"
    GT_DIR = "/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/segmentation"

    main(ROOT_DIR, GT_DIR)
