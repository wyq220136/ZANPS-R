import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = ""
SAM_CHECKPOINT = ""
SAM_MODEL_TYPE = "vit_h"
IMAGE_PATH = ""
MASK_PATH = ""

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def save_mask(mask_bool, out_path):
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    cv2.imwrite(out_path, mask_uint8)


def has_valid_pred_mask(pred_mask_frame_dir):
    if not os.path.isdir(pred_mask_frame_dir):
        return False
    for name in os.listdir(pred_mask_frame_dir):
        if name.startswith("mask_") and name.lower().endswith(".png"):
            return True
    return False


def refine_masks(anns, external_mask, iou_threshold=0.7):
    if len(anns) == 0:
        return []

    valid_anns = []
    # 预计算 external_mask 的面积，用于后续判断是否为“物体级”掩码
    external_area = np.sum(external_mask > 0)
    
    for ann in anns:
        m = ann["segmentation"]
        mask_sum = np.sum(m)
        intersection_with_ext = np.sum(m & (external_mask > 0))
        
        if intersection_with_ext < mask_sum * 0.5:
            continue

        if external_area > 0:
            containment_ratio = intersection_with_ext / external_area
            if containment_ratio > 0.9: # 阈值可根据实际效果微调（0.8-0.9）
                continue

        num_labels, _ = cv2.connectedComponents(m.astype(np.uint8))
        ann["fragmentation"] = num_labels - 1
        valid_anns.append(ann)

    valid_anns = sorted(valid_anns, key=lambda x: x["area"], reverse=True)
    keep = np.ones(len(valid_anns), dtype=bool)

    for i in range(len(valid_anns)):
        if not keep[i]:
            continue

        for j in range(i + 1, len(valid_anns)):
            if not keep[j]:
                continue

            mask_i = valid_anns[i]["segmentation"]
            mask_j = valid_anns[j]["segmentation"]

            # 使用位运算计算交集面积
            intersection = np.logical_and(mask_i, mask_j).sum()
            if intersection == 0:
                continue

            smaller_area = min(valid_anns[i]["area"], valid_anns[j]["area"])
            overlap_ratio = intersection / smaller_area
            larger_area = max(valid_anns[i]["area"], valid_anns[j]["area"])
            overlap_ratio2 = intersection / larger_area

            # 如果两个掩码高度重合，保留碎片化程度更低（连通域更少）的那个
            if overlap_ratio > iou_threshold and overlap_ratio2 > iou_threshold:
                if valid_anns[i]["fragmentation"] <= valid_anns[j]["fragmentation"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [valid_anns[i] for i in range(len(valid_anns)) if keep[i]]


def visualize_and_save_final(image, anns, output_path):
    canvas = image.copy()
    overlay = np.zeros_like(image, dtype=np.uint8)

    for ann in anns:
        mask = ann["segmentation"]
        color = np.random.randint(0, 255, (3,)).tolist()

        overlay[mask] = color

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

    result = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(12, 12))
    plt.imshow(result)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.show()


def save_masks_on_white_canvas(anns, output_dir="sam_masks_white"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(anns) == 0:
        print("No mask found")
        return

    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    h, w = sorted_anns[0]["segmentation"].shape

    print(f"Generating {len(sorted_anns)} masks")

    for i, ann in enumerate(sorted_anns):
        mask = ann["segmentation"]
        white_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 200))
        white_canvas[mask] = color

        out_file = os.path.join(output_dir, f"sam_mask_{i:03d}.png")
        cv2.imwrite(out_file, cv2.cvtColor(white_canvas, cv2.COLOR_RGB2BGR))

    print(f"Done: {output_dir}")


def create_mask_generator(
    model_cfg=MODEL_CFG,
    sam2_checkpoint=SAM2_CHECKPOINT,
    sam_checkpoint=SAM_CHECKPOINT,
    sam_model_type=SAM_MODEL_TYPE,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.92,
    min_mask_region_area=50,
):
    _ = model_cfg  # Kept for backward compatibility.

    checkpoint = sam_checkpoint or sam2_checkpoint
    if not checkpoint:
        raise ValueError("SAM checkpoint path is empty. Please set --sam-checkpoint.")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    if sam_model_type not in sam_model_registry:
        valid = ", ".join(sorted(sam_model_registry.keys()))
        raise ValueError(f"Unsupported sam_model_type={sam_model_type}, valid: {valid}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[sam_model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()

    if device == "cuda":
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
    )


def generate_candidate_masks(
    image_path,
    mask_path=None,
    model_cfg=MODEL_CFG,
    sam2_checkpoint=SAM2_CHECKPOINT,
    sam_checkpoint=SAM_CHECKPOINT,
    sam_model_type=SAM_MODEL_TYPE,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.92,
    min_mask_region_area=50,
    iou_threshold=0.5,
    mask_generator=None,
):
    if mask_generator is None:
        mask_generator = create_mask_generator(
            model_cfg=model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    external_mask = None
    if mask_path:
        external_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if external_mask is None:
            raise FileNotFoundError(f"External mask not found: {mask_path}")

    print("Generating masks...")
    with torch.inference_mode():
        raw_masks = mask_generator.generate(image_rgb)

    if external_mask is not None:
        filtered_masks = refine_masks(raw_masks, external_mask, iou_threshold=iou_threshold)
    else:
        filtered_masks = raw_masks

    return image_rgb, raw_masks, filtered_masks


def raw_segmentation(
    model_cfg=MODEL_CFG,
    sam2_checkpoint=SAM2_CHECKPOINT,
    sam_checkpoint=SAM_CHECKPOINT,
    sam_model_type=SAM_MODEL_TYPE,
    image_path=IMAGE_PATH,
    mask_path=MASK_PATH,
):
    image_rgb, raw_masks, filtered_masks = generate_candidate_masks(
        image_path=image_path,
        mask_path=mask_path,
        model_cfg=model_cfg,
        sam2_checkpoint=sam2_checkpoint,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
    )
    print(f"Raw masks: {len(raw_masks)}")
    print(f"Filtered masks: {len(filtered_masks)}")
    return image_rgb, raw_masks, filtered_masks
