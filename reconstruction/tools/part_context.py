from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from recon_utils import DatasetObject, backproject, find_image, list_parts, load_mask, mask_path_for_part_frame


def _bbox_from_mask(mask: np.ndarray) -> Optional[Dict[str, object]]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "xyxy": [x0, y0, x1, y1],
        "center": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
        "extent": [float(max(1, x1 - x0 + 1)), float(max(1, y1 - y0 + 1))],
        "area": int(len(xs)),
    }


def _find_object_mask(obj: DatasetObject, frame: str, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    candidates = []
    for folder in ("object_mask", "objectmask", "object_masks", "mask"):
        root = obj.root / folder
        if not root.is_dir():
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            candidates.append(root / f"{frame}{ext}")
    for path in candidates:
        if path.exists():
            return load_mask(path, shape_hw=shape_hw)
    return None


def _union_part_masks(obj: DatasetObject, frame: str, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    union = np.zeros(shape_hw, dtype=bool)
    found = False
    for part_name in list_parts(obj):
        path = mask_path_for_part_frame(obj, part_name, frame)
        if path is None:
            continue
        try:
            union |= load_mask(path, shape_hw=shape_hw)
            found = True
        except Exception:
            continue
    return union if found else None


def _safe_center3d(depth_m: np.ndarray, mask: np.ndarray, k: np.ndarray) -> Optional[np.ndarray]:
    pts = backproject(depth_m, mask, k)
    if len(pts) == 0:
        return None
    return np.median(pts, axis=0).astype(np.float32)


def compute_part_context(
    obj: DatasetObject,
    part_name: str,
    frame: str,
    depth_m: np.ndarray,
    part_mask: np.ndarray,
    k: np.ndarray,
) -> Dict[str, object]:
    """Compute object-level context for one part observation.

    The result is JSON-serializable and intentionally training-free. It gives the
    TSDF and downstream pose stages a compact description of where this part sits
    inside the whole object observation.
    """
    shape_hw = depth_m.shape[:2]
    part_mask = np.asarray(part_mask, dtype=bool)
    object_mask = _find_object_mask(obj, frame, shape_hw)
    object_mask_source = "object_mask"
    if object_mask is None:
        object_mask = _union_part_masks(obj, frame, shape_hw)
        object_mask_source = "part_union" if object_mask is not None else "missing"

    part_bbox = _bbox_from_mask(part_mask)
    object_bbox = _bbox_from_mask(object_mask) if object_mask is not None else None
    part_center3d = _safe_center3d(depth_m, part_mask, k)
    object_center3d = _safe_center3d(depth_m, object_mask, k) if object_mask is not None else None

    rel_center2d = None
    rel_extent2d = None
    if part_bbox is not None and object_bbox is not None:
        obj_extent = np.maximum(np.asarray(object_bbox["extent"], dtype=np.float32), 1.0)
        rel_center2d = (
            (np.asarray(part_bbox["center"], dtype=np.float32) - np.asarray(object_bbox["center"], dtype=np.float32))
            / obj_extent
        ).astype(float).tolist()
        rel_extent2d = (
            np.asarray(part_bbox["extent"], dtype=np.float32) / obj_extent
        ).astype(float).tolist()

    rel_center3d = None
    if part_center3d is not None and object_center3d is not None:
        rel_center3d = (part_center3d - object_center3d).astype(float).tolist()

    neighbor_reports = []
    for other_name in list_parts(obj):
        if other_name == part_name:
            continue
        other_path = mask_path_for_part_frame(obj, other_name, frame)
        if other_path is None:
            continue
        try:
            other_mask = load_mask(other_path, shape_hw=shape_hw)
        except Exception:
            continue
        other_bbox = _bbox_from_mask(other_mask)
        if other_bbox is None or part_bbox is None:
            continue
        pc = np.asarray(part_bbox["center"], dtype=np.float32)
        oc = np.asarray(other_bbox["center"], dtype=np.float32)
        dist = float(np.linalg.norm(pc - oc))
        neighbor_reports.append(
            {
                "part": other_name,
                "mask_pixels": int(np.count_nonzero(other_mask)),
                "bbox_center_distance_px": dist,
                "bbox": other_bbox,
            }
        )
    neighbor_reports.sort(key=lambda x: float(x["bbox_center_distance_px"]))

    return {
        "frame": str(frame),
        "part": str(part_name),
        "part_mask_pixels": int(np.count_nonzero(part_mask)),
        "object_mask_source": object_mask_source,
        "object_mask_pixels": None if object_mask is None else int(np.count_nonzero(object_mask)),
        "part_bbox": part_bbox,
        "object_bbox": object_bbox,
        "part_to_object_center_2d": rel_center2d,
        "part_to_object_extent_2d": rel_extent2d,
        "part_center_cam": None if part_center3d is None else part_center3d.astype(float).tolist(),
        "object_center_cam": None if object_center3d is None else object_center3d.astype(float).tolist(),
        "part_to_object_center_cam": rel_center3d,
        "nearest_parts": neighbor_reports[:5],
    }


def context_score(context: Dict[str, object]) -> float:
    """A conservative diagnostic score, not a hard rejection criterion."""
    part_pixels = int(context.get("part_mask_pixels") or 0)
    object_pixels = context.get("object_mask_pixels")
    if not object_pixels:
        return 0.0
    area_ratio = part_pixels / max(1, int(object_pixels))
    rel_center = context.get("part_to_object_center_2d")
    center_penalty = 0.0
    if rel_center is not None:
        center_penalty = min(1.0, float(np.linalg.norm(np.asarray(rel_center, dtype=np.float32))))
    return float(np.clip(area_ratio * (1.0 - 0.25 * center_penalty), 0.0, 1.0))

