import math
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .instance_parts import SapienCameraBufferError


def mask_is_complete_and_sized(
    mask: np.ndarray,
    min_ratio: float,
    max_ratio: float,
) -> Tuple[bool, float]:
    h, w = mask.shape
    area = float(mask.sum())
    ratio = area / float(h * w)
    if area <= 0:
        return False, 0.0
    edge_touch = (
        mask[0, :].any()
        or mask[h - 1, :].any()
        or mask[:, 0].any()
        or mask[:, w - 1].any()
    )
    ok = (not edge_touch) and (ratio >= min_ratio) and (ratio <= max_ratio)
    return ok, ratio


def choose_entity_segmentation_channel(seg: np.ndarray, valid_entity_ids: Set[int]) -> int:
    if seg.ndim != 3 or seg.shape[2] < 2:
        return 0
    valid_ids = {int(v) for v in valid_entity_ids}
    if valid_ids:
        ch1 = seg[..., 1].astype(np.int64)
        if int(np.isin(ch1, list(valid_ids)).sum()) > 0:
            return 1
    scores: List[Tuple[int, int, int]] = []
    for ch in (0, 1):
        channel = seg[..., ch].astype(np.int64)
        vals = np.unique(channel)
        overlap = sum(1 for v in vals.tolist() if int(v) in valid_ids)
        pixels = int(np.isin(channel, list(valid_ids)).sum()) if valid_ids else 0
        scores.append((pixels, overlap, ch))
    scores.sort(reverse=True)
    return scores[0][2]


def parse_link_index_from_entity_name(name: str) -> Optional[int]:
    m = re.search(r"(?:^|[^0-9])link_(\d+)(?:$|[^0-9])", name)
    if not m:
        return None
    return int(m.group(1))


def evaluate_all_parts_in_frame(
    seg_channel: np.ndarray,
    part_to_entity_ids: Dict[int, List[int]],
    min_object_coverage: float,
    max_object_coverage: float,
    union_entity_ids: Optional[np.ndarray] = None,
) -> Tuple[bool, float]:
    union_mask = compute_union_mask(
        seg_channel,
        part_to_entity_ids,
        union_entity_ids=union_entity_ids,
    )
    if int(union_mask.sum()) == 0:
        return False, 0.0
    union_ok, ratio = mask_is_complete_and_sized(
        union_mask,
        min_ratio=min_object_coverage,
        max_ratio=max_object_coverage,
    )
    return union_ok, ratio


def compute_union_mask(
    seg_channel: np.ndarray,
    part_to_entity_ids: Dict[int, List[int]],
    union_entity_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    if union_entity_ids is not None:
        if union_entity_ids.size == 0:
            return np.zeros_like(seg_channel, dtype=bool)
        return np.isin(seg_channel, union_entity_ids)
    union_mask = np.zeros_like(seg_channel, dtype=bool)
    for entity_ids in part_to_entity_ids.values():
        if not entity_ids:
            continue
        union_mask |= np.isin(seg_channel, entity_ids)
    return union_mask


def _camera_picture(camera: object, name: str) -> np.ndarray:
    try:
        return camera.get_picture(name)
    except Exception as exc:
        raise SapienCameraBufferError(name, exc) from exc


def max_visible_part_area(seg_channel: np.ndarray, part_to_entity_ids: Dict[int, List[int]]) -> int:
    max_area = 0
    for entity_ids in part_to_entity_ids.values():
        if not entity_ids:
            continue
        area = int(np.isin(seg_channel, entity_ids).sum())
        if area > max_area:
            max_area = area
    return max_area


def visible_part_areas(seg_channel: np.ndarray, part_to_entity_ids: Dict[int, List[int]]) -> Dict[int, int]:
    areas: Dict[int, int] = {}
    for part_id, entity_ids in part_to_entity_ids.items():
        if not entity_ids:
            areas[part_id] = 0
            continue
        areas[part_id] = int(np.isin(seg_channel, entity_ids).sum())
    return areas


def parts_visibility_ok(
    seg_channel: np.ndarray,
    part_to_entity_ids: Dict[int, List[int]],
    min_part_pixels: int,
    require_all_parts: bool,
) -> Tuple[bool, Dict[int, int]]:
    areas = visible_part_areas(seg_channel, part_to_entity_ids)
    if not areas:
        return False, areas
    if require_all_parts:
        ok = all(area >= int(min_part_pixels) for area in areas.values())
    else:
        ok = max(areas.values()) >= int(min_part_pixels)
    return bool(ok), areas


def part_visibility_from_pixel_counts(
    part_visible_pixels: Dict[int, int],
    min_part_pixels: int,
) -> Tuple[bool, Dict[int, bool]]:
    """Return whether at least one movable part is visibly present in a frame."""
    threshold = int(min_part_pixels)
    visibility = {
        part_id: int(pixel_count) >= threshold
        for part_id, pixel_count in part_visible_pixels.items()
    }
    return any(visibility.values()), visibility


def mask_center_offset_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 1.0
    h, w = mask.shape
    cx = float(xs.mean())
    cy = float(ys.mean())
    ox = (cx - (w - 1) * 0.5) / max(1.0, w * 0.5)
    oy = (cy - (h - 1) * 0.5) / max(1.0, h * 0.5)
    return float(math.sqrt(ox * ox + oy * oy))


def mask_bbox_center_offset_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 1.0
    h, w = mask.shape
    cx = float(xs.min() + xs.max()) * 0.5
    cy = float(ys.min() + ys.max()) * 0.5
    ox = (cx - (w - 1) * 0.5) / max(1.0, w * 0.5)
    oy = (cy - (h - 1) * 0.5) / max(1.0, h * 0.5)
    return float(math.sqrt(ox * ox + oy * oy))


def mask_edge_margin_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0
    h, w = mask.shape
    margins = (
        int(xs.min()),
        int(ys.min()),
        int(w - 1 - xs.max()),
        int(h - 1 - ys.max()),
    )
    return float(min(margins)) / max(1.0, float(min(h, w)))


def mask_framing_ok(
    mask: np.ndarray,
    ratio: float,
    min_ratio: float,
    max_ratio: float,
    max_center_offset: float,
    min_edge_margin: float,
) -> Tuple[bool, Dict[str, float]]:
    if int(mask.sum()) == 0:
        stats = {
            "ratio": 0.0,
            "center_offset": 1.0,
            "bbox_center_offset": 1.0,
            "edge_margin": 0.0,
        }
        return False, stats
    center_offset = mask_center_offset_ratio(mask)
    bbox_center_offset = mask_bbox_center_offset_ratio(mask)
    edge_margin = mask_edge_margin_ratio(mask)
    stats = {
        "ratio": float(ratio),
        "center_offset": float(center_offset),
        "bbox_center_offset": float(bbox_center_offset),
        "edge_margin": float(edge_margin),
    }
    ok = (
        min_ratio <= ratio <= max_ratio
        and max(center_offset, bbox_center_offset) <= max_center_offset
        and edge_margin >= min_edge_margin
    )
    return bool(ok), stats


def view_quality_score(mask: np.ndarray, ratio: float, target_ratio: float) -> float:
    h, w = mask.shape
    if int(mask.sum()) == 0:
        return -1e9
    edge_touch = (
        mask[0, :].any()
        or mask[h - 1, :].any()
        or mask[:, 0].any()
        or mask[:, w - 1].any()
    )
    ratio_penalty = abs(math.log(max(ratio, 1e-6) / max(target_ratio, 1e-6))) * 1.15
    center_penalty = mask_center_offset_ratio(mask) * 2.0
    score = 1.0 - ratio_penalty - center_penalty
    if edge_touch:
        score -= 2.0
    return score
