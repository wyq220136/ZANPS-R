import os,sys
import re
import json
import argparse

import cv2
import numpy as np
import torch

sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv")
try:
    from eccv.dino_match.match import DINOv2CADMatcher
except:
    from eccv.dino_match.match import DINOv2CADMatcher

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def parse_view_and_frame(frame_id: str, obj_name: str):
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix):]
        m = re.fullmatch(r"(\d+)_(\d+)", tail)
        if m:
            return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", frame_id)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return 0, 0


def bbox_from_mask(mask_np: np.ndarray):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2, y2]


def mask_stats(mask_np: np.ndarray):
    mask_bin = (mask_np > 0).astype(np.uint8)
    box = bbox_from_mask(mask_bin)
    if box is None:
        return None
    mask_area = float(mask_bin.sum())
    bbox_area = box_area_xyxy(box)
    return {
        "box": box,
        "mask_area": mask_area,
        "bbox_area": float(bbox_area),
        "fg_ratio": float(mask_area / max(bbox_area, 1.0)),
        "aspect": float(box_aspect_xyxy(box)),
    }


def load_mask_files(mask_dir):
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
    return sorted(files, key=natural_sort_key)


def _load_reference_manifest(obj_dir, ref_select_subdir="ref_select", selected_json_name="selected_parts.json"):
    subdirs = []
    if ref_select_subdir == "ref_select":
        subdirs.append("chosen_part")
    subdirs.extend([ref_select_subdir, "ref_select"])
    seen = set()
    paths = []
    for subdir in subdirs:
        if not subdir or subdir in seen:
            continue
        seen.add(subdir)
        paths.append(os.path.join(obj_dir, subdir, selected_json_name))
    path = ""
    for cand in paths:
        if os.path.exists(cand):
            path = cand
            break
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] failed to read reference manifest: {path}, err={e}")
        return {}


def _resolve_models_root(obj_dir):
    manifest = _load_reference_manifest(obj_dir)
    candidates = []
    mesh_subdir = str(manifest.get("selected_mesh_subdir", "")).strip()
    if mesh_subdir:
        candidates.append(mesh_subdir)
    candidates.extend(["selected_mesh", "models"])
    seen = set()
    for subdir in candidates:
        if not subdir or subdir in seen:
            continue
        seen.add(subdir)
        path = subdir if os.path.isabs(subdir) else os.path.join(obj_dir, subdir)
        if os.path.isdir(path):
            return path
    return os.path.join(obj_dir, "selected_mesh")


def _manifest_ref_map(obj_dir):
    manifest = _load_reference_manifest(obj_dir)
    refs = manifest.get("references", [])
    if not isinstance(refs, list):
        return {}
    out = {}
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        try:
            view_id = int(ref.get("view_id", 0))
        except Exception:
            view_id = 0
        frame_id = str(ref.get("frame_id", "")).strip()
        mask_dir_rel = str(ref.get("mask_dir", "")).strip()
        if not frame_id or not mask_dir_rel:
            continue
        mask_dir = mask_dir_rel if os.path.isabs(mask_dir_rel) else os.path.join(obj_dir, mask_dir_rel)
        if os.path.isdir(mask_dir):
            out[view_id] = {"frame_id": frame_id, "mask_dir": mask_dir, "source": "manifest"}
    return out


def _partnet_part_dirs(obj_dir):
    masks_root = os.path.join(obj_dir, "masks")
    if not os.path.isdir(masks_root):
        return []
    names = [
        d for d in os.listdir(masks_root)
        if os.path.isdir(os.path.join(masks_root, d))
    ]
    return [(name, os.path.join(masks_root, name)) for name in sorted(names, key=natural_sort_key)]


def _find_frame_file(root, frame_id, exts=(".png", ".jpg", ".jpeg")):
    if not os.path.isdir(root):
        return ""
    for ext in exts:
        path = os.path.join(root, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return ""


def _part_mask_frame_ids(obj_dir):
    frame_ids = set()
    for _, part_dir in _partnet_part_dirs(obj_dir):
        for name in os.listdir(part_dir):
            if os.path.splitext(name)[1].lower() in (".png", ".jpg", ".jpeg"):
                frame_ids.add(os.path.splitext(name)[0])
    return sorted(frame_ids, key=natural_sort_key)


def _raw_part_mask_path(obj_dir, frame_id, part_id, local_idx=None):
    part_dirs = _partnet_part_dirs(obj_dir)
    preferred = []
    for idx, (part_name, part_dir) in enumerate(part_dirs):
        pid = parse_part_id_from_name(part_name)
        if pid == int(part_id):
            preferred.append((idx, part_name, part_dir))
    if local_idx is not None and 0 <= int(local_idx) < len(part_dirs):
        item = part_dirs[int(local_idx)]
        if item not in [(p[1], p[2]) for p in preferred]:
            preferred.append((int(local_idx), item[0], item[1]))
    for idx, part_name, part_dir in preferred:
        path = _find_frame_file(part_dir, frame_id)
        if path:
            return path
    return ""


def _visible_cad_indices_from_raw_masks(obj_dir, frame_id, cad_meta, min_visible_pixels=30):
    visible = []
    invisible_ids = []
    for i, m in enumerate(cad_meta):
        pid = int(m.get("cad_part_id", -1))
        mask_path = _raw_part_mask_path(obj_dir, frame_id, pid, i)
        pix = 0
        if mask_path:
            mm = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mm is not None:
                pix = int(np.count_nonzero(mm > 0))
        if (not mask_path) or pix < int(min_visible_pixels):
            invisible_ids.append(pid)
        else:
            visible.append(i)
    return visible, invisible_ids


def _visible_cad_indices_from_gt_mask(obj_dir, frame_id, cad_meta, min_visible_pixels=30):
    return _visible_cad_indices_from_raw_masks(obj_dir, frame_id, cad_meta, min_visible_pixels)


def parse_part_id_from_name(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def find_rgb_path(obj_dir, frame_id):
    cand_png = os.path.join(obj_dir, "rgb", f"{frame_id}.png")
    cand_jpg = os.path.join(obj_dir, "rgb", f"{frame_id}.jpg")
    if os.path.exists(cand_png):
        return cand_png
    if os.path.exists(cand_jpg):
        return cand_jpg
    return None


def find_depth_path(obj_dir, frame_id):
    depth_dir = os.path.join(obj_dir, "depth")
    if not os.path.isdir(depth_dir):
        return None
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy", ".exr"):
        p = os.path.join(depth_dir, f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def find_object_mask_path(obj_dir, frame_id):
    for subdir in ("object_masks", "object_mask"):
        path = _find_frame_file(os.path.join(obj_dir, subdir), frame_id)
        if path:
            return path
    return None


def load_object_mask_for_frame(obj_dir, frame_id, image_shape, fallback_masks=None):
    mask_path = find_object_mask_path(obj_dir, frame_id)
    obj_mask = None
    if mask_path is not None:
        obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if obj_mask is None and fallback_masks:
        obj_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for m in fallback_masks:
            if m.shape[:2] != obj_mask.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (obj_mask.shape[1], obj_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            obj_mask |= (m > 0).astype(np.uint8)
    if obj_mask is None:
        return None
    if obj_mask.shape[:2] != image_shape[:2]:
        obj_mask = cv2.resize(obj_mask, (int(image_shape[1]), int(image_shape[0])), interpolation=cv2.INTER_NEAREST)
    return (obj_mask > 0).astype(np.uint8)


def _find_intrinsic_path(obj_dir):
    candidates = [
        os.path.join(obj_dir, "intrinsic.txt"),
        os.path.join(obj_dir, "cam_K.txt"),
        os.path.join(obj_dir, "camera_intrinsic.txt"),
        os.path.join(obj_dir, "K.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_intrinsic_from_obj(obj_dir):
    intrinsic_path = _find_intrinsic_path(obj_dir)
    if intrinsic_path is None:
        return None
    k = np.loadtxt(intrinsic_path, dtype=np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic shape {k.shape} from {intrinsic_path}")
    return k.astype(np.float32)


def _load_depth_any(depth_path):
    if depth_path is None:
        return None
    if depth_path.lower().endswith(".npy"):
        d = np.load(depth_path).astype(np.float32)
    else:
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        d = d.astype(np.float32)
    if d.size > 0 and np.nanmax(d) > 50.0:
        d = d / 1000.0
    d[~np.isfinite(d)] = 0.0
    d[d < 0.0] = 0.0
    return d


def _image_to_tensor(image_rgb: np.ndarray):
    return torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0


def _imagenet_mean_rgb(shape):
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    out = np.zeros((int(shape[0]), int(shape[1]), 3), dtype=np.uint8)
    out[...] = np.clip(mean, 0, 255).astype(np.uint8)
    return out


NEGATIVE_FILL_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
)


def build_negative_color_rgb(
    image_rgb: np.ndarray,
    object_mask: np.ndarray,
    removed_mask: np.ndarray,
    fill_color,
):
    obj = (object_mask > 0).astype(np.uint8)
    rem = (removed_mask > 0).astype(np.uint8)
    if obj.shape[:2] != image_rgb.shape[:2]:
        obj = cv2.resize(obj, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    if rem.shape[:2] != image_rgb.shape[:2]:
        rem = cv2.resize(rem, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = image_rgb.copy()
    out[obj <= 0] = _imagenet_mean_rgb(image_rgb.shape[:2])[obj <= 0]
    fill = np.asarray(fill_color, dtype=np.uint8).reshape(1, 1, 3)
    out[(obj > 0) & (rem > 0)] = fill
    return out


def _build_xyz_map(depth: np.ndarray, k: np.ndarray):
    h, w = depth.shape[:2]
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth.astype(np.float32)
    x = (xs - float(k[0, 2])) * z / max(float(k[0, 0]), 1e-8)
    y = (ys - float(k[1, 2])) * z / max(float(k[1, 1]), 1e-8)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def build_normal_rgb_from_depth(depth: np.ndarray | None, k: np.ndarray | None, object_mask: np.ndarray, image_shape):
    if depth is None or k is None or object_mask is None:
        return None
    if depth.shape[:2] != tuple(image_shape[:2]):
        src_h, src_w = depth.shape[:2]
        dst_h, dst_w = int(image_shape[0]), int(image_shape[1])
        sx = float(dst_w) / max(float(src_w), 1.0)
        sy = float(dst_h) / max(float(src_h), 1.0)
        depth = cv2.resize(depth, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
        k = np.asarray(k, dtype=np.float32).copy()
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
    obj = (object_mask > 0).astype(np.uint8)
    if obj.shape[:2] != depth.shape[:2]:
        obj = cv2.resize(obj, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    xyz = _build_xyz_map(depth, k)
    valid = (depth > 1e-6) & (obj > 0) & np.isfinite(depth)
    h, w = depth.shape[:2]
    normal = np.zeros((h, w, 3), dtype=np.float32)
    if h < 3 or w < 3:
        return None

    center_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    if int(np.count_nonzero(center_valid)) < 10:
        return None
    dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    n = np.cross(dx, dy).astype(np.float32)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(n_norm, 1e-8, None)

    cam_dir = -xyz[1:-1, 1:-1]
    cam_dir = cam_dir / np.clip(np.linalg.norm(cam_dir, axis=-1, keepdims=True), 1e-8, None)
    flip = np.sum(n * cam_dir, axis=-1, keepdims=True) < 0.0
    n = np.where(flip, -n, n)
    n[~center_valid] = 0.0
    normal[1:-1, 1:-1] = n

    normal_rgb = _imagenet_mean_rgb((h, w))
    color = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    normal_rgb[valid] = color[valid]
    return normal_rgb


def build_neighborhood_roi_mask(part_mask: np.ndarray, object_mask: np.ndarray):
    part = (part_mask > 0).astype(np.uint8)
    obj = (object_mask > 0).astype(np.uint8)
    if obj.shape[:2] != part.shape[:2]:
        obj = cv2.resize(obj, (part.shape[1], part.shape[0]), interpolation=cv2.INTER_NEAREST)
    stats = mask_stats(part)
    if stats is None:
        return part, bbox_from_mask(part), 0, False
    s = max(float(stats["mask_area"]), 1.0)
    radius = int(np.clip(0.45 * np.sqrt(s), 10, 32))
    x1, y1, x2, y2 = stats["box"]
    h, w = part.shape[:2]
    x1 = max(0, int(x1 - radius))
    y1 = max(0, int(y1 - radius))
    x2 = min(w, int(x2 + radius))
    y2 = min(h, int(y2 + radius))
    roi = np.zeros_like(part, dtype=np.uint8)
    roi[y1:y2, x1:x2] = 1
    roi = (roi & obj).astype(np.uint8)
    roi_box = bbox_from_mask(roi)
    if roi_box is None or int(np.count_nonzero(roi)) < 10:
        return part, stats["box"], radius, False
    return roi, roi_box, radius, True


def build_edge_rgb(image_rgb: np.ndarray, part_mask: np.ndarray, object_mask: np.ndarray):
    part = (part_mask > 0).astype(np.uint8)
    obj = (object_mask > 0).astype(np.uint8)
    if obj.shape[:2] != part.shape[:2]:
        obj = cv2.resize(obj, (part.shape[1], part.shape[0]), interpolation=cv2.INTER_NEAREST)

    edge_img = image_rgb.copy()
    edge_img[obj <= 0] = _imagenet_mean_rgb(obj.shape[:2])[obj <= 0]
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(part, cv2.MORPH_GRADIENT, kernel) > 0
    contact = (cv2.dilate(part, kernel, iterations=2) > 0) & (obj > 0) & (part <= 0)
    contact = cv2.dilate(contact.astype(np.uint8), kernel, iterations=1) > 0
    # Treat connector/contact region as normal edge: keep it in edge extraction
    # but do not highlight it with a dedicated color.
    edge_mask = boundary | contact
    edge_img[edge_mask] = np.asarray([255, 255, 0], dtype=np.uint8)
    return edge_img


def _score_from_sem_appe(sem_matrix, appe_matrix):
    return ((sem_matrix + appe_matrix) * 0.5).astype(np.float32)


def _weighted_branch_average(branches, positive_fallback):
    numerator = np.zeros_like(positive_fallback, dtype=np.float32)
    denominator = np.zeros_like(positive_fallback, dtype=np.float32)
    for matrix, valid, weight in branches:
        weight = float(weight)
        if weight <= 0.0:
            continue
        if valid is None:
            numerator += weight * matrix.astype(np.float32)
            denominator += weight
        else:
            valid_f = valid.astype(np.float32)
            numerator += weight * matrix.astype(np.float32) * valid_f
            denominator += weight * valid_f
    return np.where(denominator > 1e-8, numerator / np.clip(denominator, 1e-8, None), positive_fallback)


def matrix_to_nested_list(mat):
    return np.asarray(mat, dtype=np.float32).reshape(4, 4).tolist()


def assign_one_mask_per_cad(score_matrix: np.ndarray):
    """
    Build unique (mask, cad) pairs by score with one-mask-one-cad constraint.
    """
    pairs = []
    n_mask, n_cad = score_matrix.shape
    for i in range(n_mask):
        for j in range(n_cad):
            pairs.append((float(score_matrix[i, j]), int(i), int(j)))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_masks = set()
    used_cads = set()
    selected = []
    for s, i, j in pairs:
        if i in used_masks or j in used_cads:
            continue
        selected.append((s, i, j))
        used_masks.add(i)
        used_cads.add(j)
    return selected


def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def box_aspect_xyxy(box):
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return w / h


def build_cad_templates_with_aux(obj_dir, semantic_matcher, appearance_matcher):
    """
    Build CAD template banks with:
    - semantic descriptors
    - appearance descriptors
    - mask/bbox stats for geometry/visibility-like scoring
    """
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    models_root = _resolve_models_root(obj_dir)
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"models folder not found: {models_root}")

    frame_ids = _part_mask_frame_ids(obj_dir)
    if len(frame_ids) == 0:
        raise RuntimeError(f"No raw part mask frames found under: {obj_dir}/masks")

    def select_ref_for_view(view_id: int):
        preferred = f"{obj_name}_{view_id}_0"
        if preferred in frame_ids:
            return preferred
        candidates = []
        for fid in frame_ids:
            v, f = parse_view_and_frame(fid, obj_name)
            if v == view_id:
                candidates.append((f, fid))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return frame_ids[0]

    view_dirs = [
        d for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d)) and re.fullmatch(r"view_(\d+)", d)
    ]
    view_dirs = sorted(view_dirs, key=natural_sort_key)

    view_to_models = {}
    if view_dirs:
        for vd in view_dirs:
            m = re.fullmatch(r"view_(\d+)", vd)
            if m is None:
                continue
            view_id = int(m.group(1))
            vroot = os.path.join(models_root, vd)
            model_entries = [
                d for d in os.listdir(vroot)
                if os.path.isdir(os.path.join(vroot, d)) and os.path.exists(os.path.join(vroot, d, "model.obj"))
            ]
            model_entries = sorted(model_entries, key=natural_sort_key)
            if model_entries:
                view_to_models[view_id] = (vroot, model_entries)
    else:
        model_entries = [
            d for d in os.listdir(models_root)
            if os.path.isdir(os.path.join(models_root, d)) and os.path.exists(os.path.join(models_root, d, "model.obj"))
        ]
        model_entries = sorted(model_entries, key=natural_sort_key)
        if model_entries:
            view_to_models[-1] = (models_root, model_entries)

    if not view_to_models:
        raise RuntimeError(f"No model folders found under: {models_root}")

    manifest_refs = _manifest_ref_map(obj_dir)
    out = {}
    for view_id, (model_root_for_view, model_entries) in view_to_models.items():
        ref_info = manifest_refs.get(view_id)
        if ref_info is None and view_id < 0:
            ref_info = manifest_refs.get(0)
        ref_frame_id = ref_info["frame_id"] if ref_info else select_ref_for_view(view_id if view_id >= 0 else 0)
        ref_rgb_path = find_rgb_path(obj_dir, ref_frame_id)
        ref_mask_dir = ref_info["mask_dir"] if ref_info else ""
        if ref_rgb_path is None or (ref_mask_dir and not os.path.isdir(ref_mask_dir)):
            print(f"[WARN] skip view={view_id}: missing ref rgb/mask for {ref_frame_id}")
            continue

        image_bgr = cv2.imread(ref_rgb_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] skip view={view_id}: cannot read {ref_rgb_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_t = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        mask_files = load_mask_files(ref_mask_dir) if ref_mask_dir else []
        if ref_mask_dir and len(mask_files) == 0:
            print(f"[WARN] skip view={view_id}: no masks in {ref_mask_dir}")
            continue

        templates = []
        boxes = []
        masks = []
        cad_meta = []
        for local_idx, model_name in enumerate(model_entries):
            part_id = parse_part_id_from_name(model_name)
            if ref_mask_dir:
                preferred = f"mask_{part_id}.png"
                if preferred in mask_files:
                    mask_name = preferred
                elif local_idx < len(mask_files):
                    mask_name = mask_files[local_idx]
                else:
                    continue
                mask_path = os.path.join(ref_mask_dir, mask_name)
            else:
                mask_path = _raw_part_mask_path(obj_dir, ref_frame_id, part_id, local_idx)
                if not mask_path:
                    continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_bin = (mask > 0).astype(np.uint8)
            if np.sum(mask_bin) < 10:
                continue

            box = bbox_from_mask(mask_bin)
            if box is None:
                continue

            mask_area = float(mask_bin.sum())
            bbox_area = box_area_xyxy(box)
            fg_ratio = mask_area / max(bbox_area, 1.0)

            templates.append(image_t)
            boxes.append(box)
            masks.append(mask_bin.astype(np.float32))
            cad_meta.append(
                {
                    "view_id": int(view_id),
                    "cad_part_id": int(part_id),
                    "cad_model_dir": os.path.join(model_root_for_view, model_name),
                    "reference_frame_id": ref_frame_id,
                    "reference_mask_path": mask_path,
                    "template_bbox": [int(v) for v in box],
                    "template_bbox_area": float(bbox_area),
                    "template_mask_area": float(mask_area),
                    "template_fg_ratio": float(fg_ratio),
                    "template_aspect": float(box_aspect_xyxy(box)),
                }
            )

        if len(templates) == 0:
            print(f"[WARN] skip view={view_id}: no valid templates")
            continue

        ref_object_mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for mm in masks:
            ref_object_mask |= (mm > 0).astype(np.uint8)

        k_ref = load_intrinsic_from_obj(obj_dir)
        ref_depth_path = find_depth_path(obj_dir, ref_frame_id)
        ref_depth = _load_depth_any(ref_depth_path) if ref_depth_path else None
        ref_normal_rgb = build_normal_rgb_from_depth(ref_depth, k_ref, ref_object_mask, image_rgb.shape[:2])
        ref_object_stats = mask_stats(ref_object_mask)
        if ref_object_stats is None:
            ref_object_stats = {"box": [0, 0, image_rgb.shape[1], image_rgb.shape[0]]}

        neg_boxes = []
        neg_masks = []
        neg_valid = []
        neg_color_templates = [[] for _ in NEGATIVE_FILL_COLORS]
        neg_color_boxes = []
        neg_color_masks = []
        normal_templates = []
        normal_boxes = []
        normal_masks = []
        normal_valid = []
        edge_templates = []
        edge_boxes = []
        edge_masks = []
        edge_valid = []
        for mask_pos, meta in zip(masks, cad_meta):
            mask_pos_bin = (mask_pos > 0).astype(np.uint8)
            neg_mask = (ref_object_mask > 0).astype(np.uint8) & (mask_pos_bin == 0).astype(np.uint8)
            st = mask_stats(neg_mask)
            if st is None or st["mask_area"] < 10:
                st = {
                    "box": meta["template_bbox"],
                    "mask_area": meta["template_mask_area"],
                    "bbox_area": meta["template_bbox_area"],
                    "fg_ratio": meta["template_fg_ratio"],
                    "aspect": meta["template_aspect"],
                }
                neg_mask = mask_pos_bin
                neg_valid.append(False)
            else:
                neg_valid.append(True)
            neg_boxes.append(st["box"])
            neg_masks.append(neg_mask.astype(np.float32))
            neg_color_boxes.append(ref_object_stats["box"])
            neg_color_masks.append(ref_object_mask.astype(np.float32))
            for color_idx, color in enumerate(NEGATIVE_FILL_COLORS):
                neg_color_templates[color_idx].append(_image_to_tensor(
                    build_negative_color_rgb(image_rgb, ref_object_mask, mask_pos_bin, color)
                ))
            meta["template_negative_valid"] = bool(neg_valid[-1])
            meta["template_negative_bbox"] = [int(v) for v in st["box"]]
            meta["template_negative_bbox_area"] = float(st["bbox_area"])
            meta["template_negative_mask_area"] = float(st["mask_area"])
            meta["template_negative_fg_ratio"] = float(st["fg_ratio"])
            meta["template_negative_aspect"] = float(st["aspect"])

            roi_mask, roi_box, roi_radius, roi_ok = build_neighborhood_roi_mask(mask_pos_bin, ref_object_mask)
            normal_img = ref_normal_rgb if ref_normal_rgb is not None else image_rgb
            normal_templates.append(_image_to_tensor(normal_img))
            normal_boxes.append(roi_box)
            normal_masks.append(roi_mask.astype(np.float32))
            normal_valid.append(bool(ref_normal_rgb is not None and roi_ok))
            edge_img = build_edge_rgb(image_rgb, mask_pos_bin, ref_object_mask)
            edge_templates.append(_image_to_tensor(edge_img))
            edge_boxes.append(roi_box)
            edge_masks.append(roi_mask.astype(np.float32))
            edge_valid.append(bool(roi_ok))
            meta["template_neighborhood_radius"] = int(roi_radius)
            meta["template_normal_valid"] = bool(normal_valid[-1])
            meta["template_edge_valid"] = bool(edge_valid[-1])

        templates_t = torch.stack(templates, dim=0)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        masks_t = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32)
        neg_boxes_t = torch.as_tensor(neg_boxes, dtype=torch.float32)
        neg_masks_t = torch.as_tensor(np.stack(neg_masks, axis=0), dtype=torch.float32)
        neg_color_boxes_t = torch.as_tensor(neg_color_boxes, dtype=torch.float32)
        neg_color_masks_t = torch.as_tensor(np.stack(neg_color_masks, axis=0), dtype=torch.float32)
        normal_templates_t = torch.stack(normal_templates, dim=0)
        normal_boxes_t = torch.as_tensor(normal_boxes, dtype=torch.float32)
        normal_masks_t = torch.as_tensor(np.stack(normal_masks, axis=0), dtype=torch.float32)
        edge_templates_t = torch.stack(edge_templates, dim=0)
        edge_boxes_t = torch.as_tensor(edge_boxes, dtype=torch.float32)
        edge_masks_t = torch.as_tensor(np.stack(edge_masks, axis=0), dtype=torch.float32)

        desc_sem = semantic_matcher.encode_templates(templates_t, boxes_t, masks_t)
        desc_appe = appearance_matcher.encode_templates(templates_t, boxes_t, masks_t)
        desc_sem_neg_colors = []
        desc_appe_neg_colors = []
        for color_templates in neg_color_templates:
            color_templates_t = torch.stack(color_templates, dim=0)
            desc_sem_neg_colors.append(
                semantic_matcher.encode_templates(color_templates_t, neg_color_boxes_t, neg_color_masks_t)
            )
            desc_appe_neg_colors.append(
                appearance_matcher.encode_templates(color_templates_t, neg_color_boxes_t, neg_color_masks_t)
            )
        desc_sem_normal = semantic_matcher.encode_templates(normal_templates_t, normal_boxes_t, normal_masks_t)
        desc_appe_normal = appearance_matcher.encode_templates(normal_templates_t, normal_boxes_t, normal_masks_t)
        desc_sem_edge = semantic_matcher.encode_templates(edge_templates_t, edge_boxes_t, edge_masks_t)
        desc_appe_edge = appearance_matcher.encode_templates(edge_templates_t, edge_boxes_t, edge_masks_t)
        out[int(view_id)] = {
            "meta": cad_meta,
            "desc_sem": desc_sem.unsqueeze(1),   # [N_cad, 1, D]
            "desc_appe": desc_appe.unsqueeze(1), # [N_cad, 1, D]
            "desc_sem_neg_colors": [d.unsqueeze(1) for d in desc_sem_neg_colors],
            "desc_appe_neg_colors": [d.unsqueeze(1) for d in desc_appe_neg_colors],
            "desc_sem_normal": desc_sem_normal.unsqueeze(1),
            "desc_appe_normal": desc_appe_normal.unsqueeze(1),
            "desc_sem_edge": desc_sem_edge.unsqueeze(1),
            "desc_appe_edge": desc_appe_edge.unsqueeze(1),
        }
        print(f"[TEMPLATE] view={view_id} ref={ref_frame_id}, cad_templates={len(cad_meta)}")

    if not out:
        raise RuntimeError("No valid CAD template banks built.")
    return out


def _safe_log_ratio(a, b, eps=1e-6):
    return np.abs(np.log((a + eps) / (b + eps)))


def compute_geom_and_visible_scores(query_stats, template_stats):
    """
    SAM-6D inspired geometric+visible scoring without 3D projection:
    - geometric_score: bbox area consistency + aspect consistency
    - visible_ratio: mask occupancy consistency (fg ratio in bbox)
    """
    q_area = query_stats["bbox_area"][:, None]
    q_aspect = query_stats["aspect"][:, None]
    q_fg_ratio = query_stats["fg_ratio"][:, None]

    t_area = template_stats["bbox_area"][None, :]
    t_aspect = template_stats["aspect"][None, :]
    t_fg_ratio = template_stats["fg_ratio"][None, :]

    area_score = np.exp(-_safe_log_ratio(q_area, t_area))
    aspect_score = np.exp(-_safe_log_ratio(q_aspect, t_aspect))
    geometric_score = 0.5 * area_score + 0.5 * aspect_score

    visible_ratio = np.minimum(q_fg_ratio, t_fg_ratio) / np.maximum(q_fg_ratio, t_fg_ratio + 1e-6)
    visible_ratio = np.clip(visible_ratio, 0.0, 1.0)
    return geometric_score.astype(np.float32), visible_ratio.astype(np.float32)


def greedy_one_to_one(score_matrix: np.ndarray, score_thresh: float):
    pairs = []
    n_mask, n_cad = score_matrix.shape
    for i in range(n_mask):
        for j in range(n_cad):
            s = float(score_matrix[i, j])
            if s >= score_thresh:
                pairs.append((s, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    return pairs


def visualize_frame(image_bgr, proposals, matches, cad_meta, output_path):
    vis = image_bgr.copy()
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 255, size=(max(len(cad_meta), 1), 3), dtype=np.uint8)

    for q_idx, c_idx, score in matches:
        box = proposals[q_idx]["bbox"]
        x1, y1, x2, y2 = box
        color = tuple(int(v) for v in palette[c_idx % len(palette)].tolist())
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        txt = f"cad={cad_meta[c_idx]['cad_part_id']} score={score:.3f}"
        cv2.putText(vis, txt, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, vis)


def run_matching_for_object(
    obj_dir,
    out_dir,
    model_name="dinov2_vitl14",
    score_thresh=0.25,
    semantic_thresh=0.15,
    topk_per_frame=3,
    pred_mask_subdir="pred_mask",
    matched_mask_subdir="matched_pred_mask_sam6d",
    finalize_one_to_one=False,
    sam6d_pos_weight=0.25,
    sam6d_neg_weight=0.25,
    sam6d_normal_weight=0.25,
    sam6d_edge_weight=0.25,
    min_visible_pixels=30,
):
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    pred_mask_root = os.path.join(obj_dir, pred_mask_subdir)
    if not os.path.isdir(pred_mask_root):
        raise FileNotFoundError(f"{pred_mask_subdir} not found: {pred_mask_root}")
    os.makedirs(out_dir, exist_ok=True)
    matched_mask_root = os.path.join(obj_dir, matched_mask_subdir)
    os.makedirs(matched_mask_root, exist_ok=True)

    semantic_matcher = DINOv2CADMatcher(
        model_name=model_name,
        proposal_size=224,
        chunk_size=16,
        background_mean_fill=True,
        use_multi_layer_fusion=True,
        fusion_layers=8,
    )
    appearance_matcher = DINOv2CADMatcher(
        model_name=model_name,
        proposal_size=224,
        chunk_size=16,
        background_mean_fill=False,
        use_multi_layer_fusion=True,
        fusion_layers=8,
    )
    try:
        templates_by_view = build_cad_templates_with_aux(obj_dir, semantic_matcher, appearance_matcher)
    except Exception as e:
        print(f"[WARN] skip object {obj_name}: cannot build templates. err={e}")
        out_json = os.path.join(out_dir, "match_results_sam6d_style.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return
    print(f"[TEMPLATE-SELECT] available template keys: {sorted(list(templates_by_view.keys()))}")
    if not templates_by_view:
        out_json = os.path.join(out_dir, "match_results_sam6d_style.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return

    k_obj = load_intrinsic_from_obj(obj_dir)
    frame_ids = sorted(
        [d for d in os.listdir(pred_mask_root) if os.path.isdir(os.path.join(pred_mask_root, d))],
        key=natural_sort_key,
    )
    all_results = {}
    for frame_id in frame_ids:
        rgb_path = find_rgb_path(obj_dir, frame_id)
        if rgb_path is None:
            continue
        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        frame_view_id, _ = parse_view_and_frame(frame_id, obj_name)
        if frame_view_id in templates_by_view:
            template_key = frame_view_id
        elif 0 in templates_by_view:
            template_key = 0
        elif -1 in templates_by_view:
            template_key = -1
        else:
            template_key = sorted(templates_by_view.keys())[0]
        view_pack = templates_by_view[template_key]

        cad_meta = view_pack["meta"]
        reference_sem = view_pack["desc_sem"]   # [N_cad, 1, D]
        reference_appe = view_pack["desc_appe"] # [N_cad, 1, D]
        reference_sem_neg_colors = view_pack["desc_sem_neg_colors"]
        reference_appe_neg_colors = view_pack["desc_appe_neg_colors"]
        reference_sem_normal = view_pack["desc_sem_normal"]
        reference_appe_normal = view_pack["desc_appe_normal"]
        reference_sem_edge = view_pack["desc_sem_edge"]
        reference_appe_edge = view_pack["desc_appe_edge"]

        visible_cad_idx, invisible_cad_part_ids = _visible_cad_indices_from_gt_mask(
            obj_dir=obj_dir,
            frame_id=frame_id,
            cad_meta=cad_meta,
            min_visible_pixels=min_visible_pixels,
        )
        if len(visible_cad_idx) == 0:
            all_results[frame_id] = []
            continue
        cad_meta = [cad_meta[i] for i in visible_cad_idx]
        idx_t = torch.as_tensor(visible_cad_idx, dtype=torch.long, device=reference_sem.device)
        reference_sem = reference_sem.index_select(0, idx_t)
        reference_appe = reference_appe.index_select(0, idx_t)
        reference_sem_neg_colors = [x.index_select(0, idx_t) for x in reference_sem_neg_colors]
        reference_appe_neg_colors = [x.index_select(0, idx_t) for x in reference_appe_neg_colors]
        reference_sem_normal = reference_sem_normal.index_select(0, idx_t)
        reference_appe_normal = reference_appe_normal.index_select(0, idx_t)
        reference_sem_edge = reference_sem_edge.index_select(0, idx_t)
        reference_appe_edge = reference_appe_edge.index_select(0, idx_t)

        mask_dir = os.path.join(pred_mask_root, frame_id)
        mask_files = load_mask_files(mask_dir)
        proposals = []
        masks = []
        boxes = []
        q_bbox_area = []
        q_fg_ratio = []
        q_aspect = []
        for mask_name in mask_files:
            mask_path = os.path.join(mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_bin = (mask > 0).astype(np.uint8)
            if np.sum(mask_bin) < 10:
                continue
            box = bbox_from_mask(mask_bin)
            if box is None:
                continue

            bbox_area = box_area_xyxy(box)
            mask_area = float(mask_bin.sum())
            fg_ratio = mask_area / max(bbox_area, 1.0)
            aspect = box_aspect_xyxy(box)

            proposals.append({"mask_path": mask_path, "bbox": box, "mask_bin": mask_bin.astype(np.uint8)})
            masks.append(mask_bin.astype(np.float32))
            boxes.append(box)
            q_bbox_area.append(float(bbox_area))
            q_fg_ratio.append(float(fg_ratio))
            q_aspect.append(float(aspect))

        if len(proposals) == 0:
            all_results[frame_id] = []
            continue

        object_mask = load_object_mask_for_frame(
            obj_dir,
            frame_id,
            image_rgb.shape[:2],
            fallback_masks=[p["mask_bin"] for p in proposals],
        )
        if object_mask is None:
            object_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            for p in proposals:
                object_mask |= (p["mask_bin"] > 0).astype(np.uint8)
        object_stats = mask_stats(object_mask)
        if object_stats is None:
            object_stats = {"box": [0, 0, image_rgb.shape[1], image_rgb.shape[0]]}

        depth_path = find_depth_path(obj_dir, frame_id)
        depth = _load_depth_any(depth_path) if depth_path else None
        normal_rgb = build_normal_rgb_from_depth(depth, k_obj, object_mask, image_rgb.shape[:2])

        neg_masks = []
        neg_boxes = []
        neg_bbox_area = []
        neg_fg_ratio = []
        neg_aspect = []
        query_negative_valid = []
        neg_color_templates = [[] for _ in NEGATIVE_FILL_COLORS]
        neg_color_boxes = []
        neg_color_masks = []
        normal_masks = []
        normal_boxes = []
        query_normal_valid = []
        edge_templates = []
        edge_masks = []
        edge_boxes = []
        query_edge_valid = []
        for proposal, pos_box, pos_area, pos_fg, pos_asp in zip(
            proposals,
            boxes,
            q_bbox_area,
            q_fg_ratio,
            q_aspect,
        ):
            pos_mask = proposal["mask_bin"].astype(np.uint8)
            neg_mask = ((object_mask > 0) & (pos_mask == 0)).astype(np.uint8)
            st = mask_stats(neg_mask)
            if st is None or st["mask_area"] < 10:
                neg_mask = pos_mask
                st = {
                    "box": pos_box,
                    "bbox_area": float(pos_area),
                    "fg_ratio": float(pos_fg),
                    "aspect": float(pos_asp),
                }
                query_negative_valid.append(False)
            else:
                query_negative_valid.append(True)
            neg_masks.append(neg_mask.astype(np.float32))
            neg_boxes.append(st["box"])
            neg_bbox_area.append(float(st["bbox_area"]))
            neg_fg_ratio.append(float(st["fg_ratio"]))
            neg_aspect.append(float(st["aspect"]))
            neg_color_boxes.append(object_stats["box"])
            neg_color_masks.append(object_mask.astype(np.float32))
            for color_idx, color in enumerate(NEGATIVE_FILL_COLORS):
                neg_color_templates[color_idx].append(
                    _image_to_tensor(build_negative_color_rgb(image_rgb, object_mask, pos_mask, color))
                )

            roi_mask, roi_box, _, roi_ok = build_neighborhood_roi_mask(pos_mask, object_mask)
            normal_masks.append(roi_mask.astype(np.float32))
            normal_boxes.append(roi_box)
            query_normal_valid.append(bool(normal_rgb is not None and roi_ok))
            edge_img = build_edge_rgb(image_rgb, pos_mask, object_mask)
            edge_templates.append(_image_to_tensor(edge_img))
            edge_masks.append(roi_mask.astype(np.float32))
            edge_boxes.append(roi_box)
            query_edge_valid.append(bool(roi_ok))

        masks_t = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.float32)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        neg_masks_t = torch.as_tensor(np.stack(neg_masks, axis=0), dtype=torch.float32)
        neg_boxes_t = torch.as_tensor(neg_boxes, dtype=torch.float32)
        neg_color_boxes_t = torch.as_tensor(neg_color_boxes, dtype=torch.float32)
        neg_color_masks_t = torch.as_tensor(np.stack(neg_color_masks, axis=0), dtype=torch.float32)
        normal_masks_t = torch.as_tensor(np.stack(normal_masks, axis=0), dtype=torch.float32)
        normal_boxes_t = torch.as_tensor(normal_boxes, dtype=torch.float32)
        edge_templates_t = torch.stack(edge_templates, dim=0)
        edge_masks_t = torch.as_tensor(np.stack(edge_masks, axis=0), dtype=torch.float32)
        edge_boxes_t = torch.as_tensor(edge_boxes, dtype=torch.float32)

        query_sem = semantic_matcher.encode_proposals(image_rgb, masks_t, boxes_t)
        query_appe = appearance_matcher.encode_proposals(image_rgb, masks_t, boxes_t)
        _ = (neg_masks_t, neg_boxes_t)
        query_sem_neg_colors = []
        query_appe_neg_colors = []
        for color_templates in neg_color_templates:
            color_templates_t = torch.stack(color_templates, dim=0)
            query_sem_neg_colors.append(
                semantic_matcher.encode_templates(color_templates_t, neg_color_boxes_t, neg_color_masks_t)
            )
            query_appe_neg_colors.append(
                appearance_matcher.encode_templates(color_templates_t, neg_color_boxes_t, neg_color_masks_t)
            )
        normal_image_for_encode = normal_rgb if normal_rgb is not None else image_rgb
        query_sem_normal = semantic_matcher.encode_proposals(normal_image_for_encode, normal_masks_t, normal_boxes_t)
        query_appe_normal = appearance_matcher.encode_proposals(normal_image_for_encode, normal_masks_t, normal_boxes_t)
        query_sem_edge = semantic_matcher.encode_templates(edge_templates_t, edge_boxes_t, edge_masks_t)
        query_appe_edge = appearance_matcher.encode_templates(edge_templates_t, edge_boxes_t, edge_masks_t)

        sem_matrix = semantic_matcher.pairwise_similarity(query_sem, reference_sem).detach().cpu().numpy()[..., 0]
        appe_matrix = appearance_matcher.pairwise_similarity(query_appe, reference_appe).detach().cpu().numpy()[..., 0]
        neg_sem_matrices = []
        neg_appe_matrices = []
        for q_sem_neg, q_appe_neg, r_sem_neg, r_appe_neg in zip(
            query_sem_neg_colors,
            query_appe_neg_colors,
            reference_sem_neg_colors,
            reference_appe_neg_colors,
        ):
            neg_sem_matrices.append(
                semantic_matcher.pairwise_similarity(q_sem_neg, r_sem_neg).detach().cpu().numpy()[..., 0]
            )
            neg_appe_matrices.append(
                appearance_matcher.pairwise_similarity(q_appe_neg, r_appe_neg).detach().cpu().numpy()[..., 0]
            )
        neg_sem_stack = np.stack(neg_sem_matrices, axis=0)
        neg_appe_stack = np.stack(neg_appe_matrices, axis=0)
        normal_sem_matrix = semantic_matcher.pairwise_similarity(query_sem_normal, reference_sem_normal).detach().cpu().numpy()[..., 0]
        normal_appe_matrix = appearance_matcher.pairwise_similarity(query_appe_normal, reference_appe_normal).detach().cpu().numpy()[..., 0]
        edge_sem_matrix = semantic_matcher.pairwise_similarity(query_sem_edge, reference_sem_edge).detach().cpu().numpy()[..., 0]
        edge_appe_matrix = appearance_matcher.pairwise_similarity(query_appe_edge, reference_appe_edge).detach().cpu().numpy()[..., 0]

        # SAM-6D style pre-filter: keep proposals with sufficient semantic confidence.
        proposal_sem_max = sem_matrix.max(axis=1)
        candidate_idx = np.where(proposal_sem_max > semantic_thresh)[0]
        if len(candidate_idx) == 0:
            all_results[frame_id] = []
            continue

        sem_matrix = sem_matrix[candidate_idx]
        appe_matrix = appe_matrix[candidate_idx]
        neg_sem_stack = neg_sem_stack[:, candidate_idx, :]
        neg_appe_stack = neg_appe_stack[:, candidate_idx, :]
        normal_sem_matrix = normal_sem_matrix[candidate_idx]
        normal_appe_matrix = normal_appe_matrix[candidate_idx]
        edge_sem_matrix = edge_sem_matrix[candidate_idx]
        edge_appe_matrix = edge_appe_matrix[candidate_idx]

        query_stats = {
            "bbox_area": np.asarray(q_bbox_area, dtype=np.float32)[candidate_idx],
            "fg_ratio": np.asarray(q_fg_ratio, dtype=np.float32)[candidate_idx],
            "aspect": np.asarray(q_aspect, dtype=np.float32)[candidate_idx],
        }
        template_stats = {
            "bbox_area": np.asarray([m["template_bbox_area"] for m in cad_meta], dtype=np.float32),
            "fg_ratio": np.asarray([m["template_fg_ratio"] for m in cad_meta], dtype=np.float32),
            "aspect": np.asarray([m["template_aspect"] for m in cad_meta], dtype=np.float32),
        }
        geometric_score, visible_ratio = compute_geom_and_visible_scores(query_stats, template_stats)

        neg_query_stats = {
            "bbox_area": np.asarray(neg_bbox_area, dtype=np.float32)[candidate_idx],
            "fg_ratio": np.asarray(neg_fg_ratio, dtype=np.float32)[candidate_idx],
            "aspect": np.asarray(neg_aspect, dtype=np.float32)[candidate_idx],
        }
        neg_template_stats = {
            "bbox_area": np.asarray([m["template_negative_bbox_area"] for m in cad_meta], dtype=np.float32),
            "fg_ratio": np.asarray([m["template_negative_fg_ratio"] for m in cad_meta], dtype=np.float32),
            "aspect": np.asarray([m["template_negative_aspect"] for m in cad_meta], dtype=np.float32),
        }
        neg_geometric_score, neg_visible_ratio = compute_geom_and_visible_scores(neg_query_stats, neg_template_stats)

        positive_matrix = (sem_matrix + appe_matrix + geometric_score * visible_ratio) / (2.0 + visible_ratio)
        neg_sem_matrix = np.min(neg_sem_stack, axis=0)
        neg_appe_matrix = np.min(neg_appe_stack, axis=0)
        negative_matrix_colors = (
            neg_sem_stack + neg_appe_stack + neg_geometric_score[None] * neg_visible_ratio[None]
        ) / (2.0 + neg_visible_ratio[None])
        negative_matrix = np.min(negative_matrix_colors, axis=0)
        q_neg_valid = np.asarray(query_negative_valid, dtype=bool)[candidate_idx]
        t_neg_valid = np.asarray([bool(m.get("template_negative_valid", False)) for m in cad_meta], dtype=bool)
        neg_valid_pair = q_neg_valid[:, None] & t_neg_valid[None, :]
        negative_matrix = np.where(neg_valid_pair, negative_matrix, positive_matrix)

        normal_matrix = _score_from_sem_appe(normal_sem_matrix, normal_appe_matrix)
        edge_matrix = _score_from_sem_appe(edge_sem_matrix, edge_appe_matrix)
        q_normal_valid = np.asarray(query_normal_valid, dtype=bool)[candidate_idx]
        t_normal_valid = np.asarray([bool(m.get("template_normal_valid", False)) for m in cad_meta], dtype=bool)
        normal_valid_pair = q_normal_valid[:, None] & t_normal_valid[None, :]
        q_edge_valid = np.asarray(query_edge_valid, dtype=bool)[candidate_idx]
        t_edge_valid = np.asarray([bool(m.get("template_edge_valid", False)) for m in cad_meta], dtype=bool)
        edge_valid_pair = q_edge_valid[:, None] & t_edge_valid[None, :]

        base_matrix = _weighted_branch_average(
            [
                (positive_matrix, None, sam6d_pos_weight),
                (negative_matrix, None, sam6d_neg_weight),
                (normal_matrix, normal_valid_pair, sam6d_normal_weight),
                (edge_matrix, edge_valid_pair, sam6d_edge_weight),
            ],
            positive_fallback=positive_matrix,
        )
        # Stage-1 (SAM6D-like): only score and rank all CAD<->mask pairs.
        # One-to-one assignment is deferred to the adaptive rerank stage.
        local_to_global = {int(i_local): int(i_global) for i_local, i_global in enumerate(candidate_idx.tolist())}

        # Keep SAM6D candidate ranking pure. Optional render scoring is applied
        # later inside adaptive_weight.
        final_matrix = base_matrix

        frame_res = []
        matches_for_vis = []
        if finalize_one_to_one:
            ranked_pairs = assign_one_mask_per_cad(final_matrix)
            for rank, (score, i_local, c_idx) in enumerate(ranked_pairs):
                q_idx = int(local_to_global[int(i_local)])
                cad = cad_meta[int(c_idx)]
                src_mask_path = proposals[q_idx]["mask_path"]
                cad_part_id = int(cad["cad_part_id"])
                dst_mask_name = f"mask_{cad_part_id:04d}.png"
                frame_res.append(
                    {
                        "rank": int(rank),
                        "rank_for_cad": 0,
                        "proposal_index": int(q_idx),
                        "mask_path": src_mask_path,
                        "saved_mask_path": dst_mask_name,
                        "bbox": [int(v) for v in proposals[q_idx]["bbox"]],
                        "cad_part_id": cad_part_id,
                        "cad_model_dir": cad["cad_model_dir"],
                        "semantic_score": float(sem_matrix[int(i_local), int(c_idx)]),
                        "appearance_score": float(appe_matrix[int(i_local), int(c_idx)]),
                        "geometric_score": float(geometric_score[int(i_local), int(c_idx)]),
                        "visible_ratio": float(visible_ratio[int(i_local), int(c_idx)]),
                        "positive_sam6d_score": float(positive_matrix[int(i_local), int(c_idx)]),
                        "negative_sam6d_score": float(negative_matrix[int(i_local), int(c_idx)]),
                        "negative_valid": bool(neg_valid_pair[int(i_local), int(c_idx)]),
                        "negative_semantic_score": float(neg_sem_matrix[int(i_local), int(c_idx)]),
                        "negative_appearance_score": float(neg_appe_matrix[int(i_local), int(c_idx)]),
                        "negative_geometric_score": float(neg_geometric_score[int(i_local), int(c_idx)]),
                        "negative_visible_ratio": float(neg_visible_ratio[int(i_local), int(c_idx)]),
                        "normal_sam6d_score": float(normal_matrix[int(i_local), int(c_idx)]),
                        "normal_valid": bool(normal_valid_pair[int(i_local), int(c_idx)]),
                        "normal_semantic_score": float(normal_sem_matrix[int(i_local), int(c_idx)]),
                        "normal_appearance_score": float(normal_appe_matrix[int(i_local), int(c_idx)]),
                        "edge_sam6d_score": float(edge_matrix[int(i_local), int(c_idx)]),
                        "edge_valid": bool(edge_valid_pair[int(i_local), int(c_idx)]),
                        "edge_semantic_score": float(edge_sem_matrix[int(i_local), int(c_idx)]),
                        "edge_appearance_score": float(edge_appe_matrix[int(i_local), int(c_idx)]),
                        "sam6d_base_score": float(base_matrix[int(i_local), int(c_idx)]),
                        "sam6d_positive_weight": float(sam6d_pos_weight),
                        "sam6d_negative_weight": float(sam6d_neg_weight),
                        "sam6d_normal_weight": float(sam6d_normal_weight),
                        "sam6d_edge_weight": float(sam6d_edge_weight),
                        "score": float(score),
                    }
                )
                matches_for_vis.append((int(q_idx), int(c_idx), float(score)))
        else:
            ranked_pairs = greedy_one_to_one(final_matrix, score_thresh=float(score_thresh))
            cad_rank_counter = {}
            for rank, (score, i_local, c_idx) in enumerate(ranked_pairs):
                q_idx = int(local_to_global[int(i_local)])
                cad = cad_meta[int(c_idx)]
                src_mask_path = proposals[q_idx]["mask_path"]
                cad_part_id = int(cad["cad_part_id"])
                cad_rank = int(cad_rank_counter.get(cad_part_id, 0))
                cad_rank_counter[cad_part_id] = cad_rank + 1
                dst_mask_name = f"mask_cad_{cad_part_id:04d}_rk{cad_rank:02d}_p{int(q_idx):03d}.png"
                frame_res.append(
                    {
                        "rank": int(rank),
                        "rank_for_cad": int(cad_rank),
                        "proposal_index": int(q_idx),
                        "mask_path": src_mask_path,
                        "saved_mask_path": dst_mask_name,
                        "bbox": [int(v) for v in proposals[q_idx]["bbox"]],
                        "cad_part_id": cad_part_id,
                        "cad_model_dir": cad["cad_model_dir"],
                        "semantic_score": float(sem_matrix[int(i_local), int(c_idx)]),
                        "appearance_score": float(appe_matrix[int(i_local), int(c_idx)]),
                        "geometric_score": float(geometric_score[int(i_local), int(c_idx)]),
                        "visible_ratio": float(visible_ratio[int(i_local), int(c_idx)]),
                        "positive_sam6d_score": float(positive_matrix[int(i_local), int(c_idx)]),
                        "negative_sam6d_score": float(negative_matrix[int(i_local), int(c_idx)]),
                        "negative_valid": bool(neg_valid_pair[int(i_local), int(c_idx)]),
                        "negative_semantic_score": float(neg_sem_matrix[int(i_local), int(c_idx)]),
                        "negative_appearance_score": float(neg_appe_matrix[int(i_local), int(c_idx)]),
                        "negative_geometric_score": float(neg_geometric_score[int(i_local), int(c_idx)]),
                        "negative_visible_ratio": float(neg_visible_ratio[int(i_local), int(c_idx)]),
                        "normal_sam6d_score": float(normal_matrix[int(i_local), int(c_idx)]),
                        "normal_valid": bool(normal_valid_pair[int(i_local), int(c_idx)]),
                        "normal_semantic_score": float(normal_sem_matrix[int(i_local), int(c_idx)]),
                        "normal_appearance_score": float(normal_appe_matrix[int(i_local), int(c_idx)]),
                        "edge_sam6d_score": float(edge_matrix[int(i_local), int(c_idx)]),
                        "edge_valid": bool(edge_valid_pair[int(i_local), int(c_idx)]),
                        "edge_semantic_score": float(edge_sem_matrix[int(i_local), int(c_idx)]),
                        "edge_appearance_score": float(edge_appe_matrix[int(i_local), int(c_idx)]),
                        "sam6d_base_score": float(base_matrix[int(i_local), int(c_idx)]),
                        "sam6d_positive_weight": float(sam6d_pos_weight),
                        "sam6d_negative_weight": float(sam6d_neg_weight),
                        "sam6d_normal_weight": float(sam6d_normal_weight),
                        "sam6d_edge_weight": float(sam6d_edge_weight),
                        "score": float(score),
                    }
                )
                if cad_rank < max(1, int(topk_per_frame)):
                    matches_for_vis.append((int(q_idx), int(c_idx), float(score)))

        matched_frame_dir = os.path.join(matched_mask_root, frame_id)
        os.makedirs(matched_frame_dir, exist_ok=True)
        for item in frame_res:
            src_mask_path = item["mask_path"]
            src_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if src_mask is None:
                continue
            dst_name = os.path.basename(item["saved_mask_path"])
            dst_mask_path = os.path.join(matched_frame_dir, dst_name)
            cv2.imwrite(dst_mask_path, src_mask)
            item["saved_mask_path"] = dst_mask_path
        for it in frame_res:
            it["skipped_invisible_cad_part_ids"] = [int(x) for x in invisible_cad_part_ids]
        all_results[frame_id] = frame_res

        out_img = os.path.join(out_dir, f"{frame_id}.jpg")
        visualize_frame(image_bgr, proposals, matches_for_vis, cad_meta, out_img)
        if finalize_one_to_one:
            print(
                f"[MATCH-SAM6D-FINAL] {frame_id}: proposals={len(proposals)}, candidates={len(candidate_idx)}, "
                f"selected_pairs={len(frame_res)} -> vis:{out_img}, kept_masks:{matched_frame_dir}"
            )
        else:
            print(
                f"[MATCH-SAM6D] {frame_id}: proposals={len(proposals)}, candidates={len(candidate_idx)}, "
                f"ranked_pairs={len(frame_res)}, vis_topk_per_cad={len(matches_for_vis)} -> vis:{out_img}, kept_masks:{matched_frame_dir}"
            )

    out_json = os.path.join(out_dir, "match_results_sam6d_style.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved results to {out_json}")
