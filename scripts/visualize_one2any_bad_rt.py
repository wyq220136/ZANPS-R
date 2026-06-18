import argparse
import json
import os
import pickle
import re
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import trimesh


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def natural_sort_key(value: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(value))]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pose_txt(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4):
            return None
        return pose.astype(np.float64)
    except Exception:
        return None


def load_intrinsic(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        K = np.loadtxt(path, dtype=np.float64)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        if K.shape != (3, 3):
            return None
        return K.astype(np.float64)
    except Exception:
        return None


def build_rt_4x4(pose_rts_param: dict) -> np.ndarray:
    rot = np.asarray(pose_rts_param["R"], dtype=np.float64).reshape(3, 3)
    trans = np.asarray(pose_rts_param["T"], dtype=np.float64).reshape(3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def metric_float(record: dict, *names: str) -> Optional[float]:
    for name in names:
        if name not in record or record.get(name) is None:
            continue
        try:
            value = float(record[name])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def iter_detail_records(detail) -> Iterable[dict]:
    if isinstance(detail, dict):
        for split, split_items in detail.items():
            if not isinstance(split_items, list):
                continue
            for obj_item in split_items:
                if not isinstance(obj_item, dict):
                    continue
                obj_name = str(obj_item.get("object", ""))
                samples = obj_item.get("samples", None)
                if isinstance(samples, list):
                    for sample in samples:
                        if not isinstance(sample, dict):
                            continue
                        rec = dict(sample)
                        rec.setdefault("object", obj_name)
                        rec.setdefault("split", split)
                        yield rec
                elif obj_item.get("frame_id") is not None:
                    rec = dict(obj_item)
                    rec.setdefault("split", split)
                    yield rec
    elif isinstance(detail, list):
        for item in detail:
            if isinstance(item, dict):
                if isinstance(item.get("samples"), list):
                    obj_name = str(item.get("object", ""))
                    split = str(item.get("split", ""))
                    for sample in item["samples"]:
                        if isinstance(sample, dict):
                            rec = dict(sample)
                            rec.setdefault("object", obj_name)
                            rec.setdefault("split", split)
                            yield rec
                else:
                    yield dict(item)


def rank_score(record: dict, args) -> float:
    re_deg = metric_float(record, "re", "R_e_deg", "R_e", "rot_error")
    te = metric_float(record, "te", "T_e", "T_e_m", "trans_error")
    te_scaled = metric_float(record, "te_scaled", "T_e_scaled", "T_e_cm")
    if te_scaled is None and te is not None:
        te_scaled = te * float(args.te_unit_scale)

    re_val = -np.inf if re_deg is None else re_deg
    te_val = -np.inf if te is None else te
    te_scaled_val = -np.inf if te_scaled is None else te_scaled

    if args.sort_by == "re":
        return float(re_val)
    if args.sort_by == "te":
        return float(te_val)
    if args.sort_by == "te_scaled":
        return float(te_scaled_val)
    if args.sort_by == "sum":
        re_norm = 0.0 if re_deg is None else re_deg / float(args.re_ref)
        te_norm = 0.0 if te_scaled is None else te_scaled / float(args.te_scaled_ref)
        return float(re_norm + te_norm)

    re_norm = 0.0 if re_deg is None else re_deg / float(args.re_ref)
    te_norm = 0.0 if te_scaled is None else te_scaled / float(args.te_scaled_ref)
    return float(max(re_norm, te_norm))


def select_bad_records(records: List[dict], args) -> List[dict]:
    candidates = []
    per_object_count: Dict[str, int] = {}
    for rec in records:
        if args.require_ok and str(rec.get("status", "")).lower() not in ("ok", ""):
            continue
        re_deg = metric_float(rec, "re", "R_e_deg", "R_e", "rot_error")
        te = metric_float(rec, "te", "T_e", "T_e_m", "trans_error")
        te_scaled = metric_float(rec, "te_scaled", "T_e_scaled", "T_e_cm")
        if te_scaled is None and te is not None:
            te_scaled = te * float(args.te_unit_scale)

        if args.min_re is not None and (re_deg is None or re_deg < args.min_re):
            if args.threshold_mode == "and":
                continue
            if args.min_te_scaled is None and args.min_te is None:
                continue
        if args.min_te is not None and (te is None or te < args.min_te):
            if args.threshold_mode == "and":
                continue
            if args.min_re is None and args.min_te_scaled is None:
                continue
        if args.min_te_scaled is not None and (te_scaled is None or te_scaled < args.min_te_scaled):
            if args.threshold_mode == "and":
                continue
            if args.min_re is None and args.min_te is None:
                continue

        if args.threshold_mode == "or" and any(v is not None for v in (args.min_re, args.min_te, args.min_te_scaled)):
            hit = False
            if args.min_re is not None and re_deg is not None and re_deg >= args.min_re:
                hit = True
            if args.min_te is not None and te is not None and te >= args.min_te:
                hit = True
            if args.min_te_scaled is not None and te_scaled is not None and te_scaled >= args.min_te_scaled:
                hit = True
            if not hit:
                continue

        rec = dict(rec)
        rec["_score"] = rank_score(rec, args)
        rec["_re"] = re_deg
        rec["_te"] = te
        rec["_te_scaled"] = te_scaled
        candidates.append(rec)

    candidates.sort(key=lambda r: (r["_score"], r.get("_re") or -np.inf, r.get("_te_scaled") or -np.inf), reverse=True)
    selected = []
    for rec in candidates:
        obj = str(rec.get("object", ""))
        if args.max_per_object > 0 and per_object_count.get(obj, 0) >= args.max_per_object:
            continue
        selected.append(rec)
        per_object_count[obj] = per_object_count.get(obj, 0) + 1
        if len(selected) >= args.top_k:
            break
    return selected


def resolve_path(path: str, *bases: str) -> str:
    if not path:
        return ""
    path = str(path).replace("/", os.sep)
    if os.path.exists(path):
        return path
    if os.path.isabs(path):
        for marker in (f"objs{os.sep}", f"gt_pose_from_ann{os.sep}", f"gt_pose{os.sep}"):
            idx = path.find(marker)
            if idx >= 0:
                rel = path[idx:]
                for base in bases:
                    cand = os.path.join(base, rel)
                    if os.path.exists(cand):
                        return cand
        return path
    for base in bases:
        if not base:
            continue
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(bases[0], path) if bases else path


def resolve_split_root(dataset_root: str, split: str) -> str:
    root = os.path.abspath(dataset_root)
    if os.path.isdir(os.path.join(root, "objs")) or os.path.isdir(os.path.join(root, "gt_pose_from_ann")):
        return root
    return os.path.join(root, split)


def find_image_path(obj_dir: str, split_root: str, frame_id: str) -> str:
    folders = [
        os.path.join(obj_dir, "rgb"),
        os.path.join(split_root, "rgb"),
    ]
    for folder in folders:
        for ext in IMAGE_EXTS:
            path = os.path.join(folder, f"{frame_id}{ext}")
            if os.path.exists(path):
                return path
    return ""


def load_mapping(obj_dir: str, mapping_name: str) -> Dict[str, dict]:
    path = os.path.join(obj_dir, mapping_name)
    if not os.path.exists(path):
        return {}
    try:
        data = load_json(path)
    except Exception:
        return {}
    out = {}
    for item in data.get("mapping", []):
        part_key = str(item.get("part_key", ""))
        link_name = str(item.get("link_name", ""))
        if part_key:
            out[part_key] = item
        if link_name:
            out.setdefault(link_name, item)
    return out


def lookup_mapping(mapping: Dict[str, dict], part_key: str, link_name: str = "") -> Optional[dict]:
    for key in (part_key, link_name):
        item = mapping.get(str(key), None)
        if isinstance(item, dict):
            return item
    wanted = [trailing_int(part_key), trailing_int(link_name)]
    for key, item in mapping.items():
        idx = trailing_int(key)
        if idx is not None and idx in wanted:
            return item
    return None


def trailing_int(text: str) -> Optional[int]:
    match = re.search(r"(\d+)$", str(text))
    return int(match.group(1)) if match else None


def load_adapter_part_names(obj_dir: str) -> List[str]:
    path = os.path.join(obj_dir, "dataset_train_val_adapter_parts.json")
    if not os.path.exists(path):
        return []
    try:
        data = load_json(path)
    except Exception:
        return []
    pairs = []
    for item in data.get("parts", []):
        if item.get("index") is not None and item.get("name") is not None:
            pairs.append((int(item["index"]), str(item["name"])))
    return [name for _, name in sorted(pairs)]


def resolve_mesh_path(split_root: str, obj_dir: str, part_key: str, link_name: str, mapping: Dict[str, dict]) -> str:
    item = lookup_mapping(mapping, part_key, link_name)
    if item is not None:
        mesh_rel = str(item.get("model_path", ""))
        if mesh_rel:
            path = resolve_path(mesh_rel, split_root, obj_dir)
            if os.path.exists(path):
                return path

    part_names = load_adapter_part_names(obj_dir)
    for idx in [trailing_int(part_key), trailing_int(link_name)]:
        if idx is None:
            continue
        if idx < len(part_names):
            path = os.path.join(obj_dir, "models", part_names[idx], "model.obj")
            if os.path.exists(path):
                return path
        candidates = [
            f"model_{idx:04d}",
            f"model_{idx}",
            f"link_{idx}",
            f"part_{idx}",
            str(part_key),
            str(link_name),
        ]
        for name in candidates:
            path = os.path.join(obj_dir, "models", name, "model.obj")
            if os.path.exists(path):
                return path
    return ""


def load_mesh_bbox(mesh_path: str, cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if mesh_path in cache:
        return cache[mesh_path]
    if not mesh_path or not os.path.exists(mesh_path):
        return None
    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[0] == 0:
            return None
        bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
        cache[mesh_path] = bbox
        return bbox
    except Exception:
        return None


def load_annotation_local_bbox(split_root: str, frame_id: str, link_name: str) -> Optional[np.ndarray]:
    if not frame_id or not link_name:
        return None
    bbox_path = os.path.join(split_root, "bbox", f"{frame_id}.pkl")
    if not os.path.exists(bbox_path):
        return None
    try:
        with open(bbox_path, "rb") as f:
            bbox_data = pickle.load(f)
        bbox_pose_dict = bbox_data.get("bbox_pose_dict", {})
        if not isinstance(bbox_pose_dict, dict):
            return None
        info = bbox_pose_dict.get(link_name, None)
        if info is None:
            link_idx = trailing_int(link_name)
            for cand_name, cand_info in bbox_pose_dict.items():
                if link_idx is not None and trailing_int(cand_name) == link_idx:
                    info = cand_info
                    break
        if info is None:
            return None
        bbox_world = np.asarray(info.get("bbox", []), dtype=np.float64)
        if bbox_world.shape != (8, 3):
            return None
        pose_rts = info.get("pose_RTS_param", None)
        if pose_rts is None:
            return None
        pose_world = build_rt_4x4(pose_rts)
        local_bbox = (pose_world[:3, :3].T @ (bbox_world - pose_world[:3, 3]).T).T
        return local_bbox.astype(np.float64)
    except Exception:
        return None


def project_point(K: np.ndarray, pose: np.ndarray, point_obj: np.ndarray) -> Optional[Tuple[int, int]]:
    point_cam = (pose @ point_obj.reshape(4, 1))[:3, 0]
    if point_cam[2] <= 1e-6:
        return None
    uvw = K @ point_cam
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def draw_line(img, p0, p1, color, thickness: int):
    if p0 is None or p1 is None:
        return
    cv2.line(img, p0, p1, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_posed_bbox_3d(img, K: np.ndarray, pose: np.ndarray, bbox3d: np.ndarray, color, thickness: int):
    mn = bbox3d.min(axis=0)
    mx = bbox3d.max(axis=0)
    corners = np.array(
        [
            [mn[0], mn[1], mn[2], 1.0],
            [mx[0], mn[1], mn[2], 1.0],
            [mx[0], mx[1], mn[2], 1.0],
            [mn[0], mx[1], mn[2], 1.0],
            [mn[0], mn[1], mx[2], 1.0],
            [mx[0], mn[1], mx[2], 1.0],
            [mx[0], mx[1], mx[2], 1.0],
            [mn[0], mx[1], mx[2], 1.0],
        ],
        dtype=np.float64,
    )
    uv = [project_point(K, pose, point) for point in corners]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        draw_line(img, uv[a], uv[b], color, thickness)


def draw_posed_annotation_bbox_3d(img, K: np.ndarray, pose: np.ndarray, local_bbox: np.ndarray, color, thickness: int):
    corners = np.concatenate(
        [local_bbox.astype(np.float64), np.ones((local_bbox.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    uv = [project_point(K, pose, point) for point in corners]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        draw_line(img, uv[a], uv[b], color, thickness)


def draw_xyz_axis(
    img,
    K: np.ndarray,
    pose: np.ndarray,
    bbox3d: np.ndarray,
    palette: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
    axis_ratio: float,
    min_axis_scale: float,
    max_axis_scale: float,
    thickness: int,
    scale_mode: str = "extent",
    origin_mode: str = "pose",
):
    mn = bbox3d.min(axis=0)
    mx = bbox3d.max(axis=0)
    extent = mx - mn
    origin = np.zeros(3, dtype=np.float64)
    if origin_mode == "bbox-center":
        origin = (mn + mx) * 0.5

    center = project_point(K, pose, np.array([origin[0], origin[1], origin[2], 1.0], dtype=np.float64))
    if center is None:
        return

    if scale_mode == "diag":
        scale = float(np.clip(np.linalg.norm(extent) * axis_ratio, min_axis_scale, max_axis_scale))
        scales = np.array([scale, scale, scale], dtype=np.float64)
    else:
        scales = np.clip(extent.astype(np.float64) * axis_ratio, min_axis_scale, max_axis_scale)

    endpoints = [
        np.array([origin[0] + scales[0], origin[1], origin[2], 1.0], dtype=np.float64),
        np.array([origin[0], origin[1] + scales[1], origin[2], 1.0], dtype=np.float64),
        np.array([origin[0], origin[1], origin[2] + scales[2], 1.0], dtype=np.float64),
    ]
    for endpoint, color in zip(endpoints, palette):
        draw_line(img, center, project_point(K, pose, endpoint), color, thickness)


def annotate(img, lines: List[str]):
    y = 26
    for line in lines:
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def fmt_float(value, fmt: str, none_text: str = "nan") -> str:
    if value is None:
        return none_text
    try:
        return format(float(value), fmt)
    except Exception:
        return none_text


def visualize_record(rec: dict, args, bbox_cache: Dict[str, np.ndarray]) -> Tuple[bool, dict]:
    split = str(rec.get("split", ""))
    obj_name = str(rec.get("object", ""))
    frame_id = str(rec.get("frame_id", ""))
    part_key = str(rec.get("part_key", ""))
    link_name = str(rec.get("link_name", ""))
    split_root = resolve_split_root(args.dataset_root, split)
    obj_dir = os.path.join(split_root, "objs", obj_name)

    info = {
        "split": split,
        "object": obj_name,
        "frame_id": frame_id,
        "part_key": part_key,
        "link_name": link_name,
        "re": rec.get("_re"),
        "te": rec.get("_te"),
        "te_scaled": rec.get("_te_scaled"),
        "score": rec.get("_score"),
        "status": "pending",
    }

    K = load_intrinsic(os.path.join(obj_dir, "K.txt"))
    if K is None:
        info["status"] = "missing_K"
        return False, info

    rgb_path = find_image_path(obj_dir, split_root, frame_id)
    if not rgb_path:
        info["status"] = "missing_rgb"
        return False, info
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        info["status"] = "read_rgb_failed"
        info["rgb_path"] = rgb_path
        return False, info

    pred_pose_path = resolve_path(str(rec.get("pred_pose_path", "")), split_root, obj_dir)
    if not os.path.exists(pred_pose_path) and part_key:
        pred_pose_path = os.path.join(obj_dir, args.pred_subdir, "poses", frame_id, f"{part_key}.txt")
    pred_pose = load_pose_txt(pred_pose_path)
    if pred_pose is None:
        info["status"] = "missing_pred_pose"
        info["pred_pose_path"] = pred_pose_path
        return False, info

    gt_pose_path = resolve_path(str(rec.get("gt_pose_path", "")), split_root, obj_dir)
    if not os.path.exists(gt_pose_path) and link_name:
        gt_pose_path = os.path.join(split_root, args.gt_pose_dir, f"{frame_id}__{link_name}.txt")
    gt_pose = load_pose_txt(gt_pose_path)
    if gt_pose is None:
        info["status"] = "missing_gt_pose"
        info["gt_pose_path"] = gt_pose_path
        return False, info

    mapping = load_mapping(obj_dir, args.mapping_name)
    mesh_path = resolve_mesh_path(split_root, obj_dir, part_key, link_name, mapping)
    bbox3d = load_mesh_bbox(mesh_path, bbox_cache)
    if bbox3d is None:
        info["status"] = "missing_mesh_bbox"
        info["mesh_path"] = mesh_path
        return False, info

    gt_local_bbox = load_annotation_local_bbox(split_root, frame_id, link_name)
    if gt_local_bbox is None:
        info["status"] = "missing_gt_annotation_bbox"
        info["mesh_path"] = mesh_path
        return False, info

    draw_posed_annotation_bbox_3d(
        img,
        K,
        gt_pose,
        gt_local_bbox,
        color=(0, 255, 255),
        thickness=args.bbox_thickness,
    )
    draw_xyz_axis(
        img,
        K,
        gt_pose,
        gt_local_bbox,
        palette=((0, 0, 255), (0, 180, 0), (255, 0, 0)),
        axis_ratio=args.axis_ratio,
        min_axis_scale=args.min_axis_scale,
        max_axis_scale=args.max_axis_scale,
        thickness=args.axis_thickness,
        scale_mode=args.axis_scale_mode,
        origin_mode=args.axis_origin,
    )
    draw_posed_bbox_3d(img, K, pred_pose, bbox3d, color=(255, 0, 255), thickness=args.bbox_thickness)
    draw_xyz_axis(
        img,
        K,
        pred_pose,
        bbox3d,
        palette=((0, 128, 255), (255, 255, 0), (255, 0, 255)),
        axis_ratio=args.axis_ratio,
        min_axis_scale=args.min_axis_scale,
        max_axis_scale=args.max_axis_scale,
        thickness=args.axis_thickness,
        scale_mode=args.axis_scale_mode,
        origin_mode=args.axis_origin,
    )

    if args.draw_labels:
        annotate(
            img,
            [
                "GT annotation bbox yellow, One2Any bbox magenta",
                f"{split}/{obj_name}/{frame_id} {part_key} {link_name}",
                (
                    f"Re={fmt_float(info['re'], '.2f')} deg  "
                    f"Te={fmt_float(info['te'], '.4f')} m  "
                    f"Te_scaled={fmt_float(info['te_scaled'], '.2f')}"
                ),
            ],
        )

    safe_name = "__".join([split, obj_name, frame_id, part_key or link_name]).replace(os.sep, "_")
    out_path = os.path.join(args.output_dir, f"{safe_name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

    info.update(
        {
            "status": "ok",
            "rgb_path": rgb_path,
            "pred_pose_path": pred_pose_path,
            "gt_pose_path": gt_pose_path,
            "mesh_path": mesh_path,
            "gt_bbox_source": os.path.join(split_root, "bbox", f"{frame_id}.pkl"),
            "output_path": out_path,
        }
    )
    return True, info


def run(args):
    detail = load_json(args.detail_json)
    records = list(iter_detail_records(detail))
    selected = select_bad_records(records, args)
    os.makedirs(args.output_dir, exist_ok=True)

    bbox_cache: Dict[str, np.ndarray] = {}
    outputs = []
    ok = 0
    for rec in selected:
        success, info = visualize_record(rec, args, bbox_cache)
        outputs.append(info)
        ok += int(success)
        if success:
            print(f"[OK] {info['output_path']}")
        else:
            print(f"[SKIP] {info['split']}/{info['object']}/{info['frame_id']}/{info['part_key']}: {info['status']}")

    summary = {
        "detail_json": args.detail_json,
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "num_records_in_json": len(records),
        "num_selected": len(selected),
        "num_visualized": ok,
        "selection": {
            "top_k": args.top_k,
            "sort_by": args.sort_by,
            "re_ref": args.re_ref,
            "te_scaled_ref": args.te_scaled_ref,
            "min_re": args.min_re,
            "min_te": args.min_te,
            "min_te_scaled": args.min_te_scaled,
            "threshold_mode": args.threshold_mode,
            "max_per_object": args.max_per_object,
        },
        "records": outputs,
    }
    summary_path = os.path.join(args.output_dir, "selected_records.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] visualized={ok}/{len(selected)} summary={summary_path}")


def get_args():
    parser = argparse.ArgumentParser(
        "Select high-Re/Te One2Any cases and overlay GT/One2Any bbox+axis on RGB."
    )
    parser.add_argument("--detail-json", type=str, default="one2any_eval_rt_all_splits_detail.json")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="one2any_bad_rt_vis")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-per-object", type=int, default=3, help="0 means no per-object limit.")
    parser.add_argument("--sort-by", choices=["max", "sum", "re", "te", "te_scaled"], default="max")
    parser.add_argument("--re-ref", type=float, default=90.0, help="Re normalization for max/sum ranking.")
    parser.add_argument("--te-scaled-ref", type=float, default=10.0, help="Te_scaled normalization for max/sum ranking.")
    parser.add_argument("--te-unit-scale", type=float, default=100.0, help="Convert Te meters to Te_scaled centimeters.")
    parser.add_argument("--min-re", type=float, default=None)
    parser.add_argument("--min-te", type=float, default=None)
    parser.add_argument("--min-te-scaled", type=float, default=None)
    parser.add_argument("--threshold-mode", choices=["or", "and"], default="or")
    parser.add_argument("--require-ok", action="store_true", default=True)
    parser.add_argument("--include-non-ok", dest="require_ok", action="store_false")
    parser.add_argument("--pred-subdir", type=str, default="one2any_results")
    parser.add_argument("--gt-pose-dir", type=str, default="gt_pose_from_ann")
    parser.add_argument("--mapping-name", type=str, default="part_mapping_first_frame.json")
    parser.add_argument("--bbox-thickness", type=int, default=2)
    parser.add_argument("--axis-thickness", type=int, default=2)
    parser.add_argument("--axis-ratio", type=float, default=0.35)
    parser.add_argument("--min-axis-scale", type=float, default=0.0)
    parser.add_argument("--max-axis-scale", type=float, default=0.10)
    parser.add_argument(
        "--axis-scale-mode",
        choices=["extent", "diag"],
        default="extent",
        help="extent keeps each axis proportional to that bbox dimension; diag reproduces annotate.py-style single length.",
    )
    parser.add_argument(
        "--axis-origin",
        choices=["pose", "bbox-center"],
        default="pose",
        help="pose draws from the pose origin; bbox-center keeps the visual axis centered in the drawn bbox.",
    )
    parser.add_argument("--draw-labels", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
