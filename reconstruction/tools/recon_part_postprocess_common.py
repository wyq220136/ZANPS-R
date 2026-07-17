import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from recon_utils import (
    DatasetObject,
    ensure_dir,
    find_image,
    list_parts,
    load_depth_m,
    load_k,
    load_mask,
    load_pose,
    mask_path_for_part_frame,
    method_models_dir,
    method_object_dir,
    method_pose_ready_dir,
    model_obj_path,
    part_model_name,
    pose_path_for_part_frame,
    select_best_frame_for_part,
    write_json,
)


THIN_PART_KEYWORDS = ("door", "drawer", "lid", "cover", "panel")


def _trimesh():
    try:
        import trimesh
    except Exception as e:
        raise RuntimeError("trimesh is required for reference-only mesh part cutting.") from e
    return trimesh


def _as_mesh(mesh_obj):
    tm = _trimesh()
    if isinstance(mesh_obj, tm.Scene):
        geoms = [g for g in mesh_obj.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            raise ValueError("mesh scene is empty")
        mesh_obj = tm.util.concatenate(geoms)
    if not isinstance(mesh_obj, tm.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh_obj)!r}")
    if len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("mesh has no vertices/faces")
    return tm.Trimesh(
        vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
        faces=np.asarray(mesh_obj.faces, dtype=np.int64),
        visual=mesh_obj.visual,
        process=False,
    )


def _part_semantic(part_name: str) -> str:
    raw = str(part_name).lower().replace("-", "_")
    tokens = [t for t in raw.split("_") if t and not t.isdigit()]
    return "_".join(tokens) if tokens else raw


def _is_thin_part(part_name: str) -> bool:
    s = str(part_name).lower()
    return any(k in s for k in THIN_PART_KEYWORDS)


def _find_object_mask(obj: DatasetObject, frame: str, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    candidates = []
    for folder in ("object_masks", "object_mask", "objectmask", "mask"):
        root = obj.root / folder
        if not root.is_dir():
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            candidates.append(root / f"{frame}{ext}")
    for path in candidates:
        if path.exists():
            return load_mask(path, shape_hw=shape_hw)
    return None


def _project(points_obj: np.ndarray, ob_in_cam: np.ndarray, k: np.ndarray):
    pts_cam = (ob_in_cam[:3, :3] @ points_obj.T).T + ob_in_cam[:3, 3]
    z = pts_cam[:, 2]
    valid_z = z > 1e-6
    u = np.zeros_like(z, dtype=np.float32)
    v = np.zeros_like(z, dtype=np.float32)
    u[valid_z] = k[0, 0] * pts_cam[valid_z, 0] / z[valid_z] + k[0, 2]
    v[valid_z] = k[1, 1] * pts_cam[valid_z, 1] / z[valid_z] + k[1, 2]
    return u, v, z, valid_z


def _sample_mask(mask: np.ndarray, u: np.ndarray, v: np.ndarray, valid_z: np.ndarray):
    h, w = mask.shape[:2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside_img = valid_z & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(len(u), dtype=bool)
    idx = np.where(inside_img)[0]
    if len(idx) > 0:
        out[idx] = mask[vi[idx], ui[idx]]
    return out, inside_img, ui, vi


def _depth_inliers(depth_m: np.ndarray, ui: np.ndarray, vi: np.ndarray, inside_img: np.ndarray, z: np.ndarray, thresh: float):
    h, w = depth_m.shape[:2]
    valid = inside_img.copy()
    valid &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.zeros(len(z), dtype=bool)
    residual = np.full(len(z), np.nan, dtype=np.float32)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out, residual, np.zeros(len(z), dtype=bool)
    obs = depth_m[vi[idx], ui[idx]]
    has_depth = obs > 1e-6
    good_idx = idx[has_depth]
    if len(good_idx) > 0:
        res = np.abs(z[good_idx] - obs[has_depth])
        residual[good_idx] = res.astype(np.float32)
        out[good_idx] = res <= float(thresh)
    has_depth_mask = np.zeros(len(z), dtype=bool)
    has_depth_mask[good_idx] = True
    return out, residual, has_depth_mask


def _projection_score(mesh, mask: np.ndarray, depth_m: np.ndarray, k: np.ndarray, ob_in_cam: np.ndarray, args):
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = verts[faces].mean(axis=1)

    cu, cv, cz, cvalid_z = _project(centers, ob_in_cam, k)
    center_in_mask, center_in_img, cui, cvi = _sample_mask(mask, cu, cv, cvalid_z)
    center_depth_ok, center_depth_res, center_has_depth = _depth_inliers(
        depth_m,
        cui,
        cvi,
        center_in_img,
        cz,
        float(args.partcut_depth_thresh),
    )

    vu, vv, vz, vvalid_z = _project(verts, ob_in_cam, k)
    vert_in_mask, vert_in_img, _, _ = _sample_mask(mask, vu, vv, vvalid_z)
    face_vert_votes = vert_in_mask[faces].sum(axis=1)
    face_vert_visible = vert_in_img[faces].sum(axis=1)

    in_support = center_in_mask | (face_vert_votes >= 2)
    if bool(args.partcut_use_depth):
        has_depth = center_has_depth & center_in_mask
        depth_support = (~has_depth) | center_depth_ok
        in_support = in_support & depth_support

    visible_faces = center_in_img | (face_vert_visible > 0)
    face_keep = in_support.copy()

    stats = {
        "faces": int(len(faces)),
        "vertices": int(len(verts)),
        "center_visible_ratio": float(np.mean(center_in_img)) if len(faces) else 0.0,
        "center_inside_part_ratio": float(np.mean(center_in_mask[center_in_img])) if int(center_in_img.sum()) else 0.0,
        "face_keep_ratio": float(np.mean(face_keep)) if len(face_keep) else 0.0,
        "outside_part_ratio": float(np.mean((~center_in_mask) & center_in_img)) if len(faces) else 0.0,
        "depth_inlier_ratio": float(np.mean(center_depth_ok[center_has_depth])) if int(center_has_depth.sum()) else 0.0,
        "mean_depth_residual": float(np.nanmean(center_depth_res)) if np.any(np.isfinite(center_depth_res)) else None,
        "visible_faces": int(np.count_nonzero(visible_faces)),
        "kept_faces": int(np.count_nonzero(face_keep)),
    }
    if stats["center_inside_part_ratio"] > 0.0:
        stats["leakage_score"] = float(stats["outside_part_ratio"] / max(stats["center_inside_part_ratio"], 1e-6))
    else:
        stats["leakage_score"] = float("inf") if stats["outside_part_ratio"] > 0 else 0.0
    return face_keep, stats, {
        "center_u": cu,
        "center_v": cv,
        "center_in_img": center_in_img,
        "center_in_mask": center_in_mask,
        "face_keep": face_keep,
    }


def _component_select(mesh, mask: np.ndarray, k: np.ndarray, ob_in_cam: np.ndarray):
    comps = mesh.split(only_watertight=False)
    if not comps:
        return mesh, {"components": 0, "selected": -1}
    best_i = 0
    best_score = -1.0
    reports = []
    for i, comp in enumerate(comps):
        verts = np.asarray(comp.vertices, dtype=np.float32)
        if len(verts) == 0:
            reports.append({"index": i, "vertices": 0, "faces": int(len(comp.faces)), "inside_ratio": 0.0})
            continue
        u, v, z, valid_z = _project(verts, ob_in_cam, k)
        in_mask, in_img, _, _ = _sample_mask(mask, u, v, valid_z)
        visible = int(np.count_nonzero(in_img))
        inside_ratio = float(np.mean(in_mask[in_img])) if visible > 0 else 0.0
        # Favor semantic support first, but avoid tiny fragments when ratios tie.
        score = inside_ratio + 0.01 * np.log1p(max(1, len(comp.faces)))
        reports.append(
            {
                "index": i,
                "vertices": int(len(comp.vertices)),
                "faces": int(len(comp.faces)),
                "visible_vertices": visible,
                "inside_ratio": inside_ratio,
                "score": float(score),
            }
        )
        if score > best_score:
            best_score = score
            best_i = i
    return comps[best_i], {"components": len(comps), "selected": int(best_i), "reports": reports}


def _remove_degenerate_faces(mesh):
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
        return mesh
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def _remove_duplicate_faces(mesh):
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
        return mesh
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    return mesh


def _remove_unreferenced_vertices(mesh):
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    return mesh


def _repair_mesh(mesh):
    tm = _trimesh()
    repaired = mesh.copy()
    try:
        _remove_degenerate_faces(repaired)
    except Exception:
        pass
    try:
        _remove_duplicate_faces(repaired)
    except Exception:
        pass
    try:
        _remove_unreferenced_vertices(repaired)
    except Exception:
        pass
    try:
        tm.repair.fix_normals(repaired)
    except Exception:
        pass
    try:
        tm.repair.fill_holes(repaired)
    except Exception:
        pass
    try:
        _remove_unreferenced_vertices(repaired)
    except Exception:
        pass
    return repaired


def _write_debug_overlay(rgb_path: Path, mask: np.ndarray, projection: Dict[str, np.ndarray], out_path: Path, max_points: int = 5000):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return
    if rgb.shape[:2] != mask.shape[:2]:
        mask_vis = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    else:
        mask_vis = mask
    overlay = rgb.copy()
    overlay[mask_vis] = (0.55 * overlay[mask_vis] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    u = projection["center_u"]
    v = projection["center_v"]
    in_img = projection["center_in_img"]
    keep = projection["face_keep"]
    idx = np.where(in_img)[0]
    if len(idx) > max_points:
        idx = idx[np.linspace(0, len(idx) - 1, int(max_points)).astype(np.int64)]
    for i in idx:
        x = int(round(float(u[i])))
        y = int(round(float(v[i])))
        color = (0, 255, 0) if bool(keep[i]) else (0, 0, 255)
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            overlay[y, x] = color
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _copy_aux_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.is_file() and p.name != "model.obj":
            try:
                shutil.copy2(p, dst_dir / p.name)
            except Exception:
                pass


def _process_one_part(
    obj: DatasetObject,
    args: argparse.Namespace,
    base_obj: Path,
    out_dir: Path,
    model_dir: Path,
    part_name: str,
    part_model: str,
) -> Dict[str, object]:
    tm = _trimesh()
    out_obj = out_dir / "model.obj"
    if out_obj.exists() and not args.overwrite:
        return {"part": part_name, "part_model": part_model, "status": "cached", "model": str(out_obj)}

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = out_dir / "model_raw.obj"
    if not raw_copy.exists() or args.overwrite:
        shutil.copy2(base_obj, raw_copy)
    _copy_aux_files(base_obj.parent, out_dir)

    frame = select_best_frame_for_part(obj, part_name, args.min_mask_pixels)
    if frame is None:
        shutil.copy2(base_obj, out_obj)
        shutil.copy2(out_obj, model_dir / "model.obj")
        return {
            "part": part_name,
            "part_model": part_model,
            "status": "base_copied",
            "reason": "no_reference_frame",
            "model": str(out_obj),
        }

    rgb_path = find_image(obj.rgb_dir, frame)
    depth_path = find_image(obj.depth_dir, frame)
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    pose_path = pose_path_for_part_frame(obj, part_name, frame)
    if rgb_path is None or depth_path is None or mask_path is None or pose_path is None:
        shutil.copy2(base_obj, out_obj)
        shutil.copy2(out_obj, model_dir / "model.obj")
        return {
            "part": part_name,
            "part_model": part_model,
            "status": "base_copied",
            "reason": "missing_reference_inputs",
            "frame": frame,
            "model": str(out_obj),
        }

    depth_m = load_depth_m(depth_path, args.depth_scale)
    mask = load_mask(mask_path, depth_m.shape[:2])
    object_mask = _find_object_mask(obj, frame, depth_m.shape[:2])
    k = load_k(obj)
    ob_in_cam = load_pose(pose_path, args.pose_convention)
    mesh = _as_mesh(tm.load(str(base_obj), force="mesh", process=False))

    face_keep, stats, projection = _projection_score(mesh, mask, depth_m, k, ob_in_cam, args)
    stats.update(
        {
            "frame": str(frame),
            "part_semantic": _part_semantic(part_name),
            "thin_part": bool(_is_thin_part(part_name)),
            "mask_pixels": int(np.count_nonzero(mask)),
            "object_mask_pixels": None if object_mask is None else int(np.count_nonzero(object_mask)),
        }
    )
    if object_mask is not None and int(np.count_nonzero(object_mask)) > 0:
        stats["mask_to_object_area_ratio"] = float(np.count_nonzero(mask) / max(1, np.count_nonzero(object_mask)))

    min_faces = max(4, int(len(mesh.faces) * float(args.partcut_min_keep_ratio)))
    enough_support = int(np.count_nonzero(face_keep)) >= min_faces
    visible_ok = float(stats.get("center_visible_ratio", 0.0)) >= float(args.partcut_min_visible_ratio)
    inside_ok = float(stats.get("center_inside_part_ratio", 0.0)) >= float(args.partcut_min_inside_ratio)

    accepted = bool(enough_support and visible_ok and inside_ok)
    if accepted:
        keep_idx = np.where(face_keep)[0]
        partcut = mesh.submesh([keep_idx], append=True, repair=False)
        if len(partcut.vertices) == 0 or len(partcut.faces) == 0:
            accepted = False
            stats["reject_reason"] = "empty_after_submesh"
        else:
            partcut, component_info = _component_select(partcut, mask, k, ob_in_cam)
            stats["component_select"] = component_info
            repaired = _repair_mesh(partcut)
            if len(repaired.vertices) == 0 or len(repaired.faces) == 0:
                accepted = False
                stats["reject_reason"] = "empty_after_repair"
    else:
        reasons = []
        if not enough_support:
            reasons.append("too_few_kept_faces")
        if not visible_ok:
            reasons.append("low_visible_ratio")
        if not inside_ok:
            reasons.append("low_inside_part_ratio")
        stats["reject_reason"] = ",".join(reasons)

    if accepted:
        partcut_path = out_dir / "model_partcut.obj"
        repaired_path = out_dir / "model_repaired.obj"
        partcut.export(str(partcut_path))
        repaired.export(str(repaired_path))
        repaired.export(str(out_obj))
        status = "success"
    else:
        shutil.copy2(base_obj, out_obj)
        status = "base_copied"

    try:
        _write_debug_overlay(rgb_path, mask, projection, out_dir / "model_partcut_debug.png")
    except Exception as exc:
        stats["debug_overlay_error"] = str(exc)
    write_json(out_dir / "model_projection_score.json", stats)
    shutil.copy2(out_obj, model_dir / "model.obj")
    _copy_aux_files(out_dir, model_dir)

    return {
        "part": part_name,
        "part_model": part_model,
        "status": status,
        "frame": frame,
        "model": str(out_obj),
        "source_model": str(base_obj),
        "score": stats,
    }


def run_partcut_object(obj: DatasetObject, args: argparse.Namespace, base_method: str, method: str) -> Dict[str, object]:
    work_root = Path(args.work_root).resolve()
    base_root = method_pose_ready_dir(work_root, base_method, args.split, obj.name)
    out_pose_root = ensure_dir(method_pose_ready_dir(work_root, method, args.split, obj.name))
    out_model_root = ensure_dir(method_models_dir(work_root, method, args.split, obj.name))
    parts = list_parts(obj)
    summary = {"method": method, "base_method": base_method, "object": obj.name, "parts": []}

    missing = []
    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        base_obj = model_obj_path(base_root, part_model)
        if not base_obj.exists():
            missing.append(part_model)
            continue
        part_summary = _process_one_part(
            obj=obj,
            args=args,
            base_obj=base_obj,
            out_dir=out_pose_root / part_model,
            model_dir=out_model_root / part_model,
            part_name=part_name,
            part_model=part_model,
        )
        summary["parts"].append(part_summary)

    for part_model in missing:
        summary["parts"].append({"part_model": part_model, "status": "skipped", "reason": "base_model_missing"})
    write_json(method_object_dir(work_root, method, args.split, obj.name) / "summary.json", summary)
    return summary


def add_partcut_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partcut-depth-thresh", type=float, default=0.04)
    parser.add_argument("--partcut-use-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partcut-min-keep-ratio", type=float, default=0.02)
    parser.add_argument("--partcut-min-visible-ratio", type=float, default=0.01)
    parser.add_argument("--partcut-min-inside-ratio", type=float, default=0.01)
