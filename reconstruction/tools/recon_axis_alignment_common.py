import argparse
import itertools
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _trimesh():
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("trimesh is required for axis alignment.") from exc
    return trimesh


def _as_mesh(mesh_obj):
    tm = _trimesh()
    if isinstance(mesh_obj, tm.Scene):
        geoms = [g for g in mesh_obj.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            raise ValueError("empty mesh scene")
        mesh_obj = tm.util.concatenate(geoms)
    if not isinstance(mesh_obj, tm.Trimesh) or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("invalid mesh")
    return tm.Trimesh(
        vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
        faces=np.asarray(mesh_obj.faces, dtype=np.int64),
        visual=mesh_obj.visual,
        process=False,
    )


def _natural_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    if len(pts) < 3:
        return center.astype(np.float32), np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    local = centered @ axes
    extents = np.maximum(local.max(axis=0) - local.min(axis=0), 1e-6)
    return center.astype(np.float32), axes.astype(np.float32), extents.astype(np.float32)


def _backproject_mask_points(depth_m: np.ndarray, mask: np.ndarray, k: np.ndarray, ob_in_cam: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.where(mask & (depth_m > 1e-6))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if max_points > 0 and len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, int(max_points)).astype(np.int64)
        xs = xs[idx]
        ys = ys[idx]
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - float(k[0, 2])) * z / float(k[0, 0])
    y = (ys.astype(np.float32) - float(k[1, 2])) * z / float(k[1, 1])
    pts_cam = np.stack([x, y, z], axis=1)
    cam_in_ob = np.linalg.inv(ob_in_cam)
    pts_ob = (cam_in_ob[:3, :3] @ pts_cam.T).T + cam_in_ob[:3, 3]
    return pts_ob.astype(np.float32)


def _project(points_obj: np.ndarray, ob_in_cam: np.ndarray, k: np.ndarray):
    pts_cam = (ob_in_cam[:3, :3] @ points_obj.T).T + ob_in_cam[:3, 3]
    z = pts_cam[:, 2]
    valid_z = z > 1e-6
    u = np.zeros_like(z, dtype=np.float32)
    v = np.zeros_like(z, dtype=np.float32)
    u[valid_z] = k[0, 0] * pts_cam[valid_z, 0] / z[valid_z] + k[0, 2]
    v[valid_z] = k[1, 1] * pts_cam[valid_z, 1] / z[valid_z] + k[1, 2]
    return u, v, z, valid_z


def _projection_score(mesh, mask: np.ndarray, depth_m: np.ndarray, k: np.ndarray, ob_in_cam: np.ndarray, depth_thresh: float) -> Dict[str, float]:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = verts[faces].mean(axis=1) if len(faces) else verts
    if len(centers) == 0:
        return {"score": -1.0, "mask_iou": 0.0, "depth_consistency": 0.0, "visible_ratio": 0.0}
    u, v, z, valid_z = _project(centers, ob_in_cam, k)
    h, w = mask.shape[:2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    in_img = valid_z & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    idx = np.where(in_img)[0]
    if len(idx) == 0:
        return {"score": -1.0, "mask_iou": 0.0, "depth_consistency": 0.0, "visible_ratio": 0.0}
    in_mask = np.zeros(len(centers), dtype=bool)
    in_mask[idx] = mask[vi[idx], ui[idx]]
    obs = depth_m[vi[idx], ui[idx]]
    has_depth = obs > 1e-6
    depth_ok = np.zeros(len(centers), dtype=bool)
    if np.any(has_depth):
        good_idx = idx[has_depth]
        depth_ok[good_idx] = np.abs(z[good_idx] - obs[has_depth]) <= float(depth_thresh)
    mask_iou = float(np.mean(in_mask[idx])) if len(idx) else 0.0
    depth_consistency = float(np.mean(depth_ok[idx][has_depth])) if np.any(has_depth) else 0.0
    visible_ratio = float(len(idx) / max(1, len(centers)))
    score = 0.55 * mask_iou + 0.35 * depth_consistency + 0.10 * visible_ratio
    return {
        "score": float(score),
        "mask_iou": mask_iou,
        "depth_consistency": depth_consistency,
        "visible_ratio": visible_ratio,
    }


def _candidate_mesh(mesh, mesh_center, mesh_axes, mesh_extents, obs_center, obs_axes, obs_extents, perm, signs, allow_scale: bool):
    tm = _trimesh()
    basis = np.zeros((3, 3), dtype=np.float64)
    for mesh_axis, obs_axis in enumerate(perm):
        basis[obs_axis, mesh_axis] = float(signs[mesh_axis])
    rotation = obs_axes.astype(np.float64) @ basis @ mesh_axes.astype(np.float64).T
    if np.linalg.det(rotation) <= 0:
        return None, None
    if allow_scale:
        scale_per_mesh_axis = np.asarray([obs_extents[perm[i]] / max(mesh_extents[i], 1e-6) for i in range(3)], dtype=np.float64)
        scale = float(np.median(scale_per_mesh_axis))
    else:
        scale = 1.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation * scale
    transform[:3, 3] = obs_center - transform[:3, :3] @ mesh_center
    out = mesh.copy()
    out.vertices = (transform[:3, :3] @ np.asarray(mesh.vertices, dtype=np.float64).T).T + transform[:3, 3]
    out = tm.Trimesh(vertices=np.asarray(out.vertices, dtype=np.float32), faces=np.asarray(out.faces, dtype=np.int64), visual=out.visual, process=False)
    return out, {"transform": transform, "scale": scale}


def _write_debug_overlay(rgb_path: Optional[Path], mesh, mask: np.ndarray, k: np.ndarray, ob_in_cam: np.ndarray, out_path: Path) -> None:
    if rgb_path is None:
        return
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return
    overlay = rgb.copy()
    if overlay.shape[:2] != mask.shape[:2]:
        mask_vis = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    else:
        mask_vis = mask
    overlay[mask_vis] = (0.55 * overlay[mask_vis] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts) > 5000:
        verts = verts[np.linspace(0, len(verts) - 1, 5000).astype(np.int64)]
    u, v, _, valid_z = _project(verts, ob_in_cam, k)
    h, w = overlay.shape[:2]
    for x, y, ok in zip(u, v, valid_z):
        if not ok:
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            overlay[yi, xi] = (0, 255, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _align_one_part(obj: DatasetObject, args: argparse.Namespace, base_obj: Path, out_dir: Path, model_dir: Path, part_name: str, part_model: str) -> Dict[str, object]:
    tm = _trimesh()
    out_obj = out_dir / "model.obj"
    meta_path = out_dir / "axis_alignment.json"
    if out_obj.exists() and meta_path.exists() and not args.overwrite:
        shutil.copy2(out_obj, model_dir / "model.obj")
        return {"part": part_name, "part_model": part_model, "status": "cached", "model": str(out_obj)}

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    mesh = _as_mesh(tm.load(str(base_obj), force="mesh", process=False))

    frame = select_best_frame_for_part(obj, part_name, args.axis_align_min_mask_pixels)
    depth_path = find_image(obj.depth_dir, frame) if frame is not None else None
    mask_path = mask_path_for_part_frame(obj, part_name, frame) if frame is not None else None
    pose_path = pose_path_for_part_frame(obj, part_name, frame) if frame is not None else None
    rgb_path = find_image(obj.rgb_dir, frame) if frame is not None else None
    if frame is None or depth_path is None or mask_path is None or pose_path is None:
        shutil.copy2(base_obj, out_obj)
        shutil.copy2(out_obj, model_dir / "model.obj")
        result = {"part": part_name, "part_model": part_model, "status": "base_copied", "reason": "missing_reference_inputs", "model": str(out_obj)}
        write_json(meta_path, result)
        return result

    depth_m = load_depth_m(depth_path, args.depth_scale)
    if int(args.axis_align_depth_erode) > 0:
        kernel = np.ones((int(args.axis_align_depth_erode), int(args.axis_align_depth_erode)), np.uint8)
        valid = (depth_m > 1e-6).astype(np.uint8)
        valid = cv2.erode(valid, kernel, iterations=1) > 0
        depth_m = np.where(valid, depth_m, 0.0)
    if int(args.axis_align_bilateral_radius) > 0:
        d = int(args.axis_align_bilateral_radius) * 2 + 1
        depth_m = cv2.bilateralFilter(depth_m.astype(np.float32), d=d, sigmaColor=0.03, sigmaSpace=float(d))
    mask = load_mask(mask_path, depth_m.shape[:2])
    k = load_k(obj)
    ob_in_cam = load_pose(pose_path, args.pose_convention)
    obs_pts = _backproject_mask_points(depth_m, mask, k, ob_in_cam, int(args.axis_align_max_points))
    if len(obs_pts) < int(args.axis_align_min_points):
        shutil.copy2(base_obj, out_obj)
        shutil.copy2(out_obj, model_dir / "model.obj")
        result = {"part": part_name, "part_model": part_model, "status": "base_copied", "reason": "too_few_observed_points", "points": int(len(obs_pts)), "model": str(out_obj)}
        write_json(meta_path, result)
        return result

    mesh_points, _ = tm.sample.sample_surface(mesh, min(int(args.axis_align_mesh_samples), max(1, len(mesh.faces) * 8)))
    mesh_center, mesh_axes, mesh_extents = _natural_axes(mesh_points)
    obs_center, obs_axes, obs_extents = _natural_axes(obs_pts)

    candidates: List[Dict[str, object]] = []
    best_mesh = mesh
    best_report: Optional[Dict[str, object]] = None
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1, 1), repeat=3):
            cand_mesh, info = _candidate_mesh(
                mesh,
                mesh_center,
                mesh_axes,
                mesh_extents,
                obs_center,
                obs_axes,
                obs_extents,
                perm,
                signs,
                bool(args.axis_align_scale),
            )
            if cand_mesh is None or info is None:
                continue
            score = _projection_score(cand_mesh, mask, depth_m, k, ob_in_cam, float(args.axis_align_depth_thresh))
            report = {
                "axis_permutation": [int(x) for x in perm],
                "axis_sign": [int(x) for x in signs],
                "scale": float(info["scale"]),
                "mesh_to_aligned": np.asarray(info["transform"], dtype=float).reshape(-1).tolist(),
                "score": score,
            }
            candidates.append(report)
            if best_report is None or float(score["score"]) > float(best_report["score"]["score"]):
                best_report = report
                best_mesh = cand_mesh

    if best_report is None or float(best_report["score"]["score"]) < float(args.axis_align_score_thresh):
        best_mesh = mesh
        status = "base_copied"
        reason = "no_candidate_above_threshold"
    else:
        status = "axis_aligned"
        reason = ""

    best_mesh.export(str(out_obj))
    shutil.copy2(out_obj, model_dir / "model.obj")
    if bool(args.axis_align_debug):
        try:
            _write_debug_overlay(rgb_path, best_mesh, mask, k, ob_in_cam, out_dir / "axis_alignment_debug.png")
        except Exception as exc:
            if best_report is not None:
                best_report["debug_error"] = str(exc)
    result = {
        "part": part_name,
        "part_model": part_model,
        "status": status,
        "reason": reason or None,
        "frame": str(frame),
        "source_model": str(base_obj),
        "model": str(out_obj),
        "best": best_report,
        "candidates": candidates if bool(args.axis_align_save_candidates) else [],
    }
    write_json(meta_path, result)
    try:
        shutil.copy2(meta_path, model_dir / "axis_alignment.json")
    except Exception:
        pass
    return result


def run_axis_alignment_object(obj: DatasetObject, args: argparse.Namespace, base_method: str, method: str) -> Dict[str, object]:
    work_root = Path(args.work_root).resolve()
    base_root = method_pose_ready_dir(work_root, base_method, args.split, obj.name)
    out_pose_root = ensure_dir(method_pose_ready_dir(work_root, method, args.split, obj.name))
    out_model_root = ensure_dir(method_models_dir(work_root, method, args.split, obj.name))
    summary = {"method": method, "base_method": base_method, "object": obj.name, "parts": []}
    for part_idx, part_name in enumerate(list_parts(obj)):
        part_model = part_model_name(part_name, part_idx)
        base_obj = model_obj_path(base_root, part_model)
        if not base_obj.exists():
            summary["parts"].append({"part": part_name, "part_model": part_model, "status": "skipped", "reason": "base_model_missing"})
            continue
        if bool(args.axis_align):
            part_summary = _align_one_part(obj, args, base_obj, out_pose_root / part_model, out_model_root / part_model, part_name, part_model)
        else:
            out_dir = out_pose_root / part_model
            model_dir = out_model_root / part_model
            out_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_obj, out_dir / "model.obj")
            shutil.copy2(base_obj, model_dir / "model.obj")
            part_summary = {"part": part_name, "part_model": part_model, "status": "disabled_copied", "model": str(out_dir / "model.obj")}
        summary["parts"].append(part_summary)
    write_json(method_object_dir(work_root, method, args.split, obj.name) / "summary.json", summary)
    return summary


def add_axis_alignment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--axis-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-align-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-align-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--axis-align-save-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--axis-align-score-thresh", type=float, default=0.05)
    parser.add_argument("--axis-align-depth-thresh", type=float, default=0.04)
    parser.add_argument("--axis-align-min-mask-pixels", type=int, default=200)
    parser.add_argument("--axis-align-min-points", type=int, default=64)
    parser.add_argument("--axis-align-max-points", type=int, default=20000)
    parser.add_argument("--axis-align-mesh-samples", type=int, default=20000)
    parser.add_argument("--axis-align-depth-erode", type=int, default=2)
    parser.add_argument("--axis-align-bilateral-radius", type=int, default=2)
