#!/usr/bin/env python3
"""Smoke test for transform-aware TSDF mesh normalization.

This script does not run SAM3D/InstantMesh/Hunyuan3D or TSDF. It only reads a
small sample of existing reconstruction outputs and dataset observations, then
checks the geometry contract needed by a future normalized TSDF branch:

    x_norm = T_raw_to_norm @ x_raw
    x_raw = T_norm_to_raw @ x_norm
    pose_norm = pose_raw @ T_norm_to_raw

For any mesh point, pose_raw @ x_raw and pose_norm @ x_norm should land at the
same camera-space point. Results are written as JSON under sample_test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RECON_ROOT = SCRIPT_DIR.parent
REPO_ROOT = RECON_ROOT.parent
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from recon_utils import (  # noqa: E402
    DatasetObject,
    backproject,
    frames_for_part,
    list_objects,
    list_parts,
    load_depth_m,
    load_k,
    load_mask,
    load_pose,
    mask_path_for_part_frame,
    method_models_dir,
    method_pose_ready_dir,
    model_obj_path,
    natural_sort_key,
    part_model_name,
    pose_path_for_part_frame,
    slice_objects,
    write_json,
)


LOCAL_DATA_ROOT = REPO_ROOT / "dataset_train"
SERVER_DATA_ROOT = Path(
    "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
)
DEFAULT_DATA_ROOT = LOCAL_DATA_ROOT if LOCAL_DATA_ROOT.exists() else SERVER_DATA_ROOT
DEFAULT_WORK_ROOT = SCRIPT_DIR / "runs" / "default" / "reconstruction_runs"
DEFAULT_OBJECTS_JSON = SCRIPT_DIR / "sampled_objects.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "tsdf_normalization_smoke.json"


def _trimesh():
    import trimesh

    return trimesh


def _as_float_list(x):
    return np.asarray(x, dtype=float).tolist()


def _matrix_json(x: np.ndarray) -> List[List[float]]:
    return np.asarray(x, dtype=float).reshape(4, 4).tolist()


def _robust_stats(points: np.ndarray) -> Dict[str, object]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError(f"not enough points for stats: {len(points)}")
    lo = np.percentile(points, 2.0, axis=0)
    hi = np.percentile(points, 98.0, axis=0)
    center = np.median(points, axis=0)
    extents = np.maximum(hi - lo, 1e-8)
    centered = points - center[None]
    cov = centered.T @ centered / float(max(1, len(centered)))
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].astype(np.float32)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return {
        "center": center.astype(np.float32),
        "bbox_min": lo.astype(np.float32),
        "bbox_max": hi.astype(np.float32),
        "extents": extents.astype(np.float32),
        "diagonal": float(np.linalg.norm(extents)),
        "axes": axes,
        "eigenvalues": np.maximum(eigvals[order], 0.0).astype(np.float32),
    }


def _stats_json(stats: Dict[str, object]) -> Dict[str, object]:
    return {
        "center": _as_float_list(stats["center"]),
        "bbox_min": _as_float_list(stats["bbox_min"]),
        "bbox_max": _as_float_list(stats["bbox_max"]),
        "extents": _as_float_list(stats["extents"]),
        "diagonal": float(stats["diagonal"]),
        "eigenvalues": _as_float_list(stats["eigenvalues"]),
    }


def _load_sample_objects(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    return [str(x) for x in payload.get("objects", [])]


def _resolve_objects(args: argparse.Namespace) -> List[str]:
    split_root = args.data_root / args.split
    if args.objects.strip():
        names = [x.strip() for x in args.objects.split(",") if x.strip()]
    else:
        names = _load_sample_objects(args.objects_json)
        if not names:
            all_names = list_objects(args.data_root, args.split, "all", "")
            names = slice_objects(all_names, 0, args.max_objects)
    names = [name for name in names if (split_root / name).is_dir()]
    if not names:
        all_names = list_objects(args.data_root, args.split, "all", "")
        names = slice_objects(all_names, 0, args.max_objects)
    names = sorted(names, key=natural_sort_key)
    return names[: max(0, int(args.max_objects))]


def _mesh_path_for_method(work_root: Path, method: str, split: str, object_name: str, part_model: str) -> Optional[Path]:
    roots = [
        method_pose_ready_dir(work_root, method, split, object_name),
        method_models_dir(work_root, method, split, object_name),
    ]
    for root in roots:
        p = model_obj_path(root, part_model)
        if p.exists():
            return p
    return None


def _apply_transform(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (tf[:3, :3] @ points.T).T + tf[:3, 3]


def _safe_inverse(tf: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(tf, dtype=np.float64)).astype(np.float32)


def _axis_rotation_from_policy(mesh_stats: Dict[str, object], axis_policy: str) -> np.ndarray:
    if axis_policy == "none":
        return np.eye(3, dtype=np.float32)
    axes = np.asarray(mesh_stats["axes"], dtype=np.float32)
    if axis_policy == "pca":
        return axes.T.astype(np.float32)
    if axis_policy == "obb":
        # For smoke-test purposes PCA axes are a deterministic stand-in for OBB
        # axes. The real TSDF implementation can switch this to Open3D OBB.
        return axes.T.astype(np.float32)
    raise ValueError(f"unsupported axis policy: {axis_policy}")


def _center_from_policy(mesh_stats: Dict[str, object], center_policy: str) -> np.ndarray:
    if center_policy == "keep":
        return np.zeros(3, dtype=np.float32)
    if center_policy in {"bbox", "median"}:
        return np.asarray(mesh_stats["center"], dtype=np.float32)
    raise ValueError(f"unsupported center policy: {center_policy}")


def _scale_from_policy(
    mesh_stats: Dict[str, object],
    observation_stats: Optional[Dict[str, object]],
    scale_policy: str,
    min_ratio: float,
    max_ratio: float,
) -> Tuple[float, str]:
    if scale_policy == "none" or observation_stats is None:
        return 1.0, "disabled" if scale_policy == "none" else "no_observation_stats"
    mesh_diag = float(mesh_stats["diagonal"])
    obs_diag = float(observation_stats["diagonal"])
    ratio = mesh_diag / max(obs_diag, 1e-8)
    if scale_policy == "metric-if-abnormal" and min_ratio <= ratio <= max_ratio:
        return 1.0, "metric_scale_plausible"
    if scale_policy not in {"metric", "metric-if-abnormal"}:
        raise ValueError(f"unsupported scale policy: {scale_policy}")
    scale = float(np.clip(obs_diag / max(mesh_diag, 1e-8), 0.05, 20.0))
    return scale, "metric_scale_applied"


def build_normalization_transform(
    mesh_vertices: np.ndarray,
    observation_points_obj: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mesh_stats = _robust_stats(mesh_vertices)
    obs_stats = None
    if observation_points_obj is not None and len(observation_points_obj) >= 3:
        obs_stats = _robust_stats(observation_points_obj)

    scale, scale_status = _scale_from_policy(
        mesh_stats,
        obs_stats,
        args.scale_policy,
        args.mesh_scale_ratio_min,
        args.mesh_scale_ratio_max,
    )
    rotation = _axis_rotation_from_policy(mesh_stats, args.axis_policy)
    center = _center_from_policy(mesh_stats, args.center_policy)

    raw_to_norm = np.eye(4, dtype=np.float32)
    raw_to_norm[:3, :3] = (float(scale) * rotation).astype(np.float32)
    raw_to_norm[:3, 3] = (-(float(scale) * rotation) @ center.astype(np.float32)).astype(np.float32)
    norm_to_raw = _safe_inverse(raw_to_norm)

    norm_vertices = _apply_transform(mesh_vertices, raw_to_norm)
    norm_stats = _robust_stats(norm_vertices)
    report = {
        "axis_policy": args.axis_policy,
        "center_policy": args.center_policy,
        "scale_policy": args.scale_policy,
        "scale_status": scale_status,
        "scale_applied": float(scale),
        "center_used": _as_float_list(center),
        "mesh_before": _stats_json(mesh_stats),
        "mesh_after": _stats_json(norm_stats),
        "observation_stats": None if obs_stats is None else _stats_json(obs_stats),
        "raw_to_norm": _matrix_json(raw_to_norm),
        "norm_to_raw": _matrix_json(norm_to_raw),
    }
    return raw_to_norm, norm_to_raw, report


def _collect_observation_points(
    obj: DatasetObject,
    part_name: str,
    k: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], List[Dict[str, object]]]:
    points_obj_all: List[np.ndarray] = []
    frame_debug: List[Dict[str, object]] = []
    for frame in frames_for_part(obj, part_name, args.max_frames_per_part, args.frame_stride):
        mask_path = mask_path_for_part_frame(obj, part_name, frame)
        depth_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = obj.depth_dir / f"{frame}{ext}"
            if cand.exists():
                depth_path = cand
                break
        pose_path = pose_path_for_part_frame(obj, part_name, frame)
        if mask_path is None or depth_path is None or pose_path is None:
            frame_debug.append({"frame": frame, "status": "missing_input"})
            continue
        try:
            depth_m = load_depth_m(depth_path, args.depth_scale)
            mask = load_mask(mask_path, depth_m.shape[:2])
            points_cam = backproject(depth_m, mask, k)
            pose = load_pose(pose_path, args.pose_convention)
            if len(points_cam) < int(args.min_points):
                frame_debug.append(
                    {"frame": frame, "status": "too_few_points", "points_cam": int(len(points_cam))}
                )
                continue
            cam_to_obj = np.linalg.inv(pose).astype(np.float32)
            points_obj = _apply_transform(points_cam, cam_to_obj)
            points_obj_all.append(points_obj.astype(np.float32))
            frame_debug.append(
                {
                    "frame": frame,
                    "status": "ok",
                    "points_cam": int(len(points_cam)),
                    "pose_path": str(pose_path),
                }
            )
        except Exception as e:
            frame_debug.append({"frame": frame, "status": "failed", "error": repr(e)})
    if not points_obj_all:
        return None, frame_debug
    return np.concatenate(points_obj_all, axis=0), frame_debug


def _sample_mesh_points(mesh, count: int, seed: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(faces) > 0:
        try:
            tm = _trimesh()
            pts, _ = tm.sample.sample_surface(mesh, int(count))
            return np.asarray(pts, dtype=np.float32)
        except Exception:
            pass
    rng = np.random.default_rng(int(seed))
    n = min(len(vertices), int(count))
    idx = rng.choice(len(vertices), size=n, replace=False)
    return vertices[idx].astype(np.float32)


def _pose_contract_check(points_raw: np.ndarray, pose_raw: np.ndarray, raw_to_norm: np.ndarray, norm_to_raw: np.ndarray):
    points_norm = _apply_transform(points_raw, raw_to_norm)
    pose_norm = np.asarray(pose_raw, dtype=np.float32) @ np.asarray(norm_to_raw, dtype=np.float32)
    cam_raw = _apply_transform(points_raw, pose_raw)
    cam_norm = _apply_transform(points_norm, pose_norm)
    err = np.linalg.norm(cam_raw - cam_norm, axis=1)
    return {
        "max_error": float(np.max(err)) if len(err) else None,
        "mean_error": float(np.mean(err)) if len(err) else None,
        "num_points": int(len(err)),
        "pose_norm": _matrix_json(pose_norm),
        "ok": bool(len(err) > 0 and float(np.max(err)) <= 1e-5),
    }


def smoke_part(
    obj: DatasetObject,
    method: str,
    part_name: str,
    part_idx: int,
    k: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    part_model = part_model_name(part_name, part_idx)
    rec: Dict[str, object] = {"part": part_name, "part_model": part_model, "method": method}
    mesh_path = _mesh_path_for_method(args.work_root, method, args.split, obj.name, part_model)
    if mesh_path is None:
        rec.update({"status": "missing_mesh"})
        return rec
    rec["mesh_path"] = str(mesh_path)
    tm = _trimesh()
    try:
        mesh = tm.load(str(mesh_path), force="mesh", process=False)
        if getattr(mesh, "vertices", None) is None or len(mesh.vertices) < 3:
            rec.update({"status": "invalid_mesh", "vertices": int(len(getattr(mesh, "vertices", [])))})
            return rec
        obs_points_obj, frame_debug = _collect_observation_points(obj, part_name, k, args)
        raw_to_norm, norm_to_raw, norm_report = build_normalization_transform(
            np.asarray(mesh.vertices, dtype=np.float32),
            obs_points_obj,
            args,
        )
        rec["normalization"] = norm_report
        rec["frames"] = frame_debug

        check_pose = None
        for fd in frame_debug:
            if fd.get("status") == "ok":
                check_pose = load_pose(Path(str(fd["pose_path"])), args.pose_convention)
                rec["contract_frame"] = fd["frame"]
                break
        if check_pose is None:
            rec.update({"status": "no_pose_for_contract_check"})
            return rec
        sample_points = _sample_mesh_points(mesh, args.mesh_sample_points, seed=part_idx + 1009)
        contract = _pose_contract_check(sample_points, check_pose, raw_to_norm, norm_to_raw)
        rec["pose_contract_check"] = contract
        rec["status"] = "ok" if contract["ok"] else "contract_failed"
        return rec
    except Exception as e:
        rec.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        return rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Smoke-test transform-aware TSDF mesh normalization on a small sample.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--objects-json", type=Path, default=DEFAULT_OBJECTS_JSON)
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names; overrides --objects-json.")
    parser.add_argument("--methods", type=str, default="sam3d,instantmesh,hunyuan3d")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--max-parts", type=int, default=3)
    parser.add_argument("--max-frames-per-part", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-points", type=int, default=32)
    parser.add_argument("--mesh-sample-points", type=int, default=512)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--pose-convention", choices=["cv", "sapien"], default="sapien")
    parser.add_argument("--scale-policy", choices=["none", "metric", "metric-if-abnormal"], default="metric-if-abnormal")
    parser.add_argument("--axis-policy", choices=["none", "obb", "pca"], default="none")
    parser.add_argument("--center-policy", choices=["keep", "bbox", "median"], default="keep")
    parser.add_argument("--mesh-scale-ratio-min", type=float, default=0.2)
    parser.add_argument("--mesh-scale-ratio-max", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.work_root = args.work_root.resolve()
    args.objects_json = args.objects_json.resolve()
    args.output = args.output.resolve()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    objects = _resolve_objects(args)
    started = time.time()

    payload: Dict[str, object] = {
        "kind": "tsdf_normalization_smoke",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "data_root": str(args.data_root),
        "split": args.split,
        "work_root": str(args.work_root),
        "objects_json": str(args.objects_json),
        "objects": objects,
        "methods": methods,
        "config": {
            "scale_policy": args.scale_policy,
            "axis_policy": args.axis_policy,
            "center_policy": args.center_policy,
            "max_objects": int(args.max_objects),
            "max_parts": int(args.max_parts),
            "max_frames_per_part": int(args.max_frames_per_part),
            "mesh_scale_ratio_min": float(args.mesh_scale_ratio_min),
            "mesh_scale_ratio_max": float(args.mesh_scale_ratio_max),
        },
        "records": [],
    }

    status_counts: Dict[str, int] = {}
    for object_name in objects:
        obj = DatasetObject(data_root=args.data_root, split=args.split, name=object_name)
        obj_record: Dict[str, object] = {"object": object_name, "methods": []}
        try:
            k = load_k(obj)
            parts = list_parts(obj)[: max(0, int(args.max_parts))]
            obj_record["parts_considered"] = parts
            for method in methods:
                method_record = {"method": method, "parts": []}
                for part_idx, part_name in enumerate(parts):
                    part_record = smoke_part(obj, method, part_name, part_idx, k, args)
                    status = str(part_record.get("status", "unknown"))
                    status_counts[status] = status_counts.get(status, 0) + 1
                    method_record["parts"].append(part_record)
                obj_record["methods"].append(method_record)
        except Exception as e:
            obj_record["status"] = "failed"
            obj_record["error"] = repr(e)
            obj_record["traceback"] = traceback.format_exc()
            status_counts["object_failed"] = status_counts.get("object_failed", 0) + 1
        payload["records"].append(obj_record)

    payload["status_counts"] = status_counts
    payload["elapsed_sec"] = round(time.time() - started, 3)
    ok_count = int(status_counts.get("ok", 0))
    fail_count = sum(v for k, v in status_counts.items() if k not in {"ok", "missing_mesh"})
    payload["summary"] = {
        "ok_parts": ok_count,
        "non_missing_failures": int(fail_count),
        "missing_mesh": int(status_counts.get("missing_mesh", 0)),
    }
    write_json(args.output, payload)
    print(f"[smoke] wrote {args.output}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
