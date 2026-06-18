from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from recon_utils import (
    DatasetObject,
    backproject,
    copy_model_tree,
    ensure_dir,
    find_image,
    frames_for_part,
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
    write_json,
)
from pose_delta_optimization import PoseDeltaOptimizationConfig, optimize_pose_deltas
from pose_utility import PoseUtilityConfig, evaluate_pose_utility, pose_delta_egocentric, utility_improved


def _trimesh():
    import trimesh

    return trimesh


def _as_trimesh(mesh_obj):
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
        process=False,
    )


def _o3d_mesh_to_trimesh(mesh):
    tm = _trimesh()
    return _as_trimesh(
        tm.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.triangles, dtype=np.int64),
            process=False,
        )
    )


def _pose_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3] @ b[:3, :3].T
    cos = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _pose_translation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _score_pose_distance(
    pose: np.ndarray,
    accepted_poses: List[np.ndarray],
    min_angle: float,
    max_angle: float,
    min_translation: float,
    max_translation: float,
) -> Tuple[bool, Dict[str, float]]:
    if not accepted_poses:
        return True, {"nearest_angle_deg": 0.0, "nearest_translation": 0.0}
    angles = np.asarray([_pose_angle_deg(pose, p) for p in accepted_poses], dtype=np.float32)
    translations = np.asarray([_pose_translation(pose, p) for p in accepted_poses], dtype=np.float32)
    nearest_idx = int(np.argmin(angles + translations))
    nearest_angle = float(angles[nearest_idx])
    nearest_translation = float(translations[nearest_idx])
    too_dense = nearest_angle < float(min_angle) and nearest_translation < float(min_translation)
    too_far = float(np.min(angles)) > float(max_angle) or float(np.min(translations)) > float(max_translation)
    return (not too_dense and not too_far), {
        "nearest_angle_deg": nearest_angle,
        "nearest_translation": nearest_translation,
        "min_angle_deg": float(np.min(angles)),
        "min_translation": float(np.min(translations)),
    }


def _load_frame_observation(
    obj: DatasetObject,
    part_name: str,
    frame: str,
    k: np.ndarray,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    depth_path = find_image(obj.depth_dir, frame)
    rgb_path = find_image(obj.rgb_dir, frame)
    pose_path = pose_path_for_part_frame(obj, part_name, frame)
    if mask_path is None or depth_path is None or rgb_path is None or pose_path is None:
        return None
    depth_m = load_depth_m(depth_path, args.depth_scale)
    mask = load_mask(mask_path, depth_m.shape[:2])
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels < int(args.min_mask_pixels):
        return None
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    pose = load_pose(pose_path, args.pose_convention)
    points_cam = backproject(depth_m, mask, k)
    if len(points_cam) < int(args.iter_tsdf_min_points):
        return None
    return {
        "frame": frame,
        "rgb": rgb,
        "depth_m": depth_m,
        "mask": mask,
        "mask_pixels": mask_pixels,
        "raw_pose": pose.astype(np.float32),
        "points_cam": points_cam.astype(np.float32),
    }


def _make_rgbd(obs: Dict[str, object], args: argparse.Namespace, o3d):
    depth_m = np.asarray(obs["depth_m"], dtype=np.float32)
    mask = np.asarray(obs["mask"], dtype=bool)
    rgb = np.asarray(obs["rgb"])
    depth_masked = depth_m.copy()
    depth_masked[~mask] = 0.0
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(np.clip(depth_masked * 1000.0, 0, 65535).astype(np.uint16)),
        depth_scale=1000.0,
        depth_trunc=float(args.depth_trunc),
        convert_rgb_to_intensity=False,
    )


def _camera_intrinsic(obs: Dict[str, object], k: np.ndarray, o3d):
    depth_m = np.asarray(obs["depth_m"], dtype=np.float32)
    h, w = depth_m.shape[:2]
    return o3d.camera.PinholeCameraIntrinsic(
        int(w), int(h), float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    )


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(max_points), replace=False)]


def _points_cam_to_obj(points_cam: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    cam_to_ob = np.linalg.inv(ob_in_cam).astype(np.float32)
    return (cam_to_ob[:3, :3] @ points_cam.T).T + cam_to_ob[:3, 3]


def _points_obj_to_cam(points_obj: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    return (ob_in_cam[:3, :3] @ points_obj.T).T + ob_in_cam[:3, 3]


def _estimate_pose_from_obj_cam_points(points_obj: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    n = min(int(len(points_obj)), int(len(points_cam)))
    if n < 3:
        raise ValueError(f"not enough points for pose estimation: {n}")
    src = points_obj[:n].astype(np.float64)
    dst = points_cam[:n].astype(np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / float(n)
    u, _, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt
    t = mu_d - r @ mu_s
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = r.astype(np.float32)
    tf[:3, 3] = t.astype(np.float32)
    return tf


def _refine_pose_to_mesh(
    mesh,
    obs: Dict[str, object],
    init_pose: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    from scipy.spatial import cKDTree

    points_cam = _sample_points(
        np.asarray(obs["points_cam"], dtype=np.float32),
        int(args.iter_tsdf_refine_points),
        int(seed),
    )
    tm = _trimesh()
    mesh_pts, face_idx = tm.sample.sample_surface(mesh, max(int(args.iter_tsdf_mesh_samples), 1024))
    mesh_pts = np.asarray(mesh_pts, dtype=np.float32)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    mesh_normals = face_normals[np.asarray(face_idx, dtype=np.int64)]
    tree = cKDTree(mesh_pts)

    pose = init_pose.astype(np.float32).copy()
    mean_abs = float("inf")
    inlier_ratio = 0.0
    for _ in range(max(1, int(args.iter_tsdf_refine_iters))):
        points_obj = _points_cam_to_obj(points_cam, pose)
        dist, idx = tree.query(points_obj, k=1, workers=-1)
        keep = dist <= float(args.iter_tsdf_refine_max_dist)
        if int(np.count_nonzero(keep)) < 6:
            cutoff = np.quantile(dist, min(0.8, max(0.1, float(args.iter_tsdf_refine_trim_quantile))))
            keep = dist <= cutoff
        if int(np.count_nonzero(keep)) < 6:
            break
        src_obj = mesh_pts[idx[keep]]
        dst_cam = points_cam[keep]
        new_pose = _estimate_pose_from_obj_cam_points(src_obj, dst_cam)
        delta_angle = _pose_angle_deg(new_pose, pose)
        delta_trans = _pose_translation(new_pose, pose)
        pose = new_pose
        residual = np.sum((points_obj[keep] - mesh_pts[idx[keep]]) * mesh_normals[idx[keep]], axis=1)
        mean_abs = float(np.mean(np.abs(residual)))
        inlier_ratio = float(np.count_nonzero(keep) / max(1, len(points_cam)))
        if delta_angle < float(args.iter_tsdf_refine_angle_eps) and delta_trans < float(args.iter_tsdf_refine_trans_eps):
            break

    prior_angle = _pose_angle_deg(pose, init_pose)
    prior_trans = _pose_translation(pose, init_pose)
    prior_delta = pose_delta_egocentric(init_pose, pose)
    ok = (
        np.isfinite(mean_abs)
        and mean_abs <= float(args.iter_tsdf_refine_residual_thresh)
        and prior_angle <= float(args.iter_tsdf_max_refine_angle)
        and prior_trans <= float(args.iter_tsdf_max_refine_translation)
    )
    return pose.astype(np.float32), {
        "ok": bool(ok),
        "mean_abs_point_plane": mean_abs,
        "inlier_ratio": inlier_ratio,
        "prior_angle_deg": float(prior_angle),
        "prior_translation": float(prior_trans),
        "pose_delta": prior_delta,
    }


def _pose_utility_config_from_args(args: argparse.Namespace) -> PoseUtilityConfig:
    return PoseUtilityConfig(
        max_points=int(args.pose_utility_max_points),
        mesh_samples=int(args.pose_utility_mesh_samples),
        depth_inlier_thresh=float(args.pose_utility_depth_inlier_thresh),
        projection_depth_thresh=float(args.pose_utility_projection_depth_thresh),
        min_projected_points=int(args.pose_utility_min_projected_points),
        depth_weight=float(args.pose_utility_depth_weight),
        inlier_weight=float(args.pose_utility_inlier_weight),
        mask_weight=float(args.pose_utility_mask_weight),
        dt_weight=float(args.pose_utility_dt_weight),
        dr_weight=float(args.pose_utility_dr_weight),
    )


def _pose_delta_config_from_args(args: argparse.Namespace, anchor_frame: str) -> PoseDeltaOptimizationConfig:
    max_dt = float(args.pose_delta_max_dt)
    if max_dt <= 0:
        max_dt = float(args.iter_tsdf_max_refine_translation)
    max_dr = float(args.pose_delta_max_dr_deg)
    if max_dr <= 0:
        max_dr = float(args.iter_tsdf_max_refine_angle)
    return PoseDeltaOptimizationConfig(
        rounds=int(args.pose_delta_rounds),
        max_dt=max_dt,
        max_dr_deg=max_dr,
        anchor_frame=str(anchor_frame),
        refine_anchor=bool(args.pose_delta_refine_anchor),
    )


def _pose_map_from_accepted(accepted: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    return {
        str(item["obs"]["frame"]): np.asarray(item["pose"], dtype=np.float32).reshape(4, 4)
        for item in accepted
    }


def _obs_list_from_accepted(accepted: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [item["obs"] for item in accepted]


def _replace_accepted_poses(accepted: List[Dict[str, object]], pose_map: Dict[str, np.ndarray]) -> None:
    for item in accepted:
        frame = str(item["obs"]["frame"])
        if frame in pose_map:
            item["pose"] = np.asarray(pose_map[frame], dtype=np.float32).reshape(4, 4)


def _project_points(points_cam: np.ndarray, k: np.ndarray, shape: Tuple[int, int]):
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = np.zeros(len(points_cam), dtype=np.int64)
    v = np.zeros(len(points_cam), dtype=np.int64)
    u[valid] = np.rint(points_cam[valid, 0] * float(k[0, 0]) / z[valid] + float(k[0, 2])).astype(np.int64)
    v[valid] = np.rint(points_cam[valid, 1] * float(k[1, 1]) / z[valid] + float(k[1, 2])).astype(np.int64)
    h, w = shape
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def _depth_consistency_one_way(
    src_obs: Dict[str, object],
    src_pose: np.ndarray,
    dst_obs: Dict[str, object],
    dst_pose: np.ndarray,
    k: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    src_pts_cam = _sample_points(
        np.asarray(src_obs["points_cam"], dtype=np.float32),
        int(args.iter_tsdf_consistency_points),
        int(seed),
    )
    points_obj = _points_cam_to_obj(src_pts_cam, src_pose)
    dst_pts_cam = _points_obj_to_cam(points_obj, dst_pose)
    dst_depth = np.asarray(dst_obs["depth_m"], dtype=np.float32)
    dst_mask = np.asarray(dst_obs["mask"], dtype=bool)
    u, v, valid = _project_points(dst_pts_cam, k, dst_depth.shape[:2])
    if int(np.count_nonzero(valid)) == 0:
        return {"valid_ratio": 0.0, "mean_abs_depth": float("inf"), "inlier_ratio": 0.0}
    valid_idx = np.where(valid)[0]
    in_mask = dst_mask[v[valid_idx], u[valid_idx]]
    depth_vals = dst_depth[v[valid_idx], u[valid_idx]]
    has_depth = depth_vals > 1e-6
    keep_local = in_mask & has_depth
    if int(np.count_nonzero(keep_local)) == 0:
        return {"valid_ratio": 0.0, "mean_abs_depth": float("inf"), "inlier_ratio": 0.0}
    kept = valid_idx[keep_local]
    residual = np.abs(dst_pts_cam[kept, 2] - depth_vals[keep_local])
    return {
        "valid_ratio": float(len(kept) / max(1, len(src_pts_cam))),
        "mean_abs_depth": float(np.mean(residual)),
        "inlier_ratio": float(np.mean(residual <= float(args.iter_tsdf_depth_residual_thresh))),
    }


def _check_depth_consistency(
    candidate_obs: Dict[str, object],
    candidate_pose: np.ndarray,
    accepted: List[Dict[str, object]],
    k: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, object]]:
    if not accepted:
        return True, {"checked_pairs": 0}
    pair_logs = []
    for idx, item in enumerate(accepted):
        acc_obs = item["obs"]
        acc_pose = item["pose"]
        forward = _depth_consistency_one_way(
            candidate_obs,
            candidate_pose,
            acc_obs,
            acc_pose,
            k,
            args,
            seed=idx + 13,
        )
        backward = _depth_consistency_one_way(
            acc_obs,
            acc_pose,
            candidate_obs,
            candidate_pose,
            k,
            args,
            seed=idx + 113,
        )
        pair_logs.append({"frame": acc_obs["frame"], "candidate_to_accepted": forward, "accepted_to_candidate": backward})
    valid = [min(p["candidate_to_accepted"]["valid_ratio"], p["accepted_to_candidate"]["valid_ratio"]) for p in pair_logs]
    depth = [max(p["candidate_to_accepted"]["mean_abs_depth"], p["accepted_to_candidate"]["mean_abs_depth"]) for p in pair_logs]
    inlier = [min(p["candidate_to_accepted"]["inlier_ratio"], p["accepted_to_candidate"]["inlier_ratio"]) for p in pair_logs]
    best_idx = int(np.argmax(np.asarray(valid, dtype=np.float32) - np.asarray(depth, dtype=np.float32)))
    ok = (
        valid[best_idx] >= float(args.iter_tsdf_min_valid_proj_ratio)
        and depth[best_idx] <= float(args.iter_tsdf_depth_residual_thresh)
        and inlier[best_idx] >= float(args.iter_tsdf_min_depth_inlier_ratio)
    )
    return bool(ok), {
        "checked_pairs": len(pair_logs),
        "best_pair_frame": pair_logs[best_idx]["frame"],
        "best_valid_ratio": float(valid[best_idx]),
        "best_mean_abs_depth": float(depth[best_idx]),
        "best_inlier_ratio": float(inlier[best_idx]),
        "pairs": pair_logs,
    }


def _rebuild_tsdf_mesh(accepted: List[Dict[str, object]], out_path: Optional[Path], k: np.ndarray, args: argparse.Namespace, o3d):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.voxel_length),
        sdf_trunc=float(args.sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for item in accepted:
        obs = item["obs"]
        rgbd = _make_rgbd(obs, args, o3d)
        intrinsic = _camera_intrinsic(obs, k, o3d)
        # Open3D integrate uses camera extrinsic (world-to-camera). Here the TSDF world is object frame.
        volume.integrate(rgbd, intrinsic, np.asarray(item["pose"], dtype=np.float64))
    mesh = volume.extract_triangle_mesh()
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=True)
        return mesh
    return None


def _select_seed(observations: List[Dict[str, object]]) -> int:
    if not observations:
        return -1
    return int(np.argmax([int(o["mask_pixels"]) for o in observations]))


def _select_next_candidate(
    observations: List[Dict[str, object]],
    pending: List[int],
    accepted: List[Dict[str, object]],
    raw_pose_map: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Optional[int], Dict[str, object]]:
    accepted_raw = [raw_pose_map[str(item["obs"]["frame"])] for item in accepted]
    best = None
    rejected = []
    for idx in list(pending):
        obs = observations[idx]
        raw_pose = raw_pose_map[str(obs["frame"])]
        ok, info = _score_pose_distance(
            raw_pose,
            accepted_raw,
            float(args.iter_tsdf_min_view_angle),
            float(args.iter_tsdf_max_view_angle),
            float(args.iter_tsdf_min_translation),
            float(args.iter_tsdf_max_translation),
        )
        if not ok:
            rejected.append({"frame": obs["frame"], "reason": "view_gate", **info})
            continue
        score = float(obs["mask_pixels"]) - 0.01 * abs(info.get("nearest_angle_deg", 0.0))
        if best is None or score > best[0]:
            best = (score, idx, info)
    if best is None:
        return None, {"view_gate_rejections": rejected}
    return int(best[1]), {"selected_info": best[2], "view_gate_rejections": rejected}


def _refresh_raw_poses_after_shape_update(
    current_mesh,
    observations: List[Dict[str, object]],
    raw_pose_map: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed_offset: int,
) -> Dict[str, object]:
    logs = {}
    for idx, obs in enumerate(observations):
        frame = str(obs["frame"])
        new_pose, info = _refine_pose_to_mesh(
            current_mesh,
            obs,
            raw_pose_map[frame],
            args,
            seed=seed_offset + idx,
        )
        if bool(info.get("ok", False)):
            raw_pose_map[frame] = new_pose
        logs[frame] = info
    return logs


def _iterative_tsdf_part(
    obj: DatasetObject,
    part_name: str,
    part_model: str,
    base_obj: Path,
    out_obj: Path,
    work_root: Path,
    method: str,
    k: np.ndarray,
    args: argparse.Namespace,
    o3d,
) -> Dict[str, object]:
    iter_root = ensure_dir(method_object_dir(work_root, method, args.split, obj.name) / "pose_tsdf_iter" / part_model)
    observations = []
    for frame in frames_for_part(obj, part_name, args.max_frames, args.frame_stride):
        obs = _load_frame_observation(obj, part_name, frame, k, args)
        if obs is not None:
            observations.append(obs)
    if not observations:
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_obj, out_obj)
        return {"status": "base_copied", "reason": "no_valid_observations", "frames": 0, "model": str(out_obj)}

    tm = _trimesh()
    base_mesh = _as_trimesh(tm.load(str(base_obj), force="mesh", process=False))
    current_mesh = base_mesh
    raw_pose_map = {str(o["frame"]): np.asarray(o["raw_pose"], dtype=np.float32) for o in observations}
    seed_idx = _select_seed(observations)
    seed_obs = observations[seed_idx]
    seed_pose, seed_refine = _refine_pose_to_mesh(
        current_mesh,
        seed_obs,
        np.asarray(seed_obs["raw_pose"], dtype=np.float32),
        args,
        seed=17,
    )
    if not bool(seed_refine.get("ok", False)):
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_obj, out_obj)
        part_summary = {
            "part": part_name,
            "part_model": part_model,
            "status": "base_copied",
            "reason": "seed_pose_refine_failed",
            "frames": 0,
            "candidate_frames": len(observations),
            "seed_frame": str(seed_obs["frame"]),
            "seed_refine": seed_refine,
            "model": str(out_obj),
            "iter_root": str(iter_root),
        }
        write_json(iter_root / "summary.json", part_summary)
        return part_summary
    accepted = [{"obs": seed_obs, "pose": seed_pose}]
    accepted_frames = {str(seed_obs["frame"])}
    pending = [i for i in range(len(observations)) if i != seed_idx]
    rejected = []
    iter_logs = [
        {
            "iter": 0,
            "frame": seed_obs["frame"],
            "status": "accepted_seed",
            "mask_pixels": int(seed_obs["mask_pixels"]),
            "refine": seed_refine,
        }
    ]

    max_accept = int(args.iter_tsdf_max_frames)
    if max_accept <= 0:
        max_accept = len(observations)
    max_rounds = max(len(observations) * 2, max_accept)
    for it in range(1, max_rounds + 1):
        if len(accepted) >= max_accept or not pending:
            break
        cand_idx, select_info = _select_next_candidate(observations, pending, accepted, raw_pose_map, args)
        if cand_idx is None:
            rejected.extend(select_info.get("view_gate_rejections", []))
            break
        pending.remove(cand_idx)
        cand_obs = observations[cand_idx]
        cand_pose_init = raw_pose_map[str(cand_obs["frame"])]
        refined_pose, refine_info = _refine_pose_to_mesh(current_mesh, cand_obs, cand_pose_init, args, seed=it + 31)
        if not bool(refine_info.get("ok", False)):
            item = {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "rejected",
                "reason": "pose_refine_failed",
                "select": select_info,
                "refine": refine_info,
            }
            rejected.append(item)
            iter_logs.append(item)
            continue
        consistency_ok, consistency_info = _check_depth_consistency(cand_obs, refined_pose, accepted, k, args)
        if not consistency_ok:
            item = {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "rejected",
                "reason": "depth_consistency_failed",
                "select": select_info,
                "refine": refine_info,
                "consistency": consistency_info,
            }
            rejected.append(item)
            iter_logs.append(item)
            continue

        old_mesh = current_mesh
        accepted.append({"obs": cand_obs, "pose": refined_pose})
        accepted_frames.add(str(cand_obs["frame"]))
        mesh_o3d = _rebuild_tsdf_mesh(
            accepted,
            iter_root / f"iter_{len(accepted) - 1:03d}" / "mesh.obj",
            k,
            args,
            o3d,
        )
        if mesh_o3d is None:
            item = {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "rejected",
                "reason": "tsdf_mesh_empty_after_accept",
                "select": select_info,
                "refine": refine_info,
                "consistency": consistency_info,
            }
            rejected.append(item)
            accepted.pop()
            accepted_frames.discard(str(cand_obs["frame"]))
            iter_logs.append(item)
            continue
        proposed_mesh = _o3d_mesh_to_trimesh(mesh_o3d)
        pose_map_before_utility = _pose_map_from_accepted(accepted)
        init_pose_map_for_utility = {
            frame: np.asarray(raw_pose_map.get(frame, pose), dtype=np.float32).reshape(4, 4)
            for frame, pose in pose_map_before_utility.items()
        }
        accepted_obs = _obs_list_from_accepted(accepted)
        pose_delta_info: Dict[str, object] = {
            "status": "disabled",
            "reason": "pose_delta_opt disabled",
        }
        pose_map_after_delta = pose_map_before_utility
        if bool(args.pose_delta_opt):
            def _refine_for_pose_delta(mesh_obj, obs_obj, init_pose_obj, seed_obj):
                return _refine_pose_to_mesh(mesh_obj, dict(obs_obj), np.asarray(init_pose_obj, dtype=np.float32), args, int(seed_obj))

            pose_map_after_delta, pose_delta_info = optimize_pose_deltas(
                mesh=proposed_mesh,
                observations=accepted_obs,
                init_pose_map=pose_map_before_utility,
                refine_pose_fn=_refine_for_pose_delta,
                cfg=_pose_delta_config_from_args(args, anchor_frame=str(seed_obs["frame"])),
                seed=2000 + it * 100,
            )

        pose_utility_info: Dict[str, object] = {
            "status": "disabled",
            "reason": "pose_utility_check disabled",
        }
        if bool(args.pose_utility_check) and len(accepted) > 1:
            utility_cfg = _pose_utility_config_from_args(args)
            old_eval = evaluate_pose_utility(
                mesh=old_mesh,
                observations=accepted_obs,
                pose_map=pose_map_before_utility,
                init_pose_map=init_pose_map_for_utility,
                k=k,
                cfg=utility_cfg,
                seed=3000 + it * 100,
            )
            new_eval = evaluate_pose_utility(
                mesh=proposed_mesh,
                observations=accepted_obs,
                pose_map=pose_map_after_delta,
                init_pose_map=init_pose_map_for_utility,
                k=k,
                cfg=utility_cfg,
                seed=4000 + it * 100,
            )
            utility_ok, utility_gate = utility_improved(
                old_eval,
                new_eval,
                min_delta=float(args.pose_utility_min_score_delta),
                max_frame_drop=float(args.pose_utility_max_frame_drop),
            )
            pose_utility_info = {
                "status": "accepted" if utility_ok else "rejected",
                "gate": utility_gate,
                "old_eval": old_eval,
                "new_eval": new_eval,
            }
            if not utility_ok:
                item = {
                    "iter": it,
                    "frame": cand_obs["frame"],
                    "status": "rejected",
                    "reason": "negative_pose_utility",
                    "select": select_info,
                    "refine": refine_info,
                    "consistency": consistency_info,
                    "pose_delta_optimization": pose_delta_info,
                    "pose_utility": pose_utility_info,
                }
                rejected.append(item)
                accepted.pop()
                accepted_frames.discard(str(cand_obs["frame"]))
                iter_logs.append(item)
                continue

        _replace_accepted_poses(accepted, pose_map_after_delta)
        current_mesh = proposed_mesh
        # Raw poses are refreshed only after shape update, then used by the next
        # candidate-selection round. The selected candidate still only receives
        # pose refinement before fusion.
        raw_update_logs = _refresh_raw_poses_after_shape_update(
            current_mesh,
            observations,
            raw_pose_map,
            args,
            seed_offset=1000 + it * 100,
        )
        iter_logs.append(
            {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "accepted",
                "accepted_count": len(accepted),
                "select": select_info,
                "refine": refine_info,
                "consistency": consistency_info,
                "pose_delta_optimization": pose_delta_info,
                "pose_utility": pose_utility_info,
                "raw_pose_refresh": raw_update_logs,
            }
        )

    final_mesh_o3d = _rebuild_tsdf_mesh(accepted, out_obj, k, args, o3d)
    status = "success" if final_mesh_o3d is not None and len(accepted) > 0 else "base_copied"
    if status == "base_copied":
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_obj, out_obj)

    poses_dir = ensure_dir(iter_root / "poses")
    for item in accepted:
        np.savetxt(poses_dir / f"{item['obs']['frame']}.txt", np.asarray(item["pose"], dtype=np.float32), fmt="%.8f")
    (iter_root / "accepted_frames.txt").write_text(
        "\n".join(str(item["obs"]["frame"]) for item in accepted) + "\n",
        encoding="utf-8",
    )
    (iter_root / "rejected_frames.txt").write_text(
        "\n".join(str(item.get("frame", "")) for item in rejected if item.get("frame", "")) + "\n",
        encoding="utf-8",
    )
    part_summary = {
        "part": part_name,
        "part_model": part_model,
        "status": status,
        "frames": len(accepted) if status == "success" else 0,
        "candidate_frames": len(observations),
        "accepted_frames": [str(item["obs"]["frame"]) for item in accepted],
        "rejected": rejected,
        "seed_frame": str(seed_obs["frame"]),
        "model": str(out_obj),
        "iter_root": str(iter_root),
        "iterations": iter_logs,
    }
    write_json(iter_root / "summary.json", part_summary)
    return part_summary


def _require_base(args: argparse.Namespace, obj: DatasetObject, base_method: str) -> None:
    parts = list_parts(obj)
    base_root = method_pose_ready_dir(Path(args.work_root).resolve(), base_method, args.split, obj.name)
    missing = [
        part_model_name(p, i)
        for i, p in enumerate(parts)
        if not model_obj_path(base_root, part_model_name(p, i)).exists()
    ]
    if not missing:
        return
    if not getattr(args, "build_base_if_missing", False):
        raise FileNotFoundError(
            f"base method '{base_method}' missing models for {obj.name}: {missing[:5]}. "
            "Run the base reconstruction first or pass --build-base-if-missing."
        )
    if base_method == "sam3d":
        from run.recon_sam3d import reconstruct_object
    elif base_method == "hunyuan3d":
        from run.recon_hunyuan3d import reconstruct_object
    else:
        raise ValueError(f"unknown base method: {base_method}")
    reconstruct_object(obj, args)


def run_tsdf_object(obj: DatasetObject, args: argparse.Namespace, base_method: str, method: str) -> Dict[str, object]:
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("Open3D is required for TSDF reconstruction.") from e

    work_root = Path(args.work_root).resolve()
    _require_base(args, obj, base_method)
    base_pose_root = method_pose_ready_dir(work_root, base_method, args.split, obj.name)
    out_pose_root = ensure_dir(method_pose_ready_dir(work_root, method, args.split, obj.name))
    out_model_root = ensure_dir(method_models_dir(work_root, method, args.split, obj.name))
    copy_model_tree(base_pose_root, out_model_root, overwrite=args.overwrite)

    k = load_k(obj)
    parts = list_parts(obj)
    summary = {"method": method, "base_method": base_method, "object": obj.name, "parts": []}

    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        base_obj = model_obj_path(base_pose_root, part_model)
        out_obj = model_obj_path(out_pose_root, part_model)
        if out_obj.exists() and not args.overwrite:
            summary["parts"].append({"part": part_name, "status": "cached", "model": str(out_obj)})
            continue
        if not base_obj.exists():
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "base_model_missing"})
            continue

        try:
            part_summary = _iterative_tsdf_part(
                obj=obj,
                part_name=part_name,
                part_model=part_model,
                base_obj=base_obj,
                out_obj=out_obj,
                work_root=work_root,
                method=method,
                k=k,
                args=args,
                o3d=o3d,
            )
        except Exception as e:
            out_obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_obj, out_obj)
            part_summary = {
                "part": part_name,
                "part_model": part_model,
                "status": "base_copied",
                "reason": "iterative_tsdf_failed",
                "error": str(e),
                "frames": 0,
                "model": str(out_obj),
            }
        summary["parts"].append(
            {
                **part_summary,
                "source_model": str(base_obj),
            }
        )

    copy_model_tree(out_pose_root, out_model_root, overwrite=True)
    write_json(method_object_dir(work_root, method, args.split, obj.name) / "summary.json", summary)
    return summary


def add_tsdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build-base-if-missing", action="store_true", help="Run the base method if shared cache is missing.")
    parser.add_argument("--voxel-length", type=float, default=0.005)
    parser.add_argument("--sdf-trunc", type=float, default=0.02)
    parser.add_argument("--depth-trunc", type=float, default=10.0)
    parser.add_argument("--iter-tsdf-max-frames", type=int, default=16, help="Maximum accepted frames per part; <=0 allows all candidates.")
    parser.add_argument("--iter-tsdf-min-points", type=int, default=64, help="Minimum valid masked depth points for a candidate frame.")
    parser.add_argument("--iter-tsdf-min-view-angle", type=float, default=3.0, help="Reject frames that are too close to accepted views.")
    parser.add_argument("--iter-tsdf-max-view-angle", type=float, default=35.0, help="Reject frames whose nearest accepted view is too far in rotation.")
    parser.add_argument("--iter-tsdf-min-translation", type=float, default=0.002, help="Reject frames that are too close to accepted views in translation.")
    parser.add_argument("--iter-tsdf-max-translation", type=float, default=0.08, help="Reject frames whose nearest accepted view is too far in translation.")
    parser.add_argument("--iter-tsdf-refine-iters", type=int, default=15)
    parser.add_argument("--iter-tsdf-refine-points", type=int, default=2000)
    parser.add_argument("--iter-tsdf-mesh-samples", type=int, default=20000)
    parser.add_argument("--iter-tsdf-refine-max-dist", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-refine-trim-quantile", type=float, default=0.8)
    parser.add_argument("--iter-tsdf-refine-residual-thresh", type=float, default=0.02)
    parser.add_argument("--iter-tsdf-refine-angle-eps", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-refine-trans-eps", type=float, default=1e-4)
    parser.add_argument("--iter-tsdf-max-refine-angle", type=float, default=20.0)
    parser.add_argument("--iter-tsdf-max-refine-translation", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-consistency-points", type=int, default=1500)
    parser.add_argument("--iter-tsdf-depth-residual-thresh", type=float, default=0.02)
    parser.add_argument("--iter-tsdf-min-valid-proj-ratio", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-min-depth-inlier-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pose-utility-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each accepted TSDF candidate, validate whether the new mesh improves pose-oriented metrics.",
    )
    parser.add_argument(
        "--pose-utility-min-score-delta",
        type=float,
        default=-0.02,
        help="Minimum allowed new_score-old_score for accepting a fused mesh; negative allows small temporary drops.",
    )
    parser.add_argument(
        "--pose-utility-max-frame-drop",
        type=float,
        default=0.15,
        help="Maximum allowed drop in the worst per-frame utility score after fusion.",
    )
    parser.add_argument("--pose-utility-max-points", type=int, default=2048)
    parser.add_argument("--pose-utility-mesh-samples", type=int, default=8192)
    parser.add_argument("--pose-utility-depth-inlier-thresh", type=float, default=0.02)
    parser.add_argument("--pose-utility-projection-depth-thresh", type=float, default=0.03)
    parser.add_argument("--pose-utility-min-projected-points", type=int, default=64)
    parser.add_argument("--pose-utility-depth-weight", type=float, default=4.0)
    parser.add_argument("--pose-utility-inlier-weight", type=float, default=1.0)
    parser.add_argument("--pose-utility-mask-weight", type=float, default=1.0)
    parser.add_argument("--pose-utility-dt-weight", type=float, default=0.5)
    parser.add_argument("--pose-utility-dr-weight", type=float, default=0.01)
    parser.add_argument(
        "--pose-delta-opt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh accepted-frame ICP pose deltas against the new mesh and anchor them to the seed frame.",
    )
    parser.add_argument("--pose-delta-rounds", type=int, default=1)
    parser.add_argument(
        "--pose-delta-max-dt",
        type=float,
        default=0.0,
        help="Maximum accepted per-refresh translation delta in meters; <=0 reuses --iter-tsdf-max-refine-translation.",
    )
    parser.add_argument(
        "--pose-delta-max-dr-deg",
        type=float,
        default=0.0,
        help="Maximum accepted per-refresh rotation delta in degrees; <=0 reuses --iter-tsdf-max-refine-angle.",
    )
    parser.add_argument(
        "--pose-delta-refine-anchor",
        action="store_true",
        help="Also refine the seed/reference pose before applying the object-frame anchor offset.",
    )
