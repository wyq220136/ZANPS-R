from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PoseUtilityConfig:
    """Weights and thresholds for pose-oriented mesh validation.

    These metrics intentionally use only observations, masks, depths, and the
    currently estimated poses. They do not consume ground-truth poses for
    non-anchor frames.
    """

    max_points: int = 2048
    mesh_samples: int = 8192
    depth_inlier_thresh: float = 0.02
    projection_depth_thresh: float = 0.03
    min_projected_points: int = 64
    depth_weight: float = 4.0
    inlier_weight: float = 1.0
    mask_weight: float = 1.0
    dt_weight: float = 0.5
    dr_weight: float = 0.01


def pose_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.asarray(a[:3, :3], dtype=np.float64) @ np.asarray(b[:3, :3], dtype=np.float64).T
    cos = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_delta_egocentric(init_pose: np.ndarray, refined_pose: np.ndarray) -> Dict[str, object]:
    """FoundationPose-style egocentric pose delta.

    Both poses are object-in-camera transforms. The translation delta is
    expressed in the camera frame, and the rotation delta maps the initial
    object orientation to the refined object orientation.
    """

    init_pose = np.asarray(init_pose, dtype=np.float32).reshape(4, 4)
    refined_pose = np.asarray(refined_pose, dtype=np.float32).reshape(4, 4)
    dt = refined_pose[:3, 3] - init_pose[:3, 3]
    d_r = refined_pose[:3, :3] @ init_pose[:3, :3].T
    cos = (float(np.trace(d_r)) - 1.0) * 0.5
    angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return {
        "dt": dt.astype(np.float32).reshape(-1).tolist(),
        "dt_norm": float(np.linalg.norm(dt)),
        "dR": d_r.astype(np.float32).reshape(-1).tolist(),
        "dR_angle_deg": angle,
    }


def points_cam_to_obj(points_cam: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    cam_to_ob = np.linalg.inv(np.asarray(ob_in_cam, dtype=np.float32).reshape(4, 4))
    points_cam = np.asarray(points_cam, dtype=np.float32)
    return (cam_to_ob[:3, :3] @ points_cam.T).T + cam_to_ob[:3, 3]


def points_obj_to_cam(points_obj: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    ob_in_cam = np.asarray(ob_in_cam, dtype=np.float32).reshape(4, 4)
    points_obj = np.asarray(points_obj, dtype=np.float32)
    return (ob_in_cam[:3, :3] @ points_obj.T).T + ob_in_cam[:3, 3]


def project_points(points_cam: np.ndarray, k: np.ndarray, shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_cam = np.asarray(points_cam, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32).reshape(3, 3)
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = np.zeros(len(points_cam), dtype=np.int64)
    v = np.zeros(len(points_cam), dtype=np.int64)
    u[valid] = np.rint(points_cam[valid, 0] * float(k[0, 0]) / z[valid] + float(k[0, 2])).astype(np.int64)
    v[valid] = np.rint(points_cam[valid, 1] * float(k[1, 1]) / z[valid] + float(k[1, 2])).astype(np.int64)
    h, w = shape_hw
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def _sample_np(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(len(points), int(max_points), replace=False)
    return points[ids]


def _sample_mesh(mesh, n_points: int, seed: int) -> np.ndarray:
    tm = __import__("trimesh")
    rng_state = np.random.get_state()
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        pts, _ = tm.sample.sample_surface(mesh, max(128, int(n_points)))
    finally:
        np.random.set_state(rng_state)
    return np.asarray(pts, dtype=np.float32)


def _surface_depth_metrics(mesh, obs: Mapping[str, object], pose: np.ndarray, cfg: PoseUtilityConfig, seed: int) -> Dict[str, float]:
    from scipy.spatial import cKDTree

    points_cam = _sample_np(np.asarray(obs["points_cam"], dtype=np.float32), int(cfg.max_points), int(seed))
    if len(points_cam) < 8:
        return {"surface_mean_dist": float("inf"), "surface_inlier_ratio": 0.0, "surface_points": float(len(points_cam))}
    points_obj = points_cam_to_obj(points_cam, pose)
    mesh_pts = _sample_mesh(mesh, int(cfg.mesh_samples), seed=int(seed) + 17)
    if len(mesh_pts) < 8:
        return {"surface_mean_dist": float("inf"), "surface_inlier_ratio": 0.0, "surface_points": float(len(points_cam))}
    dist, _ = cKDTree(mesh_pts).query(points_obj, k=1, workers=-1)
    dist = np.asarray(dist, dtype=np.float32)
    return {
        "surface_mean_dist": float(np.mean(dist)),
        "surface_inlier_ratio": float(np.mean(dist <= float(cfg.depth_inlier_thresh))),
        "surface_points": float(len(points_cam)),
    }


def _mask_projection_metrics(mesh, obs: Mapping[str, object], pose: np.ndarray, k: np.ndarray, cfg: PoseUtilityConfig, seed: int) -> Dict[str, float]:
    pts_obj = _sample_mesh(mesh, int(cfg.mesh_samples), seed=int(seed) + 31)
    depth = np.asarray(obs["depth_m"], dtype=np.float32)
    mask = np.asarray(obs["mask"], dtype=bool)
    pts_cam = points_obj_to_cam(pts_obj, pose)
    u, v, valid = project_points(pts_cam, k, depth.shape[:2])
    if int(np.count_nonzero(valid)) < int(cfg.min_projected_points):
        return {
            "projected_points": float(np.count_nonzero(valid)),
            "mask_inside_ratio": 0.0,
            "projection_depth_inlier_ratio": 0.0,
            "projection_mean_abs_depth": float("inf"),
        }
    ids = np.where(valid)[0]
    in_mask = mask[v[ids], u[ids]]
    depth_vals = depth[v[ids], u[ids]]
    has_depth = depth_vals > 1e-6
    depth_res = np.abs(pts_cam[ids, 2] - depth_vals)
    depth_keep = in_mask & has_depth
    if int(np.count_nonzero(depth_keep)) == 0:
        mean_abs_depth = float("inf")
        depth_inlier = 0.0
    else:
        residual = depth_res[depth_keep]
        mean_abs_depth = float(np.mean(residual))
        depth_inlier = float(np.mean(residual <= float(cfg.projection_depth_thresh)))
    return {
        "projected_points": float(len(ids)),
        "mask_inside_ratio": float(np.mean(in_mask)),
        "projection_depth_inlier_ratio": depth_inlier,
        "projection_mean_abs_depth": mean_abs_depth,
    }


def evaluate_pose_utility(
    mesh,
    observations: Sequence[Mapping[str, object]],
    pose_map: Mapping[str, np.ndarray],
    init_pose_map: Mapping[str, np.ndarray],
    k: np.ndarray,
    cfg: PoseUtilityConfig,
    seed: int = 0,
) -> Dict[str, object]:
    """Score a mesh as a pose-estimation object for the supplied frames.

    The score intentionally combines three pose-facing signals:
    - depth residual against the mesh surface,
    - projected mesh consistency with the observed mask/depth,
    - FoundationPose-style pose delta stability relative to the frame init pose.
    """

    per_frame: List[Dict[str, object]] = []
    scores: List[float] = []
    for idx, obs in enumerate(observations):
        frame = str(obs["frame"])
        if frame not in pose_map:
            continue
        pose = np.asarray(pose_map[frame], dtype=np.float32).reshape(4, 4)
        init_pose = np.asarray(init_pose_map.get(frame, pose), dtype=np.float32).reshape(4, 4)
        surface = _surface_depth_metrics(mesh, obs, pose, cfg, seed=seed + idx * 101)
        projection = _mask_projection_metrics(mesh, obs, pose, k, cfg, seed=seed + idx * 101)
        delta = pose_delta_egocentric(init_pose, pose)
        mean_surface = float(surface["surface_mean_dist"])
        mean_projection = float(projection["projection_mean_abs_depth"])
        if not np.isfinite(mean_surface):
            mean_surface = 1.0
        if not np.isfinite(mean_projection):
            mean_projection = 1.0
        score = (
            float(cfg.inlier_weight) * float(surface["surface_inlier_ratio"])
            + float(cfg.mask_weight) * float(projection["mask_inside_ratio"])
            + 0.5 * float(projection["projection_depth_inlier_ratio"])
            - float(cfg.depth_weight) * (mean_surface + 0.5 * mean_projection)
            - float(cfg.dt_weight) * float(delta["dt_norm"])
            - float(cfg.dr_weight) * float(delta["dR_angle_deg"])
        )
        scores.append(float(score))
        per_frame.append(
            {
                "frame": frame,
                "score": float(score),
                **surface,
                **projection,
                "pose_delta": delta,
            }
        )

    if not scores:
        return {"score": float("-inf"), "frames": 0, "per_frame": per_frame}
    scores_np = np.asarray(scores, dtype=np.float32)
    return {
        "score": float(np.mean(scores_np)),
        "min_score": float(np.min(scores_np)),
        "frames": int(len(scores)),
        "per_frame": per_frame,
    }


def utility_improved(
    old_eval: Mapping[str, object],
    new_eval: Mapping[str, object],
    min_delta: float,
    max_frame_drop: float,
) -> Tuple[bool, Dict[str, float]]:
    old_score = float(old_eval.get("score", float("-inf")))
    new_score = float(new_eval.get("score", float("-inf")))
    old_min = float(old_eval.get("min_score", old_score))
    new_min = float(new_eval.get("min_score", new_score))
    delta = new_score - old_score
    min_delta_seen = new_min - old_min
    ok = np.isfinite(new_score) and (delta >= float(min_delta)) and (min_delta_seen >= -float(max_frame_drop))
    return bool(ok), {
        "old_score": old_score,
        "new_score": new_score,
        "score_delta": float(delta),
        "old_min_score": old_min,
        "new_min_score": new_min,
        "min_score_delta": float(min_delta_seen),
    }
