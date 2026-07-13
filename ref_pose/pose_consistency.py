from typing import Dict, Optional

import cv2
import numpy as np

try:
  from reference_evidence import score_reference_evidence
except Exception:
  from ref_pose.reference_evidence import score_reference_evidence


def _sample_mesh_points(mesh, max_points: int = 6000) -> np.ndarray:
  pts = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
  if len(pts) == 0:
    return pts
  if len(pts) > int(max_points):
    ids = np.linspace(0, len(pts) - 1, int(max_points)).astype(np.int64)
    pts = pts[ids]
  return pts


def _transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
  points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
  if len(points) == 0:
    return points
  homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
  return (np.asarray(pose, dtype=np.float32).reshape(4, 4) @ homo.T).T[:, :3]


def _project(points_cam: np.ndarray, K: np.ndarray):
  pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
  z = pts[:, 2]
  valid = z > 1e-6
  uvw = (np.asarray(K, dtype=np.float32).reshape(3, 3) @ pts.T).T
  uv = np.zeros((len(pts), 2), dtype=np.float32)
  uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
  return uv, valid


def _rasterize_points(uv: np.ndarray, keep: np.ndarray, H: int, W: int, radius: int = 2) -> np.ndarray:
  out = np.zeros((int(H), int(W)), dtype=np.uint8)
  if len(uv) == 0 or not np.any(keep):
    return out.astype(bool)
  ui = np.rint(uv[:, 0]).astype(np.int32)
  vi = np.rint(uv[:, 1]).astype(np.int32)
  keep = keep & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
  out[vi[keep], ui[keep]] = 1
  if radius > 0:
    k = 2 * int(radius) + 1
    out = cv2.dilate(out, np.ones((k, k), dtype=np.uint8), iterations=1)
  return out.astype(bool)


def _pose_delta_norm(init_pose: Optional[np.ndarray], pose: np.ndarray) -> Dict[str, Optional[float]]:
  if init_pose is None:
    return {"translation_delta": None, "rotation_delta_deg": None}
  try:
    a = np.asarray(init_pose, dtype=np.float32).reshape(4, 4)
    b = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    dt = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    rel = b[:3, :3] @ a[:3, :3].T
    cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))
    return {"translation_delta": dt, "rotation_delta_deg": ang}
  except Exception:
    return {"translation_delta": None, "rotation_delta_deg": None}


def score_pose(
  mesh,
  pose: np.ndarray,
  depth: np.ndarray,
  mask: np.ndarray,
  K: np.ndarray,
  init_pose: Optional[np.ndarray] = None,
  validity_mean: Optional[float] = None,
  reference_evidence: Optional[dict] = None,
  max_points: int = 6000,
  depth_thresh: float = 0.02,
) -> Dict[str, object]:
  """Observation-consistency score for a refined pose.

  This intentionally uses projected mesh/reference samples instead of a learned
  scorer so it can run as a deterministic reranking signal.
  """
  depth_m = np.asarray(depth, dtype=np.float32)
  mask_bin = np.asarray(mask).astype(bool)
  H, W = depth_m.shape[:2]
  pts_obj = _sample_mesh_points(mesh, max_points=max_points)
  if len(pts_obj) == 0:
    return {"score": -1.0, "status": "empty_mesh"}

  pts_cam = _transform_points(pose, pts_obj)
  uv, valid_z = _project(pts_cam, K)
  ui = np.rint(uv[:, 0]).astype(np.int32)
  vi = np.rint(uv[:, 1]).astype(np.int32)
  in_img = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
  rendered_mask = _rasterize_points(uv, in_img, H, W, radius=2)

  union = np.logical_or(rendered_mask, mask_bin)
  inter = np.logical_and(rendered_mask, mask_bin)
  mask_iou = float(np.sum(inter) / max(1, np.sum(union)))
  outside_mask_ratio = float(np.sum(rendered_mask & (~mask_bin)) / max(1, np.sum(rendered_mask)))
  rendered_area_ratio = float(np.sum(rendered_mask) / max(1, H * W))

  if np.any(in_img):
    ui_v = ui[in_img]
    vi_v = vi[in_img]
    z_v = pts_cam[in_img, 2]
    inside = mask_bin[vi_v, ui_v]
    depth_obs = depth_m[vi_v, ui_v]
    has_depth = depth_obs > 1e-6
    depth_err = np.abs(depth_obs - z_v)
    depth_valid = inside & has_depth
    depth_inlier = depth_valid & (depth_err <= float(depth_thresh))
    depth_inlier_ratio = float(np.sum(depth_inlier) / max(1, np.sum(depth_valid)))
    depth_l1_in_mask = float(np.mean(depth_err[depth_valid])) if np.any(depth_valid) else None
  else:
    depth_inlier_ratio = 0.0
    depth_l1_in_mask = None

  ref_score = score_reference_evidence(
    reference_evidence,
    pose,
    depth_m,
    mask_bin,
    K,
    max_points=max_points,
    depth_thresh=depth_thresh,
  )
  delta = _pose_delta_norm(init_pose, pose)
  validity = 0.0 if validity_mean is None else float(np.clip(validity_mean, 0.0, 1.0))
  delta_penalty = 0.0
  if delta["translation_delta"] is not None:
    delta_penalty += min(float(delta["translation_delta"]), 0.20)
  if delta["rotation_delta_deg"] is not None:
    delta_penalty += min(float(delta["rotation_delta_deg"]) / 180.0, 1.0) * 0.05

  score = (
    0.35 * mask_iou
    + 0.25 * depth_inlier_ratio
    + 0.15 * validity
    + 0.20 * float(ref_score.get("score", 0.0))
    - 0.20 * outside_mask_ratio
    - 0.10 * delta_penalty
  )
  return {
    "status": "ok",
    "score": float(score),
    "mask_iou": mask_iou,
    "depth_inlier_ratio": depth_inlier_ratio,
    "depth_l1_in_mask": depth_l1_in_mask,
    "outside_mask_ratio": outside_mask_ratio,
    "rendered_area_ratio": rendered_area_ratio,
    "validity_mean": None if validity_mean is None else float(validity_mean),
    "reference": ref_score,
    "pose_delta": delta,
  }
