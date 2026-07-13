import os
from typing import Dict, Optional

import cv2
import numpy as np


def _load_pose(path: str) -> Optional[np.ndarray]:
  if not path or not os.path.exists(path):
    return None
  try:
    pose = np.loadtxt(path, dtype=np.float32)
    if pose.shape == (16,):
      pose = pose.reshape(4, 4)
    if pose.shape == (4, 4):
      return pose.astype(np.float32)
  except Exception:
    return None
  return None


def _load_points(path: str) -> Optional[np.ndarray]:
  if not path or not os.path.exists(path):
    return None
  try:
    pts = np.load(path).astype(np.float32)
    pts = pts.reshape(-1, 3)
    if len(pts) > 0:
      return pts
  except Exception:
    return None
  return None


def load_reference_evidence(model_dir: str) -> Dict[str, object]:
  """Load reference evidence saved with a reconstructed part model."""
  points_obj = _load_points(os.path.join(model_dir, "reference_points_obj.npy"))
  points_cam = _load_points(os.path.join(model_dir, "reference_points_cam.npy"))
  points_default = _load_points(os.path.join(model_dir, "reference_points.npy"))
  if points_obj is None:
    points_obj = points_default

  evidence = {
    "model_dir": model_dir,
    "points_obj": points_obj,
    "points_cam": points_cam,
    "points": points_default,
    "raw_pose": _load_pose(os.path.join(model_dir, "raw_pose.txt")),
    "local_to_object": _load_pose(os.path.join(model_dir, "local_to_object.txt")),
    "local_to_reference_camera": _load_pose(os.path.join(model_dir, "local_to_reference_camera.txt")),
  }
  evidence["available"] = points_obj is not None or points_cam is not None or points_default is not None
  return evidence


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
  points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
  if len(points) == 0:
    return points
  homo = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
  return (np.asarray(pose, dtype=np.float32).reshape(4, 4) @ homo.T).T[:, :3]


def project_points(points_cam: np.ndarray, K: np.ndarray):
  pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
  if len(pts) == 0:
    return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
  z = pts[:, 2]
  valid = z > 1e-6
  uvw = (np.asarray(K, dtype=np.float32).reshape(3, 3) @ pts.T).T
  uv = np.zeros((len(pts), 2), dtype=np.float32)
  uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
  return uv, valid


def score_reference_evidence(
  evidence: Optional[Dict[str, object]],
  pose_obj_to_cam: np.ndarray,
  depth: np.ndarray,
  mask: np.ndarray,
  K: np.ndarray,
  max_points: int = 4096,
  depth_thresh: float = 0.02,
) -> Dict[str, object]:
  """Score how well reference-visible points agree with a query RGB-D observation."""
  if not evidence or not evidence.get("available", False):
    return {
      "available": False,
      "score": 0.0,
      "inside_mask_ratio": 0.0,
      "depth_inlier_ratio": 0.0,
      "mean_abs_depth_error": None,
      "valid_projected_points": 0,
    }

  points_obj = evidence.get("points_obj", None)
  if points_obj is None:
    points_obj = evidence.get("points", None)
  if points_obj is None:
    return {
      "available": False,
      "score": 0.0,
      "inside_mask_ratio": 0.0,
      "depth_inlier_ratio": 0.0,
      "mean_abs_depth_error": None,
      "valid_projected_points": 0,
    }

  pts = np.asarray(points_obj, dtype=np.float32).reshape(-1, 3)
  if len(pts) > int(max_points):
    idx = np.linspace(0, len(pts) - 1, int(max_points)).astype(np.int64)
    pts = pts[idx]

  pts_cam = transform_points(pose_obj_to_cam, pts)
  uv, valid_z = project_points(pts_cam, K)
  H, W = depth.shape[:2]
  ui = np.rint(uv[:, 0]).astype(np.int32)
  vi = np.rint(uv[:, 1]).astype(np.int32)
  in_img = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
  if not np.any(in_img):
    return {
      "available": True,
      "score": 0.0,
      "inside_mask_ratio": 0.0,
      "depth_inlier_ratio": 0.0,
      "mean_abs_depth_error": None,
      "valid_projected_points": 0,
    }

  ui = ui[in_img]
  vi = vi[in_img]
  z = pts_cam[in_img, 2]
  mask_bin = np.asarray(mask).astype(bool)
  depth_m = np.asarray(depth, dtype=np.float32)
  inside = mask_bin[vi, ui]
  depth_obs = depth_m[vi, ui]
  has_depth = depth_obs > 1e-6
  depth_err = np.abs(depth_obs - z)
  depth_valid = has_depth & inside
  depth_inlier = depth_valid & (depth_err <= float(depth_thresh))

  inside_ratio = float(np.mean(inside)) if len(inside) else 0.0
  depth_inlier_ratio = float(np.sum(depth_inlier) / max(1, np.sum(depth_valid)))
  mean_err = float(np.mean(depth_err[depth_valid])) if np.any(depth_valid) else None
  score = 0.6 * inside_ratio + 0.4 * depth_inlier_ratio
  return {
    "available": True,
    "score": float(score),
    "inside_mask_ratio": inside_ratio,
    "depth_inlier_ratio": depth_inlier_ratio,
    "mean_abs_depth_error": mean_err,
    "valid_projected_points": int(np.sum(in_img)),
  }


def reference_reliability_map(
  evidence: Optional[Dict[str, object]],
  pose_obj_to_cam: np.ndarray,
  H: int,
  W: int,
  K: np.ndarray,
  radius: int = 3,
) -> np.ndarray:
  """Rasterize projected reference-visible points into a soft reliability map."""
  out = np.zeros((int(H), int(W)), dtype=np.float32)
  if not evidence or not evidence.get("available", False):
    return out
  pts = evidence.get("points_obj", None)
  if pts is None:
    pts = evidence.get("points", None)
  if pts is None:
    return out
  pts_cam = transform_points(pose_obj_to_cam, np.asarray(pts, dtype=np.float32).reshape(-1, 3))
  uv, valid_z = project_points(pts_cam, K)
  ui = np.rint(uv[:, 0]).astype(np.int32)
  vi = np.rint(uv[:, 1]).astype(np.int32)
  keep = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
  out[vi[keep], ui[keep]] = 1.0
  if radius > 0:
    k = 2 * int(radius) + 1
    out = cv2.dilate(out, np.ones((k, k), dtype=np.uint8), iterations=1)
    out = cv2.GaussianBlur(out, (k, k), 0)
    if out.max() > 1e-6:
      out = out / out.max()
  return out.astype(np.float32)


def optimize_pose_with_reference_correspondence(
  evidence: Optional[Dict[str, object]],
  pose_obj_to_cam: np.ndarray,
  depth: np.ndarray,
  mask: np.ndarray,
  K: np.ndarray,
  max_points: int = 4096,
  max_translation_step: float = 0.03,
  depth_thresh: float = 0.05,
) -> tuple[np.ndarray, Dict[str, object]]:
  """Lightweight reference-query correspondence correction for a pose candidate.

  This is a geometric proxy for the One2Any-like correspondence stage: reference
  visible points are projected into the query view, matched to observed depth at
  their projected pixels, and the candidate receives a small robust translation
  update in camera space.
  """
  pose = np.asarray(pose_obj_to_cam, dtype=np.float32).reshape(4, 4).copy()
  if not evidence or not evidence.get("available", False):
    return pose, {"available": False, "updated": False, "reason": "no_reference_evidence"}

  points_obj = evidence.get("points_obj", None)
  if points_obj is None:
    points_obj = evidence.get("points", None)
  if points_obj is None:
    return pose, {"available": False, "updated": False, "reason": "no_reference_points"}

  pts = np.asarray(points_obj, dtype=np.float32).reshape(-1, 3)
  if len(pts) > int(max_points):
    idx = np.linspace(0, len(pts) - 1, int(max_points)).astype(np.int64)
    pts = pts[idx]

  pts_cam = transform_points(pose, pts)
  uv, valid_z = project_points(pts_cam, K)
  H, W = depth.shape[:2]
  ui = np.rint(uv[:, 0]).astype(np.int32)
  vi = np.rint(uv[:, 1]).astype(np.int32)
  mask_bin = np.asarray(mask).astype(bool)
  depth_m = np.asarray(depth, dtype=np.float32)
  in_img = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
  if not np.any(in_img):
    return pose, {"available": True, "updated": False, "reason": "no_projected_points"}

  ui_v = ui[in_img]
  vi_v = vi[in_img]
  pred = pts_cam[in_img]
  inside = mask_bin[vi_v, ui_v]
  depth_obs = depth_m[vi_v, ui_v]
  valid = inside & (depth_obs > 1e-6) & (np.abs(depth_obs - pred[:, 2]) <= float(depth_thresh))
  if int(np.sum(valid)) < 4:
    return pose, {
      "available": True,
      "updated": False,
      "reason": "too_few_correspondence_inliers",
      "inliers": int(np.sum(valid)),
      "projected": int(np.sum(in_img)),
    }

  z_delta = np.median(depth_obs[valid] - pred[valid, 2])
  pred_uv_mean = np.array([np.mean(ui_v[valid]), np.mean(vi_v[valid]), 1.0], dtype=np.float32)
  obs_z = float(np.median(depth_obs[valid]))
  pred_z = float(np.median(pred[valid, 2]))
  pred_center_cam = (np.linalg.inv(np.asarray(K, dtype=np.float32).reshape(3, 3)) @ pred_uv_mean.reshape(3, 1)).reshape(3) * pred_z
  obs_center_cam = (np.linalg.inv(np.asarray(K, dtype=np.float32).reshape(3, 3)) @ pred_uv_mean.reshape(3, 1)).reshape(3) * obs_z
  delta = obs_center_cam - pred_center_cam
  delta[2] = z_delta
  norm = float(np.linalg.norm(delta))
  if norm > float(max_translation_step):
    delta = delta * (float(max_translation_step) / max(norm, 1e-8))
  pose[:3, 3] += delta.astype(np.float32)
  return pose, {
    "available": True,
    "updated": True,
    "inliers": int(np.sum(valid)),
    "projected": int(np.sum(in_img)),
    "translation_delta": delta.astype(float).tolist(),
    "translation_delta_norm": float(np.linalg.norm(delta)),
  }
