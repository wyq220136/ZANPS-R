import math
import os
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def _bbox_from_mask(mask_np: np.ndarray):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2, y2]


def _depth_to_meter(depth: np.ndarray):
    if depth is None:
        return None
    d = depth.astype(np.float32)
    if d.size == 0:
        return d
    if np.nanmax(d) > 50.0:
        d = d / 1000.0
    return d


def _estimate_translation_from_mask(mask_np: np.ndarray, depth: np.ndarray, k: np.ndarray):
    d = _depth_to_meter(depth)
    valid = (mask_np > 0) & np.isfinite(d) & (d > 1e-6)
    ys, xs = np.where(valid)
    if len(xs) < 5:
        ys, xs = np.where(mask_np > 0)
        if len(xs) < 5:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        z = 1.0
        u = float(xs.mean())
        v = float(ys.mean())
    else:
        z = float(np.median(d[ys, xs]))
        u = float(np.mean(xs))
        v = float(np.mean(ys))
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (u - cx) * z / max(fx, 1e-8)
    y = (v - cy) * z / max(fy, 1e-8)
    return np.array([x, y, z], dtype=np.float32)


def _euler_to_rot(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rz_m @ ry_m @ rx_m).astype(np.float32)


def sample_view_rotations():
    elevs = [-30.0, -15.0, 0.0, 15.0, 30.0]
    azims = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    rots = []
    for e in elevs:
        for a in azims:
            rots.append(_euler_to_rot(math.radians(e), 0.0, math.radians(a)))
    return rots


@lru_cache(maxsize=4096)
def _load_mesh_cached(mesh_path: str, mtime_key: float):
    _ = mtime_key
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate([m for m in mesh.geometry.values()])
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    ctr = np.asarray(mesh.bounding_box.centroid, dtype=np.float32)
    return v, f, ctr


def _load_mesh_for_path(mesh_path: str):
    try:
        mtime = float(os.path.getmtime(mesh_path))
    except Exception:
        mtime = -1.0
    return _load_mesh_cached(mesh_path, mtime)


def _render_mesh_mask(vertices, faces, pose, k, out_h, out_w):
    r = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32)
    vc = (r @ vertices.T).T + t[None, :]
    z = vc[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((out_h, out_w), dtype=np.uint8)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u = (vc[:, 0] * fx / np.maximum(z, 1e-8) + cx).astype(np.float32)
    v = (vc[:, 1] * fy / np.maximum(z, 1e-8) + cy).astype(np.float32)

    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    tri_z = z[faces].mean(axis=1)
    order = np.argsort(tri_z)[::-1]  # far to near painter
    for idx in order:
        fi = faces[idx]
        if not (valid[fi[0]] and valid[fi[1]] and valid[fi[2]]):
            continue
        pts = np.array(
            [[u[fi[0]], v[fi[0]]], [u[fi[1]], v[fi[1]]], [u[fi[2]], v[fi[2]]]],
            dtype=np.float32,
        )
        pts_i = np.round(pts).astype(np.int32)
        if np.all((pts_i[:, 0] < 0) | (pts_i[:, 0] >= out_w) | (pts_i[:, 1] < 0) | (pts_i[:, 1] >= out_h)):
            continue
        cv2.fillConvexPoly(mask, pts_i, 255)
    return (mask > 0).astype(np.uint8)


def _cosine_from_desc(a: torch.Tensor, b: torch.Tensor):
    aa = F.normalize(a, dim=-1)
    bb = F.normalize(b, dim=-1)
    return float((aa * bb).sum(dim=-1).detach().cpu().item())


def compute_multiview_render_score(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    k: np.ndarray,
    query_mask: np.ndarray,
    query_sem_desc: torch.Tensor,
    query_appe_desc: torch.Tensor,
    semantic_matcher,
    appearance_matcher,
    cad_model_dir: str,
    mesh_path: str | None = None,
):
    mesh_path_use = mesh_path if mesh_path is not None else os.path.join(cad_model_dir, "model.obj")
    if not os.path.exists(mesh_path_use):
        return None
    vertices, faces, ctr = _load_mesh_for_path(mesh_path_use)
    h, w = image_rgb.shape[:2]
    t = _estimate_translation_from_mask(query_mask, depth, k)
    rotations = sample_view_rotations()

    best = None
    q_box = _bbox_from_mask((query_mask > 0).astype(np.uint8))
    if q_box is None:
        return None

    for view_idx, r in enumerate(rotations):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = r
        pose[:3, 3] = t - r @ ctr
        rmask = _render_mesh_mask(vertices, faces, pose, k, h, w)
        rb = _bbox_from_mask(rmask)
        if rb is None:
            continue

        rm_t = torch.as_tensor(rmask[None, ...], dtype=torch.float32)
        rb_t = torch.as_tensor([rb], dtype=torch.float32)
        sem_render = semantic_matcher.encode_proposals(image_rgb, rm_t, rb_t)
        appe_render = appearance_matcher.encode_proposals(image_rgb, rm_t, rb_t)
        sem_score = _cosine_from_desc(query_sem_desc, sem_render[0:1])
        appe_score = _cosine_from_desc(query_appe_desc, appe_render[0:1])

        score = 0.5 * sem_score + 0.5 * appe_score
        if (best is None) or (score > best["score"]):
            best = {
                "score": float(score),
                "semantic_score": float(sem_score),
                "appearance_score": float(appe_score),
                "view_index": int(view_idx),
                "pose": pose.astype(np.float32),
            }
    return best


def compute_render_score_at_pose(
    image_rgb: np.ndarray,
    k: np.ndarray,
    query_mask: np.ndarray,
    pose: np.ndarray,
    semantic_matcher,
    appearance_matcher,
    cad_model_dir: str,
    mesh_path: str | None = None,
):
    mesh_path_use = mesh_path if mesh_path is not None else os.path.join(cad_model_dir, "model.obj")
    if not os.path.exists(mesh_path_use):
        return None
    vertices, faces, _ = _load_mesh_for_path(mesh_path_use)
    h, w = image_rgb.shape[:2]

    q_box = _bbox_from_mask((query_mask > 0).astype(np.uint8))
    if q_box is None:
        return None

    rmask = _render_mesh_mask(vertices, faces, np.asarray(pose, dtype=np.float32).reshape(4, 4), k, h, w)
    rb = _bbox_from_mask(rmask)
    if rb is None:
        return None

    qm_t = torch.as_tensor(query_mask[None, ...], dtype=torch.float32)
    qb_t = torch.as_tensor([q_box], dtype=torch.float32)
    rm_t = torch.as_tensor(rmask[None, ...], dtype=torch.float32)
    rb_t = torch.as_tensor([rb], dtype=torch.float32)

    query_sem = semantic_matcher.encode_proposals(image_rgb, qm_t, qb_t)
    query_appe = appearance_matcher.encode_proposals(image_rgb, qm_t, qb_t)
    sem_render = semantic_matcher.encode_proposals(image_rgb, rm_t, rb_t)
    appe_render = appearance_matcher.encode_proposals(image_rgb, rm_t, rb_t)

    sem_score = _cosine_from_desc(query_sem[0:1], sem_render[0:1])
    appe_score = _cosine_from_desc(query_appe[0:1], appe_render[0:1])
    score = 0.5 * sem_score + 0.5 * appe_score
    return {
        "score": float(score),
        "semantic_score": float(sem_score),
        "appearance_score": float(appe_score),
        "pose": np.asarray(pose, dtype=np.float32).reshape(4, 4),
    }
