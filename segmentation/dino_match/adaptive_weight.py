import argparse
from collections import defaultdict
import json
import os
import re
import shutil

import cv2
import numpy as np
try:
    import trimesh
except Exception:
    trimesh = None

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None

try:
    from .match import DINOv2CADMatcher
    from .render_score import compute_render_score_at_pose
except Exception:
    try:
        from dino_match.match import DINOv2CADMatcher
        from dino_match.render_score import compute_render_score_at_pose
    except Exception:
        try:
            from match import DINOv2CADMatcher
            from render_score import compute_render_score_at_pose
        except Exception:
            DINOv2CADMatcher = None
            compute_render_score_at_pose = None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


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


def _load_intrinsic(intrinsic_path):
    k = np.loadtxt(intrinsic_path, dtype=np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic shape {k.shape} from {intrinsic_path}")
    return k.astype(np.float32)


def _find_depth_path(obj_dir, frame_id):
    depth_dir = os.path.join(obj_dir, "depth")
    if not os.path.isdir(depth_dir):
        return None
    for ext in (".png", ".tiff", ".tif", ".npy", ".exr"):
        p = os.path.join(depth_dir, f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def _find_rgb_path(obj_dir, frame_id):
    rgb_dir = os.path.join(obj_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(rgb_dir, f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def _load_depth(depth_path):
    if depth_path.lower().endswith(".npy"):
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read depth: {depth_path}")
        depth = depth.astype(np.float32)
    if depth.max() > 50:
        depth = depth / 1000.0
    depth[(~np.isfinite(depth)) | (depth <= 1e-6)] = 0.0
    return depth


def _load_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def _backproject_masked_depth(mask, depth, intrinsic):
    valid = (mask > 0) & (depth > 1e-6)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    x = (xs - cx) * z / max(fx, 1e-8)
    y = (ys - cy) * z / max(fy, 1e-8)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _sample_points(points, max_points, rng):
    n = int(points.shape[0])
    if n <= max_points:
        return points.astype(np.float32)

    centroids = np.zeros((max_points,), dtype=np.int32)
    distance = np.ones((n,), dtype=np.float64) * 1e10
    farthest = int(rng.integers(0, n))
    for i in range(max_points):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return points[centroids].astype(np.float32)


def _apply_transform_np(points, transform):
    if points.shape[0] == 0:
        return points.astype(np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points.astype(np.float32), ones], axis=1)
    out = (transform @ homo.T).T[:, :3]
    return out.astype(np.float32)


def umeyama_alignment(src, dst, with_scale=False):
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("Empty point set for Umeyama alignment.")
    if src.shape != dst.shape:
        raise ValueError(f"Umeyama expects paired points. Got {src.shape} vs {dst.shape}")

    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / max(n, 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / max(n, 1)
        scale = float(np.sum(s) / max(var_src, 1e-8))
    else:
        scale = 1.0

    t = dst_mean - scale * (r @ src_mean)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = scale * r
    transform[:3, 3] = t
    return transform


def _nn_pair_and_dist(src, dst):
    if cKDTree is not None:
        tree = cKDTree(dst)
        dists, idx = tree.query(src, k=1, workers=-1)
        return dst[idx], dists.astype(np.float32)

    dst_nn = np.zeros_like(src, dtype=np.float32)
    dists = np.zeros((src.shape[0],), dtype=np.float32)
    chunk = 1024
    for s in range(0, src.shape[0], chunk):
        e = min(s + chunk, src.shape[0])
        diff = src[s:e, None, :] - dst[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        nn_idx = np.argmin(d2, axis=1)
        dst_nn[s:e] = dst[nn_idx]
        dists[s:e] = np.sqrt(d2[np.arange(e - s), nn_idx]).astype(np.float32)
    return dst_nn, dists


def _build_nn_index(dst):
    if cKDTree is not None:
        return cKDTree(dst)
    return None


def _query_nn(index_or_none, src, dst):
    if index_or_none is not None:
        dists, idx = index_or_none.query(src, k=1, workers=-1)
        return dists.astype(np.float32), idx
    n = src.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    idx = np.zeros((n,), dtype=np.int32)
    dists = np.zeros((n,), dtype=np.float32)
    chunk = 1024
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = src[s:e, None, :] - dst[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        nn_idx = np.argmin(d2, axis=1).astype(np.int32)
        idx[s:e] = nn_idx
        dists[s:e] = np.sqrt(d2[np.arange(e - s), nn_idx]).astype(np.float32)
    return dists, idx


def _estimate_similarity_from_pointclouds(src_points, dst_points, max_points, rng, with_scale=False):
    src = _sample_points(src_points, max_points=max_points, rng=rng)
    dst = _sample_points(dst_points, max_points=max_points, rng=rng)
    if src.shape[0] < 20 or dst.shape[0] < 20:
        raise ValueError("Not enough points for alignment.")

    nn_index = _build_nn_index(dst)
    src_ctr = src.mean(axis=0)
    dst_ctr = dst.mean(axis=0)

    def scale_init():
        if not with_scale:
            return 1.0
        src_span = np.percentile(src, 95, axis=0) - np.percentile(src, 5, axis=0)
        dst_span = np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0)
        src_delta = np.linalg.norm(src - src_ctr, axis=1)
        dst_delta = np.linalg.norm(dst - dst_ctr, axis=1)
        src_std = np.mean(src_delta)
        dst_std = np.mean(dst_delta)
        src_norm = float(np.linalg.norm(src_span))
        dst_norm = float(np.linalg.norm(dst_span))
        s_std = 1.0 if src_std < 1e-8 else (dst_std / src_std)
        s_span = 1.0 if src_norm < 1e-8 else (dst_norm / src_norm)
        s0 = float(np.median([s_std, s_span]))
        return float(np.clip(s0, 0.02, 5.0))

    def make_tf(rot, scl):
        tf = np.eye(4, dtype=np.float32)
        tf[:3, :3] = (scl * rot).astype(np.float32)
        tf[:3, 3] = (dst_ctr - scl * (rot @ src_ctr)).astype(np.float32)
        return tf

    def euler_to_rot(rx, ry, rz):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        return rz_m @ ry_m @ rx_m

    def eval_trimmed_rmse(tf, q=70):
        pts = _apply_transform_np(src, tf)
        dists, _ = _query_nn(nn_index, pts, dst)
        if dists.shape[0] < 20:
            return np.inf
        cut = np.percentile(dists, q)
        keep = dists <= cut
        if int(np.sum(keep)) < 20:
            return np.inf
        return float(np.sqrt(np.mean((dists[keep]) ** 2)))

    base_scale = scale_init()
    if with_scale:
        scale_min = max(0.02, base_scale * 0.35)
        scale_max = min(5.0, base_scale * 2.5)
    else:
        scale_min, scale_max = 1.0, 1.0

    angle_set = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
    init_candidates = [make_tf(np.eye(3, dtype=np.float32), base_scale)]
    for rx in angle_set:
        for ry in angle_set:
            for rz in angle_set:
                r0 = euler_to_rot(rx, ry, rz)
                init_candidates.append(make_tf(r0, base_scale))

    scored = []
    for tf0 in init_candidates:
        scored.append((eval_trimmed_rmse(tf0, q=70), tf0))
    scored.sort(key=lambda x: x[0])
    seeds = [x[1] for x in scored[:8]]
    if len(seeds) == 0:
        raise ValueError("No valid initialization seeds for alignment.")

    best_tf = seeds[0].copy()
    best_err = np.inf
    trim_q_schedule = [70, 75, 80, 85, 88, 90]
    for seed in seeds:
        tf = seed.copy()
        for q in trim_q_schedule:
            src_now = _apply_transform_np(src, tf)
            dists, nn_idx = _query_nn(nn_index, src_now, dst)
            if dists.shape[0] < 20:
                break
            cut = np.percentile(dists, q)
            keep = dists <= cut
            if int(np.sum(keep)) < 20:
                continue
            src_k = src_now[keep]
            dst_k = dst[nn_idx[keep]]
            current_with_scale = with_scale if q > 88 else False
            delta = umeyama_alignment(src_k, dst_k, with_scale=current_with_scale)
            if current_with_scale:
                s_step = np.linalg.norm(delta[:3, 0])
                s_damped = 1.0 + (s_step - 1.0) * 0.2
                u, _, vt = np.linalg.svd(delta[:3, :3])
                delta[:3, :3] = (u @ vt) * s_damped
            tf = (delta @ tf).astype(np.float32)

            if current_with_scale:
                a = tf[:3, :3].astype(np.float64)
                det = np.linalg.det(a)
                if det > 1e-12:
                    s = float(np.cbrt(det))
                    s_clamped = float(np.clip(s, scale_min, scale_max))
                    if abs(s - s_clamped) > 1e-8:
                        r = a / max(s, 1e-8)
                        tf[:3, :3] = (s_clamped * r).astype(np.float32)

        if with_scale:
            src_final = _apply_transform_np(src, tf)
            dists_f, nn_idx_f = _query_nn(nn_index, src_final, dst)
            core_cut = np.percentile(dists_f, 70)
            core_mask = dists_f <= core_cut
            if np.sum(core_mask) > 20:
                src_core = src[core_mask]
                dst_core = dst[nn_idx_f[core_mask]]
                s_src = np.std(np.linalg.norm(src_core - src_core.mean(axis=0), axis=1))
                s_dst = np.std(np.linalg.norm(dst_core - dst_core.mean(axis=0), axis=1))
                refined_s = s_dst / (s_src + 1e-8)
                refined_s = np.clip(refined_s, scale_min, scale_max)
                u, _, vt = np.linalg.svd(tf[:3, :3])
                refined_r = u @ vt
                tf[:3, :3] = (refined_r * refined_s).astype(np.float32)
                src_mean = src_core.mean(axis=0)
                dst_mean = dst_core.mean(axis=0)
                tf[:3, 3] = (dst_mean - refined_s * (refined_r @ src_mean)).astype(np.float32)
        err = eval_trimmed_rmse(tf, q=90)
        if err < best_err:
            best_err = err
            best_tf = tf.copy()
    return best_tf


def _iterative_voting_alignment(src_points, dst_points, max_points, rng, with_scale=False, max_iter=4):
    src = _sample_points(src_points, max_points=max_points, rng=rng)
    dst = _sample_points(dst_points, max_points=max_points, rng=rng)
    if src.shape[0] < 20 or dst.shape[0] < 20:
        raise ValueError("Not enough points for iterative alignment.")

    dst_tree = _build_nn_index(dst)
    tf = _estimate_similarity_from_pointclouds(src, dst, max_points=max_points, rng=rng, with_scale=with_scale)
    n = src.shape[0]
    votes = np.zeros((n,), dtype=np.int32)
    excluded_for_fit = np.zeros((n,), dtype=bool)

    for _ in range(max_iter):
        src_now = _apply_transform_np(src, tf)
        dists, _ = _query_nn(dst_tree, src_now, dst)
        if dists.shape[0] < 20:
            break

        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med))) + 1e-8
        robust_sigma = 1.4826 * mad
        th = med + 2.5 * robust_sigma
        outlier_round = dists > th

        min_out = max(1, int(0.10 * n))
        if int(np.sum(outlier_round)) < min_out:
            k = min(n - 1, min_out)
            if k > 0:
                idx_desc = np.argsort(dists)[::-1]
                outlier_round = np.zeros((n,), dtype=bool)
                outlier_round[idx_desc[:k]] = True

        votes[outlier_round] += 1
        excluded_for_fit = np.logical_or(excluded_for_fit, outlier_round)

        fit_mask = ~excluded_for_fit
        if int(np.sum(fit_mask)) < 20:
            fit_mask = ~outlier_round
        if int(np.sum(fit_mask)) < 20:
            break

        tf = _estimate_similarity_from_pointclouds(
            src[fit_mask],
            dst,
            max_points=max_points,
            rng=rng,
            with_scale=with_scale,
        )

    max_vote = int(np.max(votes)) if votes.size > 0 else 0
    final_keep = np.ones((n,), dtype=bool)
    if max_vote > 0:
        final_keep = votes < max_vote
        if int(np.sum(final_keep)) < 20:
            order = np.argsort(votes)
            keep_n = min(n, max(20, int(0.7 * n)))
            final_keep = np.zeros((n,), dtype=bool)
            final_keep[order[:keep_n]] = True

    src_final = src[final_keep] if int(np.sum(final_keep)) >= 20 else src
    tf_precise = _estimate_similarity_from_pointclouds(
        src_final,
        dst,
        max_points=max_points,
        rng=rng,
        with_scale=with_scale,
    )
    return tf_precise.astype(np.float32), final_keep, src.astype(np.float32)


def _estimate_alignment_rmse(src_points, dst_points, max_points, rng):
    tf, keep_mask, sampled_src = _iterative_voting_alignment(
        src_points=src_points,
        dst_points=dst_points,
        max_points=max_points,
        rng=rng,
        with_scale=False,
        max_iter=4,
    )
    sampled_src_eval = sampled_src[keep_mask] if int(np.sum(keep_mask)) >= 20 else sampled_src
    src_aligned = _apply_transform_np(sampled_src_eval, tf)
    _, dists = _nn_pair_and_dist(src_aligned, dst_points.astype(np.float32))
    if dists.shape[0] < 20:
        raise ValueError("Too few inliers after alignment.")
    cut = np.percentile(dists, 85)
    keep = dists <= cut
    if int(np.sum(keep)) < 20:
        raise ValueError("Too few inliers after trimmed evaluation.")
    rmse = float(np.sqrt(np.mean((dists[keep]) ** 2)))
    return rmse, tf


def _score_from_rmse(rmse):
    return 1.0 / (1.0 + float(rmse))


def _weighted_score(orig_score, pointcloud_score, render_score, sam6d_weight, pointcloud_weight, render_weight):
    total = float(sam6d_weight)
    score = float(sam6d_weight) * float(orig_score)
    if pointcloud_score is not None:
        total += float(pointcloud_weight)
        score += float(pointcloud_weight) * float(pointcloud_score)
    if render_score is not None:
        total += float(render_weight)
        score += float(render_weight) * float(render_score)
    return float(score / max(total, 1e-8))


def _load_reference_points_for_candidate(model_dir, max_points, rng):
    if not model_dir:
        return None, "", "model_dir_missing"

    mesh_path = os.path.join(model_dir, "model.obj")
    if trimesh is None:
        return None, mesh_path, "trimesh_unavailable"
    if not os.path.exists(mesh_path):
        return None, mesh_path, "mesh_missing"
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geos = tuple(g for g in mesh.geometry.values())
            if len(geos) == 0:
                return None, mesh_path, "mesh_invalid"
            mesh = trimesh.util.concatenate(geos)
        if mesh is None or mesh.vertices is None:
            return None, mesh_path, "mesh_invalid"
        if hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0:
            points = mesh.sample(max(3 * max_points, 3000)).astype(np.float32)
        else:
            points = np.asarray(mesh.vertices, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 20:
            return None, mesh_path, "mesh_points_too_few"
        return _sample_points(points, max_points=max_points, rng=rng), mesh_path, "ok_mesh_fallback"
    except Exception:
        return None, mesh_path, "mesh_read_failed"


def _choose_candidate_by_pointcloud(
    frame_candidates,
    depth,
    intrinsic,
    max_points,
    rng,
    image_rgb=None,
    semantic_matcher=None,
    appearance_matcher=None,
    sam6d_weight=0.5,
    pointcloud_weight=0.35,
    render_weight=0.15,
):
    evals = []
    sam6d_weight = float(sam6d_weight)
    pointcloud_weight = float(pointcloud_weight)
    render_weight = float(render_weight)
    for i, cand in enumerate(frame_candidates):
        model_dir = cand.get("cad_model_dir", "")
        mask_path = cand.get("saved_mask_path", "") or cand.get("mask_path", "")
        ref_points, ref_source_path, ref_load_reason = _load_reference_points_for_candidate(
            model_dir=model_dir,
            max_points=max_points,
            rng=rng,
        )

        item = {
            "candidate_index": int(i),
            "orig_score": float(cand.get("score", 0.0)),
            "cad_part_id": int(cand.get("cad_part_id", -1)),
            "mask_path": mask_path,
            "reference_points_path": ref_source_path,
            "pointcloud_rmse": None,
            "pointcloud_score": None,
            "pointcloud_pose": None,
            "render_score": None,
            "render_semantic_score": None,
            "render_appearance_score": None,
            "adaptive_weighted_score": None,
            "valid": False,
            "reason": "",
        }

        if (not mask_path) or (not os.path.exists(mask_path)):
            item["reason"] = "mask_missing"
            evals.append(item)
            continue

        mask = _load_mask(mask_path)
        if mask is None:
            item["reason"] = "mask_read_failed"
            evals.append(item)
            continue

        if ref_points is None:
            item["reason"] = ref_load_reason
            evals.append(item)
            continue

        cur_points = _backproject_masked_depth(mask, depth, intrinsic)
        if cur_points.shape[0] < 20:
            item["reason"] = "candidate_points_too_few"
            evals.append(item)
            continue

        try:
            rmse, pose = _estimate_alignment_rmse(
                src_points=ref_points,
                dst_points=cur_points,
                max_points=max_points,
                rng=rng,
            )
            item["pointcloud_rmse"] = float(rmse)
            item["pointcloud_score"] = float(_score_from_rmse(rmse))
            item["pointcloud_pose"] = np.asarray(pose, dtype=np.float32).reshape(4, 4).tolist()
            if (
                render_weight > 0.0
                and image_rgb is not None
                and semantic_matcher is not None
                and appearance_matcher is not None
                and compute_render_score_at_pose is not None
            ):
                render_res = compute_render_score_at_pose(
                    image_rgb=image_rgb,
                    k=intrinsic,
                    query_mask=mask,
                    pose=pose,
                    semantic_matcher=semantic_matcher,
                    appearance_matcher=appearance_matcher,
                    cad_model_dir=model_dir,
                    mesh_path=os.path.join(model_dir, "model.obj"),
                )
                if render_res is not None:
                    item["render_score"] = float(render_res["score"])
                    item["render_semantic_score"] = float(render_res["semantic_score"])
                    item["render_appearance_score"] = float(render_res["appearance_score"])
            item["adaptive_weighted_score"] = _weighted_score(
                orig_score=item["orig_score"],
                pointcloud_score=item["pointcloud_score"],
                render_score=item["render_score"],
                sam6d_weight=sam6d_weight,
                pointcloud_weight=pointcloud_weight,
                render_weight=render_weight,
            )
            item["valid"] = True
        except Exception as e:
            item["reason"] = f"align_failed:{e}"

        evals.append(item)

    valid_items = [x for x in evals if x["valid"]]
    if valid_items:
        valid_items.sort(key=lambda x: float(x["adaptive_weighted_score"]), reverse=True)
        best_idx = int(valid_items[0]["candidate_index"])
    else:
        # Fallback to original match score.
        if len(frame_candidates) == 0:
            return None, evals
        best_idx = int(np.argmax([float(x.get("score", 0.0)) for x in frame_candidates]))
    return best_idx, evals


def _build_sorted_candidates_per_cad(frame_candidates):
    by_cad = defaultdict(list)
    for idx, cand in enumerate(frame_candidates):
        cad_part_id = int(cand.get("cad_part_id", -1))
        if cad_part_id < 0:
            continue
        item = dict(cand)
        item["_frame_candidate_index"] = int(idx)
        item["_proposal_index"] = int(cand.get("proposal_index", -1))
        by_cad[cad_part_id].append(item)

    out = {}
    for cad_part_id, items in by_cad.items():
        items = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        out[int(cad_part_id)] = items
    return out


def _pick_best_for_cad_with_used(
    cad_candidates_sorted,
    topk_per_cad,
    used_masks,
    depth,
    intrinsic,
    max_points,
    rng,
    image_rgb=None,
    semantic_matcher=None,
    appearance_matcher=None,
    sam6d_weight=0.5,
    pointcloud_weight=0.35,
    render_weight=0.15,
):
    available = []
    for cand in cad_candidates_sorted:
        pidx = int(cand.get("_proposal_index", -1))
        if pidx >= 0 and pidx in used_masks:
            continue
        available.append(cand)
        if len(available) >= max(1, int(topk_per_cad)):
            break
    if not available:
        return None, [], "no_available_mask_after_used_filter"

    if depth is None:
        best_local_idx = int(np.argmax([float(x.get("score", 0.0)) for x in available]))
        return available[best_local_idx], [], "depth_missing_fallback_to_original_score"

    best_local_idx, evals = _choose_candidate_by_pointcloud(
        frame_candidates=available,
        depth=depth,
        intrinsic=intrinsic,
        max_points=max_points,
        rng=rng,
        image_rgb=image_rgb,
        semantic_matcher=semantic_matcher,
        appearance_matcher=appearance_matcher,
        sam6d_weight=sam6d_weight,
        pointcloud_weight=pointcloud_weight,
        render_weight=render_weight,
    )
    if best_local_idx is None:
        return None, evals, "no_valid_candidate"
    has_valid = any(x.get("valid", False) for x in evals)
    reason = "pointcloud_rerank" if has_valid else "fallback_to_original_score"
    selected = dict(available[best_local_idx])
    selected_eval = None
    for ev in evals:
        if int(ev.get("candidate_index", -1)) == int(best_local_idx):
            selected_eval = ev
            break
    if selected_eval is not None:
        for key in (
            "pointcloud_rmse",
            "pointcloud_score",
            "pointcloud_pose",
            "render_score",
            "render_semantic_score",
            "render_appearance_score",
            "adaptive_weighted_score",
        ):
            selected[key] = selected_eval.get(key)
        if selected_eval.get("pointcloud_pose") is not None:
            selected["coarse_pose"] = selected_eval["pointcloud_pose"]
            selected["refined_pose"] = selected_eval["pointcloud_pose"]
    return selected, evals, reason


def run_pointcloud_rerank_for_object(
    obj_dir,
    match_out_dir,
    reranked_mask_subdir="matched_pred_mask_direct_match_adaptive",
    match_result_json_name="match_results_sam6d_style.json",
    reranked_json_name="match_results_adaptive_weight.json",
    topk_per_cad=3,
    max_points=2000,
    random_seed=2025,
    sam6d_weight=0.5,
    pointcloud_weight=0.35,
    render_weight=0.15,
    render_model_name="dinov2_vitl14",
):
    match_json_path = os.path.join(match_out_dir, match_result_json_name)
    if not os.path.exists(match_json_path):
        raise FileNotFoundError(f"match result json not found: {match_json_path}")

    intrinsic_path = _find_intrinsic_path(obj_dir)
    if intrinsic_path is None:
        raise FileNotFoundError(f"intrinsic file not found under: {obj_dir}")
    intrinsic = _load_intrinsic(intrinsic_path)

    with open(match_json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    out_mask_root = os.path.join(obj_dir, reranked_mask_subdir)
    os.makedirs(out_mask_root, exist_ok=True)

    semantic_matcher = None
    appearance_matcher = None
    if render_weight > 0.0 and DINOv2CADMatcher is not None and compute_render_score_at_pose is not None:
        semantic_matcher = DINOv2CADMatcher(
            model_name=render_model_name,
            proposal_size=224,
            chunk_size=16,
            background_mean_fill=True,
            use_multi_layer_fusion=True,
            fusion_layers=8,
        )
        appearance_matcher = DINOv2CADMatcher(
            model_name=render_model_name,
            proposal_size=224,
            chunk_size=16,
            background_mean_fill=False,
            use_multi_layer_fusion=True,
            fusion_layers=8,
        )

    reranked_results = {}
    frame_ids = sorted(all_results.keys(), key=natural_sort_key)
    for frame_id in frame_ids:
        frame_candidates = all_results.get(frame_id, []) or []
        frame_out_dir = os.path.join(out_mask_root, frame_id)
        os.makedirs(frame_out_dir, exist_ok=True)
        for name in os.listdir(frame_out_dir):
            if name.startswith("mask_") and name.lower().endswith(".png"):
                p = os.path.join(frame_out_dir, name)
                if os.path.isfile(p):
                    os.remove(p)

        if len(frame_candidates) == 0:
            reranked_results[frame_id] = []
            continue

        depth_path = _find_depth_path(obj_dir, frame_id)
        depth = None if depth_path is None else _load_depth(depth_path)
        rgb_path = _find_rgb_path(obj_dir, frame_id)
        image_rgb = None
        if rgb_path is not None:
            image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        stable_id = sum(ord(c) for c in str(frame_id))

        by_cad = _build_sorted_candidates_per_cad(frame_candidates)

        selected_items = []
        used_cads = set()
        used_masks = set()
        cad_order = sorted(by_cad.keys())
        for cad_part_id in cad_order:
            if cad_part_id in used_cads:
                continue
            cad_candidates_sorted = by_cad[cad_part_id]
            rng = np.random.default_rng(random_seed + (stable_id + int(cad_part_id)) % 100000)
            selected, evals, reason = _pick_best_for_cad_with_used(
                cad_candidates_sorted=cad_candidates_sorted,
                topk_per_cad=topk_per_cad,
                used_masks=used_masks,
                depth=depth,
                intrinsic=intrinsic,
                max_points=max_points,
                rng=rng,
                image_rgb=image_rgb,
                semantic_matcher=semantic_matcher,
                appearance_matcher=appearance_matcher,
                sam6d_weight=sam6d_weight,
                pointcloud_weight=pointcloud_weight,
                render_weight=render_weight,
            )
            selected_out = None
            if selected is not None:
                src_mask = selected.get("saved_mask_path", "") or selected.get("mask_path", "")
                if src_mask and os.path.exists(src_mask):
                    selected_out = os.path.join(frame_out_dir, f"mask_{int(cad_part_id):04d}.png")
                    shutil.copyfile(src_mask, selected_out)
                pidx = int(selected.get("_proposal_index", -1))
                if pidx >= 0:
                    used_masks.add(pidx)
                used_cads.add(int(cad_part_id))

            if selected is not None:
                out_item = dict(selected)
                out_item["selected_mask_saved_path"] = selected_out
                if selected_out is not None:
                    out_item["saved_mask_path"] = selected_out
                out_item["adaptive_reason"] = reason
                out_item["adaptive_candidates_eval"] = evals
                selected_items.append(out_item)

        selected_items = sorted(selected_items, key=lambda x: int(x.get("cad_part_id", -1)))
        reranked_results[frame_id] = selected_items
        print(
            f"[ADAPTIVE] {frame_id}: candidates={len(frame_candidates)}, "
            f"cad_models={len(by_cad)}, selected={len(selected_items)}, topk_per_cad={int(topk_per_cad)}, "
            f"weights=sam6d:{float(sam6d_weight):.3f},pointcloud:{float(pointcloud_weight):.3f},"
            f"render:{float(render_weight):.3f}"
        )

    out_json = os.path.join(match_out_dir, reranked_json_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] adaptive-weight result saved: {out_json}")
    return out_json


def main():
    parser = argparse.ArgumentParser(description="Point-cloud-based adaptive rerank for direct_match top-k results.")
    parser.add_argument("--obj-dir", type=str, required=True)
    parser.add_argument("--match-out-dir", type=str, required=True)
    parser.add_argument("--reranked-mask-subdir", type=str, default="matched_pred_mask_direct_match_adaptive")
    parser.add_argument("--match-result-json-name", type=str, default="match_results_sam6d_style.json")
    parser.add_argument("--reranked-json-name", type=str, default="match_results_adaptive_weight.json")
    parser.add_argument("--topk-per-cad", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--sam6d-weight", type=float, default=0.5)
    parser.add_argument("--pointcloud-weight", type=float, default=0.35)
    parser.add_argument("--render-weight", type=float, default=0.15)
    parser.add_argument("--render-model-name", type=str, default="dinov2_vitl14")
    args = parser.parse_args()

    run_pointcloud_rerank_for_object(
        obj_dir=args.obj_dir,
        match_out_dir=args.match_out_dir,
        reranked_mask_subdir=args.reranked_mask_subdir,
        match_result_json_name=args.match_result_json_name,
        reranked_json_name=args.reranked_json_name,
        topk_per_cad=args.topk_per_cad,
        max_points=args.max_points,
        random_seed=args.random_seed,
        sam6d_weight=args.sam6d_weight,
        pointcloud_weight=args.pointcloud_weight,
        render_weight=args.render_weight,
        render_model_name=args.render_model_name,
    )


if __name__ == "__main__":
    main()
