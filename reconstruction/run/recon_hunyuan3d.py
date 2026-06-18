from pathlib import Path
import sys

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse
import itertools
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import trimesh

from recon_utils import (
    DatasetObject,
    add_common_args,
    backproject,
    ensure_dir,
    find_image,
    list_parts,
    load_depth_m,
    load_k,
    load_mask,
    load_pose,
    mask_path_for_part_frame,
    method_models_dir,
    method_pose_ready_dir,
    model_obj_path,
    part_model_name,
    pose_path_for_part_frame,
    run_object_pipeline,
    select_best_frame_for_part,
)
from reconstruct_hunyuan3d import HunyuanReconstructor, _prepare_rgba_from_mask, default_hunyuan_model_path


METHOD = "hunyuan3d"


def _as_trimesh(mesh_obj) -> trimesh.Trimesh:
    if isinstance(mesh_obj, trimesh.Scene):
        geoms = [g for g in mesh_obj.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            raise ValueError("mesh scene is empty")
        mesh_obj = trimesh.util.concatenate(geoms)
    if not isinstance(mesh_obj, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh_obj)!r}")
    if len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("mesh has no vertices/faces")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
        faces=np.asarray(mesh_obj.faces, dtype=np.int64),
        process=False,
    )


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = min(int(len(src)), int(len(dst)))
    if n < 3:
        raise ValueError(f"not enough points for similarity alignment: {n}")
    if len(src) != n:
        src = src[np.linspace(0, len(src) - 1, n).astype(np.int64)]
    if len(dst) != n:
        dst = dst[np.linspace(0, len(dst) - 1, n).astype(np.int64)]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / float(n)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt
    var = float(np.mean(np.sum(xs * xs, axis=1)))
    scale = float(np.sum(s) / max(var, 1e-12))
    t = mu_d - scale * (r @ mu_s)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = (scale * r).astype(np.float32)
    tf[:3, 3] = t.astype(np.float32)
    return tf


def _apply_tf_points(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return (tf[:3, :3] @ points.T).T + tf[:3, 3]


def _signed_permutation_rotations() -> list[np.ndarray]:
    rotations = []
    for perm in itertools.permutations(range(3)):
        p = np.zeros((3, 3), dtype=np.float32)
        for i, j in enumerate(perm):
            p[i, j] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            r = p.copy()
            r[0] *= signs[0]
            r[1] *= signs[1]
            r[2] *= signs[2]
            if np.linalg.det(r) > 0.5:
                rotations.append(r.astype(np.float32))
    return rotations


def _make_initial_similarity(src: np.ndarray, dst: np.ndarray, rot: np.ndarray) -> np.ndarray:
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src0 = (rot @ (src - src_c).T).T
    src_extent = np.percentile(src0, 95, axis=0) - np.percentile(src0, 5, axis=0)
    dst_extent = np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0)
    src_norm = float(np.linalg.norm(src_extent))
    dst_norm = float(np.linalg.norm(dst_extent))
    scale = dst_norm / max(src_norm, 1e-8)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = (scale * rot).astype(np.float32)
    tf[:3, 3] = (dst_c - scale * (rot @ src_c)).astype(np.float32)
    return tf


def _fit_similarity_icp(
    src: np.ndarray,
    dst: np.ndarray,
    samples: int,
    seed: int,
    iters: int,
    trim_quantile: float,
) -> tuple[np.ndarray, Dict[str, object]]:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(int(seed))
    if len(src) > samples:
        src_fit = src[rng.choice(len(src), size=int(samples), replace=False)]
    else:
        src_fit = src
    if len(dst) > samples:
        dst_fit = dst[rng.choice(len(dst), size=int(samples), replace=False)]
    else:
        dst_fit = dst
    if len(src_fit) < 3 or len(dst_fit) < 3:
        raise ValueError("not enough points for ICP alignment")

    tree = cKDTree(dst_fit)
    best_tf = None
    best_score = float("inf")
    best_kept = 0
    rotations = _signed_permutation_rotations()
    for r in rotations:
        tf = _make_initial_similarity(src_fit, dst_fit, r)
        kept = len(src_fit)
        score = float("inf")
        for _ in range(max(1, int(iters))):
            moved = _apply_tf_points(src_fit, tf)
            d, idx = tree.query(moved, k=1, workers=-1)
            cutoff = np.quantile(d, float(trim_quantile))
            keep = d <= cutoff
            if int(np.count_nonzero(keep)) < 3:
                keep = np.ones_like(d, dtype=bool)
            delta = _umeyama_similarity(moved[keep], dst_fit[idx[keep]])
            tf = delta @ tf
            kept = int(np.count_nonzero(keep))
            score = float(np.mean(d[keep]))
        moved = _apply_tf_points(src_fit, tf)
        d, _ = tree.query(moved, k=1, workers=-1)
        score = float(np.mean(d))
        if score < best_score:
            best_score = score
            best_tf = tf.astype(np.float32)
            best_kept = kept
    if best_tf is None:
        raise RuntimeError("ICP failed to produce an alignment")
    return best_tf, {
        "icp_mean_nn_error_m": best_score,
        "icp_kept_points": int(best_kept),
        "icp_initializations": int(len(rotations)),
        "icp_iterations": int(iters),
        "icp_trim_quantile": float(trim_quantile),
    }


def _align_raw_hunyuan_to_pose_ready(
    raw_obj: Path,
    out_obj: Path,
    obj: DatasetObject,
    part_name: str,
    frame: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    depth_path = find_image(obj.depth_dir, frame)
    pose_path = pose_path_for_part_frame(obj, part_name, frame)
    if mask_path is None or depth_path is None or pose_path is None:
        raise FileNotFoundError(
            f"missing alignment input: mask={mask_path}, depth={depth_path}, pose={pose_path}"
        )

    k = load_k(obj)
    depth_m = load_depth_m(depth_path, args.depth_scale)
    mask = load_mask(mask_path, depth_m.shape[:2])
    obs_cam = backproject(depth_m, mask, k)
    if len(obs_cam) < int(args.min_alignment_points):
        raise RuntimeError(f"too few observed depth points: {len(obs_cam)} < {args.min_alignment_points}")

    mesh_raw = _as_trimesh(trimesh.load(str(raw_obj), force="mesh", process=False))
    sample_n = min(int(args.alignment_samples), max(1000, int(len(mesh_raw.faces) * 8)))
    mesh_pts, _ = trimesh.sample.sample_surface(mesh_raw, sample_n)
    mesh_pts = np.asarray(mesh_pts, dtype=np.float32)

    local_to_cam, icp_info = _fit_similarity_icp(
        mesh_pts,
        obs_cam,
        samples=int(args.alignment_samples),
        seed=int(args.alignment_seed),
        iters=int(args.alignment_icp_iters),
        trim_quantile=float(args.alignment_trim_quantile),
    )
    ob_in_cam = load_pose(pose_path, args.pose_convention)
    cam_to_ob = np.linalg.inv(ob_in_cam).astype(np.float32)
    local_to_obj = cam_to_ob @ local_to_cam

    mesh_ready = mesh_raw.copy()
    mesh_ready.apply_transform(local_to_obj)
    mesh_ready.remove_unreferenced_vertices()
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    mesh_ready.export(str(out_obj))

    aligned_cam = _apply_tf_points(mesh_pts[: min(len(mesh_pts), 5000)], local_to_cam)
    obs_eval = obs_cam
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(obs_eval)
        d, _ = tree.query(aligned_cam, k=1, workers=-1)
        rmse = float(np.sqrt(np.mean(d * d)))
        mean_err = float(np.mean(d))
    except Exception:
        rmse = float("nan")
        mean_err = float("nan")

    np.savetxt(out_obj.parent / "raw_to_camera.txt", local_to_cam, fmt="%.8f")
    np.savetxt(out_obj.parent / "raw_to_object.txt", local_to_obj, fmt="%.8f")
    return {
        "alignment": "depth_similarity_to_object_frame",
        "frame": frame,
        "mask_pixels": int(np.count_nonzero(mask)),
        "observed_points": int(len(obs_cam)),
        "sampled_mesh_points": int(len(mesh_pts)),
        "mean_nn_error_m": mean_err,
        "rmse_nn_error_m": rmse,
        "raw_to_camera": str(out_obj.parent / "raw_to_camera.txt"),
        "raw_to_object": str(out_obj.parent / "raw_to_object.txt"),
        **icp_info,
    }


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace) -> Dict[str, object]:
    reconstructor = HunyuanReconstructor(
        model_path=args.model_path,
        subfolder=args.subfolder,
        num_inference_steps=args.num_inference_steps,
        octree_resolution=args.octree_resolution,
        guidance_scale=args.guidance_scale,
    )
    work_root = Path(args.work_root).resolve()
    models_root = ensure_dir(method_models_dir(work_root, METHOD, args.split, obj.name))
    pose_ready_root = ensure_dir(method_pose_ready_dir(work_root, METHOD, args.split, obj.name))
    parts = list_parts(obj)
    summary = {"method": METHOD, "object": obj.name, "parts": []}

    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        out_obj = model_obj_path(pose_ready_root, part_model)
        if out_obj.exists() and not args.overwrite:
            summary["parts"].append({"part": part_name, "status": "cached", "model": str(out_obj)})
            continue

        frame = select_best_frame_for_part(obj, part_name, args.min_mask_pixels)
        if frame is None:
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "no_visible_frame"})
            continue
        rgb_path = find_image(obj.rgb_dir, frame)
        depth_path = find_image(obj.depth_dir, frame)
        mask_path = mask_path_for_part_frame(obj, part_name, frame)
        pose_path = pose_path_for_part_frame(obj, part_name, frame)
        if rgb_path is None or depth_path is None or mask_path is None or pose_path is None:
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "missing_rgb_depth_mask_or_pose"})
            continue

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgba = _prepare_rgba_from_mask(rgb_bgr, mask_gray)
        if rgba is None:
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "empty_rgba"})
            continue

        raw_obj = model_obj_path(models_root, part_model)
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            reconstructor.reconstruct_part(rgba, str(raw_obj))
            align_info = _align_raw_hunyuan_to_pose_ready(raw_obj, out_obj, obj, part_name, frame, args)
            ok = True
        except Exception as e:
            summary["parts"].append({"part": part_name, "status": "failed", "frame": frame, "error": str(e)})
            continue

        summary["parts"].append(
            {
                "part": part_name,
                "part_model": part_model,
                "status": "success" if ok else "failed",
                "frame": frame,
                "model": str(out_obj),
                "raw_model": str(raw_obj),
                "alignment_info": align_info,
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Hunyuan3D reconstruction with shared-cache outputs.")
    add_common_args(parser, METHOD)
    parser.add_argument("--model-path", type=str, default=default_hunyuan_model_path())
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--alignment-samples", type=int, default=50000)
    parser.add_argument("--alignment-seed", type=int, default=2026)
    parser.add_argument("--min-alignment-points", type=int, default=200)
    parser.add_argument("--alignment-icp-iters", type=int, default=30)
    parser.add_argument("--alignment-trim-quantile", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_object_pipeline(args, METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/recon_hunyuan3d.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 1
#   python reconstruction/recon_hunyuan3d.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --gpus 0,1 --num-workers 2 --mode multi_image --coord-dir /shared/recon_coord/hunyuan3d --reset-coord
#
# Key parameters:
#   --model-path/--subfolder: Hunyuan3D checkpoint source.
#   --num-inference-steps/--octree-resolution/--guidance-scale: generation speed-quality controls.
#   --work-root: shared output/cache root. TSDF/DMesh methods reuse <work-root>/hunyuan3d.
#   --gpus/--num-workers: local multi-GPU scheduling. Usually use one worker per GPU.
