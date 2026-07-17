import sys
import os
import re
import json
import argparse
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RECON_ROOT.parent
SERVER_PROJECT_ROOT = Path("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan")


def _resolve_project_dir(name):
    env_key = f"{name.upper().replace('-', '_')}_ROOT"
    env_path = os.environ.get(env_key, "").strip()
    if env_path:
        return Path(env_path)
    local = REPO_ROOT / name
    if local.exists():
        return local
    return SERVER_PROJECT_ROOT / name


SAM3D_PROJECT_ROOT = _resolve_project_dir("sam-3d-objects")
SAM3D_NOTEBOOK_ROOT = SAM3D_PROJECT_ROOT / "notebook"

for _p in (REPO_ROOT, RECON_ROOT, SAM3D_PROJECT_ROOT, SAM3D_NOTEBOOK_ROOT):
    p = str(_p)
    if _p.exists() and p not in sys.path:
        sys.path.insert(0, p)

from inference import Inference, load_image, load_single_mask
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
from pytorch3d.transforms import Transform3d
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera

WITH_MESH_POSTPROCESS = True
WITH_TEXTURE_BAKING = True
CONFIG_PATH = os.environ.get(
    "SAM3D_CONFIG_PATH",
    str(SAM3D_PROJECT_ROOT / "checkpoints" / "hf" / "pipeline.yaml"),
)
USE_POINTMAP_RECONSTRUCTION = True
RUN_ICP_WHEN_POINTMAP_USED = False


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def _build_pointmap_from_depth(depth, intrinsic, object_mask=None):
    """
    Build SAM3D pointmap tensor from metric depth + camera intrinsics.
    Output shape is (H, W, 3), in PyTorch3D camera convention.
    """
    if depth.max() > 50:
        depth = depth / 1000.0
    depth = depth.astype(np.float32)

    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    z = depth
    x = (xs - cx) * z / max(fx, 1e-8)
    y = (ys - cy) * z / max(fy, 1e-8)
    pointmap_cam = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Use official conversion from pipeline_pointmap source.
    pts = torch.from_numpy(pointmap_cam.reshape(-1, 3))
    cam_to_p3d = (
        Transform3d()
        .rotate(camera_to_pytorch3d_camera(device="cpu").rotation)
        .to("cpu")
    )
    pointmap = (
        cam_to_p3d.transform_points(pts)
        .reshape(h, w, 3)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    valid = z > 1e-6
    if object_mask is not None:
        valid = valid & (object_mask > 0)
    pointmap[~valid] = np.nan

    return torch.from_numpy(pointmap)


def build_mesh(
    image_path: str,
    mask_dir: str,
    inference: Inference,
    mask_index: int = 0,
    depth=None,
    intrinsic=None,
    object_mask=None,
):
    """Generate 3D mesh from image/mask, optionally guided by metric pointmap."""
    rgb = load_image(image_path)
    mask = load_single_mask(mask_dir, index=mask_index)
    used_pointmap = False

    pointmap = None
    if USE_POINTMAP_RECONSTRUCTION and depth is not None and intrinsic is not None:
        try:
            pointmap = _build_pointmap_from_depth(depth, intrinsic, object_mask=object_mask)
            used_pointmap = True
        except Exception as e:
            print(f"[WARN] Failed to build pointmap from depth: {e}")
            pointmap = None
            used_pointmap = False

    try:
        output = inference(rgb, mask, seed=42, pointmap=pointmap)
        output = inference._pipeline.postprocess_slat_output(
            output,
            with_mesh_postprocess=WITH_MESH_POSTPROCESS,
            with_texture_baking=WITH_TEXTURE_BAKING,
            use_vertex_color=not WITH_TEXTURE_BAKING,
        )
        mesh = output["glb"]
        build_mesh.last_used_pointmap = used_pointmap
        return mesh
    except Exception as e:
        # Fallback: keep old basic mode if pointmap path fails.
        if pointmap is not None:
            print(f"[WARN] Pointmap reconstruction failed, fallback to basic SAM3D: {e}")
            try:
                output = inference(rgb, mask, seed=42, pointmap=None)
                output = inference._pipeline.postprocess_slat_output(
                    output,
                    with_mesh_postprocess=WITH_MESH_POSTPROCESS,
                    with_texture_baking=WITH_TEXTURE_BAKING,
                    use_vertex_color=not WITH_TEXTURE_BAKING,
                )
                mesh = output["glb"]
                build_mesh.last_used_pointmap = False
                return mesh
            except Exception:
                pass
        build_mesh.last_used_pointmap = False
        return None


def load_real_world_size(mask, depth, intrinsic):
    """Estimate physical object size from depth + intrinsics."""
    if depth.max() > 50:
        depth = depth / 1000.0

    valid = (mask > 0) & (depth > 1e-6)
    if not np.any(valid):
        return np.array([0.1, 0.1, 0.1], dtype=np.float32)

    ys, xs = np.where(valid)
    z = depth[valid]

    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=1)
    min_xyz = np.percentile(pts_cam, 2, axis=0)
    max_xyz = np.percentile(pts_cam, 98, axis=0)
    real_size = max_xyz - min_xyz
    return real_size.astype(np.float32)


def _backproject_masked_depth(mask, depth, intrinsic, max_points=30000):
    """Convert masked depth to camera-frame points with light denoising."""
    if depth.max() > 50:
        depth = depth / 1000.0

    # Depth denoise for ToF/stereo spikes, keep edges via small kernel.
    depth_f = cv2.medianBlur(depth.astype(np.float32), 3)
    valid = (mask > 0) & (depth_f > 1e-6) & (depth > 0.1)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if len(xs) > max_points:
        pick = np.random.choice(len(xs), max_points, replace=False)
        xs = xs[pick]
        ys = ys[pick]

    z = depth_f[ys, xs]
    # Suppress extreme depth outliers inside mask.
    z_lo, z_hi = np.percentile(z, [2, 98])
    keep = (z >= z_lo) & (z <= z_hi)
    xs = xs[keep]
    ys = ys[keep]
    z = z[keep]

    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _voxel_downsample(points, voxel_size):
    if len(points) == 0:
        return points
    if voxel_size <= 0:
        return points
    grid = np.floor(points / voxel_size).astype(np.int32)
    _, keep = np.unique(grid, axis=0, return_index=True)
    return points[np.sort(keep)]


def _weighted_rigid_transform(src_pts, dst_pts, weights=None):
    """Solve R,t in dst ~= R*src + t."""
    if len(src_pts) < 3:
        return np.eye(4, dtype=np.float32)

    if weights is None:
        weights = np.ones((len(src_pts),), dtype=np.float32)
    w = weights.astype(np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.eye(4, dtype=np.float32)
    w /= w_sum

    src_mean = np.sum(src_pts * w[:, None], axis=0)
    dst_mean = np.sum(dst_pts * w[:, None], axis=0)
    src_c = src_pts - src_mean
    dst_c = dst_pts - dst_mean

    h = src_c.T @ (dst_c * w[:, None])
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_mean - r @ src_mean

    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = r.astype(np.float32)
    tf[:3, 3] = t.astype(np.float32)
    return tf


def _apply_tf(pts, tf):
    return (tf[:3, :3] @ pts.T).T + tf[:3, 3]


def _pca_axes(points):
    c = points - points.mean(axis=0, keepdims=True)
    cov = c.T @ c / max(len(c) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order]


def _make_camera_init_candidates(src_pts, tgt_pts):
    """Generate rigid init hypotheses: centroid and PCA frame candidates."""
    src_ctr = src_pts.mean(axis=0)
    tgt_ctr = tgt_pts.mean(axis=0)

    tf_base = np.eye(4, dtype=np.float32)
    tf_base[:3, 3] = (tgt_ctr - src_ctr).astype(np.float32)

    candidates = []
    candidates.append(tf_base.copy())

    # PCA hypotheses with sign ambiguity resolution (4 right-handed options).
    try:
        bs = _pca_axes(src_pts)
        bt = _pca_axes(tgt_pts)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                sz = sx * sy
                d = np.diag([sx, sy, sz]).astype(np.float32)
                r = (bt @ d @ bs.T).astype(np.float32)
                tf = np.eye(4, dtype=np.float32)
                tf[:3, :3] = r
                tf[:3, 3] = (tgt_ctr - r @ src_ctr).astype(np.float32)
                candidates.append(tf)
    except Exception:
        pass

    return candidates


def align_mesh_to_camera_frame(mesh, mask, depth, intrinsic):
    """
    Robust and efficient camera-frame alignment before pose conversion.
    1) good init: centroid+scale + PCA rotation hypotheses
    2) robust ICP: trimmed correspondences + Huber weights + multistage sampling
    """
    tgt = _backproject_masked_depth(mask, depth, intrinsic)
    if len(tgt) < 200:
        return False, np.inf

    src = np.asarray(mesh.vertices, dtype=np.float32)
    if len(src) < 200:
        return False, np.inf

    # Multiscale setup for speed and robustness.
    diag = np.linalg.norm(np.percentile(tgt, 95, axis=0) - np.percentile(tgt, 5, axis=0))
    voxel_coarse = float(np.clip(diag / 60.0, 0.004, 0.02))
    voxel_fine = float(np.clip(diag / 120.0, 0.002, 0.01))

    tgt_coarse = _voxel_downsample(tgt, voxel_coarse)
    tgt_fine = _voxel_downsample(tgt, voxel_fine)
    src_coarse = _voxel_downsample(src, voxel_coarse)
    src_fine = _voxel_downsample(src, voxel_fine)
    if len(tgt_coarse) < 100 or len(src_coarse) < 100:
        return False, np.inf

    init_candidates = _make_camera_init_candidates(src_coarse, tgt_coarse)

    def run_icp(src_pts, tgt_pts, init_tf, iters, trim_percent):
        tree = cKDTree(tgt_pts)
        tf_total = init_tf.copy()
        best_rmse = np.inf
        for _ in range(iters):
            src_now = _apply_tf(src_pts, tf_total)
            dists, nn_idx = tree.query(src_now, k=1, workers=-1)
            if len(dists) < 50:
                break

            cut = np.percentile(dists, trim_percent)
            keep = dists <= cut
            if keep.sum() < 50:
                break

            src_k = src_now[keep]
            dst_k = tgt_pts[nn_idx[keep]]
            err = dists[keep]

            # Huber weights to reduce depth noise and residual outliers.
            delta = float(np.percentile(err, 60))
            delta = max(delta, 1e-4)
            w = np.where(err <= delta, 1.0, delta / (err + 1e-12)).astype(np.float32)

            delta_tf = _weighted_rigid_transform(src_k, dst_k, w)
            tf_total = delta_tf @ tf_total
            best_rmse = float(np.sqrt(np.mean(err ** 2)))
        return tf_total, best_rmse

    # Evaluate candidates quickly on coarse cloud.
    best = None
    for init_tf in init_candidates:
        tf_c, rmse_c = run_icp(src_coarse, tgt_coarse, init_tf, iters=8, trim_percent=75)
        if best is None or rmse_c < best[1]:
            best = (tf_c, rmse_c)

    tf_init = best[0]
    tf_f, rmse_f = run_icp(src_fine, tgt_fine, tf_init, iters=10, trim_percent=85)

    if not np.isfinite(rmse_f):
        return False, np.inf

    # Reject clearly wrong convergence to avoid catastrophic projection drift.
    if diag > 1e-8 and rmse_f > 0.35 * diag:
        return False, np.inf
    mesh.apply_transform(tf_f)
    return True, rmse_f


def preprocess_mesh_for_foundationpose(mesh, target_size, output_path):
    """
    Scale mesh to estimated real size in object-local frame.

    Keep scaling around object origin so gt_pose (ob_in_cam) remains consistent with
    eval_reconstruct projection logic.
    """
    current_size = mesh.extents.copy().astype(np.float32)
    current_size[current_size <= 1e-8] = 1e-8

    target_norm = float(np.linalg.norm(target_size))
    current_norm = float(np.linalg.norm(current_size))
    if current_norm <= 1e-8 or target_norm <= 1e-8:
        final_scale = 1.0
    else:
        final_scale = target_norm / current_norm

    # Preserve object center in object frame to avoid pose translation drift after scaling.
    centroid = mesh.bounding_box.centroid.copy().astype(np.float32)
    mesh.apply_translation(-centroid)
    mesh.apply_scale(final_scale)
    
    mesh.apply_translation(centroid)
    mesh.export(output_path)
    return final_scale


def _load_pose_txt_candidates(gt_root: str, frame_id: str):
    if not gt_root:
        return None

    candidates = [
        os.path.join(gt_root, f"{frame_id}.txt"),
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                pose = np.loadtxt(p, dtype=np.float32).reshape(4, 4)
                return pose
            except Exception as e:
                print(f"[WARN] Failed to parse pose txt: {p}, error: {e}")
                return None
    return None


def _load_world_to_camera_from_meta(gt_root: str, frame_id: str):
    if not gt_root:
        return None

    meta_path = os.path.join(gt_root, f"{frame_id}.json")
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if "world2camera_rotation" not in meta:
            return None

        c_world = np.array(
            meta.get("camera_pos", meta.get("camera2world_translation")),
            dtype=np.float32,
        ).reshape(3)
        r_c2w = np.array(meta["world2camera_rotation"], dtype=np.float32).reshape(3, 3)
        r_w2c = r_c2w.T
        t_w2c = -r_w2c @ c_world

        world_to_cam = np.eye(4, dtype=np.float32)
        world_to_cam[:3, :3] = r_w2c
        world_to_cam[:3, 3] = t_w2c
        return world_to_cam
    except Exception as e:
        print(f"[WARN] Failed to parse GT meta: {meta_path}, error: {e}")
        return None


def apply_extrinsic_pose_correction(mesh, gt_root: str, frame_id: str):
    """
    Align reconstructed mesh into object-local coordinates for FoundationPose.

    Eval logic expects gt_pose txt to be ob_in_cam. Therefore we strictly use:
      cam_to_obj = inv(ob_in_cam)

    Priority:
    1) <gt_root>/<frame_id>.txt (ob_in_cam)
    2) <gt_root>/<frame_id>.json (fallback world2camera, backward compatibility)
    """
    pose_raw = _load_pose_txt_candidates(gt_root, frame_id)
    if pose_raw is not None:
        cam_to_obj = np.linalg.inv(pose_raw)
        mesh.apply_transform(cam_to_obj)
        return True

    world_to_cam = _load_world_to_camera_from_meta(gt_root, frame_id)
    if world_to_cam is not None:
        cam_to_world = np.linalg.inv(world_to_cam)
        mesh.apply_transform(cam_to_world)
        return True

    return False


def p3d_to_opencv_camera(mesh):
    R_p2o = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # R_p2o = np.array([
    #     [1,  0,  0, 0],
    #     [0,  1,  0, 0],
    #     [0,  0,  1, 0],
    #     [0,  0,  0, 1]
    # ], dtype=np.float32)
    
    
    R_fix = np.array([
        [ 1, 0, 0, 0],
        [ 0, 0, -1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32)

    R_final = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    R = R_fix @ R_p2o    
    # mesh.apply_transform(R)
    mesh.apply_transform(R_final)



def align_mesh(root_dir: str, index: int, inference: Inference, save_dir: str = "model", gt_root: str = None):
    """Process one frame data."""
    os.makedirs(save_dir, exist_ok=True)
    mask_dir = os.path.join(root_dir, "gt_mask")
    rgb_dir = os.path.join(root_dir, "rgb")
    depth_dir = os.path.join(root_dir, "depth")

    mask_list = sorted(os.listdir(mask_dir), key=natural_sort_key)
    rgb_list = sorted(os.listdir(rgb_dir), key=natural_sort_key)
    depth_list = sorted(os.listdir(depth_dir), key=natural_sort_key)

    if index >= len(mask_list):
        return

    mask_path = os.path.join(mask_dir, mask_list[index])
    print(f"mask_path: {mask_path}")
    masks = [os.path.join(mask_path, m) for m in sorted(os.listdir(mask_path), key=natural_sort_key)]

    if not masks:
        print("No valid mask exists!")
        return

    rgb_path = os.path.join(rgb_dir, rgb_list[index])
    depth_path = os.path.join(depth_dir, depth_list[index])
    intrinsic_path = os.path.join(root_dir, "K.txt")

    base_name = Path(mask_path).stem
    # frame_dir = os.path.join(save_dir, base_name)
    # os.makedirs(frame_dir, exist_ok=True)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if not os.path.exists(intrinsic_path):
        print(f"[SKIP] Intrinsic file not found: {intrinsic_path}")
        return
    intrinsic = np.loadtxt(intrinsic_path)

    # Prefer local gt_pose; fallback to user-specified gt_root.
    local_pose_root = os.path.join(root_dir, "gt_pose")
    pose_roots = []
    if os.path.isdir(local_pose_root):
        pose_roots.append(local_pose_root)
    if gt_root:
        pose_roots.append(gt_root)

    for idx, mfile in enumerate(masks):
        mask = cv2.imread(mfile, cv2.IMREAD_GRAYSCALE)
        model_path = os.path.join(save_dir, f"model_{idx:04d}")
        save_obj_file = os.path.join(model_path, "model.obj")
        if os.path.exists(save_obj_file):
            print(f"[SKIP] {base_name} {idx}th model has been saved")
            continue

        print(f"  -> Processing component {idx}: {Path(mfile).name}")
        if np.sum(mask > 0) < 10:
            print("invalid mask")
            continue

        mesh = build_mesh(
            rgb_path,
            mask_path,
            inference,
            idx,
            depth=depth,
            intrinsic=intrinsic,
            object_mask=mask,
        )
        if mesh is None:
            continue
        
        # Prefer metric pointmap alignment from SAM3D itself; run ICP only when needed.
        used_pointmap = bool(getattr(build_mesh, "last_used_pointmap", False))
        if used_pointmap and (not RUN_ICP_WHEN_POINTMAP_USED):
            cam_aligned, rmse = True, 0.0
            print("    [ALIGN] skip ICP (pointmap-guided reconstruction)")
        else:
            cam_aligned, rmse = align_mesh_to_camera_frame(mesh, mask, depth, intrinsic)
            if cam_aligned:
                print(f"    [ALIGN] camera-frame ICP rmse={rmse:.5f}m")
            else:
                print("    [ALIGN] camera-frame ICP skipped/failed")

        if used_pointmap:
            p3d_to_opencv_camera(mesh) 
            print("[FIX] Converted P3D mesh to OpenCV frame")
        
        aligned = False
        for pose_root in pose_roots:
            if apply_extrinsic_pose_correction(mesh, pose_root, base_name):
                aligned = True
                break
        if not aligned:
            print(f"[WARN] Pose file missing for frame: {base_name}, keep original mesh frame")

        real_size = load_real_world_size(mask, depth, intrinsic)
        os.makedirs(model_path, exist_ok=True)
        preprocess_mesh_for_foundationpose(mesh, real_size, save_obj_file)


def inference_obj(obj_dir, filenum, save_path, inference, gt_root=None):
    for index in range(filenum):
        try:
            align_mesh(obj_dir, index, inference, save_path, gt_root=gt_root)
        except RuntimeError as e:
            if "Cuda error: 2" in str(e) or "out of memory" in str(e).lower():
                print(f"Warning: object {index} OOM, skipped. Error: {e}")
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                continue
            else:
                print(f"error: {e}")
                raise e
