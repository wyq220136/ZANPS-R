import os
import re
import sys
import gc
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch3d.transforms import Transform3d
from scipy.spatial import cKDTree
import trimesh
# --------------------------
# Global configuration
# --------------------------
SAM3D_PROJECT_ROOT = "/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/sam-3d-objects"
SAM3D_NOTEBOOK_ROOT = "/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/sam-3d-objects/notebook"
CONFIG_PATH = "/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/sam-3d-objects/checkpoints/hf/pipeline.yaml"


USE_UMEYAMA_ALIGNMENT = True
UMEYAMA_MAX_POINTS = 2000
MESH_ALIGNMENT_SAMPLE_POINTS = 50000
RANDOM_SEED = 42
MIN_VALID_PIXELS = 30

# SAM3D's GLB/OBJ export rotates the internal reconstruction mesh:
# internal [x, y, z] -> exported [x, z, -y].
# For alignment against SAM3D's depth/pointmap local frame, undo that export rotation.
SAM3D_INTERNAL_TO_EXPORTED_MESH = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# --------------------------
# SAM3D inference setup
# --------------------------
if SAM3D_PROJECT_ROOT not in sys.path:
    sys.path.append(SAM3D_PROJECT_ROOT)
if SAM3D_NOTEBOOK_ROOT not in sys.path:
    sys.path.append(SAM3D_NOTEBOOK_ROOT)

from inference import Inference, load_image, load_single_mask
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera

from reconstruct import (
    align_mesh_to_camera_frame,
    apply_extrinsic_pose_correction,
    load_real_world_size,
    natural_sort_key,
    preprocess_mesh_for_foundationpose,
)


_INFERENCE = None


def get_inference():
    global _INFERENCE
    if _INFERENCE is None:
        _INFERENCE = Inference(CONFIG_PATH, compile=False, use_depth_model=False)
    return _INFERENCE

# /root/.cache/torch/hub/facebookresearch_dinov2_main
def build_raw_points(
    image_path,
    depth,
    mask_dir,
    mask_index,
    intrinsic,
    inference: Inference,
    object_mask=None,
    mask_path: str | None = None,
):
    image = load_image(image_path)
    # Prefer explicit mask file path (dataset-dependent naming like mask_0001.png),
    # then fallback to legacy load_single_mask(mask_dir, index).
    if mask_path and os.path.exists(mask_path):
        mask = load_mask_image(mask_path).astype(np.uint8)
    else:
        try:
            mask = load_single_mask(mask_dir, index=mask_index)
        except Exception:
            # Last-resort fallback for non-legacy mask naming under mask_dir.
            mask_files = _list_mask_files(mask_dir)
            if 0 <= int(mask_index) < len(mask_files):
                mask = load_mask_image(mask_files[int(mask_index)]).astype(np.uint8)
            else:
                raise
    # Normalize mask format and ensure shape is compatible with depth/image pipeline.
    if mask is None:
        raise ValueError("Loaded mask is None.")
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Keep mask as 0/1. Inference.merge_mask_to_rgba() multiplies by 255 internally.
    # Passing 0/255 here would overflow uint8 when multiplied again (255*255 -> 1).
    mask = (mask > 0).astype(np.uint8)

    # Use GT part mask consistently for both inference and pointmap validity.
    # Do not mix external object_mask into this reconstruction stage.
    gt_mask = mask.copy()

    if depth is not None and hasattr(depth, "shape") and len(depth.shape) >= 2:
        h, w = int(depth.shape[0]), int(depth.shape[1])
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if gt_mask.shape[:2] != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    fg = int(np.count_nonzero(mask))
    if fg < 10:
        raise ValueError(f"Mask too small/empty before inference (fg={fg}).")

    pointmap = None
    try:
        pointmap, valid_count = depth_to_pointmap(gt_mask, depth, intrinsic)
        if valid_count < MIN_VALID_PIXELS:
            raise ValueError(f"too few valid depth pixels for real-depth pointmap: {valid_count}")
        used_pointmap = True
    except Exception as e:
        raise RuntimeError(f"Failed to build pointmap from real depth: {e}") from e

    output = None
    err_msgs = []

    try:
        output = inference(image, mask, seed=42, pointmap=pointmap)
        used_pointmap = True
    except Exception as e:
        err_msgs.append(str(e))
        output = None

    if output is None:
        raise RuntimeError(
            "SAM3D inference failed with real-depth pointmap. "
            f"frame_mask={mask_path if mask_path else os.path.join(mask_dir, str(mask_index))}, "
            f"fg={int(np.count_nonzero(mask))}, errs={err_msgs[:2]}"
        )

    gs = output["gs"]
    output_mesh = inference._pipeline.postprocess_slat_output(
        output,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
    )
    mesh = output_mesh["glb"]
    build_raw_points.last_used_pointmap = used_pointmap
    return gs, mesh


def depth_to_pointmap(mask, depth, intrinsic):
    if depth.max() > 50:
        depth = depth / 1000.0
    depth = depth.astype(np.float32)

    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    z = depth.astype(float)
    x = (xs - cx) * z / max(fx, 1e-8)
    y = (ys - cy) * z / max(fy, 1e-8)
    pointmap_cam = np.stack([x, y, z], axis=-1).astype(np.float32)

    pts = torch.from_numpy(pointmap_cam.reshape(-1, 3))
    cam_to_p3d = Transform3d().rotate(camera_to_pytorch3d_camera(device="cpu").rotation).to("cpu")
    pointmap = cam_to_p3d.transform_points(pts).reshape(h, w, 3).cpu().numpy().astype(np.float32)

    if mask is not None:
        valid = (z > 1e-8) & (mask > 0)
    else:
        valid = z > 1e-8
    valid_count = int(np.count_nonzero(valid))

    pointmap[~valid] = np.nan
    return torch.from_numpy(pointmap), valid_count


def load_intrinsic(path):
    k = np.loadtxt(path, dtype=np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    return k


def load_mask_image(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bin = (mask > 0).astype(np.uint8) * 255
    # Some indexed-color PNG masks may appear empty after OpenCV conversion;
    # fallback to PIL raw indices to preserve label ids.
    if int(np.count_nonzero(mask_bin)) == 0:
        try:
            pil_arr = np.array(Image.open(path))
            if pil_arr.ndim == 3:
                pil_arr = pil_arr[..., 0]
            mask_bin = (pil_arr > 0).astype(np.uint8) * 255
        except Exception:
            pass
    return mask_bin


def gaussian_xyz_to_numpy(gaussian):
    xyz = gaussian.get_xyz.detach().cpu().numpy()
    if not np.isfinite(xyz).all():
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz


def mesh_exported_to_sam3d_internal(mesh):
    """Undo SAM3D GLB/OBJ export axis rotation so mesh matches SAM3D local points."""
    mesh_local = mesh.copy()
    verts = np.asarray(mesh_local.vertices, dtype=np.float32)
    mesh_local.vertices = verts @ SAM3D_INTERNAL_TO_EXPORTED_MESH.T
    return mesh_local


def sample_mesh_surface_points(mesh, max_points=MESH_ALIGNMENT_SAMPLE_POINTS, rng=None):
    max_points = max(1, int(max_points))
    try:
        count = min(max_points, max(5000, int(len(mesh.faces) * 4)))
        points, _ = trimesh.sample.sample_surface(mesh, count)
        points = np.asarray(points, dtype=np.float32)
    except Exception:
        points = np.asarray(mesh.vertices, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] > max_points:
        if rng is None:
            rng = np.random.default_rng(RANDOM_SEED)
        points = points[rng.choice(points.shape[0], size=max_points, replace=False)]
    return points.astype(np.float32)


def apply_transform_to_gaussian(gaussian, transform):
    xyz = gaussian.get_xyz
    tf_t = torch.tensor(transform, device=xyz.device, dtype=xyz.dtype)
    r = tf_t[:3, :3]
    t = tf_t[:3, 3]
    gaussian.from_xyz(xyz @ r.T + t)
    return gaussian


def sample_points(points, max_points, rng):
    """
    浣跨敤鏈€杩滅偣閲囨牱 (FPS) 鏇夸唬闅忔満閲囨牱銆?    杈撳叆杈撳嚭淇濇寔涓€鑷淬€俽ng 浠呯敤浜庣‘瀹氱涓€涓捣濮嬬偣銆?    """
    num_points = points.shape[0]
    
    if num_points <= max_points:
        return points
    centroids = np.zeros(max_points, dtype=np.int32)
    distance = np.ones(num_points) * 1e10
    
    farthest = rng.choice(num_points)
    
    for i in range(max_points):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return points[centroids]


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


def estimate_umeyama_from_pointclouds(src_points, dst_points, max_points, rng):
    src = sample_points(src_points, max_points, rng)
    dst = sample_points(dst_points, max_points, rng)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("Not enough points for Umeyama alignment.")

    diff = src[:, None, :] - dst[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    nn_idx = np.argmin(d2, axis=1)
    dst_nn = dst[nn_idx]
    return umeyama_alignment(src, dst_nn, with_scale=False)


def _estimate_similarity_from_pointclouds(src_points, dst_points, max_points, rng, with_scale):
    src = sample_points(src_points, max_points, rng)
    dst = sample_points(dst_points, max_points, rng)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("Not enough points for alignment.")

    tree = cKDTree(dst)
    src_ctr = src.mean(axis=0)
    dst_ctr = dst.mean(axis=0)

    def scale_init():
        if not with_scale:
            return 1.0
        src_span = np.percentile(src, 95, axis=0) - np.percentile(src, 5, axis=0)
        dst_span = np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0)
        src_delta = np.linalg.norm(src-src_ctr, axis=1)
        dst_delta = np.linalg.norm(dst-dst_ctr, axis=1)
        src_std = np.mean(src_delta)
        dst_std = np.mean(dst_delta)
        src_norm = float(np.linalg.norm(src_span))
        dst_norm = float(np.linalg.norm(dst_span))
        s_std = 1.0 if src_std < 1e-8 else (dst_std / src_std)
        s_span = 1.0 if src_norm < 1e-8 else (dst_norm / src_norm)
        # Median of two estimators is more stable on tiny parts.
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
        dists, _ = tree.query(pts, k=1, workers=-1)
        if dists.shape[0] < 20:
            return np.inf
        cut = np.percentile(dists, q)
        keep = dists <= cut
        if int(np.sum(keep)) < 20:
            return np.inf
        return float(np.sqrt(np.mean((dists[keep]) ** 2)))

    # Multi-rotation initialization (like compare.py) + center+scale prior.
    base_scale = scale_init()
    # Adaptive bounds: avoid over-expansion on tiny parts while keeping flexibility.
    if with_scale:
        scale_min = max(0.02, base_scale * 0.35)
        scale_max = min(5.0, base_scale * 2.5)
    else:
        scale_min, scale_max = 1.0, 1.0
    angle_set = [0.0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi, 5 * np.pi/4, 3 * np.pi/2, 7 * np.pi/4]
    init_candidates = [make_tf(np.eye(3, dtype=np.float32), base_scale)]
    for rx in angle_set:
        for ry in angle_set:
            for rz in angle_set:
                r0 = euler_to_rot(rx, ry, rz)
                init_candidates.append(make_tf(r0, base_scale))

    # Keep top-K initializations to avoid bad local minima.
    scored = []
    for tf0 in init_candidates:
        scored.append((eval_trimmed_rmse(tf0, q=70), tf0))
    scored.sort(key=lambda x: x[0])
    seeds = [x[1] for x in scored[:8]]

    best_tf = seeds[0].copy()
    best_err = np.inf
    trim_q_schedule = [70, 75, 80, 85, 88, 90]
    for seed in seeds:
        tf = seed.copy()
        for q in trim_q_schedule:
            src_now = _apply_transform_np(src, tf)
            dists, nn_idx = tree.query(src_now, k=1, workers=-1)
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
                s_damped = 1.0 + (s_step - 1.0) * 0.2 # 闃诲凹绯绘暟
                U, _, Vt = np.linalg.svd(delta[:3, :3])
                delta[:3, :3] = (U @ Vt) * s_damped
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
            dists_f, nn_idx_f = tree.query(src_final, k=1)
            core_cut = np.percentile(dists_f, 70)
            core_mask = dists_f <= core_cut
            if np.sum(core_mask) > 20:
                src_core = src[core_mask]
                dst_core = dst[nn_idx_f[core_mask]]
                s_src = np.std(np.linalg.norm(src_core - src_core.mean(axis=0), axis=1))
                s_dst = np.std(np.linalg.norm(dst_core - dst_core.mean(axis=0), axis=1))
                refined_s = s_dst / (s_src + 1e-8)
                refined_s = np.clip(refined_s, scale_min, scale_max)
                U, _, Vt = np.linalg.svd(tf[:3, :3])
                refined_r = U @ Vt
                tf[:3, :3] = (refined_r * refined_s).astype(np.float32)
                # Keep transform self-consistent after scale update.
                src_mean = src_core.mean(axis=0)
                dst_mean = dst_core.mean(axis=0)
                tf[:3, 3] = (dst_mean - refined_s * (refined_r @ src_mean)).astype(np.float32)
        
        err = eval_trimmed_rmse(tf, q=90)
        if err < best_err:
            best_err = err
            best_tf = tf.copy()

    return best_tf


def _extract_part_id(mask_path, fallback_idx):
    # Keep part indexing consistent with reconstruction/reconstruct.py:
    # model folders follow enumeration order model_0000, model_0001, ...
    return int(fallback_idx)


def _backproject_masked_depth(mask, depth, intrinsic):
    if depth.max() > 50:
        depth = depth / 1000.0
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


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros((3,), dtype=np.float32)
    return (v / n).astype(np.float32)


def _connector_vector_from_points(points: np.ndarray, global_center: np.ndarray, near_ratio: float = 0.15) -> np.ndarray:
    if points is None or points.ndim != 2 or points.shape[0] < 20:
        return np.zeros((3,), dtype=np.float32)
    center = points.mean(axis=0)
    d = np.linalg.norm(points - global_center.reshape(1, 3), axis=1)
    k = int(max(20, min(points.shape[0], round(points.shape[0] * float(np.clip(near_ratio, 0.02, 0.5))))))
    idx = np.argpartition(d, k - 1)[:k]
    near_pts = points[idx]
    anchor = near_pts.mean(axis=0)
    return _safe_normalize(anchor - center)


def _load_ref_points_map(reference_model_dir: str) -> dict:
    ref_map = {}
    if not os.path.isdir(reference_model_dir):
        return ref_map
    model_dirs = sorted([d for d in os.listdir(reference_model_dir) if d.startswith("model_")], key=natural_sort_key)
    for d in model_dirs:
        try:
            pid = int(d.split("_")[-1])
        except Exception:
            continue
        p = os.path.join(reference_model_dir, d, "reference_points.npy")
        if not os.path.exists(p):
            continue
        try:
            pts = np.load(p).astype(np.float32)
        except Exception:
            continue
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] >= 20:
            ref_map[pid] = pts
    return ref_map


def _build_connector_vectors_from_part_points(part_points: dict, near_ratio: float = 0.15) -> dict:
    if not part_points:
        return {}
    all_pts = [v for v in part_points.values() if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3 and v.shape[0] > 0]
    if not all_pts:
        return {}
    global_center = np.concatenate(all_pts, axis=0).mean(axis=0)
    out = {}
    for pid, pts in part_points.items():
        out[int(pid)] = _connector_vector_from_points(pts, global_center, near_ratio=near_ratio)
    return out


def _angle_deg_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    aa = _safe_normalize(a)
    bb = _safe_normalize(b)
    if float(np.linalg.norm(aa)) < 1e-8 or float(np.linalg.norm(bb)) < 1e-8:
        return 0.0
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _edge_gate_accept(tf_ref2cur: np.ndarray, ref_vec_obj: np.ndarray, query_vec_cam: np.ndarray, max_angle_deg: float) -> tuple[bool, float]:
    if tf_ref2cur is None:
        return True, 0.0
    r = tf_ref2cur[:3, :3].astype(np.float32)
    ref_vec_cam = _safe_normalize(r @ ref_vec_obj.reshape(3))
    query_vec_cam = _safe_normalize(query_vec_cam.reshape(3))
    if float(np.linalg.norm(ref_vec_cam)) < 1e-8 or float(np.linalg.norm(query_vec_cam)) < 1e-8:
        return True, 0.0
    angle = _angle_deg_between_vectors(ref_vec_cam, query_vec_cam)
    return bool(angle < float(max_angle_deg)), angle


def _preprocess_depth_for_alignment(depth):
    """
    Lightweight depth denoise for robust point-cloud alignment.
    Keep behavior conservative to avoid changing pipeline semantics.
    """
    if depth is None:
        return depth
    d = depth.astype(np.float32).copy()
    if d.max() > 50:
        d = d / 1000.0
    # Remove clearly invalid values then denoise with small median kernel.
    d[(~np.isfinite(d)) | (d < 0.1) | (d > 10.0)] = 0.0
    d = cv2.medianBlur(d, 3)
    return d

def _visibility_mask_in_camera(points_cam, mask, depth, intrinsic):
    """
    Determine whether transformed points are visible in current frame.
    Visible = in image + inside object mask + not obviously behind observed depth.
    """
    if points_cam.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    h, w = depth.shape[:2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    valid_z = z > 1e-6
    u = (x * fx / np.maximum(z, 1e-8) + cx).astype(np.int32)
    v = (y * fy / np.maximum(z, 1e-8) + cy).astype(np.int32)
    inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return np.zeros_like(valid_z, dtype=bool)

    uu = u[inside]
    vv = v[inside]
    depth_obs = depth[vv, uu]
    mask_obs = mask[vv, uu] > 0
    has_depth = depth_obs > 1e-6
    z_in = z[inside]

    # Allow small positive slack; occluded points (far behind) are filtered out.
    tol = np.maximum(0.01, 0.02 * depth_obs)
    depth_ok = (z_in <= (depth_obs + tol)) & (z_in >= (depth_obs - 0.08))

    vis_inside = mask_obs & has_depth & depth_ok
    vis = np.zeros_like(valid_z, dtype=bool)
    vis[np.where(inside)[0]] = vis_inside
    return vis


def _filter_src_points_by_depth_consistency(
    src_points_local,
    coarse_tf_obj,
    ob_in_cam,
    mask,
    depth,
    intrinsic,
    tau_abs=0.06,
    tau_rel=0.08,
):
    """
    Conservative pre-filter for SAM3D points.
    Keep points whose coarse-projected depth is close to observed depth.
    Uses a relatively loose threshold to avoid over-pruning.
    """
    if src_points_local is None or src_points_local.shape[0] < 50:
        return src_points_local, np.ones((0,), dtype=bool)
    if coarse_tf_obj is None or ob_in_cam is None:
        return src_points_local, np.ones((src_points_local.shape[0],), dtype=bool)

    src_obj = _apply_transform_np(src_points_local, coarse_tf_obj)
    src_cam = _apply_transform_np(src_obj, ob_in_cam)
    h, w = depth.shape[:2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    z = src_cam[:, 2]
    valid_z = z > 1e-6
    u = (src_cam[:, 0] * fx / np.maximum(z, 1e-8) + cx).astype(np.int32)
    v = (src_cam[:, 1] * fy / np.maximum(z, 1e-8) + cy).astype(np.int32)
    inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    keep = np.zeros((src_points_local.shape[0],), dtype=bool)
    if not np.any(inside):
        return src_points_local, np.ones((src_points_local.shape[0],), dtype=bool)

    uu = u[inside]
    vv = v[inside]
    z_pred = z[inside]
    z_obs = depth[vv, uu]
    in_mask = mask[vv, uu] > 0
    has_depth = z_obs > 1e-6
    # Loose threshold: absolute + relative term.
    tau = np.maximum(float(tau_abs), float(tau_rel) * np.maximum(z_obs, 1e-3))
    depth_ok = np.abs(z_pred - z_obs) <= tau
    keep_inside = in_mask & has_depth & depth_ok
    keep[np.where(inside)[0]] = keep_inside

    # Adaptive fallback to avoid overly strict filtering.
    keep_count = int(np.sum(keep))
    min_keep = max(80, int(0.25 * src_points_local.shape[0]))
    if keep_count < min_keep:
        tau2 = np.maximum(float(tau_abs) * 1.8, float(tau_rel) * 1.8 * np.maximum(z_obs, 1e-3))
        keep_inside2 = in_mask & has_depth & (np.abs(z_pred - z_obs) <= tau2)
        keep = np.zeros((src_points_local.shape[0],), dtype=bool)
        keep[np.where(inside)[0]] = keep_inside2
        keep_count = int(np.sum(keep))

    if keep_count < max(50, int(0.12 * src_points_local.shape[0])):
        # Final fallback: keep all points if depth filtering becomes too aggressive.
        keep = np.ones((src_points_local.shape[0],), dtype=bool)

    return src_points_local[keep], keep


def _compute_alignment_score(src_points, dst_tree, tf, visible_mask=None):
    pts_t = _apply_transform_np(src_points, tf)
    if visible_mask is not None and visible_mask.shape[0] == pts_t.shape[0]:
        pts_eval = pts_t[visible_mask]
        invis_ratio = 1.0 - float(np.mean(visible_mask.astype(np.float32)))
    else:
        pts_eval = pts_t
        invis_ratio = 0.0
    if pts_eval.shape[0] < 20:
        return np.inf
    dists, _ = dst_tree.query(pts_eval, k=1, workers=-1)
    cut = np.percentile(dists, 85)
    keep = dists <= cut
    if int(np.sum(keep)) < 20:
        return np.inf
    rmse = float(np.sqrt(np.mean((dists[keep]) ** 2)))
    return rmse + 0.25 * invis_ratio


def _iterative_visibility_alignment(
    src_points,
    dst_points,
    max_points,
    rng,
    with_scale,
    visibility_fn=None,
    max_iter=4,
):
    """
    Iterative alignment with residual-voting outlier rejection.
    Notes:
    - All points participate in residual voting each iteration.
    - Points flagged as outliers are excluded from the next fit step.
    - After iterations, points with the highest vote count are removed once,
      then a final precise alignment is solved.
    - `visibility_fn` is kept only for API compatibility and is intentionally
      not used for hard visibility filtering.
    """
    src = sample_points(src_points, max_points, rng)
    dst = sample_points(dst_points, max_points, rng)
    if src.shape[0] < 20 or dst.shape[0] < 20:
        raise ValueError("Not enough points for iterative alignment.")

    dst_tree = cKDTree(dst)
    tf = _estimate_similarity_from_pointclouds(src, dst, max_points=max_points, rng=rng, with_scale=with_scale)
    n = src.shape[0]
    votes = np.zeros((n,), dtype=np.int32)
    excluded_for_fit = np.zeros((n,), dtype=bool)

    for _ in range(max_iter):
        src_now = _apply_transform_np(src, tf)
        dists, _ = dst_tree.query(src_now, k=1, workers=-1)
        if dists.shape[0] < 20:
            break

        # Robust residual thresholding to detect structurally inconsistent points.
        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med))) + 1e-8
        robust_sigma = 1.4826 * mad
        th = med + 2.5 * robust_sigma
        outlier_round = dists > th

        # Ensure each iteration contributes enough voting signal.
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
            # Fallback to current-round inliers only.
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

    # Vote-based final pruning: remove points with the highest vote count.
    max_vote = int(np.max(votes)) if votes.size > 0 else 0
    final_keep = np.ones((n,), dtype=bool)
    if max_vote > 0:
        final_keep = votes < max_vote
        if int(np.sum(final_keep)) < 20:
            # Keep most reliable points if max-vote removal is too aggressive.
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


def _load_ob_part(frame_id, part_idx, pose_roots):
    fallback_reason = None
    for pose_root in pose_roots:
        if not pose_root:
            continue
        part_list_path = os.path.join(pose_root, f"{frame_id}__parts.txt")
        if not os.path.exists(part_list_path):
            continue

        # Read per-frame part pose file list (one txt filename per line).
        with open(part_list_path, "r", encoding="utf-8") as f:
            part_list = [line.strip() for line in f.readlines() if line.strip()]

        if not part_list:
            fallback_reason = f"empty parts list: {part_list_path}"
            continue
        if int(part_idx) < 0 or int(part_idx) >= len(part_list):
            fallback_reason = (
                f"part index out of range for {part_list_path}: "
                f"part_idx={part_idx}, num_parts={len(part_list)}"
            )
            continue

        ob_in_cam_name = part_list[int(part_idx)]
        pose_path = os.path.join(pose_root, ob_in_cam_name)
        print(f"pose_path: {pose_path}")
        if not os.path.exists(pose_path):
            fallback_reason = f"pose file missing: {pose_path}"
            continue
        try:
            pose = np.loadtxt(pose_path, dtype=np.float32)
            if pose.shape == (16,):
                pose = pose.reshape(4, 4)
            if pose.shape == (4, 4):
                return pose.astype(np.float32)
            fallback_reason = f"invalid pose shape: {pose_path}, shape={pose.shape}"
        except Exception as e:
            fallback_reason = f"failed to read pose: {pose_path}, err={e}"
            continue

    if fallback_reason is not None:
        print(
            f"[WARN] Missing/unmatched gt_pose for frame={frame_id}, part_idx={part_idx}. "
            f"Fallback to identity pose. reason={fallback_reason}"
        )
        return np.eye(4, dtype=np.float32)
    return None


def _load_ob_in_cam(frame_id, pose_roots):
    for pose_root in pose_roots:
        if not pose_root:
            continue
        pose_path = os.path.join(pose_root, f"{frame_id}.txt")
        if not os.path.exists(pose_path):
            continue
        try:
            pose = np.loadtxt(pose_path, dtype=np.float32)
            if pose.shape == (16,):
                pose = pose.reshape(4, 4)
            if pose.shape == (4, 4):
                return pose.astype(np.float32)
        except Exception as e:
            print(f"[WARN] Failed to read pose: {pose_path}, err={e}")
    return None


def _apply_transform_np(points, tf):
    if points.shape[0] == 0:
        return points
    r = tf[:3, :3]
    t = tf[:3, 3]
    return (points @ r.T + t.reshape(1, 3)).astype(np.float32)


def _project_points_cam_to_image(points_cam, intrinsic, h, w):
    if points_cam.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)
    z = points_cam[:, 2]
    valid = z > 1e-6
    pts = points_cam[valid]
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=bool)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = (pts[:, 0] * fx / pts[:, 2] + cx).astype(np.int32)
    v = (pts[:, 1] * fy / pts[:, 2] + cy).astype(np.int32)
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = np.stack([u[in_img], v[in_img]], axis=1) if np.any(in_img) else np.zeros((0, 2), dtype=np.int32)
    return uv, valid


def _save_alignment_vis(rgb_bgr, intrinsic, ob_in_cam, points_aligned_obj, points_tgt_obj, save_path):
    vis = rgb_bgr.copy()
    h, w = vis.shape[:2]

    def obj_to_cam(points_obj):
        return _apply_transform_np(points_obj, ob_in_cam)

    aligned_cam = obj_to_cam(points_aligned_obj) if points_aligned_obj.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
    tgt_cam = obj_to_cam(points_tgt_obj) if points_tgt_obj.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

    uv_tgt, _ = _project_points_cam_to_image(tgt_cam, intrinsic, h, w)
    uv_aligned, _ = _project_points_cam_to_image(aligned_cam, intrinsic, h, w)

    if uv_tgt.shape[0] > 5000:
        ids = np.random.choice(uv_tgt.shape[0], 5000, replace=False)
        uv_tgt = uv_tgt[ids]
    if uv_aligned.shape[0] > 5000:
        ids = np.random.choice(uv_aligned.shape[0], 5000, replace=False)
        uv_aligned = uv_aligned[ids]

    for u, v in uv_tgt:
        cv2.circle(vis, (int(u), int(v)), 1, (0, 0, 255), -1)     # red: target
    for u, v in uv_aligned:
        cv2.circle(vis, (int(u), int(v)), 1, (0, 255, 0), -1)     # green: aligned

    cv2.putText(vis, "red=target depth, green=aligned recon", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2)
    cv2.imwrite(save_path, vis)


def _save_alignment_vis_cam(rgb_bgr, intrinsic, points_aligned_cam, points_tgt_cam, save_path):
    vis = rgb_bgr.copy()
    h, w = vis.shape[:2]
    uv_tgt, _ = _project_points_cam_to_image(points_tgt_cam, intrinsic, h, w)
    uv_aligned, _ = _project_points_cam_to_image(points_aligned_cam, intrinsic, h, w)

    if uv_tgt.shape[0] > 5000:
        ids = np.random.choice(uv_tgt.shape[0], 5000, replace=False)
        uv_tgt = uv_tgt[ids]
    if uv_aligned.shape[0] > 5000:
        ids = np.random.choice(uv_aligned.shape[0], 5000, replace=False)
        uv_aligned = uv_aligned[ids]

    for u, v in uv_tgt:
        cv2.circle(vis, (int(u), int(v)), 1, (0, 0, 255), -1)  # red: current depth cloud
    for u, v in uv_aligned:
        cv2.circle(vis, (int(u), int(v)), 1, (0, 255, 0), -1)  # green: coarse aligned ref cloud

    cv2.putText(vis, "red=current depth, green=coarse aligned ref", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2)
    cv2.imwrite(save_path, vis)


def _resolve_frame_file(frame_dir, frame_id):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(frame_dir, f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    return ""


def _list_mask_files(frame_mask_dir):
    if not os.path.isdir(frame_mask_dir):
        return []
    # Follow dataset frame-dir semantics: accept all image files as mask candidates,
    # then sort naturally for stable part indexing.
    files = []
    for name in sorted(os.listdir(frame_mask_dir), key=natural_sort_key):
        p = os.path.join(frame_mask_dir, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg"}:
            files.append(p)
    return files


def _frame_context(mask_dir, index):
    frame_dirs = []
    if os.path.isdir(mask_dir):
        frame_dirs = [
            d for d in sorted(os.listdir(mask_dir), key=natural_sort_key)
            if os.path.isdir(os.path.join(mask_dir, d))
        ]

    if frame_dirs:
        if index < 0 or index >= len(frame_dirs):
            return None
        frame_id = frame_dirs[index]
        frame_mask_dir = os.path.join(mask_dir, frame_id)
        return {
            "frame_id": frame_id,
            "frame_mask_dir": frame_mask_dir,
            "frame_dirs": frame_dirs,
            "frame_index": index,
            "mask_root_mode": True,
        }

    return {
        "frame_id": os.path.basename(os.path.normpath(mask_dir)),
        "frame_mask_dir": mask_dir,
        "frame_dirs": [],
        "frame_index": 0,
        "mask_root_mode": False,
    }


def raw_pose_estimation(intrinsic_path, rgb_path: str, index: int, depth_path: str, mask_dir: str, inference:Inference, save_dir: str = "model", gt_root=None, flat_output: bool = True):
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(intrinsic_path):
        print(f"[SKIP] Intrinsic file not found: {intrinsic_path}")
        return None
    if not os.path.exists(rgb_path):
        print(f"[SKIP] RGB file not found: {rgb_path}")
        return None
    if not os.path.exists(depth_path):
        print(f"[SKIP] Depth file not found: {depth_path}")
        return None

    ctx = _frame_context(mask_dir, index)
    if ctx is None:
        print(f"[SKIP] Invalid frame index={index} for mask_dir={mask_dir}")
        return None

    masks = _list_mask_files(ctx["frame_mask_dir"])
    if not masks:
        print(f"[SKIP] No masks under {ctx['frame_mask_dir']}")
        return None

    
    intrinsic = load_intrinsic(intrinsic_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"[SKIP] Failed to read depth: {depth_path}")
        return None
    depth = depth.astype(np.float32)
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        print(f"[SKIP] Failed to read rgb: {rgb_path}")
        return None

    frame_dir = save_dir if flat_output else os.path.join(save_dir, ctx["frame_id"])
    os.makedirs(frame_dir, exist_ok=True)

    root_dir = os.path.dirname(intrinsic_path)
    pose_roots = []
    if gt_root:
        pose_roots.append(gt_root)

    ref_points_cache = {}
    if USE_UMEYAMA_ALIGNMENT and ctx["mask_root_mode"] and ctx["frame_index"] > 0:
        ref_frame_id = ctx["frame_dirs"][0]
        ref_mask_dir = os.path.join(mask_dir, ref_frame_id)
        ref_masks = _list_mask_files(ref_mask_dir)
        ref_rgb = _resolve_frame_file(os.path.join(root_dir, "rgb"), ref_frame_id)
        ref_depth_path = _resolve_frame_file(os.path.join(root_dir, "depth"), ref_frame_id)
        if ref_rgb and ref_depth_path and ref_masks:
            ref_depth = cv2.imread(ref_depth_path, cv2.IMREAD_UNCHANGED)
            if ref_depth is not None:
                ref_depth = ref_depth.astype(np.float32)
                max_ref_parts = min(len(masks), len(ref_masks))
                for part_idx in range(max_ref_parts):
                    try:
                        ref_mask = load_mask_image(ref_masks[part_idx])
                        if np.count_nonzero((ref_mask > 0) & (ref_depth > 0)) < MIN_VALID_PIXELS:
                            continue
                        _, ref_mesh = build_raw_points(
                            ref_rgb,
                            ref_depth,
                            ref_mask_dir,
                            part_idx,
                            intrinsic,
                            inference,
                            ref_mask,
                            mask_path=ref_masks[part_idx],
                        )
                        # 閲嶅缓鐐逛簯骞惰浆鍖栦负numpy
                        ref_mesh = mesh_exported_to_sam3d_internal(ref_mesh)
                        ref_points_cache[part_idx] = sample_mesh_surface_points(
                            ref_mesh,
                            MESH_ALIGNMENT_SAMPLE_POINTS,
                            np.random.default_rng(RANDOM_SEED),
                        )
                    except Exception as e:
                        print(f"[WARN] Failed to build reference for part {part_idx}: {e}")

    rng = np.random.default_rng(RANDOM_SEED + max(ctx["frame_index"], 0))

    for part_idx, mask_path in enumerate(masks):
        part_id = _extract_part_id(mask_path, part_idx)
        model_path = os.path.join(frame_dir, f"model_{part_id:04d}")
        save_obj_file = os.path.join(model_path, "model.obj")
        ref_points_file = os.path.join(model_path, "reference_points.npy")
        align_vis_file = os.path.join(model_path, "align_vis.png")
        if os.path.exists(save_obj_file) and os.path.exists(ref_points_file) and os.path.exists(align_vis_file):
            print(f"[REBUILD] {ctx['frame_id']} part={part_id} existing model/reference will be overwritten")

        object_mask = load_mask_image(mask_path)
        # print(np.unique(depth))
        valid_pixels = int(np.count_nonzero((object_mask > 0) & (depth > 0)))
        if valid_pixels < MIN_VALID_PIXELS:
            print(
                f"[WARN] {ctx['frame_id']} part={part_idx}: "
                f"too few valid depth pixels ({valid_pixels} < {MIN_VALID_PIXELS}), skip"
            )
            continue

        try:
            _, mesh = build_raw_points(
                rgb_path,
                depth,
                ctx["frame_mask_dir"],
                part_idx,
                intrinsic,
                inference,
                object_mask,
                mask_path=mask_path,
            )
        except Exception as e:
            fg = int(np.count_nonzero(object_mask > 0))
            print(
                f"[WARN] build_raw_points failed: frame={ctx['frame_id']} part={part_idx} "
                f"mask={mask_path} fg={fg} valid_depth={valid_pixels} err={e}"
            )
            continue

        mesh = mesh_exported_to_sam3d_internal(mesh)
        raw_pose = np.eye(4, dtype=np.float32)
        src_points = sample_mesh_surface_points(mesh, MESH_ALIGNMENT_SAMPLE_POINTS, rng)
        used_gt_alignment = False

        # Reference-frame path: align sampled SAM3D-local mesh points to observed
        # camera-depth points, then move the mesh into object coordinates by gt pose.
        print(pose_roots)
        ob_in_cam = _load_ob_in_cam(ctx["frame_id"], pose_roots)
        if ob_in_cam is None:
            ob_in_cam = _load_ob_part(ctx["frame_id"], part_idx, pose_roots)
        print("ob_in_cam: ", ob_in_cam)
        if ob_in_cam is not None:
            tgt_cam = _backproject_masked_depth(object_mask, depth, intrinsic)
            if tgt_cam.shape[0] > 50 and src_points.shape[0] > 50:
                try:
                    tf_cam, keep_mask, sampled_src = _iterative_visibility_alignment(
                        src_points,
                        tgt_cam,
                        max_points=UMEYAMA_MAX_POINTS,
                        rng=rng,
                        with_scale=True,
                        visibility_fn=lambda src_cam_now: _visibility_mask_in_camera(
                            src_cam_now, object_mask, depth, intrinsic
                        ),
                        max_iter=4,
                    )
                    tf_cam = tf_cam.astype(np.float32)  # SAM3D local mesh -> reference camera
                    tf_obj = (np.linalg.inv(ob_in_cam).astype(np.float32) @ tf_cam).astype(np.float32)
                    raw_pose = ob_in_cam.astype(np.float32)  # saved mesh object frame -> reference camera
                    scale_est = float(np.cbrt(max(np.linalg.det(tf_cam[:3, :3].astype(np.float64)), 1e-12)))

                    mesh.apply_transform(tf_obj)
                    src_for_save = sampled_src[keep_mask] if int(np.sum(keep_mask)) >= 20 else sampled_src
                    aligned_points_cam = _apply_transform_np(src_for_save, tf_cam)
                    aligned_points_obj = _apply_transform_np(src_for_save, tf_obj)
                    tgt_obj = _apply_transform_np(tgt_cam, np.linalg.inv(ob_in_cam))
                    os.makedirs(model_path, exist_ok=True)
                    np.save(os.path.join(model_path, "reference_points.npy"), aligned_points_obj.astype(np.float32))
                    np.save(os.path.join(model_path, "reference_points_obj.npy"), aligned_points_obj.astype(np.float32))
                    np.save(os.path.join(model_path, "reference_points_cam.npy"), aligned_points_cam.astype(np.float32))
                    np.savetxt(os.path.join(model_path, "local_to_object.txt"), tf_obj, fmt="%.6f")
                    np.savetxt(os.path.join(model_path, "local_to_reference_camera.txt"), tf_cam, fmt="%.6f")
                    
                    _save_alignment_vis(
                        rgb_bgr,
                        intrinsic,
                        ob_in_cam,
                        aligned_points_obj,
                        tgt_obj,
                        os.path.join(model_path, "align_vis.png"),
                    )
                    used_gt_alignment = True
                    print(
                        f"[ALIGN-MESH-GT] frame={ctx['frame_id']} part={part_id} "
                        f"mesh local->camera->object, scale={scale_est:.4f}, saved_obj_frame"
                    )
                except Exception as e:
                    print(f"[WARN] mesh gt alignment failed: frame={ctx['frame_id']} part={part_id} err={e}")

        if (not used_gt_alignment) and (part_idx in ref_points_cache):
            try:
                transform = estimate_umeyama_from_pointclouds(
                    src_points,
                    ref_points_cache[part_idx],
                    UMEYAMA_MAX_POINTS,
                    rng,
                )
                raw_pose = transform
                mesh.apply_transform(transform)
                print(f"[ALIGN-INIT] frame={ctx['frame_id']} part={part_id} aligned to first-frame reference cloud")
            except Exception as e:
                print(f"[WARN] Umeyama failed: frame={ctx['frame_id']} part={part_id} err={e}")

        if not used_gt_alignment:
            cam_aligned, rmse = align_mesh_to_camera_frame(mesh, object_mask, depth, intrinsic)
            if cam_aligned:
                print(f"[ALIGN-CAM] frame={ctx['frame_id']} part={part_id} rmse={rmse:.5f}m")
            else:
                print(f"[ALIGN-CAM] frame={ctx['frame_id']} part={part_id} skipped/failed")

            if gt_root:
                aligned = False
                for pose_root in pose_roots:
                    if apply_extrinsic_pose_correction(mesh, pose_root, ctx["frame_id"]):
                        aligned = True
                        break
                if not aligned:
                    print(f"[WARN] Pose file missing for frame: {ctx['frame_id']}, keep mesh frame")

        # Use classic depth-size normalization only when gt similarity alignment is absent.
        if not used_gt_alignment:
            real_size = load_real_world_size(object_mask, depth, intrinsic)
            os.makedirs(model_path, exist_ok=True)
            preprocess_mesh_for_foundationpose(mesh, real_size, save_obj_file)
        else:
            os.makedirs(model_path, exist_ok=True)
            mesh.export(save_obj_file)
        np.savetxt(os.path.join(model_path, "raw_pose.txt"), raw_pose, fmt="%.6f")
        gc.collect()
        torch.cuda.empty_cache()

    return frame_dir


def estimate_frame_init_poses(
    intrinsic_path: str,
    rgb_path: str,
    depth_path: str,
    mask_dir: str,
    inference: Inference,
    reference_model_dir: str,
    random_seed: int = RANDOM_SEED,
    raw_est_output_dir: str | None = None,
    edge_gate: bool = False,
    edge_max_angle_deg: float = 90.0,
    edge_near_ratio: float = 0.15,
):
    out = {}
    if not os.path.exists(intrinsic_path) or (not os.path.exists(rgb_path)) or (not os.path.exists(depth_path)):
        return out
    if not os.path.isdir(mask_dir) or (not os.path.isdir(reference_model_dir)):
        return out

    intrinsic = load_intrinsic(intrinsic_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return out
    depth = _preprocess_depth_for_alignment(depth)
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR) if raw_est_output_dir else None
    if raw_est_output_dir:
        os.makedirs(raw_est_output_dir, exist_ok=True)
    
    rng = np.random.default_rng(random_seed)
    ref_vec_map = {}
    query_vec_map = {}
    if edge_gate:
        ref_points_map = _load_ref_points_map(reference_model_dir)
        ref_vec_map = _build_connector_vectors_from_part_points(ref_points_map, near_ratio=edge_near_ratio)

    masks = _list_mask_files(mask_dir)
    if edge_gate:
        q_part_points = {}
        for mask_index, mask_path in enumerate(masks):
            pid = _extract_part_id(mask_path, mask_index)
            try:
                m = load_mask_image(mask_path)
            except Exception:
                continue
            q_pts = _backproject_masked_depth(m, depth, intrinsic)
            if q_pts.shape[0] >= 20:
                q_part_points[int(pid)] = q_pts
        query_vec_map = _build_connector_vectors_from_part_points(q_part_points, near_ratio=edge_near_ratio)
    for mask_index, mask_path in enumerate(masks):
        part_id = _extract_part_id(mask_path, mask_index)
        ref_points_path = os.path.join(reference_model_dir, f"model_{part_id:04d}", "reference_points.npy")
        if not os.path.exists(ref_points_path):
            continue
        try:
            ref_points = np.load(ref_points_path).astype(np.float32)
        except Exception:
            continue
        if ref_points.ndim != 2 or ref_points.shape[1] != 3 or ref_points.shape[0] < 10:
            continue

        object_mask = load_mask_image(mask_path)
        if np.count_nonzero((object_mask > 0) & (depth > 0)) < MIN_VALID_PIXELS:
            continue

        try:
            _, cur_mesh = build_raw_points(
                rgb_path,
                depth,
                mask_dir,
                mask_index,
                intrinsic,
                inference,
                object_mask,
                mask_path=mask_path,
            )
            cur_mesh = mesh_exported_to_sam3d_internal(cur_mesh)
            src_points = sample_mesh_surface_points(cur_mesh, MESH_ALIGNMENT_SAMPLE_POINTS, rng)
            cur_points = _backproject_masked_depth(object_mask, depth, intrinsic)  # current camera frame
            if cur_points.shape[0] < 30 or src_points.shape[0] < 30:
                continue

            # Stage-1: local -> current camera
            tf_l2cur, keep_l2cur, sampled_src = _iterative_visibility_alignment(
                src_points,
                cur_points,
                max_points=UMEYAMA_MAX_POINTS,
                rng=rng,
                with_scale=True,
                visibility_fn=lambda src_cam_now: _visibility_mask_in_camera(src_cam_now, object_mask, depth, intrinsic),
                max_iter=3,
            )
            # Stage-2: local -> reference object/model frame
            tf_l2ref, _, _ = _iterative_visibility_alignment(
                src_points,
                ref_points,
                max_points=UMEYAMA_MAX_POINTS,
                rng=rng,
                with_scale=True,
                visibility_fn=None,
                max_iter=3,
            )

            # Compose to get: object/model frame -> current camera frame
            tf_ref2cur = tf_l2cur @ np.linalg.inv(tf_l2ref)

            # FoundationPose expects rigid init; remove residual scale/shear.
            a = tf_ref2cur[:3, :3].astype(np.float64)
            u, _, vt = np.linalg.svd(a)
            r = u @ vt
            if np.linalg.det(r) < 0:
                u[:, -1] *= -1
                r = u @ vt
            tf_ref2cur[:3, :3] = r.astype(np.float32)

            if edge_gate:
                ref_vec = ref_vec_map.get(int(part_id), np.zeros((3,), dtype=np.float32))
                q_vec = query_vec_map.get(int(part_id), np.zeros((3,), dtype=np.float32))
                ok_gate, gate_angle = _edge_gate_accept(
                    tf_ref2cur=tf_ref2cur,
                    ref_vec_obj=ref_vec,
                    query_vec_cam=q_vec,
                    max_angle_deg=edge_max_angle_deg,
                )
                if not ok_gate:
                    print(f"[EDGE-GATE] reject part={part_id} angle={gate_angle:.2f} >= {float(edge_max_angle_deg):.2f}")
                    continue

            out[part_id] = tf_ref2cur.astype(np.float32)

            if raw_est_output_dir and (rgb_bgr is not None):
                final_src_local = sampled_src[keep_l2cur] if int(np.sum(keep_l2cur)) >= 20 else sampled_src
                aligned_ref_cam = _apply_transform_np(final_src_local, tf_l2cur)
                vis_path = os.path.join(raw_est_output_dir, f"part_{part_id:04d}.png")
                _save_alignment_vis_cam(rgb_bgr, intrinsic, aligned_ref_cam, cur_points, vis_path)
        except Exception as e:
            print(f"[WARN] init-pose alignment failed: part={part_id}, err={e}")
    return out


def estimate_frame_init_poses_fast(
    intrinsic_path: str,
    depth_path: str,
    mask_dir: str,
    reference_model_dir: str,
    random_seed: int = RANDOM_SEED,
    rgb_path: str | None = None,
    raw_est_output_dir: str | None = None,
    edge_gate: bool = False,
    edge_max_angle_deg: float = 90.0,
    edge_near_ratio: float = 0.15,
):
    """
    Fast init for pose estimation:
    no per-frame SAM reconstruction, directly align
    reference point cloud (object/model coordinate) to current observed local cloud (camera coordinate).
    """
    out = {}
    if not os.path.exists(intrinsic_path) or (not os.path.exists(depth_path)):
        return out
    if not os.path.isdir(mask_dir) or (not os.path.isdir(reference_model_dir)):
        return out

    intrinsic = load_intrinsic(intrinsic_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return out
    depth = _preprocess_depth_for_alignment(depth)
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR) if (raw_est_output_dir and rgb_path and os.path.exists(rgb_path)) else None
    if raw_est_output_dir:
        os.makedirs(raw_est_output_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    ref_vec_map = {}
    query_vec_map = {}
    if edge_gate:
        ref_points_map = _load_ref_points_map(reference_model_dir)
        ref_vec_map = _build_connector_vectors_from_part_points(ref_points_map, near_ratio=edge_near_ratio)

    masks = _list_mask_files(mask_dir)
    if edge_gate:
        q_part_points = {}
        for mask_index, mask_path in enumerate(masks):
            pid = _extract_part_id(mask_path, mask_index)
            try:
                m = load_mask_image(mask_path)
            except Exception:
                continue
            q_pts = _backproject_masked_depth(m, depth, intrinsic)
            if q_pts.shape[0] >= 20:
                q_part_points[int(pid)] = q_pts
        query_vec_map = _build_connector_vectors_from_part_points(q_part_points, near_ratio=edge_near_ratio)
    for mask_index, mask_path in enumerate(masks):
        part_id = _extract_part_id(mask_path, mask_index)
        ref_points_path = os.path.join(reference_model_dir, f"model_{part_id:04d}", "reference_points.npy")
        if not os.path.exists(ref_points_path):
            continue

        try:
            ref_points = np.load(ref_points_path).astype(np.float32)
        except Exception:
            continue
        if ref_points.ndim != 2 or ref_points.shape[1] != 3 or ref_points.shape[0] < 10:
            continue

        object_mask = load_mask_image(mask_path)
        if np.count_nonzero((object_mask > 0) & (depth > 0)) < MIN_VALID_PIXELS:
            continue
        cur_points = _backproject_masked_depth(object_mask, depth, intrinsic)
        if cur_points.shape[0] < 30:
            continue

        try:
            tf_ref2cur, keep_ref, sampled_ref = _iterative_visibility_alignment(
                ref_points,
                cur_points,
                max_points=UMEYAMA_MAX_POINTS,
                rng=rng,
                with_scale=False,
                visibility_fn=lambda ref_cam_now: _visibility_mask_in_camera(ref_cam_now, object_mask, depth, intrinsic),
                max_iter=3,
            )
            if edge_gate:
                ref_vec = ref_vec_map.get(int(part_id), np.zeros((3,), dtype=np.float32))
                q_vec = query_vec_map.get(int(part_id), np.zeros((3,), dtype=np.float32))
                ok_gate, gate_angle = _edge_gate_accept(
                    tf_ref2cur=tf_ref2cur,
                    ref_vec_obj=ref_vec,
                    query_vec_cam=q_vec,
                    max_angle_deg=edge_max_angle_deg,
                )
                if not ok_gate:
                    print(f"[EDGE-GATE] reject part={part_id} angle={gate_angle:.2f} >= {float(edge_max_angle_deg):.2f}")
                    continue
            out[part_id] = tf_ref2cur.astype(np.float32)

            if raw_est_output_dir and (rgb_bgr is not None):
                final_ref_points = sampled_ref[keep_ref] if int(np.sum(keep_ref)) >= 20 else sampled_ref
                aligned_ref_cam = _apply_transform_np(final_ref_points, tf_ref2cur)
                vis_path = os.path.join(raw_est_output_dir, f"part_{part_id:04d}.png")
                _save_alignment_vis_cam(rgb_bgr, intrinsic, aligned_ref_cam, cur_points, vis_path)
        except Exception as e:
            print(f"[WARN] fast init alignment failed: part={part_id}, err={e}")
    return out
