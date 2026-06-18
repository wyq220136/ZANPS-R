import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh


SAPIENCAM_TO_CVCAM = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scheme C experiment: render SAM3D prior mesh into synthetic depth maps, "
            "integrate them as TSDF observations, then integrate real masked RGB-D frames."
        )
    )
    parser.add_argument("--data-root", type=str, default=r"D:\research\PartNet\data")
    parser.add_argument("--split", type=str, default="test_inter")
    parser.add_argument("--object", type=str, default="door_8897")
    parser.add_argument("--model-source", type=str, default="models/view_0")
    parser.add_argument("--mask-source", type=str, default="gt_mask")
    parser.add_argument("--pose-source", type=str, default="gt_pose_from_ann")
    parser.add_argument("--pose-fallback", type=str, default="cam_params")
    parser.add_argument("--output-root", type=str, default="data/tsdf_prior_experiments/scheme_c_rendered_prior_tsdf")
    parser.add_argument("--parts", type=str, default="0")
    parser.add_argument("--fusion-frame", choices=["camera", "object"], default="object")
    parser.add_argument("--reference-frame", type=str, default="")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--trunc-mult", type=float, default=4.0)
    parser.add_argument("--padding", type=float, default=0.03)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--prior-weight", type=float, default=0.20)
    parser.add_argument("--obs-weight", type=float, default=1.0)
    parser.add_argument("--render-views", type=int, default=24)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--render-margin", type=float, default=1.35)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--pose-convention", choices=["cv", "sapien"], default="cv")
    parser.add_argument(
        "--pose-direction",
        choices=["obj_to_cam", "cam_to_obj"],
        default="obj_to_cam",
        help="Direction stored in pose files after applying --pose-convention.",
    )
    parser.add_argument(
        "--obs-free-space",
        action="store_true",
        help="Enable free-space carving for real observations. Default is narrow surface band only.",
    )
    parser.add_argument("--max-voxels", type=int, default=40_000_000)
    parser.add_argument("--save-rendered-depth", action="store_true")
    parser.add_argument("--save-debug-points", action="store_true")
    return parser.parse_args()


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def parse_view_and_frame(frame_id: str, obj_name: str) -> Tuple[int, int]:
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix) :]
        m = re.fullmatch(r"(\d+)_(\d+)", tail)
        if m:
            return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", frame_id)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return 0, 0


def frame_part_visibility_count(object_dir: Path, frame_id: str, min_mask_pixels: int) -> int:
    mask_dir = object_dir / "gt_mask" / frame_id
    if not mask_dir.is_dir():
        return 0
    count = 0
    for path in mask_dir.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and int(np.count_nonzero(mask > 0)) >= int(min_mask_pixels):
            count += 1
    return count


def select_best_frame_per_view(object_dir: Path, min_mask_pixels: int) -> Dict[int, str]:
    grouped: Dict[int, List[Tuple[int, int, str]]] = {}
    for path in sorted([p for p in (object_dir / "gt_mask").iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name)):
        view_id, frame_idx = parse_view_and_frame(path.name, object_dir.name)
        vis = frame_part_visibility_count(object_dir, path.name, min_mask_pixels)
        grouped.setdefault(view_id, []).append((vis, frame_idx, path.name))
    refs = {}
    for view_id, candidates in grouped.items():
        candidates.sort(key=lambda x: (-x[0], x[1]))
        refs[view_id] = candidates[0][2]
    return refs


def infer_view_id(model_dir: Path) -> int:
    for parent in [model_dir.parent, *model_dir.parents]:
        m = re.fullmatch(r"view_(\d+)", parent.name)
        if m:
            return int(m.group(1))
    return 0


def model_part_id(model_dir: Path) -> int:
    return int(model_dir.name.rsplit("_", 1)[-1])


def selected_model_dirs(object_dir: Path, model_source: str, parts: str) -> List[Path]:
    root = object_dir / model_source
    if not root.exists():
        raise FileNotFoundError(f"model source not found: {root}")
    models = sorted([p.parent for p in root.rglob("model.obj") if p.parent.name.startswith("model_")], key=lambda p: natural_sort_key(str(p)))
    if not parts.strip():
        return models
    wanted = {int(x.strip()) for x in parts.split(",") if x.strip()}
    return [p for p in models if model_part_id(p) in wanted]


def load_k(object_dir: Path) -> np.ndarray:
    k_path = object_dir / "K.txt"
    if not k_path.exists():
        raise FileNotFoundError(f"K.txt not found: {k_path}")
    k = np.loadtxt(k_path).astype(np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    return k


def list_frames(object_dir: Path, mask_source: str, frame_stride: int, max_frames: int) -> List[str]:
    mask_root = object_dir / mask_source
    depth_root = object_dir / "depth"
    if not mask_root.exists():
        raise FileNotFoundError(f"mask source not found: {mask_root}")
    frames = [p.name for p in sorted(mask_root.iterdir(), key=lambda x: x.name) if p.is_dir() and (depth_root / f"{p.name}.png").exists()]
    frames = frames[:: max(1, int(frame_stride))]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def _pose_roots(object_dir: Path, source: str) -> List[Path]:
    if not str(source).strip():
        return []
    source_path = Path(source)
    if source_path.is_absolute():
        return [source_path]
    split_dir = object_dir.parent.parent
    return [split_dir / source, object_dir / source]


def _pose_from_parts_file(root: Path, frame: str, part_id: int) -> Optional[Path]:
    parts_path = root / f"{frame}__parts.txt"
    if not parts_path.exists():
        return None
    try:
        lines = [line.strip() for line in parts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return None
    if int(part_id) < 0 or int(part_id) >= len(lines):
        return None
    pose_path = Path(lines[int(part_id)])
    if not pose_path.is_absolute():
        pose_path = root / pose_path
    return pose_path if pose_path.exists() else None


def resolve_pose_path(object_dir: Path, pose_source: str, pose_fallback: str, frame: str, part_id: int) -> Path:
    for root in _pose_roots(object_dir, pose_source):
        mapped = _pose_from_parts_file(root, frame, part_id)
        if mapped is not None:
            return mapped

    candidates = []
    for root in _pose_roots(object_dir, pose_source) + _pose_roots(object_dir, pose_fallback):
        candidates.extend(
            [
                root / f"{frame}__link_{part_id}.txt",
                root / f"{frame}.txt",
                root / frame / f"link_{part_id}.txt",
                root / frame / "pose.txt",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"pose for frame {frame}, part {part_id} not found in {pose_source} or {pose_fallback}"
    )


def load_pose(path: Path, convention: str, direction: str = "obj_to_cam") -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if convention == "sapien":
        pose = SAPIENCAM_TO_CVCAM @ pose
    if direction == "cam_to_obj":
        pose = np.linalg.inv(pose).astype(np.float32)
    return pose


def load_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"failed to read depth: {path}")
    depth = depth.astype(np.float32)
    if depth.max() > 50:
        depth = depth / float(depth_scale)
    return depth


def load_mask(frame_mask_dir: Path, part_id: int, threshold: int) -> np.ndarray:
    candidates = [
        frame_mask_dir / f"mask_{part_id}.png",
        frame_mask_dir / f"mask_{part_id:04d}.png",
        frame_mask_dir / f"{part_id}.png",
    ]
    for path in candidates:
        if path.exists():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"failed to read mask: {path}")
            return mask > int(threshold)
    raise FileNotFoundError(f"mask for part {part_id} not found under {frame_mask_dir}")


def transform_points(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    pts_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (tf @ pts_h.T).T[:, :3].astype(np.float32)


def backproject(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    valid = mask & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def observed_points(
    object_dir: Path,
    frames: Sequence[str],
    part_id: int,
    args: argparse.Namespace,
    k: np.ndarray,
    ref_ob_in_cv: Optional[np.ndarray],
) -> np.ndarray:
    points = []
    for frame in frames:
        depth = load_depth_m(object_dir / "depth" / f"{frame}.png", args.depth_scale)
        mask = load_mask(object_dir / args.mask_source / frame, part_id, args.mask_threshold)
        pts_cv = backproject(depth, mask, k)
        if len(pts_cv) == 0:
            continue
        ob_in_cv = load_pose(
            resolve_pose_path(object_dir, args.pose_source, args.pose_fallback, frame, part_id),
            args.pose_convention,
            args.pose_direction,
        )
        if args.fusion_frame == "camera":
            cur_to_ref = ref_ob_in_cv @ np.linalg.inv(ob_in_cv)
            points.append(transform_points(cur_to_ref, pts_cv))
        else:
            points.append(transform_points(np.linalg.inv(ob_in_cv), pts_cv))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def make_grid(bmin: np.ndarray, bmax: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    dims = np.ceil((bmax - bmin) / float(voxel_size)).astype(np.int32) + 1
    dims = np.maximum(dims, 2)
    xs = bmin[0] + np.arange(dims[0], dtype=np.float32) * voxel_size
    ys = bmin[1] + np.arange(dims[1], dtype=np.float32) * voxel_size
    zs = bmin[2] + np.arange(dims[2], dtype=np.float32) * voxel_size
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    return dims, grid.astype(np.float32)


def look_at_cv(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera_pos
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(up_hint, forward)
    right = right / max(np.linalg.norm(right), 1e-8)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-8)
    rot = np.stack([right, down, forward], axis=0).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = -rot @ camera_pos.astype(np.float32)
    return pose


def virtual_cameras(mesh: trimesh.Trimesh, view_count: int, margin: float) -> List[np.ndarray]:
    center = mesh.bounding_box.centroid.astype(np.float32)
    radius = float(np.linalg.norm(mesh.extents)) * 0.5 * float(margin)
    radius = max(radius, 0.1)
    views = []
    elevs = [math.radians(20.0), math.radians(-20.0)]
    per_ring = max(4, int(math.ceil(view_count / len(elevs))))
    for elev in elevs:
        for i in range(per_ring):
            if len(views) >= view_count:
                break
            az = 2.0 * math.pi * i / per_ring
            pos = center + radius * np.asarray(
                [math.cos(elev) * math.cos(az), math.sin(elev), math.cos(elev) * math.sin(az)],
                dtype=np.float32,
            )
            views.append(look_at_cv(pos, center))
    return views


def synthetic_intrinsics(mesh: trimesh.Trimesh, render_size: int, margin: float) -> np.ndarray:
    extent = max(float(np.max(mesh.extents)), 1e-3)
    radius = max(float(np.linalg.norm(mesh.extents)) * 0.5 * margin, 0.1)
    f = 0.5 * render_size * radius / (0.5 * extent * margin)
    k = np.asarray(
        [[f, 0.0, render_size * 0.5], [0.0, f, render_size * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return k


def rasterize_mesh_depth(mesh: trimesh.Trimesh, ob_in_cv: np.ndarray, k: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    verts_cv = transform_points(ob_in_cv, np.asarray(mesh.vertices, dtype=np.float32))
    faces = np.asarray(mesh.faces, dtype=np.int32)
    depth = np.zeros((height, width), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    for tri in faces:
        pts = verts_cv[tri]
        if np.any(pts[:, 2] <= 1e-6):
            continue
        uv = np.empty((3, 2), dtype=np.float32)
        uv[:, 0] = k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2]
        uv[:, 1] = k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]
        xmin = max(0, int(np.floor(np.min(uv[:, 0]))))
        xmax = min(width - 1, int(np.ceil(np.max(uv[:, 0]))))
        ymin = max(0, int(np.floor(np.min(uv[:, 1]))))
        ymax = min(height - 1, int(np.ceil(np.max(uv[:, 1]))))
        if xmax < xmin or ymax < ymin:
            continue
        x0, y0 = uv[0]
        x1, y1 = uv[1]
        x2, y2 = uv[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-8:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float32)
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue
        zz = w0 * pts[0, 2] + w1 * pts[1, 2] + w2 * pts[2, 2]
        patch_z = zbuf[ymin : ymax + 1, xmin : xmax + 1]
        update = inside & (zz < patch_z)
        patch_z[update] = zz[update]
    valid = np.isfinite(zbuf)
    depth[valid] = zbuf[valid]
    return depth, valid


def integrate_observation(
    tsdf: np.ndarray,
    weight: np.ndarray,
    grid_points: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    k: np.ndarray,
    ob_in_cv: np.ndarray,
    trunc: float,
    obs_weight: float,
    obs_free_space: bool,
) -> int:
    h, w = depth.shape
    pts_cv = transform_points(ob_in_cv, grid_points)
    z = pts_cv[:, 2]
    valid_z = z > 1e-6
    u = np.zeros_like(z, dtype=np.float32)
    v = np.zeros_like(z, dtype=np.float32)
    u[valid_z] = k[0, 0] * pts_cv[valid_z, 0] / z[valid_z] + k[0, 2]
    v[valid_z] = k[1, 1] * pts_cv[valid_z, 1] / z[valid_z] + k[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    valid = valid_z & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not np.any(valid):
        return 0
    idx = np.where(valid)[0]
    depth_obs = depth[vi[idx], ui[idx]]
    mask_obs = mask[vi[idx], ui[idx]]
    ok = mask_obs & (depth_obs > 0)
    if not np.any(ok):
        return 0
    idx = idx[ok]
    sdf = depth_obs[ok] - z[idx]
    if obs_free_space:
        valid_sdf = sdf >= -trunc
    else:
        valid_sdf = np.abs(sdf) <= trunc
    if not np.any(valid_sdf):
        return 0
    idx = idx[valid_sdf]
    tsdf_new = np.clip(sdf[valid_sdf] / trunc, -1.0, 1.0).astype(np.float32)
    flat_tsdf = tsdf.reshape(-1)
    flat_weight = weight.reshape(-1)
    old_w = flat_weight[idx]
    new_w = old_w + float(obs_weight)
    flat_tsdf[idx] = (flat_tsdf[idx] * old_w + tsdf_new * float(obs_weight)) / new_w
    flat_weight[idx] = new_w
    return int(len(idx))


def write_point_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def write_mesh_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces.astype(np.int64) + 1:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")


def extract_mesh(tsdf: np.ndarray, bmin: np.ndarray, voxel_size: float) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
    try:
        from skimage import measure
    except Exception as e:
        return None, f"skimage_unavailable:{e}"
    if float(np.nanmin(tsdf)) > 0.0 or float(np.nanmax(tsdf)) < 0.0:
        return None, f"no_zero_crossing:min={float(np.nanmin(tsdf)):.5f},max={float(np.nanmax(tsdf)):.5f}"
    try:
        verts, faces, _, _ = measure.marching_cubes(tsdf, level=0.0, spacing=(voxel_size, voxel_size, voxel_size))
    except Exception as e:
        return None, f"marching_cubes_failed:{e}"
    return (verts.astype(np.float32) + bmin.reshape(1, 3), faces.astype(np.int32)), "ok"


def resolve_reference_pose(object_dir: Path, model_dir: Path, part_id: int, args: argparse.Namespace) -> Tuple[str, np.ndarray, Path]:
    if args.reference_frame.strip():
        frame = args.reference_frame.strip()
    else:
        view_id = infer_view_id(model_dir)
        refs = select_best_frame_per_view(object_dir, args.min_mask_pixels)
        if view_id not in refs:
            raise RuntimeError(f"No reference frame for view_{view_id}; refs={refs}")
        frame = refs[view_id]
    pose_path = resolve_pose_path(object_dir, args.pose_source, args.pose_fallback, frame, part_id)
    return frame, load_pose(pose_path, args.pose_convention, args.pose_direction), pose_path


def run_part(object_dir: Path, model_dir: Path, out_dir: Path, args: argparse.Namespace, k_real: np.ndarray, frames: Sequence[str]) -> Dict[str, object]:
    part_id = model_part_id(model_dir)
    mesh = trimesh.load(model_dir / "model.obj", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"empty prior mesh: {model_dir / 'model.obj'}")

    ref_frame = ""
    ref_pose_path = None
    ref_ob_in_cv = None
    if args.fusion_frame == "camera":
        ref_frame, ref_ob_in_cv, ref_pose_path = resolve_reference_pose(object_dir, model_dir, part_id, args)
        print(f"  [FusionFrame] camera ref={ref_frame} pose={ref_pose_path}", flush=True)
    else:
        print("  [FusionFrame] object", flush=True)

    observed = observed_points(object_dir, frames, part_id, args, k_real, ref_ob_in_cv)
    bounds_arrays = [np.asarray(mesh.bounds, dtype=np.float32)]
    if len(observed) > 0:
        bounds_arrays.append(np.stack([observed.min(axis=0), observed.max(axis=0)], axis=0))
    all_bounds = np.stack(bounds_arrays, axis=0)
    bmin = all_bounds[:, 0, :].min(axis=0) - float(args.padding)
    bmax = all_bounds[:, 1, :].max(axis=0) + float(args.padding)
    dims, grid_points = make_grid(bmin.astype(np.float32), bmax.astype(np.float32), args.voxel_size)
    total_voxels = int(np.prod(dims))
    if total_voxels > int(args.max_voxels):
        raise RuntimeError(f"volume too large for {model_dir.name}: dims={dims.tolist()}, voxels={total_voxels}")

    trunc = float(args.voxel_size) * float(args.trunc_mult)
    tsdf = np.ones(tuple(int(x) for x in dims), dtype=np.float32)
    weight = np.zeros_like(tsdf)

    part_out = out_dir / model_dir.relative_to(object_dir / args.model_source)
    part_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "model.obj", part_out / "prior_mesh.obj")

    k_syn = synthetic_intrinsics(mesh, args.render_size, args.render_margin)
    synthetic_poses = virtual_cameras(mesh, args.render_views, args.render_margin)
    prior_integrated = 0
    render_root = part_out / "rendered_prior"
    if args.save_rendered_depth:
        render_root.mkdir(parents=True, exist_ok=True)
    for vidx, ob_in_cv in enumerate(synthetic_poses):
        depth, mask = rasterize_mesh_depth(mesh, ob_in_cv, k_syn, args.render_size, args.render_size)
        n = integrate_observation(tsdf, weight, grid_points, depth, mask, k_syn, ob_in_cv, trunc, args.prior_weight, True)
        prior_integrated += n
        if args.save_rendered_depth:
            depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            cv2.imwrite(str(render_root / f"depth_{vidx:03d}.png"), depth_mm)
            cv2.imwrite(str(render_root / f"mask_{vidx:03d}.png"), (mask.astype(np.uint8) * 255))
            np.savetxt(render_root / f"ob_in_cv_{vidx:03d}.txt", ob_in_cv, fmt="%.8f")

    np.savez_compressed(
        part_out / "prior_render_tsdf.npz",
        tsdf=tsdf,
        weight=weight,
        bounds_min=bmin,
        bounds_max=bmax,
        voxel_size=np.float32(args.voxel_size),
        trunc=np.float32(trunc),
        k_synthetic=k_syn,
    )

    obs_integrated = 0
    touched = 0
    for frame in frames:
        try:
            depth = load_depth_m(object_dir / "depth" / f"{frame}.png", args.depth_scale)
            mask = load_mask(object_dir / args.mask_source / frame, part_id, args.mask_threshold)
            pose_path = resolve_pose_path(object_dir, args.pose_source, args.pose_fallback, frame, part_id)
            ob_in_cv = load_pose(pose_path, args.pose_convention, args.pose_direction)
            if args.fusion_frame == "camera":
                grid_to_current = ob_in_cv @ np.linalg.inv(ref_ob_in_cv)
            else:
                grid_to_current = ob_in_cv
            print(f"  [Integrate] frame={frame} pose={pose_path.name}", flush=True)
            n = integrate_observation(
                tsdf,
                weight,
                grid_points,
                depth,
                mask,
                k_real,
                grid_to_current,
                trunc,
                args.obs_weight,
                args.obs_free_space,
            )
        except FileNotFoundError:
            n = 0
        obs_integrated += n
        touched += int(n > 0)

    np.savez_compressed(
        part_out / "fused_tsdf.npz",
        tsdf=tsdf,
        weight=weight,
        bounds_min=bmin,
        bounds_max=bmax,
        voxel_size=np.float32(args.voxel_size),
        trunc=np.float32(trunc),
        frames=np.asarray(frames),
    )
    if args.save_debug_points and len(observed) > 0:
        write_point_ply(part_out / "debug_observed_points_obj.ply", observed)
        samples, _ = trimesh.sample.sample_surface(mesh, min(100_000, max(10_000, len(mesh.faces) * 4)))
        write_point_ply(part_out / "debug_prior_surface_points.ply", samples.astype(np.float32))

    extracted, mesh_status = extract_mesh(tsdf, bmin, float(args.voxel_size))
    mesh_path = ""
    if extracted is not None:
        verts, faces = extracted
        mesh_path = str(part_out / "fused_mesh.obj")
        write_mesh_obj(Path(mesh_path), verts, faces)

    return {
        "part": part_id,
        "model_dir": str(model_dir),
        "frames": len(frames),
        "touched_real_frames": touched,
        "prior_views": len(synthetic_poses),
        "prior_integrated_voxel_observations": prior_integrated,
        "real_integrated_voxel_observations": obs_integrated,
        "dims": dims.tolist(),
        "bounds_min": bmin.tolist(),
        "bounds_max": bmax.tolist(),
        "mesh": mesh_path,
        "mesh_status": mesh_status,
        "fusion_frame": args.fusion_frame,
        "pose_source": args.pose_source,
        "pose_fallback": args.pose_fallback,
        "pose_convention": args.pose_convention,
        "pose_direction": args.pose_direction,
        "obs_free_space": bool(args.obs_free_space),
        "reference_frame": ref_frame,
        "reference_pose_path": str(ref_pose_path) if ref_pose_path is not None else "",
    }


def main() -> None:
    args = parse_args()
    object_dir = Path(args.data_root) / args.split / "objs" / args.object
    if not object_dir.exists():
        raise FileNotFoundError(f"object dir not found: {object_dir}")
    out_dir = Path(args.output_root) / args.object
    out_dir.mkdir(parents=True, exist_ok=True)
    k = load_k(object_dir)
    frames = list_frames(object_dir, args.mask_source, args.frame_stride, args.max_frames)
    if not frames:
        raise RuntimeError(f"no frames found under {object_dir / args.mask_source}")

    summaries = []
    models = selected_model_dirs(object_dir, args.model_source, args.parts)
    if not models:
        raise RuntimeError(
            f"No SAM3D prior meshes selected under {object_dir / args.model_source}. "
            "Expected directories like model_0000/model.obj. "
            "Check --model-source and --parts."
        )
    for idx, model_dir in enumerate(models, start=1):
        print(f"[Scheme C {idx}/{len(models)}] {args.object} {model_dir.name}")
        summary = run_part(object_dir, model_dir, out_dir, args, k, frames)
        summaries.append(summary)
        print(
            f"  dims={summary['dims']} prior_views={summary['prior_views']} "
            f"touched={summary['touched_real_frames']} mesh={summary['mesh'] or summary['mesh_status']}"
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme": "C_rendered_prior_tsdf_fusion",
                "object_dir": str(object_dir),
                "args": vars(args),
                "parts": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Done] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
