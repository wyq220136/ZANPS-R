import argparse
import json
import re
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
            "Initialize a TSDF volume from a SAM3D prior mesh, then update it with real masked RGB-D "
            "observations. The final saved fused_mesh is extracted from the fused TSDF only."
        )
    )
    parser.add_argument("--data-root", type=str, default=r"D:\research\PartNet\data")
    parser.add_argument("--split", type=str, default="test_inter")
    parser.add_argument("--object", type=str, default="Door_8897", help="Object folder name, e.g. Box_100189.")
    parser.add_argument("--model-source", type=str, default="models")
    parser.add_argument("--mask-source", type=str, default="gt_mask")
    parser.add_argument("--pose-source", type=str, default="gt_pose_from_ann")
    parser.add_argument("--pose-fallback", type=str, default="cam_params")
    parser.add_argument("--output-root", type=str, default="./scheme_b_prior_tsdf")
    parser.add_argument("--parts", type=str, default="0", help="Comma-separated part ids/model ids, e.g. 0,2.")
    parser.add_argument("--fusion-frame", choices=["camera", "object"], default="object")
    parser.add_argument(
        "--prior-mesh-frame",
        choices=["camera", "object", "raw_pose"],
        default="raw_pose",
        help=(
            "Coordinate frame of the input SAM3D prior mesh. evaluate/build=1 writes "
            "reference-camera-frame meshes with raw_pose.txt, so the default uses raw_pose "
            "to recover local coordinates before converting to object frame."
        ),
    )
    parser.add_argument("--reference-frame", type=str, default="")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--trunc-mult", type=float, default=4.0)
    parser.add_argument("--padding", type=float, default=0.03)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--prior-weight", type=float, default=0.25)
    parser.add_argument("--obs-weight", type=float, default=1.0)
    parser.add_argument("--prior-mesh-scale", type=float, default=1.0)
    parser.add_argument(
        "--prior-align-to-observed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Before TSDF initialization, align prior mesh bounds to masked RGB-D observations.",
    )
    parser.add_argument(
        "--obs-free-space",
        action="store_true",
        help=(
            "Also integrate voxels in front of the observed surface. Disabled by default "
            "because part masks often create silhouette curtain artifacts."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--pose-convention", choices=["cv", "sapien"], default="cv")
    parser.add_argument(
        "--pose-direction",
        choices=["obj_to_cam", "cam_to_obj"],
        default="obj_to_cam",
        help="Direction stored in pose files after applying --pose-convention.",
    )
    parser.add_argument("--max-voxels", type=int, default=40_000_000)
    parser.add_argument(
        "--sdf-backend",
        choices=["auto", "open3d", "trimesh"],
        default="auto",
        help="Backend for prior mesh signed-distance initialization.",
    )
    parser.add_argument(
        "--sdf-chunk-size",
        type=int,
        default=50_000,
        help="Number of grid points per signed-distance query chunk.",
    )
    parser.add_argument("--save-debug-points", action="store_true")
    return parser.parse_args()


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def parse_view_and_frame(frame_id: str, obj_name: str) -> Tuple[int, int]:
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix):]
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
    gt_mask_root = object_dir / "gt_mask"
    if not gt_mask_root.exists():
        return {}
    for path in sorted([p for p in gt_mask_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name)):
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
    token = model_dir.name.rsplit("_", 1)[-1]
    return int(token)


def selected_model_dirs(object_dir: Path, model_source: str, parts: str) -> List[Path]:
    root = object_dir / model_source
    if not root.exists():
        raise FileNotFoundError(f"model source not found: {root}")
    models = sorted(
        [p.parent for p in root.rglob("model.obj") if p.parent.name.startswith("model_")],
        key=lambda p: natural_sort_key(str(p)),
    )
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
    frames = []
    for p in sorted([x for x in mask_root.iterdir() if x.is_dir()], key=lambda x: x.name):
        if (depth_root / f"{p.name}.png").exists():
            frames.append(p.name)
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


def bounds_align_transform(src_bounds: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Align a prior mesh bbox to observed points with uniform scale and translation."""
    tf = np.eye(4, dtype=np.float32)
    if dst_points.size == 0:
        return tf
    src_bounds = np.asarray(src_bounds, dtype=np.float32).reshape(2, 3)
    dst_bounds = np.stack([dst_points.min(axis=0), dst_points.max(axis=0)], axis=0).astype(np.float32)
    src_extent = src_bounds[1] - src_bounds[0]
    dst_extent = dst_bounds[1] - dst_bounds[0]
    valid = (src_extent > 1e-6) & (dst_extent > 1e-6)
    scale = float(np.median(dst_extent[valid] / src_extent[valid])) if np.any(valid) else 1.0
    src_center = (src_bounds[0] + src_bounds[1]) * 0.5
    dst_center = (dst_bounds[0] + dst_bounds[1]) * 0.5
    tf[:3, :3] *= np.float32(scale)
    tf[:3, 3] = dst_center - np.float32(scale) * src_center
    return tf


def backproject(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    valid = mask & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def make_grid(bmin: np.ndarray, bmax: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    dims = np.ceil((bmax - bmin) / float(voxel_size)).astype(np.int32) + 1
    dims = np.maximum(dims, 2)
    xs = bmin[0] + np.arange(dims[0], dtype=np.float32) * voxel_size
    ys = bmin[1] + np.arange(dims[1], dtype=np.float32) * voxel_size
    zs = bmin[2] + np.arange(dims[2], dtype=np.float32) * voxel_size
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    return dims, grid.astype(np.float32)


def observed_points_and_bounds(
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
        pose_path = resolve_pose_path(object_dir, args.pose_source, args.pose_fallback, frame, part_id)
        ob_in_cv = load_pose(pose_path, args.pose_convention, getattr(args, "pose_direction", "obj_to_cam"))
        if args.fusion_frame == "camera":
            cur_to_ref = ref_ob_in_cv @ np.linalg.inv(ob_in_cv)
            points.append(transform_points(cur_to_ref, pts_cv))
        else:
            points.append(transform_points(np.linalg.inv(ob_in_cv), pts_cv))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def reference_depth_points_obj(
    object_dir: Path,
    frame: str,
    part_id: int,
    args: argparse.Namespace,
    k: np.ndarray,
    ref_ob_in_cv: np.ndarray,
) -> np.ndarray:
    depth = load_depth_m(object_dir / "depth" / f"{frame}.png", args.depth_scale)
    mask = load_mask(object_dir / args.mask_source / frame, part_id, args.mask_threshold)
    pts_cv = backproject(depth, mask, k)
    if len(pts_cv) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return transform_points(np.linalg.inv(ref_ob_in_cv), pts_cv)


def initialize_tsdf_from_mesh(
    tsdf: np.ndarray,
    weight: np.ndarray,
    grid_points: np.ndarray,
    mesh: trimesh.Trimesh,
    trunc: float,
    prior_weight: float,
    backend: str,
    chunk_size: int,
) -> None:
    if backend == "auto":
        try:
            import open3d  # noqa: F401
            backend = "open3d"
        except Exception:
            backend = "trimesh"

    print(
        f"  [Prior SDF] backend={backend} points={len(grid_points)} "
        f"chunk_size={chunk_size}",
        flush=True,
    )

    o3d_scene = None
    if backend == "open3d":
        try:
            import open3d as o3d

            mesh_o3d = o3d.t.geometry.TriangleMesh()
            mesh_o3d.vertex.positions = o3d.core.Tensor(
                np.asarray(mesh.vertices, dtype=np.float32),
                dtype=o3d.core.Dtype.Float32,
            )
            mesh_o3d.triangle.indices = o3d.core.Tensor(
                np.asarray(mesh.faces, dtype=np.int32),
                dtype=o3d.core.Dtype.Int32,
            )
            o3d_scene = o3d.t.geometry.RaycastingScene()
            o3d_scene.add_triangles(mesh_o3d)
        except Exception as e:
            if backend != "auto":
                raise RuntimeError(f"Failed to initialize Open3D RaycastingScene: {e}") from e
            backend = "trimesh"

    flat_tsdf = tsdf.reshape(-1)
    flat_weight = weight.reshape(-1)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(grid_points), chunk_size):
        pts = grid_points[start:start + chunk_size]
        if backend == "open3d":
            import open3d as o3d
            sdf = o3d_scene.compute_signed_distance(
                o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
            ).numpy().astype(np.float32)
        else:
            try:
                signed = trimesh.proximity.signed_distance(mesh, pts).astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    "Failed to compute signed distance from prior mesh. "
                    "Install trimesh proximity dependencies such as rtree/open3d, "
                    "or run with --sdf-backend open3d."
                ) from e
            # trimesh returns positive inside and negative outside.
            sdf = -signed

        tsdf_new = np.clip(sdf / float(trunc), -1.0, 1.0).astype(np.float32)
        sl = slice(start, start + len(pts))
        flat_tsdf[sl] = tsdf_new
        flat_weight[sl] = float(prior_weight)

        if start == 0 or start + chunk_size >= len(grid_points) or (start // chunk_size) % 20 == 0:
            done = min(start + len(pts), len(grid_points))
            print(f"  [Prior SDF] {done}/{len(grid_points)}", flush=True)


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


def save_debug_clouds(
    part_out: Path,
    observed: np.ndarray,
    object_dir: Path,
    ref_frame: str,
    ref_ob_in_cv: Optional[np.ndarray],
    part_id: int,
    args: argparse.Namespace,
    k: np.ndarray,
) -> Dict[str, object]:
    debug = {}

    if len(observed) > 0:
        write_point_ply(part_out / "debug_all_observed_points_obj.ply", observed)
        debug["observed_bounds"] = [observed.min(axis=0).tolist(), observed.max(axis=0).tolist()]

    if ref_frame and ref_ob_in_cv is not None:
        ref_pts = reference_depth_points_obj(object_dir, ref_frame, part_id, args, k, ref_ob_in_cv)
        if len(ref_pts) > 0:
            write_point_ply(part_out / "debug_reference_depth_points_obj.ply", ref_pts)
            debug["reference_depth_bounds"] = [ref_pts.min(axis=0).tolist(), ref_pts.max(axis=0).tolist()]

    return debug


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
        verts, faces, _, _ = measure.marching_cubes(
            tsdf,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size),
        )
    except Exception as e:
        return None, f"marching_cubes_failed:{e}"

    return (verts.astype(np.float32) + bmin.reshape(1, 3), faces.astype(np.int32)), "ok"


def load_raw_pose(model_dir: Path) -> np.ndarray:
    path = model_dir / "raw_pose.txt"
    if not path.exists():
        raise FileNotFoundError(f"raw_pose.txt not found for raw_pose prior transform: {path}")
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid raw_pose shape {pose.shape}: {path}")
    return pose


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
    return frame, load_pose(pose_path, args.pose_convention, getattr(args, "pose_direction", "obj_to_cam")), pose_path


def run_part(object_dir: Path, model_dir: Path, out_dir: Path, args: argparse.Namespace, k: np.ndarray, frames: Sequence[str]) -> Dict[str, object]:
    part_id = model_part_id(model_dir)

    # SAM3D prior mesh: used only for TSDF initialization and fusion-space bounds.
    mesh = trimesh.load(model_dir / "model.obj", force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"empty prior mesh: {model_dir / 'model.obj'}")
    prior_mesh_scale = float(getattr(args, "prior_mesh_scale", 1.0))
    prior_mesh_scale_transform = np.eye(4, dtype=np.float32)
    if abs(prior_mesh_scale - 1.0) > 1e-8:
        prior_mesh_scale_transform[:3, :3] *= np.float32(prior_mesh_scale)
        mesh.apply_transform(prior_mesh_scale_transform)
        print(f"  [PriorScale] prior_mesh_scale={prior_mesh_scale:.8g}", flush=True)

    ref_frame = ""
    ref_pose_path = None
    ref_ob_in_cv = None
    need_ref_pose = args.fusion_frame == "camera" or args.prior_mesh_frame in ("camera", "raw_pose")
    if need_ref_pose:
        ref_frame, ref_ob_in_cv, ref_pose_path = resolve_reference_pose(object_dir, model_dir, part_id, args)
        print(
            f"  [FusionFrame] {args.fusion_frame} ref={ref_frame} "
            f"pose={ref_pose_path} prior_mesh_frame={args.prior_mesh_frame}",
            flush=True,
        )
    else:
        print(f"  [FusionFrame] object prior_mesh_frame={args.prior_mesh_frame}", flush=True)

    raw_pose = None
    prior_transform = np.eye(4, dtype=np.float32)

    if args.prior_mesh_frame == "raw_pose":
        raw_pose = load_raw_pose(model_dir)
        if args.fusion_frame == "object":
            prior_transform = (np.linalg.inv(ref_ob_in_cv) @ raw_pose).astype(np.float32)
        else:
            prior_transform = raw_pose.astype(np.float32)
        mesh.apply_transform(prior_transform)
    elif args.fusion_frame == "object" and args.prior_mesh_frame == "camera":
        prior_transform = np.linalg.inv(ref_ob_in_cv).astype(np.float32)
        mesh.apply_transform(prior_transform)
    elif args.fusion_frame == "camera" and args.prior_mesh_frame == "object":
        prior_transform = ref_ob_in_cv.astype(np.float32)
        mesh.apply_transform(prior_transform)

    observed = observed_points_and_bounds(object_dir, frames, part_id, args, k, ref_ob_in_cv)
    prior_observed_align = np.eye(4, dtype=np.float32)
    if bool(getattr(args, "prior_align_to_observed", False)) and len(observed) > 0:
        prior_observed_align = bounds_align_transform(np.asarray(mesh.bounds, dtype=np.float32), observed)
        mesh.apply_transform(prior_observed_align)
        print(
            f"  [PriorAlign] bounds_to_observed scale={float(prior_observed_align[0, 0]):.6g} "
            f"translation={prior_observed_align[:3, 3].round(6).tolist()}",
            flush=True,
        )
    print(
        f"  [PriorFrame] prior_mesh_frame={args.prior_mesh_frame} "
        f"fusion_frame={args.fusion_frame} prior_transform_max_abs_delta_from_I="
        f"{float(np.max(np.abs(prior_transform - np.eye(4, dtype=np.float32)))):.6g} "
        f"prior_align_max_abs_delta_from_I="
        f"{float(np.max(np.abs(prior_observed_align - np.eye(4, dtype=np.float32)))):.6g}",
        flush=True,
    )
    print(
        f"  [Bounds] mesh_min={np.asarray(mesh.bounds[0]).round(5).tolist()} "
        f"mesh_max={np.asarray(mesh.bounds[1]).round(5).tolist()} "
        f"observed_points={len(observed)}",
        flush=True,
    )

    bounds_arrays = [np.asarray(mesh.bounds, dtype=np.float32)]
    if len(observed) > 0:
        bounds_arrays.append(np.stack([observed.min(axis=0), observed.max(axis=0)], axis=0))
    all_bounds = np.stack(bounds_arrays, axis=0)

    bmin = all_bounds[:, 0, :].min(axis=0) - float(args.padding)
    bmax = all_bounds[:, 1, :].max(axis=0) + float(args.padding)

    dims, grid_points = make_grid(bmin.astype(np.float32), bmax.astype(np.float32), args.voxel_size)
    total_voxels = int(np.prod(dims))
    print(
        f"  [Grid] bounds_min={bmin.round(5).tolist()} bounds_max={bmax.round(5).tolist()} "
        f"dims={dims.tolist()} voxels={total_voxels} voxel_size={args.voxel_size}",
        flush=True,
    )

    if total_voxels > int(args.max_voxels):
        raise RuntimeError(f"volume too large for {model_dir.name}: dims={dims.tolist()}, voxels={total_voxels}")

    trunc = float(args.voxel_size) * float(args.trunc_mult)
    tsdf = np.ones(tuple(int(x) for x in dims), dtype=np.float32)
    weight = np.zeros_like(tsdf)

    # Step 1: initialize TSDF from SAM3D prior mesh
    initialize_tsdf_from_mesh(
        tsdf,
        weight,
        grid_points,
        mesh,
        trunc,
        args.prior_weight,
        args.sdf_backend,
        args.sdf_chunk_size,
    )

    part_out = out_dir / model_dir.relative_to(object_dir / args.model_source)
    part_out.mkdir(parents=True, exist_ok=True)

    # Keep IO structure unchanged, but do not export/save prior mesh as final fusion geometry.
    np.savez_compressed(
        part_out / "initialized_tsdf.npz",
        tsdf=tsdf,
        weight=weight,
        bounds_min=bmin,
        bounds_max=bmax,
        voxel_size=np.float32(args.voxel_size),
        trunc=np.float32(trunc),
    )

    touched = 0
    integrated = 0

    # Step 2: integrate masked real RGB-D observations into TSDF
    for frame in frames:
        try:
            depth = load_depth_m(object_dir / "depth" / f"{frame}.png", args.depth_scale)
            mask = load_mask(object_dir / args.mask_source / frame, part_id, args.mask_threshold)
            pose_path = resolve_pose_path(object_dir, args.pose_source, args.pose_fallback, frame, part_id)
            print(f"  [Integrate] frame={frame} pose={pose_path.name}", flush=True)

            ob_in_cv = load_pose(pose_path, args.pose_convention, getattr(args, "pose_direction", "obj_to_cam"))
            if args.fusion_frame == "camera":
                grid_to_current = ob_in_cv @ np.linalg.inv(ref_ob_in_cv)
            else:
                grid_to_current = ob_in_cv

            n = integrate_observation(
                tsdf,
                weight,
                grid_points,
                depth,
                mask,
                k,
                grid_to_current,
                trunc,
                args.obs_weight,
                args.obs_free_space,
            )
        except FileNotFoundError:
            n = 0

        integrated += n
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

    debug_info = {}
    if args.save_debug_points:
        debug_info = save_debug_clouds(
            part_out=part_out,
            observed=observed,
            object_dir=object_dir,
            ref_frame=ref_frame,
            ref_ob_in_cv=ref_ob_in_cv,
            part_id=part_id,
            args=args,
            k=k,
        )

    # Step 3: extract final mesh strictly from fused TSDF
    mesh_status = "not_run"
    mesh_path = ""
    extracted, mesh_status = extract_mesh(tsdf, bmin, float(args.voxel_size))
    if extracted is not None:
        verts, faces = extracted
        mesh_path = str(part_out / "fused_mesh.obj")
        write_mesh_obj(Path(mesh_path), verts, faces)

    return {
        "part": part_id,
        "model_dir": str(model_dir),
        "frames": len(frames),
        "touched_frames": touched,
        "integrated_voxel_observations": integrated,
        "dims": dims.tolist(),
        "bounds_min": bmin.tolist(),
        "bounds_max": bmax.tolist(),
        "mesh": mesh_path,
        "mesh_status": mesh_status,
        "mesh_source": "tsdf_only",
        "fusion_frame": args.fusion_frame,
        "prior_mesh_frame": args.prior_mesh_frame,
        "prior_mesh_scale": prior_mesh_scale,
        "prior_mesh_scale_transform": prior_mesh_scale_transform.tolist(),
        "prior_transform": prior_transform.tolist(),
        "prior_observed_align": prior_observed_align.tolist(),
        "prior_align_to_observed": bool(getattr(args, "prior_align_to_observed", False)),
        "pose_direction": getattr(args, "pose_direction", "obj_to_cam"),
        "reference_frame": ref_frame,
        "reference_pose_path": str(ref_pose_path) if ref_pose_path is not None else "",
        "debug": debug_info,
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
        print(f"[Scheme B {idx}/{len(models)}] {args.object} {model_dir.name}")
        summary = run_part(object_dir, model_dir, out_dir, args, k, frames)
        summaries.append(summary)
        print(
            f"  dims={summary['dims']} touched={summary['touched_frames']} "
            f"mesh={summary['mesh'] or summary['mesh_status']}",
            flush=True,
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scheme": "B_prior_tsdf_initialization_tsdf_only_output",
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
