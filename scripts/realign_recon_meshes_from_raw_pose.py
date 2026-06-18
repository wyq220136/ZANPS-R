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

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process evaluate/build=1 SAM3D meshes without re-running reconstruction: "
            "model.obj -> inv(raw_pose) local mesh -> align local mesh to reference-frame "
            "masked depth in camera frame -> transform to GT object frame."
        )
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test_intra")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--model-source", type=str, default="models")
    parser.add_argument("--pose-source", type=str, default="gt_pose_from_ann")
    parser.add_argument("--pose-fallback", type=str, default="gt_pose")
    parser.add_argument("--output-root", type=str, default="data/tsdf_prior_experiments/raw_pose_realigned_models")
    parser.add_argument("--parts", type=str, default="", help="Comma-separated model ids, e.g. 0,2.")
    parser.add_argument("--reference-frame", type=str, default="", help="Override reference frame for all views.")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-source-points", type=int, default=20000)
    parser.add_argument("--max-target-points", type=int, default=30000)
    parser.add_argument("--icp-iters", type=int, default=16)
    parser.add_argument("--trim-percent", type=float, default=85.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


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
    mask_root = object_dir / "gt_mask"
    if not mask_root.is_dir():
        raise FileNotFoundError(f"gt_mask not found: {mask_root}")
    grouped: Dict[int, List[Tuple[int, int, str]]] = {}
    for path in sorted([p for p in mask_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name)):
        view_id, frame_idx = parse_view_and_frame(path.name, object_dir.name)
        vis = frame_part_visibility_count(object_dir, path.name, min_mask_pixels)
        grouped.setdefault(view_id, []).append((vis, frame_idx, path.name))
    refs = {}
    for view_id, candidates in grouped.items():
        candidates.sort(key=lambda x: (-x[0], x[1]))
        refs[view_id] = candidates[0][2]
    return refs


def model_part_id(model_dir: Path) -> int:
    return int(model_dir.name.rsplit("_", 1)[-1])


def selected_model_dirs(object_dir: Path, model_source: str, parts: str) -> List[Path]:
    root = object_dir / model_source
    if not root.is_dir():
        raise FileNotFoundError(f"model source not found: {root}")
    model_dirs = sorted(
        [p.parent for p in root.rglob("model.obj") if p.parent.name.startswith("model_")],
        key=lambda p: natural_sort_key(str(p)),
    )
    if parts.strip():
        wanted = {int(x.strip()) for x in parts.split(",") if x.strip()}
        model_dirs = [p for p in model_dirs if model_part_id(p) in wanted]
    if not model_dirs:
        raise RuntimeError(f"No model_*/model.obj found under {root}")
    return model_dirs


def infer_view_id(model_dir: Path) -> int:
    for parent in [model_dir.parent, *model_dir.parents]:
        m = re.fullmatch(r"view_(\d+)", parent.name)
        if m:
            return int(m.group(1))
    return 0


def pose_roots(object_dir: Path, source: str) -> List[Path]:
    source_path = Path(source)
    if source_path.is_absolute():
        return [source_path]
    split_dir = object_dir.parent.parent
    return [split_dir / source, object_dir / source]


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid pose shape {pose.shape}: {path}")
    return pose


def resolve_pose_by_part_order(
    object_dir: Path,
    pose_source: str,
    pose_fallback: str,
    frame: str,
    part_order: int,
) -> Tuple[Path, str]:
    roots = pose_roots(object_dir, pose_source) + pose_roots(object_dir, pose_fallback)
    for root in roots:
        parts_file = root / f"{frame}__parts.txt"
        if parts_file.exists():
            names = [line.strip() for line in parts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if 0 <= int(part_order) < len(names):
                pose_path = root / names[int(part_order)]
                if pose_path.exists():
                    return pose_path, "parts_order"
    for root in roots:
        candidates = [
            root / f"{frame}__link_{part_order}.txt",
            root / f"{frame}.txt",
            root / frame / f"link_{part_order}.txt",
            root / frame / "pose.txt",
        ]
        for path in candidates:
            if path.exists():
                return path, "fallback_link"
    raise FileNotFoundError(f"pose not found for frame={frame}, part_order={part_order}")


def list_mask_files(frame_mask_dir: Path) -> List[Path]:
    if not frame_mask_dir.is_dir():
        raise FileNotFoundError(f"frame mask dir not found: {frame_mask_dir}")
    files = [
        p
        for p in frame_mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    files = sorted(files, key=lambda p: natural_sort_key(p.name))
    if not files:
        raise RuntimeError(f"no mask files under {frame_mask_dir}")
    return files


def load_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"failed to read depth: {path}")
    depth = depth.astype(np.float32)
    if depth.max() > 50:
        depth = depth / float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def backproject(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    valid = (mask > 0) & np.isfinite(depth) & (depth > 1e-6)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - float(k[0, 2])) * z / max(float(k[0, 0]), 1e-9)
    y = (ys.astype(np.float32) - float(k[1, 2])) * z / max(float(k[1, 1]), 1e-9)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def load_reference_depth_points_cam(
    object_dir: Path,
    frame: str,
    part_order: int,
    depth_scale: float,
) -> Tuple[np.ndarray, Path]:
    depth = load_depth_m(object_dir / "depth" / f"{frame}.png", depth_scale)
    mask_files = list_mask_files(object_dir / "gt_mask" / frame)
    if part_order < 0 or part_order >= len(mask_files):
        raise IndexError(f"part_order={part_order} outside mask list of {len(mask_files)} for {frame}")
    mask_path = mask_files[part_order]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {mask_path}")
    if mask.shape[:2] != depth.shape[:2]:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    k = np.loadtxt(object_dir / "K.txt").astype(np.float32).reshape(3, 3)
    return backproject(depth, mask, k), mask_path


def transform_points(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3).astype(np.float32)
    pts_h = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (tf @ pts_h.T).T[:, :3].astype(np.float32)


def sample_array(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) <= max_points:
        return points
    ids = rng.choice(len(points), size=int(max_points), replace=False)
    return points[ids]


def sample_mesh_points(mesh: trimesh.Trimesh, max_points: int, rng: np.random.Generator) -> np.ndarray:
    count = min(max_points, max(3000, int(len(mesh.faces) * 4)))
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, count)
        pts = np.asarray(pts, dtype=np.float32)
    except Exception:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    return sample_array(pts, max_points, rng)


def nearest_query(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cKDTree is not None:
        dists, idx = cKDTree(dst).query(src, k=1, workers=-1)
        return dists.astype(np.float32), idx.astype(np.int64)
    idx = np.zeros((len(src),), dtype=np.int64)
    dists = np.zeros((len(src),), dtype=np.float32)
    chunk = 1024
    for start in range(0, len(src), chunk):
        pts = src[start : start + chunk]
        d2 = np.sum((pts[:, None, :] - dst[None, :, :]) ** 2, axis=2)
        nn = np.argmin(d2, axis=1)
        idx[start : start + len(pts)] = nn
        dists[start : start + len(pts)] = np.sqrt(d2[np.arange(len(pts)), nn]).astype(np.float32)
    return dists, idx


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"invalid alignment arrays: {src.shape} vs {dst.shape}")
    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)
    src_mean = src64.mean(axis=0)
    dst_mean = dst64.mean(axis=0)
    src_c = src64 - src_mean
    dst_c = dst64 - dst_mean
    cov = (dst_c.T @ src_c) / max(len(src64), 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    scale = 1.0
    if with_scale:
        var_src = float(np.sum(src_c * src_c) / max(len(src64), 1))
        scale = float(np.sum(s) / max(var_src, 1e-12))
        scale = float(np.clip(scale, 0.02, 50.0))
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = (scale * r).astype(np.float32)
    tf[:3, 3] = (dst_mean - scale * (r @ src_mean)).astype(np.float32)
    return tf


def pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    axes = vecs[:, np.argsort(vals)[::-1]].astype(np.float32)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return axes


def make_alignment_seeds(src: np.ndarray, dst: np.ndarray, raw_pose: Optional[np.ndarray]) -> List[np.ndarray]:
    seeds = []
    if raw_pose is not None:
        seeds.append(raw_pose.astype(np.float32))
    src_ctr = src.mean(axis=0)
    dst_ctr = dst.mean(axis=0)
    src_span = np.percentile(src, 95, axis=0) - np.percentile(src, 5, axis=0)
    dst_span = np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0)
    scale0 = float(np.linalg.norm(dst_span) / max(np.linalg.norm(src_span), 1e-8))
    scale0 = float(np.clip(scale0, 0.02, 50.0))
    base = np.eye(4, dtype=np.float32)
    base[:3, :3] *= scale0
    base[:3, 3] = (dst_ctr - scale0 * src_ctr).astype(np.float32)
    seeds.append(base)
    try:
        src_axes = pca_axes(src)
        dst_axes = pca_axes(dst)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    signs = np.diag([sx, sy, sz]).astype(np.float32)
                    if np.linalg.det(signs) < 0:
                        continue
                    r = (dst_axes @ signs @ src_axes.T).astype(np.float32)
                    tf = np.eye(4, dtype=np.float32)
                    tf[:3, :3] = scale0 * r
                    tf[:3, 3] = (dst_ctr - scale0 * (r @ src_ctr)).astype(np.float32)
                    seeds.append(tf)
    except Exception:
        pass
    return seeds


def score_alignment(tf: np.ndarray, src: np.ndarray, dst: np.ndarray, trim_percent: float) -> float:
    src_t = transform_points(tf, src)
    dists, _ = nearest_query(src_t, dst)
    if len(dists) < 50:
        return np.inf
    cut = np.percentile(dists, float(trim_percent))
    keep = dists <= cut
    if int(np.sum(keep)) < 50:
        return np.inf
    return float(np.sqrt(np.mean(dists[keep] ** 2)))


def align_local_to_ref_cam(
    mesh_local: trimesh.Trimesh,
    target_cam: np.ndarray,
    raw_pose: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(args.seed))
    src = sample_mesh_points(mesh_local, int(args.max_source_points), rng)
    dst = sample_array(target_cam, int(args.max_target_points), rng)
    if len(src) < 50 or len(dst) < 50:
        raise RuntimeError(f"too few points for alignment: src={len(src)} dst={len(dst)}")

    best_tf = None
    best_score = np.inf
    for seed in make_alignment_seeds(src, dst, raw_pose):
        tf = seed.copy()
        for _ in range(max(1, int(args.icp_iters))):
            src_now = transform_points(tf, src)
            dists, nn_idx = nearest_query(src_now, dst)
            cut = np.percentile(dists, float(args.trim_percent))
            keep = dists <= cut
            if int(np.sum(keep)) < 50:
                continue
            delta = umeyama_similarity(src_now[keep], dst[nn_idx[keep]], with_scale=True)
            tf = (delta @ tf).astype(np.float32)
        score = score_alignment(tf, src, dst, float(args.trim_percent))
        if score < best_score:
            best_score = score
            best_tf = tf.copy()

    if best_tf is None or not np.isfinite(best_score):
        raise RuntimeError("failed to align local mesh to reference camera points")
    return best_tf.astype(np.float32), float(best_score)


def load_raw_pose(model_dir: Path) -> np.ndarray:
    path = model_dir / "raw_pose.txt"
    if not path.exists():
        raise FileNotFoundError(f"raw_pose.txt not found: {path}")
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid raw_pose shape {pose.shape}: {path}")
    return pose


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.name == "model.obj":
            continue
        dst = dst_dir / path.name
        if path.is_file():
            shutil.copy2(path, dst)


def write_point_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def process_model(
    object_dir: Path,
    model_dir: Path,
    out_object_dir: Path,
    refs_by_view: Dict[int, str],
    args: argparse.Namespace,
) -> dict:
    part_order = model_part_id(model_dir)
    view_id = infer_view_id(model_dir)
    ref_frame = args.reference_frame.strip() or refs_by_view.get(view_id, "")
    if not ref_frame:
        raise RuntimeError(f"No reference frame for view_{view_id}; refs={refs_by_view}")

    mesh_saved = trimesh.load(model_dir / "model.obj", force="mesh", process=False)
    if isinstance(mesh_saved, trimesh.Scene):
        mesh_saved = trimesh.util.concatenate(tuple(mesh_saved.geometry.values()))
    if len(mesh_saved.vertices) == 0 or len(mesh_saved.faces) == 0:
        raise RuntimeError(f"empty mesh: {model_dir / 'model.obj'}")

    raw_pose = load_raw_pose(model_dir)
    mesh_local = mesh_saved.copy()
    mesh_local.apply_transform(np.linalg.inv(raw_pose).astype(np.float32))

    target_cam, mask_path = load_reference_depth_points_cam(
        object_dir=object_dir,
        frame=ref_frame,
        part_order=part_order,
        depth_scale=float(args.depth_scale),
    )
    pose_path, pose_mode = resolve_pose_by_part_order(
        object_dir=object_dir,
        pose_source=args.pose_source,
        pose_fallback=args.pose_fallback,
        frame=ref_frame,
        part_order=part_order,
    )
    ref_ob_in_cam = load_pose(pose_path)

    local_to_ref_cam, rmse = align_local_to_ref_cam(mesh_local, target_cam, raw_pose, args)
    local_to_obj = (np.linalg.inv(ref_ob_in_cam) @ local_to_ref_cam).astype(np.float32)

    mesh_obj = mesh_local.copy()
    mesh_obj.apply_transform(local_to_obj)

    rel = model_dir.relative_to(object_dir / args.model_source)
    dst_dir = out_object_dir / args.model_source / rel
    dst_obj = dst_dir / "model.obj"
    if dst_obj.exists() and not args.overwrite:
        print(f"[SKIP] exists: {dst_obj}")
    else:
        copy_sidecars(model_dir, dst_dir)
        mesh_obj.export(dst_obj)
        np.savetxt(dst_dir / "raw_pose_original.txt", raw_pose, fmt="%.8f")
        np.savetxt(dst_dir / "local_to_ref_cam_realigned.txt", local_to_ref_cam, fmt="%.8f")
        np.savetxt(dst_dir / "ref_ob_in_cam.txt", ref_ob_in_cam, fmt="%.8f")
        np.savetxt(dst_dir / "local_to_obj_realigned.txt", local_to_obj, fmt="%.8f")
        with (dst_dir / "reference_frame.txt").open("w", encoding="utf-8") as f:
            f.write(ref_frame + "\n")
        with (dst_dir / "coordinate_frame.txt").open("w", encoding="utf-8") as f:
            f.write("object\n")
        if args.save_debug:
            src_debug = sample_mesh_points(mesh_local, min(20000, int(args.max_source_points)), np.random.default_rng(args.seed))
            aligned_cam = transform_points(local_to_ref_cam, src_debug)
            aligned_obj = transform_points(local_to_obj, src_debug)
            write_point_ply(dst_dir / "debug_ref_depth_points_cam.ply", target_cam)
            write_point_ply(dst_dir / "debug_aligned_surface_ref_cam.ply", aligned_cam)
            write_point_ply(dst_dir / "debug_aligned_surface_obj.ply", aligned_obj)
            mesh_local.export(dst_dir / "debug_mesh_local.obj")
        print(f"[OK] {model_dir} -> {dst_obj} ref={ref_frame} part_order={part_order} rmse={rmse:.6f}")

    return {
        "src_model_dir": str(model_dir),
        "dst_model_dir": str(dst_dir),
        "view_id": int(view_id),
        "part_order": int(part_order),
        "reference_frame": ref_frame,
        "mask_path": str(mask_path),
        "pose_path": str(pose_path),
        "pose_lookup": pose_mode,
        "alignment_rmse_ref_cam": float(rmse),
    }


def main() -> None:
    args = parse_args()
    object_dir = Path(args.data_root) / args.split / "objs" / args.object
    if not object_dir.is_dir():
        raise FileNotFoundError(f"object dir not found: {object_dir}")

    out_object_dir = Path(args.output_root) / args.split / args.object
    out_object_dir.mkdir(parents=True, exist_ok=True)

    refs_by_view = select_best_frame_per_view(object_dir, int(args.min_mask_pixels))
    model_dirs = selected_model_dirs(object_dir, args.model_source, args.parts)
    summaries = []
    for model_dir in model_dirs:
        summaries.append(process_model(object_dir, model_dir, out_object_dir, refs_by_view, args))

    summary = {
        "object_dir": str(object_dir),
        "output_object_dir": str(out_object_dir),
        "model_source": args.model_source,
        "pose_source": args.pose_source,
        "pose_fallback": args.pose_fallback,
        "coordinate_frame": "object",
        "models": summaries,
    }
    with (out_object_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Done] wrote {out_object_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
