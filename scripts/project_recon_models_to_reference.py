import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project evaluate/build=1 reconstructed models onto their reference frames "
            "by projecting saved reference_points.npy."
        )
    )
    parser.add_argument("--data-root", type=str, default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan")
    parser.add_argument("--split", type=str, default="test_intra")
    parser.add_argument("--objects", type=str, default="Box_100189", help="Comma-separated objects; empty means all.")
    parser.add_argument("--object", type=str, default="", help="Single object alias.")
    parser.add_argument("--model-source", type=str, default="models/view_0")
    parser.add_argument("--output-root", type=str, default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/recon_model_reference_projection")
    parser.add_argument("--parts", type=str, default="0", help="Comma-separated model ids, e.g. 0,2.")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
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


def infer_view_id(model_dir: Path) -> int:
    for parent in [model_dir.parent, *model_dir.parents]:
        m = re.fullmatch(r"view_(\d+)", parent.name)
        if m:
            return int(m.group(1))
    return 0


def model_part_id(model_dir: Path) -> int:
    return int(model_dir.name.rsplit("_", 1)[-1])


def load_matrix(path: Path) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid matrix shape {pose.shape}: {path}")
    return pose


def list_model_dirs(object_dir: Path, model_source: str, parts: str) -> List[Path]:
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
    return model_dirs


def load_k(object_dir: Path) -> np.ndarray:
    k = np.loadtxt(object_dir / "K.txt").astype(np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    return k.reshape(3, 3)


def find_rgb(object_dir: Path, frame: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = object_dir / "rgb" / f"{frame}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"rgb not found for frame={frame}")


def sample_mesh_points(mesh: trimesh.Trimesh, max_points: int) -> np.ndarray:
    max_points = max(1, int(max_points))
    try:
        count = min(max_points, max(5000, int(len(mesh.faces) * 4)))
        pts, _ = trimesh.sample.sample_surface(mesh, count)
        pts = np.asarray(pts, dtype=np.float32)
    except Exception:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(12345)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    return pts.astype(np.float32)


def project_points(points_cam: np.ndarray, k: np.ndarray, height: int, width: int) -> np.ndarray:
    z = points_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    pts = points_cam[valid]
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    u = k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2]
    v = k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(inside):
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([ui[inside], vi[inside]], axis=1).astype(np.int32)


def transform_points(tf: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3).astype(np.float32)
    pts_h = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (tf @ pts_h.T).T[:, :3].astype(np.float32)


def overlay_model(object_dir: Path, model_dir: Path, ref_frame: str, out_path: Path, args: argparse.Namespace) -> dict:
    rgb_path = find_rgb(object_dir, ref_frame)
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read rgb: {rgb_path}")
    h, w = image.shape[:2]
    k = load_k(object_dir)
    points_path = model_dir / "reference_points.npy"
    if not points_path.exists():
        raise FileNotFoundError(f"reference_points.npy not found: {points_path}")
    points_cam = np.load(points_path).astype(np.float32).reshape(-1, 3)
    if len(points_cam) > int(args.max_points):
        rng = np.random.default_rng(12345)
        points_cam = points_cam[rng.choice(len(points_cam), size=int(args.max_points), replace=False)]
    uv = project_points(points_cam, k, h, w)

    vis = image.copy()
    radius = max(1, int(args.point_radius))
    for u, v in uv:
        cv2.circle(vis, (int(u), int(v)), radius, (0, 255, 0), -1)
    cv2.putText(
        vis,
        f"{model_dir.name} reference_points.npy projection, ref={ref_frame}, points={len(uv)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return {
        "model_dir": str(model_dir),
        "reference_frame": ref_frame,
        "rgb_path": str(rgb_path),
        "reference_points_path": str(points_path),
        "output": str(out_path),
        "sampled_points": int(len(points_cam)),
        "projected_points": int(len(uv)),
        "reference_points_bounds": [
            points_cam.min(axis=0).tolist() if len(points_cam) else [],
            points_cam.max(axis=0).tolist() if len(points_cam) else [],
        ],
        "projection_source": "reference_points.npy",
    }


def collect_objects(split_root: Path, args: argparse.Namespace) -> List[str]:
    raw = args.object.strip() or args.objects.strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    objs_root = split_root / "objs"
    return sorted(
        [p.name for p in objs_root.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=natural_sort_key,
    )


def main() -> None:
    args = parse_args()
    split_root = Path(args.data_root) / args.split
    objs_root = split_root / "objs"
    if not objs_root.is_dir():
        raise FileNotFoundError(f"objs root not found: {objs_root}")
    out_root = Path(args.output_root) / args.split
    summaries = []
    for obj_name in collect_objects(split_root, args):
        object_dir = objs_root / obj_name
        refs_by_view = select_best_frame_per_view(object_dir, args.min_mask_pixels)
        model_dirs = list_model_dirs(object_dir, args.model_source, args.parts)
        for model_dir in model_dirs:
            view_id = infer_view_id(model_dir)
            if view_id not in refs_by_view:
                raise RuntimeError(f"No reference frame for view_{view_id}: {model_dir}")
            ref_frame = refs_by_view[view_id]
            rel = model_dir.relative_to(object_dir / args.model_source)
            out_path = out_root / obj_name / args.model_source / rel / "direct_projection.png"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] exists: {out_path}")
                continue
            summary = overlay_model(object_dir, model_dir, ref_frame, out_path, args)
            summaries.append(summary)
            print(f"[OK] {model_dir} -> {out_path} projected={summary['projected_points']}")

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "project_recon_models_to_reference_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "model_source": args.model_source,
                "projection_transform": "reference_points.npy",
                "models": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Done] wrote {summary_path}")


if __name__ == "__main__":
    main()
