import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tsdf_prior_init_from_sam3d as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3D-prior TSDF fusion directly on dataset_train/val layout "
            "(models as prior, masks as part masks, cam_params as obj_in_cam)."
        )
    )
    parser.add_argument(
        "--work-root",
        type=str,
        default=str(REPO_ROOT / "dataset_train" / "val"),
        help="dataset_train/val root. Kept as --work-root for compatibility.",
    )
    parser.add_argument("--data-root", type=str, default="", help="Alias for --work-root.")
    parser.add_argument("--objects", type=str, default="", help="Comma-separated objects; empty means all.")
    parser.add_argument("--object", type=str, default="bottle_3517", help="Single object alias.")
    parser.add_argument("--model-source", type=str, default="models")
    parser.add_argument("--mask-source", type=str, default="masks")
    parser.add_argument("--pose-source", type=str, default="cam_params")
    parser.add_argument("--pose-fallback", type=str, default="")
    parser.add_argument("--output-root", type=str, default="", help="Default: <work-root>/scheme_b_prior_tsdf.")
    parser.add_argument("--parts", type=str, default="0", help="Comma-separated part ids/model ids, e.g. 0,2.")
    parser.add_argument("--fusion-frame", choices=["camera", "object"], default="object")
    parser.add_argument(
        "--prior-mesh-frame",
        choices=["camera", "object", "raw_pose"],
        default="object",
        help="Current run reconstruction saves model.obj in object coordinates.",
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
    parser.add_argument(
        "--target-max-extent",
        type=float,
        default=0.6,
        help="Same value used by scripts/sapien_render.py; used to scale raw exported prior models during fusion.",
    )
    parser.add_argument(
        "--prior-scale-from-target-extent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale prior model.obj by the SAPIEN object_scale computed from bounding_box.json and --target-max-extent.",
    )
    parser.add_argument(
        "--prior-align-to-observed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Align dataset_train/val prior model bounds to masked RGB-D observations before TSDF init. "
            "The model.obj files are not always already in the cam_params part coordinate frame."
        ),
    )
    parser.add_argument(
        "--obs-free-space",
        action="store_true",
        help="Enable free-space carving during observation fusion. Default is narrow surface band only.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--pose-convention", choices=["cv", "sapien"], default="sapien")
    parser.add_argument(
        "--pose-direction",
        choices=["obj_to_cam", "cam_to_obj"],
        default="obj_to_cam",
        help="Direction stored in pose files after applying --pose-convention.",
    )
    parser.add_argument("--max-voxels", type=int, default=40_000_000)
    parser.add_argument("--sdf-backend", choices=["auto", "open3d", "trimesh"], default="auto")
    parser.add_argument("--sdf-chunk-size", type=int, default=50_000)
    parser.add_argument("--save-debug-points", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def compute_object_scale_from_target_extent(object_dir: Path, target_max_extent: float) -> float:
    meta_path = object_dir / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    source_dir_raw = str(meta.get("source_dir", "")).strip()
    if source_dir_raw:
        bbox_path = Path(source_dir_raw) / "bounding_box.json"
        if bbox_path.exists():
            bbox = load_json(bbox_path)
            bmin = np.asarray(bbox.get("min", []), dtype=np.float32)
            bmax = np.asarray(bbox.get("max", []), dtype=np.float32)
            if bmin.shape == (3,) and bmax.shape == (3,):
                extent = bmax - bmin
                max_extent = float(np.max(extent))
                if max_extent > 1e-6:
                    return float(np.clip(float(target_max_extent) / max_extent, 0.25, 4.0))
    render = meta.get("render", {})
    if isinstance(render, dict) and "object_scale" in render:
        return float(render["object_scale"])
    return 1.0


def collect_objects(work_root: Path, args: argparse.Namespace) -> List[str]:
    raw = args.object.strip() or args.objects.strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return sorted(
        [
            p.name
            for p in work_root.iterdir()
            if p.is_dir()
            and (p / "K.txt").exists()
            and (p / args.mask_source).is_dir()
            and (p / args.model_source).is_dir()
            and (p / args.pose_source).is_dir()
        ],
        key=base.natural_sort_key,
    )


def train_val_part_key(model_dir: Path) -> str:
    return model_dir.name


def train_val_selected_model_dirs(object_dir: Path, model_source: str, parts: str) -> List[Path]:
    root = object_dir / model_source
    if not root.is_dir():
        raise FileNotFoundError(f"model source not found: {root}")
    models = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "model.obj").exists()],
        key=lambda p: base.natural_sort_key(p.name),
    )
    if not parts.strip():
        return models
    wanted_raw = [x.strip() for x in parts.split(",") if x.strip()]
    wanted_names = set(wanted_raw)
    wanted_indices = {int(x) for x in wanted_raw if x.isdigit()}
    selected = []
    for idx, model_dir in enumerate(models):
        if model_dir.name in wanted_names or idx in wanted_indices:
            selected.append(model_dir)
    return selected


def train_val_list_frames(object_dir: Path, mask_source: str, frame_stride: int, max_frames: int) -> List[str]:
    depth_root = object_dir / "depth"
    rgb_root = object_dir / "rgb"
    if not depth_root.is_dir():
        raise FileNotFoundError(f"depth dir not found: {depth_root}")
    frames = []
    for p in sorted(depth_root.iterdir(), key=lambda x: base.natural_sort_key(x.name)):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        frame = p.stem
        if (rgb_root / f"{frame}.png").exists() or (rgb_root / f"{frame}.jpg").exists() or (rgb_root / f"{frame}.jpeg").exists():
            frames.append(frame)
    frames = frames[:: max(1, int(frame_stride))]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def train_val_load_mask(frame_mask_dir: Path, part_id, threshold: int):
    # base.run_part passes object_dir / mask_source / frame. In dataset_train/val
    # masks are laid out as object_dir / masks / <part_name> / <frame>.png.
    mask_root = frame_mask_dir.parent
    object_dir = mask_root.parent
    frame = frame_mask_dir.name
    part_key = str(part_id)
    candidates = [
        object_dir / "masks" / part_key / f"{frame}.png",
        object_dir / "masks" / part_key / f"{frame}.jpg",
        object_dir / "masks" / part_key / f"{frame}.jpeg",
    ]
    import cv2

    for path in candidates:
        if path.exists():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"failed to read mask: {path}")
            return mask > int(threshold)
    raise FileNotFoundError(f"mask for part={part_key} frame={frame} not found under {object_dir / 'masks' / part_key}")


def train_val_resolve_pose_path(object_dir: Path, pose_source: str, pose_fallback: str, frame: str, part_id) -> Path:
    part_key = str(part_id)
    roots = []
    pose_root = Path(pose_source)
    roots.append(pose_root if pose_root.is_absolute() else object_dir / pose_root)
    if str(pose_fallback).strip():
        fallback_root = Path(pose_fallback)
        roots.append(fallback_root if fallback_root.is_absolute() else object_dir / fallback_root)
    for root in roots:
        candidates = [
            root / part_key / f"{frame}.txt",
            root / f"{frame}.txt",
            root / frame / f"{part_key}.txt",
            root / frame / "pose.txt",
        ]
        for path in candidates:
            if path.exists():
                return path
    raise FileNotFoundError(f"pose for part={part_key} frame={frame} not found in {pose_source}")


def install_train_val_layout_hooks() -> None:
    base.model_part_id = train_val_part_key
    base.selected_model_dirs = train_val_selected_model_dirs
    base.list_frames = train_val_list_frames
    base.load_mask = train_val_load_mask
    base.resolve_pose_path = train_val_resolve_pose_path


def make_base_args(args: argparse.Namespace, object_dir: Path, out_root: Path) -> argparse.Namespace:
    prior_mesh_scale = (
        compute_object_scale_from_target_extent(object_dir, args.target_max_extent)
        if args.prior_scale_from_target_extent
        else 1.0
    )
    return argparse.Namespace(
        model_source=args.model_source,
        mask_source=args.mask_source,
        pose_source=args.pose_source,
        pose_fallback=args.pose_fallback,
        output_root=str(out_root),
        parts=args.parts,
        fusion_frame=args.fusion_frame,
        prior_mesh_frame=args.prior_mesh_frame,
        reference_frame=args.reference_frame,
        min_mask_pixels=args.min_mask_pixels,
        voxel_size=args.voxel_size,
        trunc_mult=args.trunc_mult,
        padding=args.padding,
        depth_scale=args.depth_scale,
        mask_threshold=args.mask_threshold,
        prior_weight=args.prior_weight,
        obs_weight=args.obs_weight,
        prior_mesh_scale=prior_mesh_scale,
        prior_align_to_observed=args.prior_align_to_observed,
        obs_free_space=args.obs_free_space,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        pose_convention=args.pose_convention,
        pose_direction=args.pose_direction,
        max_voxels=args.max_voxels,
        sdf_backend=args.sdf_backend,
        sdf_chunk_size=args.sdf_chunk_size,
        save_debug_points=args.save_debug_points,
    )


def run_object(object_dir: Path, out_root: Path, args: argparse.Namespace) -> dict:
    base_args = make_base_args(args, object_dir, out_root)
    out_dir = out_root / object_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Paths] object={object_dir.name} "
        f"models={object_dir / base_args.model_source} "
        f"masks={object_dir / base_args.mask_source} "
        f"poses={object_dir / base_args.pose_source} "
        f"depth={object_dir / 'depth'} "
        f"prior_mesh_scale={base_args.prior_mesh_scale:.8g}",
        flush=True,
    )

    k = base.load_k(object_dir)
    frames = base.list_frames(object_dir, base_args.mask_source, base_args.frame_stride, base_args.max_frames)
    if not frames:
        raise RuntimeError(f"no frames found under {object_dir / base_args.mask_source}")

    models = base.selected_model_dirs(object_dir, base_args.model_source, base_args.parts)
    if not models:
        raise RuntimeError(f"no model.obj found under {object_dir / base_args.model_source}")

    summaries = []
    for idx, model_dir in enumerate(models, start=1):
        print(f"[TSDF train_val {idx}/{len(models)}] {object_dir.name} {model_dir.name}", flush=True)
        summary = base.run_part(object_dir, model_dir, out_dir, base_args, k, frames)
        summaries.append(summary)
        print(
            f"  dims={summary['dims']} touched={summary['touched_frames']} "
            f"mesh={summary['mesh'] or summary['mesh_status']}",
            flush=True,
        )

    payload = {
        "scheme": "B_prior_tsdf_initialization_dataset_train_val",
        "object_dir": str(object_dir),
        "args": vars(args),
        "resolved_model_source": str(object_dir / base_args.model_source),
        "resolved_mask_source": str(object_dir / base_args.mask_source),
        "resolved_pose_source": str(object_dir / base_args.pose_source),
        "parts": summaries,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    args = parse_args()
    if args.data_root.strip():
        args.work_root = args.data_root
    install_train_val_layout_hooks()
    work_root = Path(args.work_root)
    if not work_root.is_dir():
        raise FileNotFoundError(f"work root not found: {work_root}")
    out_root = Path(args.output_root) if args.output_root.strip() else work_root / "scheme_b_prior_tsdf"
    out_root.mkdir(parents=True, exist_ok=True)

    objects = collect_objects(work_root, args)
    if not objects:
        raise RuntimeError(f"no objects selected under {work_root}")

    all_summaries = []
    for obj_name in objects:
        object_dir = work_root / obj_name
        if not object_dir.is_dir():
            print(f"[SKIP] object dir not found: {object_dir}", flush=True)
            continue
        all_summaries.append(run_object(object_dir, out_root, args))

    summary_path = out_root / "summary_all.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"work_root": str(work_root), "objects": all_summaries}, f, indent=2, ensure_ascii=False)
    print(f"[Done] wrote {summary_path}")


if __name__ == "__main__":
    main()
