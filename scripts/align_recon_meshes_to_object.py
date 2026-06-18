import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process SAM3D reconstruction meshes from reference-camera frame "
            "into GT object frame without modifying the original models directory."
        )
    )
    parser.add_argument("--data-root", type=str, default=r"D:\research\PartNet\data")
    parser.add_argument("--split", type=str, default="test_inter")
    parser.add_argument("--object", type=str, default="Door_8897")
    parser.add_argument("--model-source", type=str, default="models/view_0")
    parser.add_argument("--pose-source", type=str, default="gt_pose_from_ann")
    parser.add_argument("--pose-fallback", type=str, default="gt_pose")
    parser.add_argument("--output-root", type=str, default="./tsdf_prior_experiments/object_aligned_models")
    parser.add_argument("--parts", type=str, default="0", help="Comma-separated part ids, e.g. 0,2.")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
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
    obj_name = object_dir.name
    mask_root = object_dir / "gt_mask"
    if not mask_root.is_dir():
        raise FileNotFoundError(f"gt_mask not found: {mask_root}")
    grouped: Dict[int, List[Tuple[int, int, str]]] = {}
    for path in sorted([p for p in mask_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name)):
        view_id, frame_idx = parse_view_and_frame(path.name, obj_name)
        vis_count = frame_part_visibility_count(object_dir, path.name, min_mask_pixels)
        grouped.setdefault(view_id, []).append((vis_count, frame_idx, path.name))
    refs: Dict[int, str] = {}
    for view_id, candidates in grouped.items():
        candidates.sort(key=lambda x: (-x[0], x[1]))
        refs[view_id] = candidates[0][2]
    return refs


def model_part_id(model_dir: Path) -> int:
    return int(model_dir.name.rsplit("_", 1)[-1])


def list_model_dirs(object_dir: Path, model_source: str, parts: str) -> List[Path]:
    root = object_dir / model_source
    if not root.is_dir():
        raise FileNotFoundError(f"model source not found: {root}")
    models = sorted([p for p in root.rglob("model.obj")], key=lambda p: natural_sort_key(str(p)))
    model_dirs = [p.parent for p in models if p.parent.name.startswith("model_")]
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


def resolve_pose_path(object_dir: Path, pose_source: str, pose_fallback: str, frame: str, part_id: int) -> Path:
    candidates = []
    for root in pose_roots(object_dir, pose_source) + pose_roots(object_dir, pose_fallback):
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
    raise FileNotFoundError(f"pose not found for frame={frame}, part={part_id}")


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid pose shape {pose.shape}: {path}")
    return pose


def transform_obj_vertices(src_obj: Path, dst_obj: Path, tf: np.ndarray) -> None:
    dst_obj.parent.mkdir(parents=True, exist_ok=True)
    r = tf[:3, :3].astype(np.float64)
    t = tf[:3, 3].astype(np.float64)
    with src_obj.open("r", encoding="utf-8", errors="ignore") as f_in, dst_obj.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith("v "):
                parts = line.rstrip("\n").split()
                if len(parts) >= 4:
                    xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    out = r @ xyz + t
                    rest = " ".join(parts[4:])
                    suffix = f" {rest}" if rest else ""
                    f_out.write(f"v {out[0]:.8f} {out[1]:.8f} {out[2]:.8f}{suffix}\n")
                    continue
            f_out.write(line)


def copy_sidecars(src_dir: Path, dst_dir: Path, skip_obj: str = "model.obj") -> None:
    for path in src_dir.iterdir():
        if path.name == skip_obj:
            continue
        dst = dst_dir / path.name
        if path.is_file():
            shutil.copy2(path, dst)


def main() -> None:
    args = parse_args()
    object_dir = Path(args.data_root) / args.split / "objs" / args.object
    project_dir = Path(args.data_root) / args.split
    if not object_dir.is_dir():
        raise FileNotFoundError(f"object dir not found: {object_dir}")
    out_object_dir = Path(args.output_root) / args.split / args.object
    out_object_dir.mkdir(parents=True, exist_ok=True)

    refs_by_view = select_best_frame_per_view(object_dir, args.min_mask_pixels)
    model_dirs = list_model_dirs(object_dir, args.model_source, args.parts)
    summaries = []
    for model_dir in model_dirs:
        part_id = model_part_id(model_dir)
        view_id = infer_view_id(model_dir)
        if view_id not in refs_by_view:
            raise RuntimeError(f"No reference frame for view_{view_id}; refs={refs_by_view}")
        ref_frame = refs_by_view[view_id]
        pose_path = resolve_pose_path(project_dir, args.pose_source, args.pose_fallback, ref_frame, part_id)
        ob_in_cam = load_pose(pose_path)
        cam_to_obj = np.linalg.inv(ob_in_cam).astype(np.float32)

        rel = model_dir.relative_to(object_dir / args.model_source)
        dst_dir = out_object_dir / args.model_source / rel
        dst_obj = dst_dir / "model.obj"
        if dst_obj.exists() and not args.overwrite:
            print(f"[SKIP] exists: {dst_obj}")
        else:
            copy_sidecars(model_dir, dst_dir)
            transform_obj_vertices(model_dir / "model.obj", dst_obj, cam_to_obj)
            np.savetxt(dst_dir / "cam_to_obj.txt", cam_to_obj, fmt="%.8f")
            np.savetxt(dst_dir / "ref_ob_in_cam.txt", ob_in_cam, fmt="%.8f")
            with (dst_dir / "reference_frame.txt").open("w", encoding="utf-8") as f:
                f.write(ref_frame + "\n")
            print(f"[OK] {model_dir} -> {dst_obj} ref={ref_frame} part={part_id}")

        summaries.append(
            {
                "src_model_dir": str(model_dir),
                "dst_model_dir": str(dst_dir),
                "view_id": view_id,
                "part_id": part_id,
                "reference_frame": ref_frame,
                "pose_path": str(pose_path),
            }
        )

    with (out_object_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "object_dir": str(object_dir),
                "model_source": args.model_source,
                "pose_source": args.pose_source,
                "models": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Done] wrote {out_object_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
