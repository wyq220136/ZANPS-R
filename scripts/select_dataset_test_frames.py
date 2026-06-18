import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def frame_id_from_name(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def list_frame_ids(obj_dir: Path) -> List[str]:
    rgb_dir = obj_dir / "rgb"
    if rgb_dir.is_dir():
        frames = [
            frame_id_from_name(p.name)
            for p in rgb_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        return sorted(set(frames), key=natural_sort_key)

    frames = set()
    masks_root = obj_dir / "masks"
    if masks_root.is_dir():
        for part_dir in masks_root.iterdir():
            if not part_dir.is_dir():
                continue
            for p in part_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    frames.add(frame_id_from_name(p.name))
    return sorted(frames, key=natural_sort_key)


def first_part_pose_dir(obj_dir: Path) -> Optional[Path]:
    cam_root = obj_dir / "cam_params"
    if not cam_root.is_dir():
        return None
    part_dirs = sorted([p for p in cam_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name))
    return part_dirs[0] if part_dirs else None


def load_pose_direction(obj_dir: Path, frame_id: str) -> Optional[np.ndarray]:
    part_dir = first_part_pose_dir(obj_dir)
    if part_dir is None:
        return None
    pose_path = part_dir / f"{frame_id}.txt"
    if not pose_path.exists():
        return None
    try:
        pose = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    t = pose[:3, 3].astype(np.float64)
    norm = float(np.linalg.norm(t))
    if norm < 1e-9:
        return None
    return t / norm


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def select_diverse_frames(
    obj_dir: Path,
    frame_ids: List[str],
    num_frames: int,
    seed: int,
    candidate_pool: int,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    rng = random.Random(seed)
    dirs = {fid: load_pose_direction(obj_dir, fid) for fid in frame_ids}
    valid = [fid for fid in frame_ids if dirs[fid] is not None]
    missing = [fid for fid in frame_ids if dirs[fid] is None]
    if len(frame_ids) <= num_frames:
        return list(frame_ids), {k: v for k, v in dirs.items() if v is not None}
    if not valid:
        picked = sorted(rng.sample(frame_ids, num_frames), key=natural_sort_key)
        return picked, {}

    picked = [rng.choice(valid)]
    remaining = [f for f in valid if f not in picked]
    while remaining and len(picked) < min(num_frames, len(valid)):
        pool_size = min(max(1, int(candidate_pool)), len(remaining))
        pool = rng.sample(remaining, pool_size)
        best = max(
            pool,
            key=lambda fid: min(angular_distance(dirs[fid], dirs[p]) for p in picked),
        )
        picked.append(best)
        remaining.remove(best)

    if len(picked) < num_frames and missing:
        need = min(num_frames - len(picked), len(missing))
        picked.extend(rng.sample(missing, need))
    return sorted(picked, key=natural_sort_key), {k: v for k, v in dirs.items() if v is not None}


def remove_unselected_files(obj_dir: Path, keep: Set[str]) -> int:
    removed = 0
    frame_file_roots = [obj_dir / "rgb", obj_dir / "depth", obj_dir / "object_mask"]
    part_file_roots = [obj_dir / "masks", obj_dir / "cam_params"]

    for root in frame_file_roots:
        if not root.is_dir():
            continue
        for p in list(root.iterdir()):
            if p.is_file() and frame_id_from_name(p.name) not in keep:
                p.unlink()
                removed += 1

    for root in part_file_roots:
        if not root.is_dir():
            continue
        for part_dir in root.iterdir():
            if not part_dir.is_dir():
                continue
            for p in list(part_dir.iterdir()):
                if p.is_file() and frame_id_from_name(p.name) not in keep:
                    p.unlink()
                    removed += 1
    return removed


def summarize_angles(selected: List[str], dirs: Dict[str, np.ndarray]) -> Dict[str, float]:
    vals = []
    for i, a in enumerate(selected):
        if a not in dirs:
            continue
        for b in selected[i + 1 :]:
            if b in dirs:
                vals.append(angular_distance(dirs[a], dirs[b]))
    if not vals:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def build_parser():
    p = argparse.ArgumentParser("Select a diverse 50-frame subset in dataset_train/test objects.")
    p.add_argument("--test-root", type=str, default="dataset_train/test")
    p.add_argument("--num-frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--candidate-pool", type=int, default=24)
    p.add_argument("--objects", type=str, default="", help="Comma-separated object names; empty means all.")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--manifest-name", type=str, default="selected_50_frames.json")
    return p


def main():
    args = build_parser().parse_args()
    test_root = Path(args.test_root).resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"test root not found: {test_root}")
    keep_objects = {x.strip() for x in args.objects.split(",") if x.strip()}
    objects = sorted(
        [
            p
            for p in test_root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and (not keep_objects or p.name in keep_objects)
        ],
        key=lambda p: natural_sort_key(p.name),
    )
    print(
        f"[select] root={test_root} objects={len(objects)} num_frames={args.num_frames} "
        f"seed={args.seed} dry_run={args.dry_run}"
    )

    for obj_idx, obj_dir in enumerate(objects):
        frame_ids = list_frame_ids(obj_dir)
        seed = int(args.seed) + obj_idx * 9973
        selected, dirs = select_diverse_frames(
            obj_dir=obj_dir,
            frame_ids=frame_ids,
            num_frames=int(args.num_frames),
            seed=seed,
            candidate_pool=int(args.candidate_pool),
        )
        angle_stats = summarize_angles(selected, dirs)
        print(
            f"[object] {obj_dir.name}: total={len(frame_ids)} selected={len(selected)} "
            f"pose_valid={len(dirs)} angle_deg min/mean/max="
            f"{angle_stats['min']:.1f}/{angle_stats['mean']:.1f}/{angle_stats['max']:.1f}"
        )
        print(f"  selected: {','.join(selected[:10])}{' ...' if len(selected) > 10 else ''}")

        manifest = {
            "object": obj_dir.name,
            "test_root": str(test_root),
            "num_total_frames": int(len(frame_ids)),
            "num_selected_frames": int(len(selected)),
            "seed": seed,
            "dry_run": bool(args.dry_run),
            "selected_frames": selected,
            "removed_frames": [f for f in frame_ids if f not in set(selected)],
            "angle_deg": angle_stats,
        }
        if args.write_manifest:
            out_path = obj_dir / args.manifest_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"  manifest: {out_path}")

        if args.dry_run:
            print(f"  dry-run: would remove {len(manifest['removed_frames'])} frame ids")
        else:
            removed = remove_unselected_files(obj_dir, set(selected))
            print(f"  applied: removed_files={removed}")


if __name__ == "__main__":
    main()
