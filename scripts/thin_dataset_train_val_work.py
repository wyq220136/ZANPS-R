import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2


SKIP_OBJECT_CHILD_DIRS = {
    "models",
    "_heldout_frames",
    "__pycache__",
}


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def parse_frame_index(frame_id: str, obj_name: str) -> int:
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix):]
        m = re.fullmatch(r"\d+_(\d+)", tail)
        if m:
            return int(m.group(1))
    nums = re.findall(r"\d+", frame_id)
    return int(nums[-1]) if nums else 0


def source_frame_stem(frame_id: str, obj_name: str) -> str:
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix):]
        if "_" in tail:
            return tail.split("_", 1)[1]
    return Path(frame_id).stem


def frame_ids(obj_dir: Path) -> List[str]:
    mask_root = obj_dir / "gt_mask"
    if not mask_root.is_dir():
        return []
    return sorted([p.name for p in mask_root.iterdir() if p.is_dir()], key=natural_sort_key)


def frame_visible_part_count(obj_dir: Path, frame_id: str, min_mask_pixels: int) -> int:
    mask_dir = obj_dir / "gt_mask" / frame_id
    if not mask_dir.is_dir():
        return 0
    count = 0
    for p in mask_dir.iterdir():
        if not p.name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is not None and int((mask > 0).sum()) >= int(min_mask_pixels):
            count += 1
    return count


def select_reference_frame(obj_dir: Path, frames: Sequence[str], min_mask_pixels: int) -> str:
    obj_name = obj_dir.name
    if not frames:
        return ""
    ranked = []
    for fid in frames:
        ranked.append(
            (
                -frame_visible_part_count(obj_dir, fid, min_mask_pixels=min_mask_pixels),
                parse_frame_index(fid, obj_name),
                fid,
            )
        )
    ranked.sort(key=lambda x: (x[0], x[1], natural_sort_key(x[2])))
    return ranked[0][2]


def choose_keep_frames(
    obj_dir: Path,
    target_frames: int,
    stride: int,
    min_mask_pixels: int,
) -> Tuple[List[str], str]:
    frames = frame_ids(obj_dir)
    if len(frames) <= target_frames:
        ref = select_reference_frame(obj_dir, frames, min_mask_pixels=min_mask_pixels)
        return frames, ref

    ref = select_reference_frame(obj_dir, frames, min_mask_pixels=min_mask_pixels)
    keep: List[str] = []
    seen: Set[str] = set()
    if ref:
        keep.append(ref)
        seen.add(ref)

    for fid in frames[:: max(1, int(stride))]:
        if fid in seen:
            continue
        keep.append(fid)
        seen.add(fid)
        if len(keep) >= int(target_frames):
            return sorted(keep, key=natural_sort_key), ref

    for fid in frames:
        if fid in seen:
            continue
        keep.append(fid)
        seen.add(fid)
        if len(keep) >= int(target_frames):
            break

    return sorted(keep, key=natural_sort_key), ref


def safe_move(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def move_frame_entries_in_dir(
    child_dir: Path,
    heldout_child_dir: Path,
    remove_frames: Set[str],
    dry_run: bool,
) -> int:
    moved = 0
    if not child_dir.is_dir():
        return moved
    for item in sorted(child_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
        if item.name in remove_frames:
            moved += int(safe_move(item, heldout_child_dir / item.name, dry_run=dry_run))
            continue
        if item.is_file() and item.stem in remove_frames:
            moved += int(safe_move(item, heldout_child_dir / item.name, dry_run=dry_run))
    return moved


def materialize_symlink_dir(path: Path, dry_run: bool) -> bool:
    if not path.is_symlink():
        return False
    if dry_run:
        return True
    src = Path(os.readlink(path))
    if not src.is_absolute():
        src = (path.parent / src).resolve()
    tmp = path.with_name(f"{path.name}.tmp_materialized")
    if tmp.exists():
        raise FileExistsError(f"temporary materialization path already exists: {tmp}")
    shutil.copytree(src, tmp, dirs_exist_ok=False)
    path.unlink()
    tmp.rename(path)
    return True


def move_cam_params_entries(
    cam_params_dir: Path,
    heldout_cam_params_dir: Path,
    remove_frames: Set[str],
    obj_name: str,
    dry_run: bool,
) -> int:
    moved = 0
    if not cam_params_dir.is_dir():
        return moved

    materialize_symlink_dir(cam_params_dir, dry_run=dry_run)
    remove_stems = {source_frame_stem(fid, obj_name) for fid in remove_frames}
    remove_names = set(remove_frames) | remove_stems

    for part_dir in sorted(cam_params_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
        if not part_dir.is_dir():
            if part_dir.is_file() and part_dir.stem in remove_names:
                moved += int(safe_move(part_dir, heldout_cam_params_dir / part_dir.name, dry_run=dry_run))
            continue
        for item in sorted(part_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
            if item.name in remove_names or (item.is_file() and item.stem in remove_names):
                moved += int(
                    safe_move(
                        item,
                        heldout_cam_params_dir / part_dir.name / item.name,
                        dry_run=dry_run,
                    )
                )
    return moved


def process_object(obj_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    frames = frame_ids(obj_dir)
    keep, ref = choose_keep_frames(
        obj_dir,
        target_frames=args.target_frames,
        stride=args.stride,
        min_mask_pixels=args.min_mask_pixels,
    )
    keep_set = set(keep)
    remove_set = set(frames) - keep_set
    heldout_root = obj_dir / args.heldout_dir_name
    moved = 0

    if remove_set:
        for child in sorted(obj_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
            if not child.is_dir():
                continue
            if child.name in SKIP_OBJECT_CHILD_DIRS or child.name == args.heldout_dir_name:
                continue
            if child.name == "cam_params":
                moved += move_cam_params_entries(
                    child,
                    heldout_root / child.name,
                    remove_frames=remove_set,
                    obj_name=obj_dir.name,
                    dry_run=args.dry_run,
                )
                continue
            moved += move_frame_entries_in_dir(
                child,
                heldout_root / child.name,
                remove_frames=remove_set,
                dry_run=args.dry_run,
            )

    summary = {
        "object": obj_dir.name,
        "total_frames_before": int(len(frames)),
        "target_frames": int(args.target_frames),
        "kept_frames": keep,
        "kept_count": int(len(keep)),
        "removed_count": int(len(remove_set)),
        "reference_frame": ref,
        "moved_entries": int(moved),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        heldout_root.mkdir(parents=True, exist_ok=True)
        with (heldout_root / "thin_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def collect_objects(work_root: Path, objects_arg: str) -> List[Path]:
    names = sorted(
        [
            p.name
            for p in work_root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name not in {"gt_pose_from_ann"}
        ],
        key=natural_sort_key,
    )
    if objects_arg.strip():
        wanted = {x.strip() for x in objects_arg.split(",") if x.strip()}
        names = [n for n in names if n in wanted]
    return [work_root / n for n in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin dataset_train_val_work after reconstruction by moving non-evaluation "
            "frames out of each object directory while preserving reconstructed models."
        )
    )
    parser.add_argument("--work-root", type=str, default="dataset_train_val_work")
    parser.add_argument("--objects", type=str, default="", help="Comma-separated objects; empty means all.")
    parser.add_argument("--target-frames", type=int, default=25)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--heldout-dir-name", type=str, default="_heldout_frames")
    parser.add_argument("--write-root-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root).resolve()
    if not work_root.is_dir():
        raise FileNotFoundError(f"work root not found: {work_root}")
    objects = collect_objects(work_root, args.objects)
    summaries = []
    for obj_dir in objects:
        summary = process_object(obj_dir, args)
        summaries.append(summary)
        print(
            "[thin] {object}: before={total_frames_before} keep={kept_count} "
            "remove={removed_count} ref={reference_frame} moved={moved_entries}".format(**summary)
        )

    if args.write_root_summary and not args.dry_run:
        out_path = work_root / "thin_dataset_train_val_work_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"work_root": str(work_root), "objects": summaries}, f, ensure_ascii=False, indent=2)
        print(f"[done] wrote {out_path}")
    elif args.dry_run:
        print("[dry-run] no files or directories were moved/created")


if __name__ == "__main__":
    main()
