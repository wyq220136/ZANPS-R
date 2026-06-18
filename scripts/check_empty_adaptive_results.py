import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def collect_objects(objs_root: Path, objects_arg: str, start: int, end: Optional[int]) -> List[Path]:
    objects = [
        p
        for p in objs_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name not in {"gt_pose", "gt_pose_from_ann", "bbox"}
    ]
    objects = sorted(objects, key=lambda p: natural_sort_key(p.name))
    if objects_arg.strip():
        keep = {x.strip() for x in objects_arg.split(",") if x.strip()}
        objects = [p for p in objects if p.name in keep]
    end_i = len(objects) if end is None else int(end)
    return objects[int(start):end_i]


def frame_ids_from_rgb(obj_dir: Path) -> List[str]:
    rgb_dir = obj_dir / "rgb"
    if not rgb_dir.is_dir():
        return []
    frames = []
    for p in rgb_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            frames.append(p.stem)
    return sorted(frames, key=natural_sort_key)


def mask_files(frame_dir: Path) -> List[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        [
            p
            for p in frame_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.stat().st_size > 0
        ],
        key=lambda p: natural_sort_key(p.name),
    )


def check_object(obj_dir: Path, adaptive_subdir: str) -> Dict[str, object]:
    frame_ids = frame_ids_from_rgb(obj_dir)
    adaptive_root = obj_dir / adaptive_subdir
    if not adaptive_root.is_dir():
        return {
            "object": obj_dir.name,
            "status": "missing_adaptive_dir",
            "adaptive_dir": str(adaptive_root),
            "num_frames": len(frame_ids),
            "num_nonempty_frames": 0,
            "empty_frames": frame_ids,
        }

    nonempty_frames = []
    empty_frames = []
    for frame_id in frame_ids:
        files = mask_files(adaptive_root / frame_id)
        if files:
            nonempty_frames.append(frame_id)
        else:
            empty_frames.append(frame_id)

    status = "ok" if not empty_frames and frame_ids else "empty_or_incomplete"
    if not frame_ids:
        status = "no_rgb_frames"
    return {
        "object": obj_dir.name,
        "status": status,
        "adaptive_dir": str(adaptive_root),
        "num_frames": len(frame_ids),
        "num_nonempty_frames": len(nonempty_frames),
        "num_empty_frames": len(empty_frames),
        "empty_frames": empty_frames,
    }


def resolve_split_root(root: Path, split: str) -> Path:
    if (root / "objs").is_dir():
        return root
    return root / split


def run_split(args: argparse.Namespace, split: str) -> Dict[str, object]:
    split_root = resolve_split_root(Path(args.root).resolve(), split)
    objs_root = split_root / "objs"
    if not objs_root.is_dir():
        raise FileNotFoundError(f"objs root not found: {objs_root}")

    objects = collect_objects(objs_root, args.objects, args.start, args.end)
    results = [check_object(obj_dir, args.adaptive_subdir) for obj_dir in objects]
    bad = [r for r in results if r["status"] != "ok"]
    return {
        "split": split,
        "split_root": str(split_root),
        "adaptive_subdir": args.adaptive_subdir,
        "num_objects": len(objects),
        "num_bad_objects": len(bad),
        "bad_objects": bad,
        "objects": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Check empty or missing adaptive result folders without modifying files.")
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--split", choices=["test_intra", "test_inter"], default="test_intra")
    p.add_argument("--all-splits", action="store_true")
    p.add_argument("--objects", type=str, default="")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--adaptive-subdir", type=str, default="matched_pred_mask_direct_match_adaptive")
    p.add_argument("--output-json", type=str, default="adaptive_empty_check.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["test_intra", "test_inter"] if args.all_splits else [args.split]
    summaries = [run_split(args, split) for split in splits]
    payload = {"root": str(Path(args.root).resolve()), "splits": summaries}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True) if out_path.parent != Path(".") else None
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    for summary in summaries:
        print(
            f"[{summary['split']}] objects={summary['num_objects']} "
            f"bad={summary['num_bad_objects']} adaptive_subdir={summary['adaptive_subdir']}"
        )
        for item in summary["bad_objects"]:
            print(
                f"  {item['object']}: {item['status']} "
                f"nonempty={item['num_nonempty_frames']}/{item['num_frames']}"
            )
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
