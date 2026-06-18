import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def natural_sort_key(s: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect data/<split>/depth_bg against object depth and masks. "
            "Useful for checking whether background-depth can filter depth noise/leakage."
        )
    )
    parser.add_argument("--data-root", type=str, default=r"D:\research\PartNet\data")
    parser.add_argument("--split", type=str, default="test_inter")
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--parts", type=str, default="", help="Part ids or names. Empty means all available masks.")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument(
        "--bg-margin",
        type=float,
        default=0.01,
        help="Meters. Masked depth within this margin of depth_bg is counted as background leakage.",
    )
    parser.add_argument(
        "--fg-min-delta",
        type=float,
        default=0.003,
        help="Meters. Clean foreground keeps depth < depth_bg - fg_min_delta when depth_bg exists.",
    )
    parser.add_argument("--output-root", type=str, default="scripts/depth_bg_check")
    parser.add_argument("--save-cleaned-depth", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def load_npz_depth(path: Path) -> np.ndarray:
    z = np.load(path)
    key = "depth_map" if "depth_map" in z.files else z.files[0]
    depth = z[key].astype(np.float32)
    return depth


def load_png_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"failed to read depth: {path}")
    depth = depth.astype(np.float32)
    if depth.max() > 50:
        depth /= float(depth_scale)
    return depth


def load_mask(path: Path, threshold: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    return mask > int(threshold)


def list_frames(object_dir: Path, split_root: Path, max_frames: int, frame_stride: int) -> List[str]:
    candidates = []
    depth_dir = object_dir / "depth"
    for p in depth_dir.glob("*.png"):
        if (split_root / "depth_bg" / f"{p.stem}.npz").exists():
            candidates.append(p.stem)
    frames = sorted(candidates, key=natural_sort_key)[:: max(1, int(frame_stride))]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def list_part_masks(object_dir: Path, frame: str, parts: str) -> Dict[str, Path]:
    mask_root = object_dir / "gt_mask" / frame
    if not mask_root.is_dir():
        mask_root = object_dir / "masks"
    found = {}
    if (object_dir / "gt_mask" / frame).is_dir():
        for p in sorted(mask_root.iterdir(), key=lambda x: natural_sort_key(x.name)):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                name = p.stem
                if name.startswith("mask_"):
                    name = name[5:]
                found[name] = p
    elif mask_root.is_dir():
        for part_dir in sorted([p for p in mask_root.iterdir() if p.is_dir()], key=lambda x: natural_sort_key(x.name)):
            for ext in (".png", ".jpg", ".jpeg"):
                p = part_dir / f"{frame}{ext}"
                if p.exists():
                    found[part_dir.name] = p
                    break

    if parts.strip():
        wanted = {x.strip() for x in parts.split(",") if x.strip()}
        found = {k: v for k, v in found.items() if k in wanted or f"mask_{k}" in wanted}
    return found


def colorize_depth(depth: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    if valid is None:
        valid = depth > 0
    out = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        out[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_TURBO)


def write_vis(out_path: Path, depth_obj: np.ndarray, depth_bg: np.ndarray, mask: np.ndarray, clean_mask: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj_vis = colorize_depth(depth_obj, depth_obj > 0)
    bg_vis = colorize_depth(depth_bg, depth_bg > 0)
    diff = np.zeros_like(depth_obj, dtype=np.float32)
    valid = (depth_obj > 0) & (depth_bg > 0)
    diff[valid] = depth_bg[valid] - depth_obj[valid]
    diff_vis = colorize_depth(np.clip(diff, 0, np.percentile(diff[valid], 98) if np.any(valid) else 1.0), valid)
    mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_vis[mask] = (0, 0, 255)
    mask_vis[clean_mask] = (0, 255, 0)
    canvas = np.concatenate([obj_vis, bg_vis, diff_vis, mask_vis], axis=1)
    cv2.imwrite(str(out_path), canvas)


def analyze_frame_part(
    object_dir: Path,
    split_root: Path,
    frame: str,
    part_name: str,
    mask_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, object]:
    depth_obj = load_png_depth_m(object_dir / "depth" / f"{frame}.png", args.depth_scale)
    depth_bg = load_npz_depth(split_root / "depth_bg" / f"{frame}.npz")
    mask = load_mask(mask_path, args.mask_threshold)
    if depth_bg.shape != depth_obj.shape:
        depth_bg = cv2.resize(depth_bg, (depth_obj.shape[1], depth_obj.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid_mask_depth = mask & (depth_obj > 0)
    valid_bg = valid_mask_depth & (depth_bg > 0)
    delta = np.full(depth_obj.shape, np.nan, dtype=np.float32)
    delta[valid_bg] = depth_bg[valid_bg] - depth_obj[valid_bg]

    likely_bg = valid_bg & (np.abs(delta) <= float(args.bg_margin))
    behind_bg = valid_bg & (depth_obj >= depth_bg - float(args.fg_min_delta))
    clean_mask = valid_mask_depth & (~behind_bg)

    stats = {
        "frame": frame,
        "part": part_name,
        "mask_path": str(mask_path),
        "depth_path": str(object_dir / "depth" / f"{frame}.png"),
        "depth_bg_path": str(split_root / "depth_bg" / f"{frame}.npz"),
        "mask_pixels": int(np.count_nonzero(mask)),
        "valid_mask_depth": int(np.count_nonzero(valid_mask_depth)),
        "valid_bg_overlap": int(np.count_nonzero(valid_bg)),
        "likely_background_pixels_abs_delta": int(np.count_nonzero(likely_bg)),
        "would_remove_pixels": int(np.count_nonzero(behind_bg)),
        "clean_pixels": int(np.count_nonzero(clean_mask)),
    }
    if np.any(valid_bg):
        d = delta[valid_bg]
        stats.update(
            {
                "delta_bg_minus_obj_m": {
                    "min": float(np.nanmin(d)),
                    "p01": float(np.nanpercentile(d, 1)),
                    "p05": float(np.nanpercentile(d, 5)),
                    "median": float(np.nanmedian(d)),
                    "p95": float(np.nanpercentile(d, 95)),
                    "max": float(np.nanmax(d)),
                }
            }
        )
    if args.save_cleaned_depth:
        cleaned = depth_obj.copy()
        cleaned[mask & (~clean_mask)] = 0.0
        cleaned_mm = np.clip(cleaned * float(args.depth_scale), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        clean_path = out_dir / "cleaned_depth" / frame / f"{part_name}.png"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(clean_path), cleaned_mm)
        stats["cleaned_depth_path"] = str(clean_path)
    if args.save_vis:
        vis_path = out_dir / "vis" / f"{frame}_{part_name}.png"
        write_vis(vis_path, depth_obj, depth_bg, mask, clean_mask)
        stats["vis_path"] = str(vis_path)
    return stats


def main() -> None:
    args = parse_args()
    split_root = Path(args.data_root) / args.split
    object_dir = split_root / "objs" / args.object
    if not object_dir.is_dir():
        raise FileNotFoundError(f"object dir not found: {object_dir}")
    out_dir = Path(args.output_root) / args.split / args.object
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frames(object_dir, split_root, args.max_frames, args.frame_stride)
    if not frames:
        raise RuntimeError(f"no frames with depth_bg found for {object_dir}")

    rows = []
    for frame in frames:
        masks = list_part_masks(object_dir, frame, args.parts)
        for part_name, mask_path in masks.items():
            try:
                rows.append(analyze_frame_part(object_dir, split_root, frame, part_name, mask_path, args, out_dir))
            except FileNotFoundError as e:
                print(f"[SKIP] {e}", flush=True)

    summary = {
        "object": args.object,
        "split_root": str(split_root),
        "bg_margin": args.bg_margin,
        "fg_min_delta": args.fg_min_delta,
        "frames": len(frames),
        "records": rows,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Done] wrote {summary_path}")
    if rows:
        total_valid = sum(r["valid_mask_depth"] for r in rows)
        total_removed = sum(r["would_remove_pixels"] for r in rows)
        print(
            f"[Stats] records={len(rows)} valid_mask_depth={total_valid} "
            f"would_remove={total_removed} ratio={total_removed / max(total_valid, 1):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
