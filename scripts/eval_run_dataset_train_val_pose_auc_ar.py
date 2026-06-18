import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate run_dataset_train_val.py pose-est outputs with ADD/ADD-S AUC/AR, "
            "using dataset_train/val-compatible work directory layout."
        )
    )
    parser.add_argument("--work-root", type=str, default=str(REPO_ROOT / "dataset_train_val_work"))
    parser.add_argument("--source-root", type=str, default=str(REPO_ROOT / "dataset_train" / "val"))
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names; empty means all.")
    parser.add_argument("--object", type=str, default="", help="Single object alias.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end; <=0 means all remaining.")
    parser.add_argument("--pose-subdir", type=str, default="ob_in_cam2")
    parser.add_argument("--output-tag", type=str, default="", help="Alternative to --pose-subdir; ablation -> ob_in_cam2_ablation.")
    parser.add_argument("--gt-pose-root", type=str, default="", help="Default: <work-root>/gt_pose_from_ann.")
    parser.add_argument("--target-max-extent", type=float, default=0.6)
    parser.add_argument(
        "--model-scale-mode",
        choices=["auto", "target-max-extent", "meta", "none"],
        default="auto",
        help="dataset_train/val cam_params are rendered after object scale; default recovers that scale for ADD metrics.",
    )
    parser.add_argument("--max-model-points", type=int, default=20000)
    parser.add_argument("--te-unit-scale", type=float, default=100.0, help="Default converts meters to cm.")
    parser.add_argument("--output-summary", type=str, default="run_dataset_train_val_pose_auc_ar_summary.json")
    parser.add_argument("--output-detail", type=str, default="run_dataset_train_val_pose_auc_ar_detail.json")
    return parser.parse_args()


def object_category(obj_name: str) -> str:
    return str(obj_name).split("_", 1)[0]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pose(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        return pose.reshape(4, 4).astype(np.float64)
    except Exception:
        return None


def load_obj_vertices(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    pts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            toks = line.strip().split()
            if len(toks) >= 4:
                try:
                    pts.append([float(toks[1]), float(toks[2]), float(toks[3])])
                except ValueError:
                    pass
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float64)


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points > 0 and len(points) > int(max_points):
        rng = np.random.default_rng(12345)
        points = points[rng.choice(len(points), size=int(max_points), replace=False)]
    return points


def source_object_dir(work_obj_dir: Path, source_root: Path) -> Path:
    meta_path = work_obj_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            source_dir = str(meta.get("source_dir", "")).strip()
            if source_dir:
                p = Path(source_dir)
                if p.is_dir():
                    return p
        except Exception:
            pass
    return source_root / work_obj_dir.name


def scale_from_bbox(source_obj_dir: Path, target_max_extent: float) -> Optional[float]:
    bbox_path = source_obj_dir / "bounding_box.json"
    if not bbox_path.exists():
        return None
    try:
        data = load_json(bbox_path)
        mn = np.asarray(data.get("min", data.get("bbox_min")), dtype=np.float64)
        mx = np.asarray(data.get("max", data.get("bbox_max")), dtype=np.float64)
        if mn.shape != (3,) or mx.shape != (3,):
            return None
        extent = float(np.max(mx - mn))
        if extent <= 1e-12:
            return None
        return float(np.clip(float(target_max_extent) / extent, 0.25, 4.0))
    except Exception:
        return None


def scale_from_meta(work_obj_dir: Path) -> Optional[float]:
    meta_path = work_obj_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = load_json(meta_path)
        render = meta.get("render", {})
        val = render.get("object_scale", meta.get("object_scale", None))
        return float(val) if val is not None else None
    except Exception:
        return None


def resolve_model_scale(work_obj_dir: Path, source_root: Path, args: argparse.Namespace) -> Tuple[float, str]:
    if args.model_scale_mode == "none":
        return 1.0, "none"
    src_obj = source_object_dir(work_obj_dir, source_root)
    if args.model_scale_mode in ("auto", "target-max-extent"):
        val = scale_from_bbox(src_obj, args.target_max_extent)
        if val is not None:
            return float(val), f"target-max-extent:{args.target_max_extent}"
        if args.model_scale_mode == "target-max-extent":
            return 1.0, "target-max-extent-missing-bbox"
    if args.model_scale_mode in ("auto", "meta"):
        val = scale_from_meta(work_obj_dir)
        if val is not None:
            return float(val), "meta.render.object_scale"
    return 1.0, "fallback:1.0"


def load_part_names(work_obj_dir: Path) -> List[str]:
    manifest = work_obj_dir / "dataset_train_val_adapter_parts.json"
    if manifest.exists():
        try:
            data = load_json(manifest)
            pairs = []
            for item in data.get("parts", []):
                pairs.append((int(item["index"]), str(item["name"])))
            return [name for _, name in sorted(pairs)]
        except Exception:
            pass
    masks_dir = work_obj_dir / "masks"
    if masks_dir.is_dir():
        return sorted([p.name for p in masks_dir.iterdir() if p.is_dir()], key=natural_sort_key)
    models_dir = work_obj_dir / "models"
    if models_dir.is_dir():
        return sorted([p.name for p in models_dir.iterdir() if (p / "model.obj").exists()], key=natural_sort_key)
    return []


def model_path_for_part(work_obj_dir: Path, source_obj_dir_: Path, part_names: Sequence[str], part_idx: int) -> Path:
    candidates = []
    if 0 <= part_idx < len(part_names):
        candidates.append(work_obj_dir / "models" / part_names[part_idx] / "model.obj")
        candidates.append(source_obj_dir_ / "models" / part_names[part_idx] / "model.obj")
    for name in (f"link_{part_idx}", f"model_{part_idx:04d}", f"model_{part_idx}", str(part_idx)):
        candidates.append(work_obj_dir / "models" / name / "model.obj")
        candidates.append(source_obj_dir_ / "models" / name / "model.obj")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] if candidates else work_obj_dir / "models" / f"link_{part_idx}" / "model.obj"


def gt_pose_path(gt_root: Path, frame_id: str, part_idx: int) -> Path:
    parts_path = gt_root / f"{frame_id}__parts.txt"
    if parts_path.exists():
        lines = [x.strip() for x in parts_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if 0 <= part_idx < len(lines):
            p = Path(lines[part_idx])
            return p if p.is_absolute() else gt_root / p
    return gt_root / f"{frame_id}__link_{part_idx}.txt"


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (points @ pose[:3, :3].T) + pose[:3, 3]


def project_points(points_cam: np.ndarray, k: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
    if points_cam.size == 0 or not np.any(valid):
        return None
    pts = points_cam[valid]
    u = k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2]
    v = k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]
    return np.stack([u, v], axis=1)


def nearest_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(b).query(a, k=1, workers=-1)
        return float(np.mean(d))
    except Exception:
        chunk = 2048
        mins = []
        for i in range(0, len(a), chunk):
            diff = a[i : i + chunk, None, :] - b[None, :, :]
            mins.append(np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1))
        return float(np.mean(np.concatenate(mins))) if mins else math.inf


def rotation_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    r = pred[:3, :3] @ gt[:3, :3].T
    cos_theta = (float(np.trace(r)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def translation_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred[:3, 3] - gt[:3, 3]))


def calc_auc(errors: Sequence[float], max_threshold: float = 0.1) -> Optional[float]:
    vals = [float(e) for e in errors if np.isfinite(e)]
    if not vals:
        return None
    t = max(float(max_threshold), 1e-12)
    return float(np.mean([max(0.0, 1.0 - min(e, t) / t) for e in vals]))


def calc_recall(errors: Sequence[float], threshold: float) -> Optional[float]:
    vals = [float(e) for e in errors if np.isfinite(e)]
    if not vals:
        return None
    return float(np.mean([e < float(threshold) for e in vals]))


def ar_over_thresholds(errors: Sequence[float], thresholds: Sequence[float]) -> Optional[float]:
    recalls = [calc_recall(errors, t) for t in thresholds]
    recalls = [r for r in recalls if r is not None]
    return float(np.mean(recalls)) if recalls else None


def summarize_samples(samples: List[dict]) -> dict:
    ok = [s for s in samples if s.get("status") == "ok"]
    add = [float(s["add"]) for s in ok if s.get("add") is not None]
    adds = [float(s["adds"]) for s in ok if s.get("adds") is not None]
    add_norm = [float(s["add_norm"]) for s in ok if s.get("add_norm") is not None]
    adds_norm = [float(s["adds_norm"]) for s in ok if s.get("adds_norm") is not None]
    mssd_norm = [float(s["mssd_norm"]) for s in ok if s.get("mssd_norm") is not None]
    mspd = [float(s["mspd"]) for s in ok if s.get("mspd") is not None]
    ar_add = ar_over_thresholds(add_norm, np.arange(0.05, 0.51, 0.05))
    ar_adds = ar_over_thresholds(adds_norm, np.arange(0.05, 0.51, 0.05))
    ar_mssd = ar_over_thresholds(mssd_norm, np.arange(0.05, 0.51, 0.05))
    ar_mspd = ar_over_thresholds(mspd, np.arange(5.0, 51.0, 5.0))
    ar_bop_vals = [x for x in (ar_mssd, ar_mspd) if x is not None]
    return {
        "num_records": int(len(samples)),
        "num_ok": int(len(ok)),
        "mean_re": float(np.mean([s["re"] for s in ok])) if ok else None,
        "mean_te": float(np.mean([s["te"] for s in ok])) if ok else None,
        "mean_te_scaled": float(np.mean([s["te_scaled"] for s in ok])) if ok else None,
        "mean_add": float(np.mean(add)) if add else None,
        "mean_adds": float(np.mean(adds)) if adds else None,
        "auc_add": calc_auc(add_norm, 0.1),
        "auc_adds": calc_auc(adds_norm, 0.1),
        "add_0.1d": calc_recall(add_norm, 0.1),
        "adds_0.1d": calc_recall(adds_norm, 0.1),
        "ar_add": ar_add,
        "ar_adds": ar_adds,
        "ar_mssd": ar_mssd,
        "ar_mspd": ar_mspd,
        "ar_bop": float(np.mean(ar_bop_vals)) if ar_bop_vals else None,
        "ar": float(np.mean([x for x in (ar_add, ar_adds) if x is not None])) if (ar_add is not None or ar_adds is not None) else None,
    }


def collect_objects(work_root: Path, args: argparse.Namespace) -> List[str]:
    raw = args.object.strip() or args.objects.strip()
    if raw:
        names = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        names = sorted(
            [
                p.name
                for p in work_root.iterdir()
                if p.is_dir() and not p.name.startswith("_") and (p / "K.txt").exists()
            ],
            key=natural_sort_key,
        )
    start = max(0, int(args.start))
    end = int(args.end)
    return names[start:] if end <= 0 else names[start:end]


def eval_object(work_obj_dir: Path, source_root: Path, gt_root: Path, args: argparse.Namespace) -> dict:
    pose_subdir = f"ob_in_cam2_{args.output_tag}" if args.output_tag.strip() else args.pose_subdir
    pred_root = work_obj_dir / pose_subdir
    if not pred_root.is_dir():
        return {"object": work_obj_dir.name, "status": "missing_pred_pose_dir", "samples": [], **summarize_samples([])}

    part_names = load_part_names(work_obj_dir)
    source_obj = source_object_dir(work_obj_dir, source_root)
    model_scale, model_scale_source = resolve_model_scale(work_obj_dir, source_root, args)
    k = None
    k_path = work_obj_dir / "K.txt"
    if k_path.exists():
        try:
            k = np.loadtxt(k_path, dtype=np.float64).reshape(3, 3)
        except Exception:
            k = None
    model_cache: Dict[Tuple[str, float], Optional[dict]] = {}
    samples = []
    frame_dirs = sorted([p for p in pred_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name))
    for frame_dir in frame_dirs:
        for pose_path in sorted(frame_dir.glob("pose_*.txt"), key=lambda p: natural_sort_key(p.name)):
            m = re.search(r"pose_(\d+)\.txt$", pose_path.name)
            if not m:
                continue
            part_idx = int(m.group(1))
            gt_path = gt_pose_path(gt_root, frame_dir.name, part_idx)
            pred_pose = load_pose(pose_path)
            gt_pose = load_pose(gt_path)
            rec = {
                "object": work_obj_dir.name,
                "category": object_category(work_obj_dir.name),
                "frame_id": frame_dir.name,
                "part_idx": int(part_idx),
                "part_name": part_names[part_idx] if 0 <= part_idx < len(part_names) else "",
                "pred_pose_path": str(pose_path),
                "gt_pose_path": str(gt_path),
                "model_scale": float(model_scale),
                "model_scale_source": model_scale_source,
            }
            if pred_pose is None:
                rec["status"] = "missing_pred_pose"
                samples.append(rec)
                continue
            if gt_pose is None:
                rec["status"] = "missing_gt_pose"
                samples.append(rec)
                continue
            model_path = model_path_for_part(work_obj_dir, source_obj, part_names, part_idx)
            cache_key = (str(model_path), float(model_scale))
            if cache_key not in model_cache:
                pts = load_obj_vertices(model_path)
                if pts is not None:
                    pts = subsample_points(pts * float(model_scale), args.max_model_points)
                    diameter = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
                    model_cache[cache_key] = {"pts": pts, "diameter": max(diameter, 1e-12)}
                else:
                    model_cache[cache_key] = None
            model = model_cache[cache_key]
            if model is None:
                rec["status"] = "missing_model"
                rec["model_path"] = str(model_path)
                samples.append(rec)
                continue

            pts = model["pts"]
            pred_pts = transform_points(pred_pose, pts)
            gt_pts = transform_points(gt_pose, pts)
            point_dists = np.linalg.norm(pred_pts - gt_pts, axis=1)
            add = float(np.linalg.norm(pred_pts - gt_pts, axis=1).mean())
            adds = nearest_mean_distance(pred_pts, gt_pts)
            diameter = float(model["diameter"])
            mspd = None
            if k is not None:
                valid_z = (
                    np.isfinite(pred_pts[:, 2])
                    & np.isfinite(gt_pts[:, 2])
                    & (pred_pts[:, 2] > 1e-9)
                    & (gt_pts[:, 2] > 1e-9)
                )
                uv_pred = project_points(pred_pts, k, valid_z)
                uv_gt = project_points(gt_pts, k, valid_z)
                if uv_pred is not None and uv_gt is not None:
                    mspd = float(np.linalg.norm(uv_pred - uv_gt, axis=1).max())
            rec.update(
                {
                    "status": "ok",
                    "re": rotation_error_deg(pred_pose, gt_pose),
                    "te": translation_error(pred_pose, gt_pose),
                    "te_scaled": translation_error(pred_pose, gt_pose) * float(args.te_unit_scale),
                    "model_path": str(model_path),
                    "diameter": diameter,
                    "add": add,
                    "adds": adds,
                    "add_norm": add / diameter,
                    "adds_norm": adds / diameter,
                    "mssd": float(point_dists.max()),
                    "mssd_norm": float(point_dists.max() / diameter),
                    "mspd": mspd,
                }
            )
            samples.append(rec)

    return {
        "object": work_obj_dir.name,
        "category": object_category(work_obj_dir.name),
        "status": "ok",
        "pose_subdir": pose_subdir,
        "source_object_dir": str(source_obj),
        "model_scale": float(model_scale),
        "model_scale_source": model_scale_source,
        "samples": samples,
        **summarize_samples(samples),
    }


def group_summary(name_key: str, name: str, samples: List[dict], extra: Optional[dict] = None) -> dict:
    out = {name_key: name, **summarize_samples(samples)}
    if extra:
        out.update(extra)
    return out


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root)
    source_root = Path(args.source_root)
    gt_root = Path(args.gt_pose_root) if args.gt_pose_root.strip() else work_root / "gt_pose_from_ann"
    if not work_root.is_dir():
        raise FileNotFoundError(f"work root not found: {work_root}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"gt pose root not found: {gt_root}")

    object_names = collect_objects(work_root, args)
    per_object = [eval_object(work_root / obj_name, source_root, gt_root, args) for obj_name in object_names]
    all_samples = [s for obj in per_object for s in obj.get("samples", [])]
    per_category = []
    for cat in sorted({object_category(o) for o in object_names}, key=natural_sort_key):
        cat_samples = [s for s in all_samples if s.get("category") == cat]
        cat_objects = sorted([o for o in object_names if object_category(o) == cat], key=natural_sort_key)
        per_category.append(group_summary("category", cat, cat_samples, {"objects": cat_objects, "num_objects": len(cat_objects)}))

    summary = {
        "work_root": str(work_root),
        "source_root": str(source_root),
        "gt_pose_root": str(gt_root),
        "pose_subdir": f"ob_in_cam2_{args.output_tag}" if args.output_tag.strip() else args.pose_subdir,
        "target_max_extent": float(args.target_max_extent),
        "model_scale_mode": args.model_scale_mode,
        "te_unit_scale": float(args.te_unit_scale),
        "objects": object_names,
        "num_objects": int(len(object_names)),
        "total": summarize_samples(all_samples),
        "per_category": per_category,
        "per_object": [
            {k: v for k, v in obj.items() if k != "samples"}
            for obj in per_object
        ],
    }

    with (work_root / args.output_summary).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (work_root / args.output_detail).open("w", encoding="utf-8") as f:
        json.dump(per_object, f, ensure_ascii=False, indent=2)
    print(
        f"[Done] ok={summary['total']['num_ok']} records={summary['total']['num_records']} "
        f"summary={work_root / args.output_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
