import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import trimesh

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
REPO_ROOT = RECON_ROOT.parent
DEFAULT_BOP_ROOT = REPO_ROOT / "related_works" / "bop_toolkit"
for _p in (RECON_ROOT, TOOLS_ROOT, DEFAULT_BOP_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from recon_utils import DatasetObject, list_objects, list_parts, load_k, load_pose, model_obj_path, natural_sort_key, part_model_name  # noqa: E402

PIPELINE_METHODS = [
    "sam3d",
    "sam3d_tsdf",
    "sam3d_tsdf_dmesh",
    "sam3d_partcut_tsdf_dmesh",
    "hunyuan3d",
    "hunyuan3d_tsdf",
    "hunyuan3d_tsdf_dmesh",
    "hunyuan3d_partcut_tsdf_dmesh",
    "instantmesh",
    "instantmesh_tsdf",
    "instantmesh_tsdf_dmesh",
    "instantmesh_partcut_tsdf_dmesh",
]


def default_pose_root(work_root: str | Path) -> Path:
    work_root = Path(work_root).resolve()
    return work_root.parent / "reconstruction_pose_est"


def _import_bop_pose_error(bop_root: str | Path):
    bop_root = Path(bop_root).resolve()
    if str(bop_root) not in sys.path:
        sys.path.insert(0, str(bop_root))
    from bop_toolkit_lib import pose_error  # noqa: E402

    return pose_error


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_mesh_points(path: Path, samples: int, seed: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            return None
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    if len(mesh.faces) > 0:
        np.random.seed(int(seed))
        pts, _ = trimesh.sample.sample_surface(mesh, max(1, int(samples)))
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    return np.asarray(pts, dtype=np.float64)


def _diameter(pts: np.ndarray) -> float:
    if pts is None or len(pts) == 0:
        return 0.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _auc(errors: List[float], max_threshold: float) -> float:
    vals = np.asarray([e for e in errors if np.isfinite(e)], dtype=np.float64)
    if vals.size == 0 or max_threshold <= 0:
        return 0.0
    vals = np.minimum(vals, max_threshold)
    thresholds = np.linspace(0.0, max_threshold, 101)
    recalls = np.asarray([np.mean(vals <= t) for t in thresholds], dtype=np.float64)
    return float(np.trapz(recalls, thresholds) / max_threshold)


def _mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def _category_name(obj_name: str) -> str:
    return str(obj_name).split("_")[0]


def _gt_model_path(obj: DatasetObject, part_name: str) -> Path:
    return model_obj_path(obj.gt_models_dir, part_name)


def _build_gt_point_cache(args, objects: List[str]) -> Dict[str, Dict[str, object]]:
    cache = {}
    data_root = Path(args.data_root).resolve()
    for obj_name in objects:
        obj = DatasetObject(data_root=data_root, split=args.split, name=obj_name)
        for part_idx, part_name in enumerate(list_parts(obj)):
            key = f"{obj_name}/{part_name}"
            path = _gt_model_path(obj, part_name)
            pts = _load_mesh_points(path, int(args.model_samples), seed=part_idx + 123)
            if pts is None:
                cache[key] = {"path": str(path), "pts": None, "diameter": 0.0}
            else:
                cache[key] = {"path": str(path), "pts": pts, "diameter": _diameter(pts)}
    return cache


def _eval_row(row: Dict[str, object], args, pose_error, gt_cache: Dict[str, Dict[str, object]]) -> Optional[Dict[str, object]]:
    if row.get("status") != "ok":
        return None
    obj_name = str(row["object"])
    part_name = str(row["part"])
    frame = str(row["frame"])
    gt_key = f"{obj_name}/{part_name}"
    gt_info = gt_cache.get(gt_key)
    if not gt_info or gt_info.get("pts") is None:
        return None
    gt_pose_path = row.get("gt_pose_path")
    if not gt_pose_path:
        obj = DatasetObject(data_root=Path(args.data_root).resolve(), split=args.split, name=obj_name)
        from recon_utils import pose_path_for_part_frame  # local to avoid polluting imports

        p = pose_path_for_part_frame(obj, part_name, frame)
        gt_pose_path = None if p is None else str(p)
    if not gt_pose_path:
        return None

    est = np.asarray(row["pose"], dtype=np.float64).reshape(4, 4)
    gt = load_pose(Path(gt_pose_path), args.pose_convention).astype(np.float64)
    pts = np.asarray(gt_info["pts"], dtype=np.float64)
    diameter = float(gt_info["diameter"])
    R_est, t_est = est[:3, :3], est[:3, 3]
    R_gt, t_gt = gt[:3, :3], gt[:3, 3]
    syms = [{"R": np.eye(3, dtype=np.float64), "t": np.zeros(3, dtype=np.float64)}]

    add = float(pose_error.add(R_est, t_est, R_gt, t_gt, pts))
    adds = float(pose_error.adi(R_est, t_est, R_gt, t_gt, pts))
    re = float(pose_error.re(R_est, R_gt))
    te = float(pose_error.te(t_est, t_gt))
    mssd = float(pose_error.mssd(R_est, t_est, R_gt, t_gt, pts, syms))
    try:
        obj_for_k = DatasetObject(data_root=Path(args.data_root).resolve(), split=args.split, name=obj_name)
        k = load_k(obj_for_k).astype(np.float64)
        mspd = float(pose_error.mspd(R_est, t_est, R_gt, t_gt, k, pts, syms))
    except Exception:
        mspd = float("nan")
    norm_add = add / max(diameter, 1e-9)
    norm_adds = adds / max(diameter, 1e-9)
    norm_mssd = mssd / max(diameter, 1e-9)
    return {
        "object": obj_name,
        "category": _category_name(obj_name),
        "part": part_name,
        "frame": frame,
        "add": add,
        "adds": adds,
        "add_norm": norm_add,
        "adds_norm": norm_adds,
        "mssd": mssd,
        "mssd_norm": norm_mssd,
        "mspd": mspd,
        "re": re,
        "te": te,
        "diameter": diameter,
        "acc_5deg_2cm": bool(re <= 5.0 and te <= 0.02),
        "acc_10deg_5cm": bool(re <= 10.0 and te <= 0.05),
    }


def _aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "count": 0,
            "auc_add": 0.0,
            "auc_adds": 0.0,
            "ar_bop": 0.0,
            "re": None,
            "te": None,
            "acc_5deg_2cm": 0.0,
            "acc_10deg_5cm": 0.0,
        }
    add_norm = [r["add_norm"] for r in rows]
    adds_norm = [r["adds_norm"] for r in rows]
    mssd_norm = [r["mssd_norm"] for r in rows]
    mspd = [r["mspd"] for r in rows if np.isfinite(float(r["mspd"]))]
    bop_mssd_thresholds = np.arange(0.05, 0.55, 0.05)
    bop_mspd_thresholds = np.arange(5.0, 55.0, 5.0)
    ar_mssd = float(np.mean([np.mean(np.asarray(mssd_norm) <= t) for t in bop_mssd_thresholds]))
    ar_mspd = float(np.mean([np.mean(np.asarray(mspd) <= t) for t in bop_mspd_thresholds])) if mspd else 0.0
    return {
        "count": int(len(rows)),
        "auc_add": _auc(add_norm, 0.1),
        "auc_adds": _auc(adds_norm, 0.1),
        "ar_bop": float((ar_mssd + ar_mspd) * 0.5),
        "ar_bop_mssd": ar_mssd,
        "ar_bop_mspd": ar_mspd,
        "ar_bop_note": "BOP-style AR averaged over MSSD normalized thresholds 0.05..0.50 and MSPD pixel thresholds 5..50; VSD is not included because no renderer is required here.",
        "re": _mean(r["re"] for r in rows),
        "te": _mean(r["te"] for r in rows),
        "add": _mean(r["add"] for r in rows),
        "adds": _mean(r["adds"] for r in rows),
        "mssd": _mean(r["mssd"] for r in rows),
        "mspd": _mean(r["mspd"] for r in rows),
        "acc_5deg_2cm": float(np.mean([bool(r["acc_5deg_2cm"]) for r in rows])),
        "acc_10deg_5cm": float(np.mean([bool(r["acc_10deg_5cm"]) for r in rows])),
    }


def evaluate_method(method: str, args, pose_error, gt_cache: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    pose_root = Path(args.pose_root).resolve() if str(args.pose_root).strip() else default_pose_root(args.work_root)
    pose_file = pose_root / method / "poses.jsonl"
    eval_rows = []
    total_rows = 0
    for row in _iter_jsonl(pose_file) or []:
        total_rows += 1
        item = _eval_row(row, args, pose_error, gt_cache)
        if item is not None:
            eval_rows.append(item)

    by_category = {}
    for cat in sorted({r["category"] for r in eval_rows}, key=natural_sort_key):
        by_category[cat] = _aggregate([r for r in eval_rows if r["category"] == cat])
    by_object = {}
    for obj in sorted({r["object"] for r in eval_rows}, key=natural_sort_key):
        by_object[obj] = _aggregate([r for r in eval_rows if r["object"] == obj])
    return {
        "method": method,
        "pose_file": str(pose_file),
        "total_pose_rows": int(total_rows),
        "evaluated_rows": int(len(eval_rows)),
        "overall": _aggregate(eval_rows),
        "by_category": by_category,
        "by_object": by_object,
        "details": eval_rows if bool(args.include_details) else [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate FoundationPose outputs for reconstruction pipelines.")
    p.add_argument("--data-root", type=str, default="dataset_train")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--work-root", type=str, default="reconstruction_runs")
    p.add_argument("--pose-root", type=str, default="")
    p.add_argument("--eval-root", type=str, default="", help="Default: sibling of work-root named reconstruction_pose_eval.")
    p.add_argument("--bop-toolkit-root", type=str, default=str(DEFAULT_BOP_ROOT))
    p.add_argument("--methods", type=str, default=",".join(PIPELINE_METHODS))
    p.add_argument("--objects", type=str, default="")
    p.add_argument("--model-samples", type=int, default=10000)
    p.add_argument("--pose-convention", choices=["cv", "sapien"], default="sapien")
    p.add_argument("--include-details", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pose_error = _import_bop_pose_error(args.bop_toolkit_root)
    data_root = Path(args.data_root).resolve()
    work_root = Path(args.work_root).resolve()
    eval_root = Path(args.eval_root).resolve() if str(args.eval_root).strip() else work_root.parent / "reconstruction_pose_eval"
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    objects = [x.strip() for x in args.objects.split(",") if x.strip()] or list_objects(data_root, args.split, "all", "")
    gt_cache = _build_gt_point_cache(args, objects)

    eval_root.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for method in methods:
        result = evaluate_method(method, args, pose_error, gt_cache)
        out_path = eval_root / f"{method}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        all_summary[method] = {
            "overall": result["overall"],
            "evaluated_rows": result["evaluated_rows"],
            "json": str(out_path),
        }
        print(f"[eval] method={method} rows={result['evaluated_rows']} json={out_path}")
    with (eval_root / "summary_all_methods.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(f"[eval] wrote {eval_root}")


if __name__ == "__main__":
    main()
