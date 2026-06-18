import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from recon_utils import (  # noqa: E402
    DatasetObject,
    backproject,
    list_objects,
    list_parts,
    load_depth_m,
    load_k,
    load_mask,
    load_pose,
    mask_path_for_part_frame,
    method_pose_ready_dir,
    model_obj_path,
    natural_sort_key,
    part_model_name,
    pose_path_for_part_frame,
)


def load_mesh(path: Path) -> Optional[trimesh.Trimesh]:
    if not path.exists():
        return None
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if len(g.vertices) and len(g.faces)]
        mesh = trimesh.util.concatenate(geoms) if geoms else None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)


def sample_surface(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, max(1, int(n)))
    return np.asarray(pts, dtype=np.float32)


def similarity_align_umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = min(len(src), len(dst))
    if n < 3:
        return np.eye(4, dtype=np.float32)
    if len(src) != n:
        src = src[np.linspace(0, len(src) - 1, n).astype(np.int64)]
    if len(dst) != n:
        dst = dst[np.linspace(0, len(dst) - 1, n).astype(np.int64)]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / float(n)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt
    var = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.sum(s) / max(var, 1e-12))
    t = mu_d - scale * (r @ mu_s)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = (scale * r).astype(np.float32)
    tf[:3, 3] = t.astype(np.float32)
    return tf


def apply_tf(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return (tf[:3, :3] @ points.T).T + tf[:3, 3]


def chamfer_and_fscore(cand: trimesh.Trimesh, gt: trimesh.Trimesh, samples: int, thresh: float) -> Dict[str, float]:
    cand_pts = sample_surface(cand, samples)
    gt_pts = sample_surface(gt, samples)
    tf = similarity_align_umeyama(cand_pts, gt_pts)
    cand_pts = apply_tf(cand_pts, tf)
    tree_gt = cKDTree(gt_pts)
    tree_c = cKDTree(cand_pts)
    d_c2g, _ = tree_gt.query(cand_pts, k=1, workers=-1)
    d_g2c, _ = tree_c.query(gt_pts, k=1, workers=-1)
    chamfer = float(np.mean(d_c2g) + np.mean(d_g2c))
    precision = float(np.mean(d_c2g <= thresh))
    recall = float(np.mean(d_g2c <= thresh))
    f = 0.0 if precision + recall <= 1e-12 else float(2 * precision * recall / (precision + recall))
    return {"chamfer": chamfer, "fscore": f, "precision": precision, "recall": recall}


def mesh_quality(mesh: trimesh.Trimesh) -> Dict[str, float]:
    comps = mesh.split(only_watertight=False)
    edges = mesh.edges_unique_length if len(mesh.edges_unique) else np.asarray([np.nan])
    area = mesh.area_faces
    degenerate = float(np.mean(area <= 1e-12)) if len(area) else math.nan
    return {
        "vertices": float(len(mesh.vertices)),
        "faces": float(len(mesh.faces)),
        "components": float(len(comps)),
        "edge_mean": float(np.nanmean(edges)),
        "edge_std": float(np.nanstd(edges)),
        "degenerate_ratio": degenerate,
    }


def visible_depth_proxy(obj: DatasetObject, mesh: trimesh.Trimesh, part_name: str, frame: str, pose_convention: str) -> Dict[str, float]:
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    depth_path = obj.depth_dir / f"{frame}.png"
    pose_path = pose_path_for_part_frame(obj, part_name, frame)
    if mask_path is None or (not depth_path.exists()) or pose_path is None:
        return {"visible_chamfer": math.nan, "observed_points": 0.0}
    k = load_k(obj)
    depth = load_depth_m(depth_path)
    mask = load_mask(mask_path, depth.shape[:2])
    obs_cam = backproject(depth, mask, k)
    if len(obs_cam) < 20:
        return {"visible_chamfer": math.nan, "observed_points": float(len(obs_cam))}
    ob_in_cam = load_pose(pose_path, pose_convention)
    mesh_pts_obj = sample_surface(mesh, min(5000, max(500, len(obs_cam))))
    mesh_pts_cam = apply_tf(mesh_pts_obj, ob_in_cam)
    valid = mesh_pts_cam[:, 2] > 1e-6
    mesh_pts_cam = mesh_pts_cam[valid]
    if len(mesh_pts_cam) < 20:
        return {"visible_chamfer": math.nan, "observed_points": float(len(obs_cam))}
    tree_obs = cKDTree(obs_cam)
    tree_mesh = cKDTree(mesh_pts_cam)
    d_m2o, _ = tree_obs.query(mesh_pts_cam, k=1, workers=-1)
    d_o2m, _ = tree_mesh.query(obs_cam, k=1, workers=-1)
    return {"visible_chamfer": float(np.mean(d_m2o) + np.mean(d_o2m)), "observed_points": float(len(obs_cam))}


def eval_one_method_object(args, method: str, obj_name: str) -> List[Dict[str, object]]:
    data_root = Path(args.data_root).resolve()
    work_root = Path(args.work_root).resolve()
    obj = DatasetObject(data_root=data_root, split=args.split, name=obj_name)
    rows = []
    parts = list_parts(obj)
    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        cand_path = model_obj_path(method_pose_ready_dir(work_root, method, args.split, obj.name), part_model)
        gt_path = model_obj_path(obj.gt_models_dir, part_name)
        row = {"method": method, "object": obj.name, "part": part_name, "candidate": str(cand_path), "gt": str(gt_path)}
        cand = load_mesh(cand_path)
        gt = load_mesh(gt_path)
        if cand is None:
            row["status"] = "missing_candidate"
            rows.append(row)
            continue
        if gt is None:
            row["status"] = "missing_gt"
            row.update(mesh_quality(cand))
            rows.append(row)
            continue
        try:
            row.update(chamfer_and_fscore(cand, gt, args.samples, args.fscore_thresh))
            row.update(mesh_quality(cand))
            frames = []
            part_dir = obj.masks_dir / part_name
            if part_dir.is_dir():
                frames = sorted([p.stem for p in part_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=natural_sort_key)
            if args.max_eval_frames > 0:
                frames = frames[: args.max_eval_frames]
            vis_vals = []
            for frame in frames[:: max(1, args.frame_stride)]:
                vp = visible_depth_proxy(obj, cand, part_name, frame, args.pose_convention)
                if np.isfinite(vp["visible_chamfer"]):
                    vis_vals.append(vp["visible_chamfer"])
            row["visible_chamfer"] = float(np.mean(vis_vals)) if vis_vals else math.nan
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
        rows.append(row)
    return rows


def write_tables(rows: List[Dict[str, object]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    detail = out_root / "per_part_metrics.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with detail.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    methods = sorted({str(r.get("method")) for r in rows})
    summary_rows = []
    for method in methods:
        mr = [r for r in rows if r.get("method") == method and r.get("status") == "ok"]
        item = {"method": method, "count": len(mr)}
        for key in ("chamfer", "fscore", "visible_chamfer", "components", "degenerate_ratio"):
            vals = [float(r[key]) for r in mr if key in r and np.isfinite(float(r[key]))]
            item[key] = float(np.mean(vals)) if vals else math.nan
        summary_rows.append(item)
    summary = out_root / "summary_table.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "count", "chamfer", "fscore", "visible_chamfer", "components", "degenerate_ratio"])
        writer.writeheader()
        writer.writerows(summary_rows)
    md = out_root / "summary_table.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("| method | count | chamfer↓ | fscore↑ | visible_chamfer↓ | components↓ | degenerate_ratio↓ |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['method']} | {r['count']} | {r['chamfer']:.6g} | {r['fscore']:.6g} | "
                f"{r['visible_chamfer']:.6g} | {r['components']:.6g} | {r['degenerate_ratio']:.6g} |\n"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate reconstruction methods and output comparison tables.")
    p.add_argument("--data-root", type=str, default="dataset_train")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--work-root", type=str, default="reconstruction_runs")
    p.add_argument("--methods", type=str, default="sam3d,sam3d_tsdf,sam3d_dmesh,hunyuan3d,hunyuan3d_tsdf,hunyuan3d_dmesh")
    p.add_argument("--objects", type=str, default="")
    p.add_argument("--output-root", type=str, default="reconstruction_eval")
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--fscore-thresh", type=float, default=0.005)
    p.add_argument("--max-eval-frames", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--pose-convention", choices=["cv", "sapien"], default="sapien")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    objects = [x.strip() for x in args.objects.split(",") if x.strip()] or list_objects(data_root, args.split, "all", "")
    rows = []
    for method in methods:
        for obj_name in objects:
            rows.extend(eval_one_method_object(args, method, obj_name))
    write_tables(rows, Path(args.output_root).resolve())
    print(f"[Done] wrote {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/eval/eval_all_methods.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --output-root reconstruction_eval
#   python reconstruction/eval/eval_all_methods.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --methods sam3d,sam3d_tsdf,hunyuan3d,hunyuan3d_tsdf --output-root /shared/recon_eval
#
# Key parameters:
#   --data-root: dataset root with split/object folders. Each object should contain Models, masks, object_mask/objectmask.
#   --work-root: shared reconstruction output root containing method folders.
#   --methods: comma-separated method names to compare.
#   --samples: surface samples per mesh for Chamfer/F-score.
#   --fscore-thresh: metric threshold for F-score, e.g. 0.005 means 5mm.
#   --max-eval-frames: max frames per part for the visible-depth proxy; 0 evaluates all frames.
