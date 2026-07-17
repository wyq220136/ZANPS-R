import argparse
import contextlib
import json
import logging
import os
import queue
import sys
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
REPO_ROOT = RECON_ROOT.parent
REF_POSE_ROOT = REPO_ROOT / "ref_pose"
for _p in (RECON_ROOT, TOOLS_ROOT, REF_POSE_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from recon_utils import (  # noqa: E402
    DatasetObject,
    find_image,
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


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            raise RuntimeError(f"empty mesh scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _mesh_diameter(mesh: trimesh.Trimesh) -> float:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if len(verts) == 0:
        return 0.0
    return float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))


def _import_foundationpose():
    from foundationpose import FoundationPose  # noqa: E402
    from ref_pose.ablation.predict_pose_refine_original import OriginalPoseRefinePredictor  # noqa: E402

    try:
        import nvdiffrast.torch as dr  # noqa: E402
    except Exception:
        dr = None
    return FoundationPose, OriginalPoseRefinePredictor, dr


def _build_estimator(mesh: trimesh.Trimesh, args: argparse.Namespace):
    with _quiet_foundationpose_io(args):
        FoundationPose, OriginalPoseRefinePredictor, dr = _import_foundationpose()
        glctx = None
        if bool(args.use_nvdiffrast) and dr is not None:
            glctx = dr.RasterizeCudaContext()
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if normals.shape != np.asarray(mesh.vertices).shape:
            normals = np.zeros_like(np.asarray(mesh.vertices, dtype=np.float32))
        return FoundationPose(
            model_pts=np.asarray(mesh.vertices, dtype=np.float32),
            model_normals=normals,
            mesh=mesh,
            glctx=glctx,
            debug=int(args.debug),
            debug_dir=str(args.debug_dir),
            load_render_models=bool(args.use_nvdiffrast),
            refiner=OriginalPoseRefinePredictor() if bool(args.use_nvdiffrast) else None,
        )


@contextlib.contextmanager
def _quiet_foundationpose_io(args: argparse.Namespace):
    if not bool(getattr(args, "quiet_foundationpose", True)):
        yield
        return
    prev_disable = logging.root.manager.disable
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            logging.disable(logging.INFO)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    finally:
        logging.disable(prev_disable)


def _progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "progress", True)) and (tqdm is not None)


def _worker_position(worker_label: str) -> int:
    if not worker_label:
        return 0
    try:
        return int(str(worker_label).split("_")[-1])
    except Exception:
        return 0


def _progress_write(msg: str, args: argparse.Namespace) -> None:
    if _progress_enabled(args):
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _progress_message(args: argparse.Namespace, progress_queue, text: str) -> None:
    if progress_queue is not None:
        progress_queue.put({"kind": "message", "text": str(text)})
    else:
        _progress_write(text, args)


def _progress_frame(args: argparse.Namespace, progress_queue, row: Dict[str, object]) -> None:
    if progress_queue is not None:
        progress_queue.put(
            {
                "kind": "frame",
                "status": str(row.get("status", "")),
                "object": str(row.get("object", "")),
                "part": str(row.get("part", "")),
                "frame": str(row.get("frame", "")),
            }
        )


def _frame_ids_for_part(obj: DatasetObject, part_name: str, args: argparse.Namespace) -> List[str]:
    part_dir = obj.masks_dir / part_name
    if not part_dir.is_dir():
        return []
    frames = sorted(
        [p.stem for p in part_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=natural_sort_key,
    )
    frames = frames[:: max(1, int(args.frame_stride))]
    if int(args.max_frames_per_part) > 0:
        frames = frames[: int(args.max_frames_per_part)]
    return frames


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_VIS_BBOX_EDGES = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _part_vis_color(index: int) -> Tuple[int, int, int]:
    colors = (
        (30, 220, 255),
        (80, 255, 80),
        (255, 120, 60),
        (255, 80, 220),
        (80, 160, 255),
        (180, 255, 80),
        (255, 220, 60),
        (120, 120, 255),
        (60, 255, 180),
        (220, 120, 255),
    )
    return colors[int(index) % len(colors)]


def _to_homo(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)


def _project_homo_point(point_h: np.ndarray, k: np.ndarray, ob_in_cam: np.ndarray) -> Tuple[int, int]:
    projected = k @ ((ob_in_cam @ point_h.reshape(4, 1))[:3, :])
    projected = projected.reshape(-1)
    projected = projected / max(float(projected[2]), 1e-8)
    return tuple(projected[:2].round().astype(int).tolist())


def _draw_foundationpose_xyz_axis(
    image_bgr: np.ndarray,
    ob_in_cam: np.ndarray,
    k: np.ndarray,
    scale: float,
    thickness: int = 3,
    transparency: float = 0.0,
) -> np.ndarray:
    xx = np.array([scale, 0, 0, 1], dtype=np.float32)
    yy = np.array([0, scale, 0, 1], dtype=np.float32)
    zz = np.array([0, 0, scale, 1], dtype=np.float32)
    origin = _project_homo_point(np.array([0, 0, 0, 1], dtype=np.float32), k, ob_in_cam)
    xx_uv = _project_homo_point(xx, k, ob_in_cam)
    yy_uv = _project_homo_point(yy, k, ob_in_cam)
    zz_uv = _project_homo_point(zz, k, ob_in_cam)
    out = image_bgr.copy()
    for uv, color in ((xx_uv, (0, 0, 255)), (yy_uv, (0, 255, 0)), (zz_uv, (255, 0, 0))):
        tmp = out.copy()
        tmp = cv2.arrowedLine(tmp, origin, uv, color=color, thickness=thickness, line_type=cv2.LINE_AA, tipLength=0)
        mask = np.linalg.norm(tmp.astype(np.float32) - out.astype(np.float32), axis=-1) > 0
        out[mask] = out[mask] * transparency + tmp[mask] * (1.0 - transparency)
    return out.astype(np.uint8)


def _draw_foundationpose_3d_box(
    image_bgr: np.ndarray,
    ob_in_cam: np.ndarray,
    k: np.ndarray,
    bbox: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    linewidth: int = 2,
) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32).reshape(2, 3)
    min_xyz = bbox.min(axis=0)
    max_xyz = bbox.max(axis=0)
    xmin, ymin, zmin = min_xyz
    xmax, ymax, zmax = max_xyz
    out = image_bgr

    def draw_line3d(start, end):
        nonlocal out
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts_cam = (ob_in_cam @ _to_homo(pts).T).T[:, :3]
        if np.any(~np.isfinite(pts_cam)) or np.any(pts_cam[:, 2] <= 1e-8):
            return
        projected = (k @ pts_cam.T).T
        uv = np.round(projected[:, :2] / projected[:, 2:3]).astype(int)
        out = cv2.line(out, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)

    for y in (ymin, ymax):
        for z in (zmin, zmax):
            start = np.array([xmin, y, z], dtype=np.float32)
            draw_line3d(start, start + np.array([xmax - xmin, 0, 0], dtype=np.float32))
    for x in (xmin, xmax):
        for z in (zmin, zmax):
            start = np.array([x, ymin, z], dtype=np.float32)
            draw_line3d(start, start + np.array([0, ymax - ymin, 0], dtype=np.float32))
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            start = np.array([x, y, zmin], dtype=np.float32)
            draw_line3d(start, start + np.array([0, 0, zmax - zmin], dtype=np.float32))
    return out


def _foundationpose_box_pose(mesh: trimesh.Trimesh, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3).astype(np.float32)
    center_pose = np.asarray(pose, dtype=np.float32).reshape(4, 4) @ np.linalg.inv(to_origin).astype(np.float32)
    return center_pose, bbox


def _write_object_pose_visualizations(
    obj: DatasetObject,
    obj_rows: List[Dict[str, object]],
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    if not bool(args.save_vis):
        return
    ok_rows = [r for r in obj_rows if r.get("status") == "ok" and r.get("pose") is not None]
    if not ok_rows:
        return

    by_frame: Dict[str, List[Dict[str, object]]] = {}
    for row in ok_rows:
        by_frame.setdefault(str(row["frame"]), []).append(row)

    try:
        k = load_k(obj)
    except Exception as exc:
        print(f"[pose-vis][warn] {obj.name}: failed to load K for visualization: {exc}", flush=True)
        return

    vis_root = out_root / obj.name / "vis"
    vis_root.mkdir(parents=True, exist_ok=True)
    mesh_cache: Dict[str, trimesh.Trimesh] = {}

    for frame, rows in sorted(by_frame.items(), key=lambda x: natural_sort_key(x[0])):
        rgb_path = find_image(obj.rgb_dir, frame)
        if rgb_path is None:
            continue
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]

        for part_idx, row in enumerate(sorted(rows, key=lambda r: natural_sort_key(str(r.get("part", ""))))):
            mesh_path = str(row.get("mesh_path", ""))
            if not mesh_path:
                continue
            try:
                if mesh_path not in mesh_cache:
                    mesh_cache[mesh_path] = _load_mesh(Path(mesh_path))
                mesh = mesh_cache[mesh_path]
                pose = np.asarray(row["pose"], dtype=np.float32).reshape(4, 4)
            except Exception as exc:
                print(
                    f"[pose-vis][warn] {obj.name} frame={frame} part={row.get('part')}: {exc}",
                    flush=True,
                )
                continue

            color = _part_vis_color(part_idx)
            center_pose, bbox = _foundationpose_box_pose(mesh, pose)
            axis_scale = max(0.02, min(0.15, float(np.linalg.norm(bbox[1] - bbox[0])) * 0.35))
            image = _draw_foundationpose_3d_box(k=k, image_bgr=image, ob_in_cam=center_pose, bbox=bbox, line_color=color)
            image = _draw_foundationpose_xyz_axis(image, center_pose, k, scale=axis_scale, thickness=3, transparency=0.0)

        cv2.imwrite(str(vis_root / f"{frame}.png"), image)


def _estimate_one_frame(
    estimator,
    obj: DatasetObject,
    part_name: str,
    frame: str,
    mesh_path: Path,
    mesh_diameter: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    rgb_path = find_image(obj.rgb_dir, frame)
    depth_path = find_image(obj.depth_dir, frame)
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    gt_pose_path = pose_path_for_part_frame(obj, part_name, frame)
    if rgb_path is None or depth_path is None or mask_path is None:
        return {
            "status": "skipped",
            "reason": "missing_rgb_depth_mask",
            "object": obj.name,
            "part": part_name,
            "frame": frame,
        }

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"failed to read rgb: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = load_depth_m(depth_path, args.depth_scale)
    mask = load_mask(mask_path, depth.shape[:2]).astype(np.uint8)
    k = load_k(obj)
    init_pose = None
    if bool(args.use_gt_init) and gt_pose_path is not None:
        init_pose = load_pose(gt_pose_path, args.pose_convention)

    t0 = time.time()
    with _quiet_foundationpose_io(args):
        pose = estimator.register(
            K=k,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=int(args.refine_iterations),
            use_nvdiffrast=bool(args.use_nvdiffrast),
            init_pose=init_pose,
        )
    elapsed = time.time() - t0
    row = {
        "status": "ok",
        "object": obj.name,
        "category": obj.name.split("_")[0],
        "part": part_name,
        "frame": frame,
        "mesh_path": str(mesh_path),
        "mesh_diameter": float(mesh_diameter),
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "mask_path": str(mask_path),
        "gt_pose_path": None if gt_pose_path is None else str(gt_pose_path),
        "time_sec": float(elapsed),
        "pose": np.asarray(pose, dtype=float).reshape(4, 4).tolist(),
    }
    return row


def _resolve_object_names(args: argparse.Namespace) -> List[str]:
    data_root = Path(args.data_root).resolve()
    object_names = [x.strip() for x in args.objects.split(",") if x.strip()]
    if not object_names:
        object_names = list_objects(data_root, args.split, "all", "")
    if args.start or args.end is not None:
        end = len(object_names) if args.end is None else int(args.end)
        object_names = object_names[int(args.start) : end]
    return object_names


def _count_pose_frames(args: argparse.Namespace, object_names: List[str]) -> Tuple[int, int]:
    method = str(args.method)
    data_root = Path(args.data_root).resolve()
    work_root = Path(args.work_root).resolve()
    total = 0
    missing_mesh = 0
    for obj_name in object_names:
        obj = DatasetObject(data_root=data_root, split=args.split, name=obj_name)
        for part_idx, part_name in enumerate(list_parts(obj)):
            part_model = part_model_name(part_name, part_idx)
            mesh_path = model_obj_path(method_pose_ready_dir(work_root, method, args.split, obj.name), part_model)
            if not mesh_path.exists():
                missing_mesh += 1
                continue
            total += len(_frame_ids_for_part(obj, part_name, args))
    return int(total), int(missing_mesh)


def _estimate_objects(
    args: argparse.Namespace,
    object_names: List[str],
    worker_label: str = "",
    progress_queue=None,
    progress_total: Optional[int] = None,
) -> Dict[str, object]:
    method = str(args.method)
    data_root = Path(args.data_root).resolve()
    work_root = Path(args.work_root).resolve()
    pose_root = Path(args.pose_root).resolve() if str(args.pose_root).strip() else default_pose_root(work_root)
    out_root = pose_root / method
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    summary = {
        "method": method,
        "data_root": str(data_root),
        "work_root": str(work_root),
        "pose_root": str(pose_root),
        "worker": str(worker_label),
        "objects": [],
    }

    local_bar = None
    if progress_queue is None and _progress_enabled(args):
        if progress_total is None:
            progress_total, _ = _count_pose_frames(args, object_names)
        local_bar = tqdm(
            total=int(progress_total),
            desc=f"pose-est {method}",
            unit="frame",
            dynamic_ncols=True,
            leave=True,
        )

    for obj_name in object_names:
        if local_bar is not None:
            local_bar.set_postfix(object=obj_name)
        obj = DatasetObject(data_root=data_root, split=args.split, name=obj_name)
        obj_rows: List[Dict[str, object]] = []
        obj_summary = {"object": obj_name, "parts": []}
        for part_idx, part_name in enumerate(list_parts(obj)):
            part_model = part_model_name(part_name, part_idx)
            mesh_path = model_obj_path(method_pose_ready_dir(work_root, method, args.split, obj.name), part_model)
            part_summary = {"part": part_name, "part_model": part_model, "mesh_path": str(mesh_path), "frames": 0, "ok": 0}
            if not mesh_path.exists():
                part_summary["status"] = "missing_mesh"
                obj_summary["parts"].append(part_summary)
                _progress_message(
                    args,
                    progress_queue,
                    f"[pose-est][warn] method={method} object={obj_name} part={part_name} missing mesh: {mesh_path}",
                )
                continue
            try:
                mesh = _load_mesh(mesh_path)
                estimator = _build_estimator(mesh, args)
                diameter = _mesh_diameter(mesh)
                frames = _frame_ids_for_part(obj, part_name, args)
                part_summary["frames"] = len(frames)
                for frame in frames:
                    try:
                        row = _estimate_one_frame(estimator, obj, part_name, frame, mesh_path, diameter, args)
                    except Exception as exc:
                        row = {
                            "status": "failed",
                            "object": obj.name,
                            "category": obj.name.split("_")[0],
                            "part": part_name,
                            "frame": frame,
                            "mesh_path": str(mesh_path),
                            "error": str(exc),
                            "traceback": traceback.format_exc() if bool(args.save_traceback) else "",
                        }
                        _progress_message(
                            args,
                            progress_queue,
                            f"[pose-est][error] method={method} object={obj.name} part={part_name} "
                            f"frame={frame}: {exc}",
                        )
                    obj_rows.append(row)
                    all_rows.append(row)
                    part_summary["ok"] += int(row.get("status") == "ok")
                    _progress_frame(args, progress_queue, row)
                    if local_bar is not None:
                        local_bar.update(1)
                        local_bar.set_postfix(
                            object=obj_name,
                            part=part_name,
                            ok=part_summary["ok"],
                        )
                part_summary["status"] = "ok"
            except Exception as exc:
                part_summary["status"] = "failed"
                part_summary["error"] = str(exc)
                if bool(args.save_traceback):
                    part_summary["traceback"] = traceback.format_exc()
                _progress_message(args, progress_queue, f"[pose-est][error] method={method} object={obj_name} part={part_name}: {exc}")
            obj_summary["parts"].append(part_summary)
        _write_jsonl(out_root / "objects" / f"{obj_name}.jsonl", obj_rows)
        _write_object_pose_visualizations(obj, obj_rows, out_root, args)
        summary["objects"].append(obj_summary)

    if local_bar is not None:
        local_bar.close()

    if worker_label:
        worker_root = out_root / "worker_outputs"
        _write_jsonl(worker_root / f"{worker_label}.jsonl", all_rows)
        with (worker_root / f"{worker_label}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        _write_jsonl(out_root / "poses.jsonl", all_rows)
    summary["total_rows"] = len(all_rows)
    summary["ok_rows"] = int(sum(1 for r in all_rows if r.get("status") == "ok"))
    if not worker_label:
        with (out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[pose-est] wrote {out_root}")
    return summary


def _gpu_worker(worker_idx: int, args: argparse.Namespace, object_names: List[str], gpu_ids: List[str], progress_queue=None) -> None:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[worker_idx % len(gpu_ids)])
    worker_label = f"worker_{worker_idx:03d}"
    _estimate_objects(args, object_names, worker_label=worker_label, progress_queue=progress_queue)


def _merge_worker_outputs(args: argparse.Namespace, worker_count: int) -> Dict[str, object]:
    method = str(args.method)
    work_root = Path(args.work_root).resolve()
    pose_root = Path(args.pose_root).resolve() if str(args.pose_root).strip() else default_pose_root(work_root)
    out_root = pose_root / method
    worker_root = out_root / "worker_outputs"
    all_rows: List[Dict[str, object]] = []
    objects = []
    for idx in range(worker_count):
        label = f"worker_{idx:03d}"
        all_rows.extend(_read_jsonl(worker_root / f"{label}.jsonl"))
        summary_path = worker_root / f"{label}_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                worker_summary = json.load(f)
            objects.extend(worker_summary.get("objects", []))
    _write_jsonl(out_root / "poses.jsonl", all_rows)
    summary = {
        "method": method,
        "data_root": str(Path(args.data_root).resolve()),
        "work_root": str(work_root),
        "pose_root": str(pose_root),
        "num_workers": int(worker_count),
        "objects": objects,
        "total_rows": len(all_rows),
        "ok_rows": int(sum(1 for r in all_rows if r.get("status") == "ok")),
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[pose-est] merged workers={worker_count} rows={len(all_rows)} wrote {out_root}", flush=True)
    return summary


def estimate_method(args: argparse.Namespace) -> Dict[str, object]:
    method = str(args.method)
    if method not in PIPELINE_METHODS:
        raise ValueError(f"unknown method={method}; expected one of {PIPELINE_METHODS}")

    object_names = _resolve_object_names(args)
    if not object_names:
        raise RuntimeError("no objects to process")

    gpu_ids = [x.strip() for x in str(args.gpu_ids).split(",") if x.strip()]
    if int(args.num_workers) > 0:
        worker_count = int(args.num_workers)
    elif gpu_ids:
        worker_count = max(1, len(gpu_ids) * max(1, int(args.workers_per_gpu)))
    else:
        worker_count = 1

    total_frames, missing_mesh = _count_pose_frames(args, object_names)
    frames_per_worker = float(total_frames) / float(max(1, worker_count))
    if _progress_enabled(args):
        tqdm.write(
            f"[pose-est] method={method} objects={len(object_names)} "
            f"frames={total_frames} workers={worker_count} "
            f"frames_per_worker={frames_per_worker:.1f} missing_mesh_parts={missing_mesh}"
        )
    else:
        print(
            f"[pose-est] method={method} objects={len(object_names)} "
            f"frames={total_frames} workers={worker_count} "
            f"frames_per_worker={frames_per_worker:.1f} missing_mesh_parts={missing_mesh}",
            flush=True,
        )

    if worker_count <= 1:
        return _estimate_objects(args, object_names, progress_total=total_frames)

    chunks = [object_names[i::worker_count] for i in range(worker_count)]
    chunks = [c for c in chunks if c]
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    procs = []
    for worker_idx, chunk in enumerate(chunks):
        p = ctx.Process(target=_gpu_worker, args=(worker_idx, args, chunk, gpu_ids, progress_queue), daemon=False)
        p.start()
        procs.append(p)

    ok_count = 0
    failed_count = 0
    skipped_count = 0
    completed_count = 0
    pbar = None
    if _progress_enabled(args):
        pbar = tqdm(
            total=int(total_frames),
            desc=f"pose-est {method}",
            unit="frame",
            dynamic_ncols=True,
            leave=True,
        )

    def drain_progress_queue():
        nonlocal ok_count, failed_count, skipped_count, completed_count
        while True:
            try:
                item = progress_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind == "frame":
                completed_count += 1
                status = item.get("status")
                if status == "ok":
                    ok_count += 1
                elif status == "failed":
                    failed_count += 1
                else:
                    skipped_count += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        object=item.get("object", ""),
                        part=item.get("part", ""),
                        ok=ok_count,
                        failed=failed_count,
                    )
            elif kind == "message":
                msg = str(item.get("text", ""))
                if msg:
                    if pbar is not None:
                        tqdm.write(msg)
                    else:
                        print(msg, flush=True)

    while any(p.is_alive() for p in procs):
        drain_progress_queue()
        time.sleep(0.2)
    drain_progress_queue()
    if pbar is not None:
        pbar.close()

    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((p.pid, p.exitcode))
    if failed:
        raise RuntimeError(f"pose-est workers failed: {failed}")
    return _merge_worker_outputs(args, len(chunks))


def build_parser(default_method: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run FoundationPose pose estimation for reconstruction pipeline outputs.")
    parser.add_argument("--method", type=str, default=default_method)
    parser.add_argument("--data-root", type=str, default="dataset_train")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--work-root", type=str, default="reconstruction_runs")
    parser.add_argument("--pose-root", type=str, default="", help="Default: sibling of work-root named reconstruction_pose_est.")
    parser.add_argument("--objects", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-part", type=int, default=0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--pose-convention", choices=["cv", "sapien"], default="sapien")
    parser.add_argument("--refine-iterations", type=int, default=8)
    parser.add_argument("--use-nvdiffrast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gt-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug-dir", type=str, default="/tmp/foundationpose_debug")
    parser.add_argument("--save-traceback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vis-max-points", type=int, default=1500)
    parser.add_argument("--quiet-foundationpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU ids. Workers are assigned round-robin.")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Used when --num-workers is 0 and --gpu-ids is set.")
    parser.add_argument("--num-workers", type=int, default=1, help="Total pose-est worker processes. 1 disables multiprocessing.")
    return parser


def main(default_method: str = "") -> None:
    args = build_parser(default_method).parse_args()
    if not str(args.method).strip():
        raise ValueError("--method is required")
    estimate_method(args)
