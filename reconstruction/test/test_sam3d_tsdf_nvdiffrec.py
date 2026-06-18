import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_dataset(args: argparse.Namespace) -> Tuple[Path, str]:
    if args.dataset_path:
        split_root = Path(args.dataset_path).resolve()
        if not split_root.is_dir():
            raise FileNotFoundError(f"dataset split folder not found: {split_root}")
        return split_root.parent, split_root.name
    return Path(args.data_root).resolve(), args.split


def _has_minimal_dataset_shape(object_dir: Path) -> bool:
    required_dirs = ("rgb", "depth", "masks", "cam_params")
    if not all((object_dir / name).is_dir() for name in required_dirs):
        return False
    if not (object_dir / "K.txt").is_file():
        return False
    return any((object_dir / "masks").iterdir())


def _split_object_names(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _natural_sort_key(value: object) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(value))]


def _part_model_name(part_name: str, fallback_idx: int) -> str:
    match = re.search(r"(\d+)", str(part_name))
    part_id = int(match.group(1)) if match else int(fallback_idx)
    return f"model_{part_id:04d}"


def _find_image(folder: Path, frame: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        path = folder / f"{frame}{ext}"
        if path.exists():
            return path
    return None


def _load_pose(path: Path, convention: str) -> "object":
    import numpy as np

    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if convention == "sapien":
        sapiencam_to_cvcam = np.asarray(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose = sapiencam_to_cvcam @ pose
    return pose.astype(np.float32)


def _project_points(points_cam: "object", k: "object", shape_hw: Tuple[int, int]) -> Tuple["object", "object", "object"]:
    import numpy as np

    z = points_cam[:, 2]
    valid = z > 1e-6
    u = np.zeros(len(points_cam), dtype=np.int64)
    v = np.zeros(len(points_cam), dtype=np.int64)
    u[valid] = np.rint(points_cam[valid, 0] * float(k[0, 0]) / z[valid] + float(k[0, 2])).astype(np.int64)
    v[valid] = np.rint(points_cam[valid, 1] * float(k[1, 1]) / z[valid] + float(k[1, 2])).astype(np.int64)
    h, w = shape_hw
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def _classify_mesh_faces(
    mesh: "object",
    k: "object",
    mask: "object",
    depth_m: "object",
    ob_in_cam: "object",
    depth_margin: float,
    mask_dilate: int,
) -> Dict[str, "object"]:
    import cv2
    import numpy as np

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    centers = verts[faces].mean(axis=1)
    points_cam = (ob_in_cam[:3, :3] @ centers.T).T + ob_in_cam[:3, 3]
    u, v, valid = _project_points(points_cam, k, depth_m.shape[:2])

    mask_eval = mask.astype(bool)
    if int(mask_dilate) > 0:
        kernel_size = max(1, int(mask_dilate) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_eval = cv2.dilate(mask_eval.astype(np.uint8), kernel, iterations=1).astype(bool)

    cls = np.zeros(len(faces), dtype=np.uint8)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return {"class": cls, "stats": {"faces": int(len(faces)), "projected": 0}}

    uu = u[valid_idx]
    vv = v[valid_idx]
    in_mask = mask_eval[vv, uu]
    depth_vals = depth_m[vv, uu]
    has_depth = depth_vals > 1e-6
    z = points_cam[valid_idx, 2]
    margin = float(depth_margin)

    reliable = in_mask & has_depth & (np.abs(z - depth_vals) <= margin)
    front_conflict = in_mask & has_depth & (z < depth_vals - margin)
    behind = in_mask & has_depth & (z > depth_vals + margin)
    outside = ~in_mask

    cls[valid_idx[reliable]] = 1
    cls[valid_idx[behind]] = 2
    cls[valid_idx[front_conflict | outside]] = 3
    stats = {
        "faces": int(len(faces)),
        "projected": int(len(valid_idx)),
        "reliable": int(np.count_nonzero(cls == 1)),
        "behind_uncertain": int(np.count_nonzero(cls == 2)),
        "conflict": int(np.count_nonzero(cls == 3)),
        "unobserved": int(np.count_nonzero(cls == 0)),
    }
    return {"class": cls, "stats": stats}


def _write_colored_mesh(mesh: "object", face_class: "object", out_path: Path, new_support: Optional["object"] = None) -> None:
    import numpy as np
    import trimesh

    colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    colors[:] = np.asarray([150, 150, 150, 255], dtype=np.uint8)  # unobserved
    colors[face_class == 1] = np.asarray([40, 190, 90, 255], dtype=np.uint8)  # reliable
    colors[face_class == 2] = np.asarray([240, 190, 45, 255], dtype=np.uint8)  # behind/uncertain
    colors[face_class == 3] = np.asarray([220, 65, 65, 255], dtype=np.uint8)  # conflict
    if new_support is not None:
        colors[new_support.astype(bool)] = np.asarray([40, 170, 240, 255], dtype=np.uint8)  # newly observed
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    out.visual.face_colors = colors
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(out_path))


def _debug_reliability_for_object(
    data_root: Path,
    split: str,
    object_name: str,
    work_root: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    import cv2
    import numpy as np
    import trimesh

    object_dir = data_root / split / object_name
    masks_root = object_dir / "masks"
    depth_root = object_dir / "depth"
    cam_root = object_dir / "cam_params"
    k_path = object_dir / "K.txt"
    if not (masks_root.is_dir() and depth_root.is_dir() and cam_root.is_dir() and k_path.is_file()):
        return None

    k = np.loadtxt(k_path).astype(np.float32).reshape(3, 3)
    debug_root = work_root / "sam3d_tsdf_dmesh" / split / object_name / "debug" / "reliability"
    summary: Dict[str, object] = {
        "object": object_name,
        "legend": {
            "green": "supported by current/previous depth+mask",
            "cyan": "newly supported by this accepted TSDF frame",
            "yellow": "behind observed depth; uncertain/unobserved back-side geometry",
            "red": "outside mask or in front of observed depth; conflict",
            "gray": "not projected/validated",
        },
        "parts": [],
    }

    part_dirs = sorted([p for p in masks_root.iterdir() if p.is_dir()], key=lambda p: _natural_sort_key(p.name))
    for part_idx, part_dir in enumerate(part_dirs):
        part_name = part_dir.name
        part_model = _part_model_name(part_name, part_idx)
        sam3d_model = work_root / "sam3d" / split / object_name / "pose_ready_models" / "view_0" / part_model / "model.obj"
        iter_root = work_root / "sam3d_tsdf" / split / object_name / "pose_tsdf_iter" / part_model
        iter_summary_path = iter_root / "summary.json"
        if not sam3d_model.is_file():
            continue

        frame_names = sorted([p.stem for p in part_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=_natural_sort_key)
        if int(args.debug_reliability_max_frames) > 0:
            frame_names = frame_names[: int(args.debug_reliability_max_frames)]
        iter_summary = {}
        if iter_summary_path.is_file():
            with iter_summary_path.open("r", encoding="utf-8") as f:
                iter_summary = json.load(f)
        seed_frame = str(iter_summary.get("seed_frame") or (frame_names[0] if frame_names else ""))
        accepted_frames = [str(x) for x in iter_summary.get("accepted_frames", [])]
        if not accepted_frames and seed_frame:
            accepted_frames = [seed_frame]

        part_report: Dict[str, object] = {
            "part": part_name,
            "part_model": part_model,
            "sam3d_model": str(sam3d_model),
            "seed_frame": seed_frame,
            "accepted_frames": accepted_frames,
            "visualizations": [],
        }

        seed_depth_path = _find_image(depth_root, seed_frame)
        seed_mask_path = _find_image(part_dir, seed_frame)
        seed_pose_path = cam_root / part_name / f"{seed_frame}.txt"
        if seed_depth_path and seed_mask_path and seed_pose_path.is_file():
            mesh = trimesh.load(str(sam3d_model), force="mesh", process=False)
            depth = cv2.imread(str(seed_depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            if depth.size > 0 and float(np.nanmax(depth)) > 50.0:
                depth = depth / float(args.debug_reliability_depth_scale)
            mask = cv2.imread(str(seed_mask_path), cv2.IMREAD_GRAYSCALE) > 127
            pose = _load_pose(seed_pose_path, args.pose_convention)
            classified = _classify_mesh_faces(
                mesh,
                k,
                mask,
                depth,
                pose,
                float(args.debug_reliability_depth_margin),
                int(args.debug_reliability_mask_dilate),
            )
            out_path = debug_root / part_model / "sam3d_prior" / f"{seed_frame}_reliability.ply"
            _write_colored_mesh(mesh, classified["class"], out_path)
            part_report["visualizations"].append(
                {"stage": "sam3d_prior", "frame": seed_frame, "mesh": str(out_path), "stats": classified["stats"]}
            )

        support_count = None
        for accepted_idx, frame in enumerate(accepted_frames):
            if accepted_idx == 0:
                mesh_path = sam3d_model
            else:
                mesh_path = iter_root / f"iter_{accepted_idx:03d}" / "mesh.obj"
                if not mesh_path.is_file():
                    mesh_path = work_root / "sam3d_tsdf" / split / object_name / "pose_ready_models" / "view_0" / part_model / "model.obj"
            depth_path = _find_image(depth_root, frame)
            mask_path = _find_image(part_dir, frame)
            pose_path = iter_root / "poses" / f"{frame}.txt"
            if not pose_path.is_file():
                pose_path = cam_root / part_name / f"{frame}.txt"
            if not (mesh_path.is_file() and depth_path and mask_path and pose_path.is_file()):
                continue
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            if depth.size > 0 and float(np.nanmax(depth)) > 50.0:
                depth = depth / float(args.debug_reliability_depth_scale)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            pose = _load_pose(pose_path, "cv" if pose_path.parent.name == "poses" else args.pose_convention)
            classified = _classify_mesh_faces(
                mesh,
                k,
                mask,
                depth,
                pose,
                float(args.debug_reliability_depth_margin),
                int(args.debug_reliability_mask_dilate),
            )
            face_class = classified["class"]
            if support_count is None or len(support_count) != len(face_class):
                support_count = np.zeros(len(face_class), dtype=np.int32)
            reliable = face_class == 1
            new_support = reliable & (support_count == 0)
            support_count[reliable] += 1
            cumulative = face_class.copy()
            cumulative[support_count > 0] = 1
            out_path = debug_root / part_model / "tsdf_incremental" / f"{accepted_idx:03d}_{frame}_new_support.ply"
            _write_colored_mesh(mesh, cumulative, out_path, new_support=new_support)
            part_report["visualizations"].append(
                {
                    "stage": "tsdf_incremental",
                    "accepted_idx": int(accepted_idx),
                    "frame": frame,
                    "mesh": str(out_path),
                    "new_supported_faces": int(np.count_nonzero(new_support)),
                    "cumulative_supported_faces": int(np.count_nonzero(support_count > 0)),
                    "stats": classified["stats"],
                }
            )

        summary["parts"].append(part_report)

    if not summary["parts"]:
        return None
    debug_root.mkdir(parents=True, exist_ok=True)
    with (debug_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return debug_root


def _pick_objects(data_root: Path, split: str, requested: str, max_objects: int) -> List[str]:
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"split folder not found: {split_root}")

    requested_names = _split_object_names(requested)
    if requested_names:
        missing = [name for name in requested_names if not (split_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"requested object folders not found under {split_root}: {missing}"
            )
        return requested_names[: max(1, int(max_objects))] if int(max_objects) > 0 else requested_names

    picked: List[str] = []
    for object_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        if _has_minimal_dataset_shape(object_dir):
            picked.append(object_dir.name)
            if int(max_objects) > 0 and len(picked) >= int(max_objects):
                break
    if picked:
        return picked
    raise RuntimeError(
        f"No usable object found under {split_root}. Expected object/rgb, depth, masks, cam_params, and K.txt."
    )


def _find_first_summary(work_root: Path, split: str, object_name: str) -> Optional[Path]:
    summary = work_root / "sam3d_tsdf_dmesh" / split / object_name / "summary.json"
    return summary if summary.exists() else None


def _recon_entry(repo: Path) -> Path:
    candidates = [
        repo / "reconstruction" / "run" / "recon_sam3d_tsdf_dmesh.py",
        repo / "reconstruction" / "recon_sam3d_tsdf_dmesh.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "recon_sam3d_tsdf_dmesh.py not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _print_summary(summary_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    parts = data.get("parts", [])
    print(f"[SUMMARY] {summary_path}")
    print(f"[SUMMARY] parts={len(parts)}")
    for part in parts[:5]:
        name = part.get("part", "unknown")
        status = part.get("status", "unknown")
        model = part.get("output_model") or part.get("model")
        dlmesh = part.get("dlmesh_result")
        if isinstance(dlmesh, dict):
            status = dlmesh.get("status", status)
            model = dlmesh.get("output_model", model)
        print(f"[SUMMARY] part={name} status={status} model={model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Fast smoke test for SAM3D + TSDF + nvdiffrast/DLMesh reconstruction on one object."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Direct split folder containing object dirs, e.g. dataset_train/test. Overrides --data-root/--split.",
    )
    parser.add_argument("--data-root", type=str, default="dataset_train")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--object", type=str, default="", help="Deprecated alias for --objects.")
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names. Defaults to first usable object.")
    parser.add_argument("--max-objects", type=int, default=1, help="Number of usable objects to test when --objects is empty. <=0 means all.")
    parser.add_argument("--work-root", type=str, default="reconstruction_runs_test")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Allow existing reconstruction outputs to be reused. By default this smoke test rebuilds everything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag; rebuild is already the default unless --reuse-existing is set.",
    )
    parser.add_argument(
        "--no-build-base",
        action="store_true",
        help="Require an existing SAM3D cache instead of building it during this smoke test.",
    )

    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--voxel-length", type=float, default=0.02)
    parser.add_argument("--sdf-trunc", type=float, default=0.08)
    parser.add_argument("--iter-tsdf-max-frames", type=int, default=2)
    parser.add_argument("--iter-tsdf-refine-iters", type=int, default=2)
    parser.add_argument("--iter-tsdf-refine-points", type=int, default=256)
    parser.add_argument("--iter-tsdf-mesh-samples", type=int, default=1024)
    parser.add_argument("--iter-tsdf-consistency-points", type=int, default=256)

    parser.add_argument("--dlmesh-device", type=str, default="cuda:0")
    parser.add_argument("--dlmesh-outer-iters", type=int, default=1)
    parser.add_argument("--dlmesh-steps-per-stage", type=int, default=8)
    parser.add_argument("--dlmesh-target-faces", type=int, default=1000)
    parser.add_argument("--dlmesh-max-keyframes", type=int, default=2)
    parser.add_argument("--dlmesh-stage-size", type=int, default=2)
    parser.add_argument("--dlmesh-min-mask-pixels", type=int, default=64)
    parser.add_argument("--dlmesh-min-points", type=int, default=64)
    parser.add_argument("--dlmesh-icp-iters", type=int, default=3)
    parser.add_argument("--dlmesh-icp-points", type=int, default=256)
    parser.add_argument("--dlmesh-mesh-samples", type=int, default=512)
    parser.add_argument("--dlmesh-pose-points", type=int, default=256)
    parser.add_argument("--dlmesh-pose-steps", type=int, default=5)
    parser.add_argument("--dlmesh-stage-points-per-frame", type=int, default=256)
    parser.add_argument("--dlmesh-project-vertices", type=int, default=512)
    parser.add_argument(
        "--debug-reliability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After reconstruction, export colored mesh reliability visualizations under the debug folder.",
    )
    parser.add_argument(
        "--debug-reliability-depth-margin",
        type=float,
        default=0.015,
        help="Depth tolerance in meters for marking a face as supported by mask/depth.",
    )
    parser.add_argument(
        "--debug-reliability-mask-dilate",
        type=int,
        default=3,
        help="Dilate part masks by this many pixels before reverse projection checks.",
    )
    parser.add_argument(
        "--debug-reliability-max-frames",
        type=int,
        default=0,
        help="Maximum dataset frames used for fallback debug ordering; 0 keeps all.",
    )
    parser.add_argument("--debug-reliability-depth-scale", type=float, default=1000.0)
    parser.add_argument(
        "--copy-base-as-placeholder",
        action="store_true",
        help="Only test IO by copying TSDF meshes; this does not exercise nvdiffrast/DLMesh.",
    )
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    data_root, split = _resolve_dataset(args)
    requested_objects = args.objects.strip() or args.object.strip()
    object_names = _pick_objects(data_root, split, requested_objects, args.max_objects)
    objects_arg = ",".join(object_names)
    work_root = Path(args.work_root).resolve()

    cmd = [
        sys.executable,
        str(_recon_entry(repo)),
        "--data-root",
        str(data_root),
        "--split",
        split,
        "--work-root",
        str(work_root),
        "--objects",
        objects_arg,
        "--num-workers",
        str(args.num_workers),
        "--gpus",
        args.gpus,
        "--max-frames",
        str(args.max_frames),
        "--voxel-length",
        str(args.voxel_length),
        "--sdf-trunc",
        str(args.sdf_trunc),
        "--iter-tsdf-max-frames",
        str(args.iter_tsdf_max_frames),
        "--iter-tsdf-refine-iters",
        str(args.iter_tsdf_refine_iters),
        "--iter-tsdf-refine-points",
        str(args.iter_tsdf_refine_points),
        "--iter-tsdf-mesh-samples",
        str(args.iter_tsdf_mesh_samples),
        "--iter-tsdf-consistency-points",
        str(args.iter_tsdf_consistency_points),
        "--dlmesh-device",
        args.dlmesh_device,
        "--dlmesh-outer-iters",
        str(args.dlmesh_outer_iters),
        "--dlmesh-steps-per-stage",
        str(args.dlmesh_steps_per_stage),
        "--dlmesh-target-faces",
        str(args.dlmesh_target_faces),
        "--dlmesh-max-keyframes",
        str(args.dlmesh_max_keyframes),
        "--dlmesh-stage-size",
        str(args.dlmesh_stage_size),
        "--dlmesh-min-mask-pixels",
        str(args.dlmesh_min_mask_pixels),
        "--dlmesh-min-points",
        str(args.dlmesh_min_points),
        "--dlmesh-icp-iters",
        str(args.dlmesh_icp_iters),
        "--dlmesh-icp-points",
        str(args.dlmesh_icp_points),
        "--dlmesh-mesh-samples",
        str(args.dlmesh_mesh_samples),
        "--dlmesh-pose-points",
        str(args.dlmesh_pose_points),
        "--dlmesh-pose-steps",
        str(args.dlmesh_pose_steps),
        "--dlmesh-stage-points-per-frame",
        str(args.dlmesh_stage_points_per_frame),
        "--dlmesh-project-vertices",
        str(args.dlmesh_project_vertices),
        "--reset-coord",
    ]
    if not args.reuse_existing or args.overwrite:
        cmd.append("--overwrite")
    if not args.no_build_base:
        cmd.append("--build-base-if-missing")
    if args.copy_base_as_placeholder:
        cmd.append("--copy-base-as-placeholder")
    cmd.extend(args.extra_args)

    print(f"[TEST] dataset={data_root / split}")
    print(f"[TEST] objects={objects_arg}")
    print("[TEST] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    missing_summaries = []
    for object_name in object_names:
        summary = _find_first_summary(work_root, split, object_name)
        if summary is None:
            missing_summaries.append(object_name)
            continue
        _print_summary(summary)
    if missing_summaries:
        raise FileNotFoundError(
            f"reconstruction command finished but summary.json was not found for "
            f"{missing_summaries} under {work_root}"
        )

    if args.debug_reliability:
        debug_roots = []
        for object_name in object_names:
            debug_root = _debug_reliability_for_object(
                data_root=data_root,
                split=split,
                object_name=object_name,
                work_root=work_root,
                args=args,
            )
            if debug_root is not None:
                debug_roots.append(debug_root)
                print(f"[DEBUG] reliability={debug_root}")
        if not debug_roots:
            print("[DEBUG] reliability visualization skipped: no usable SAM3D/TSDF part outputs found")


if __name__ == "__main__":
    main()
