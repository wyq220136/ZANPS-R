from pathlib import Path
import sys

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse
from typing import Dict

import cv2

from recon_utils import (
    DatasetObject,
    add_common_args,
    ensure_dir,
    find_image,
    list_parts,
    mask_path_for_part_frame,
    method_models_dir,
    method_pose_ready_dir,
    model_obj_path,
    part_model_name,
    pose_path_for_part_frame,
    run_object_pipeline,
    select_best_frame_for_part,
)
from reconstruct_hunyuan3d import _prepare_rgba_from_mask
from reconstruct_instantmesh import (
    InstantMeshReconstructor,
    default_instantmesh_config_path,
    default_instantmesh_root,
)
from run.recon_hunyuan3d import _align_raw_hunyuan_to_pose_ready


METHOD = "instantmesh"


def add_instantmesh_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instantmesh-root", type=str, default=default_instantmesh_root())
    parser.add_argument("--instantmesh-config-path", type=str, default=default_instantmesh_config_path())
    parser.add_argument("--instantmesh-diffusion-model", type=str, default="sudo-ai/zero123plus-v1.2")
    parser.add_argument("--instantmesh-dino-model", type=str, default="")
    parser.add_argument("--instantmesh-unet-path", type=str, default="")
    parser.add_argument("--instantmesh-model-path", type=str, default="")
    parser.add_argument("--instantmesh-diffusion-steps", type=int, default=75)
    parser.add_argument("--instantmesh-seed", type=int, default=42)
    parser.add_argument("--instantmesh-scale", type=float, default=1.0)
    parser.add_argument("--instantmesh-view", type=int, choices=[4, 6], default=6)
    parser.add_argument("--instantmesh-foreground-ratio", type=float, default=0.85)
    parser.add_argument("--instantmesh-export-texmap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alignment-samples", type=int, default=50000)
    parser.add_argument("--alignment-seed", type=int, default=2026)
    parser.add_argument("--min-alignment-points", type=int, default=64)
    parser.add_argument("--alignment-icp-iters", type=int, default=30)
    parser.add_argument("--alignment-trim-quantile", type=float, default=0.8)


def _make_reconstructor(args: argparse.Namespace) -> InstantMeshReconstructor:
    return InstantMeshReconstructor(
        instantmesh_root=args.instantmesh_root,
        config_path=args.instantmesh_config_path,
        diffusion_model=args.instantmesh_diffusion_model,
        dino_model=args.instantmesh_dino_model,
        unet_path=args.instantmesh_unet_path,
        model_path=args.instantmesh_model_path,
        diffusion_steps=args.instantmesh_diffusion_steps,
        seed=args.instantmesh_seed,
        scale=args.instantmesh_scale,
        view=args.instantmesh_view,
        foreground_ratio=args.instantmesh_foreground_ratio,
        export_texmap=args.instantmesh_export_texmap,
    )


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace) -> Dict[str, object]:
    work_root = Path(args.work_root).resolve()
    models_root = ensure_dir(method_models_dir(work_root, METHOD, args.split, obj.name))
    pose_ready_root = ensure_dir(method_pose_ready_dir(work_root, METHOD, args.split, obj.name))
    parts = list_parts(obj)
    reconstructor = None
    summary = {
        "method": METHOD,
        "object": obj.name,
        "instantmesh": {
            "root": str(args.instantmesh_root),
            "config_path": str(args.instantmesh_config_path),
            "diffusion_model": str(args.instantmesh_diffusion_model),
            "dino_model": str(args.instantmesh_dino_model),
            "diffusion_steps": int(args.instantmesh_diffusion_steps),
            "view": int(args.instantmesh_view),
            "scale": float(args.instantmesh_scale),
            "foreground_ratio": float(args.instantmesh_foreground_ratio),
            "export_texmap": bool(args.instantmesh_export_texmap),
        },
        "parts": [],
    }

    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        out_obj = model_obj_path(pose_ready_root, part_model)
        if out_obj.exists() and not args.overwrite:
            summary["parts"].append({"part": part_name, "status": "cached", "model": str(out_obj)})
            continue

        frame = select_best_frame_for_part(obj, part_name, args.min_mask_pixels)
        if frame is None:
            summary["parts"].append(
                {
                    "part": part_name,
                    "part_model": part_model,
                    "status": "skipped",
                    "reason": "no_visible_frame",
                    "min_mask_pixels": int(args.min_mask_pixels),
                }
            )
            continue
        rgb_path = find_image(obj.rgb_dir, frame)
        depth_path = find_image(obj.depth_dir, frame)
        mask_path = mask_path_for_part_frame(obj, part_name, frame)
        pose_path = pose_path_for_part_frame(obj, part_name, frame)
        if rgb_path is None or depth_path is None or mask_path is None or pose_path is None:
            summary["parts"].append(
                {
                    "part": part_name,
                    "part_model": part_model,
                    "status": "skipped",
                    "reason": "missing_rgb_depth_mask_or_pose",
                    "frame": frame,
                    "rgb": None if rgb_path is None else str(rgb_path),
                    "depth": None if depth_path is None else str(depth_path),
                    "mask": None if mask_path is None else str(mask_path),
                    "pose": None if pose_path is None else str(pose_path),
                }
            )
            continue

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or mask_gray is None:
            summary["parts"].append(
                {
                    "part": part_name,
                    "part_model": part_model,
                    "status": "skipped",
                    "reason": "unreadable_rgb_or_mask",
                    "frame": frame,
                    "rgb": str(rgb_path),
                    "mask": str(mask_path),
                }
            )
            continue
        rgba = _prepare_rgba_from_mask(rgb_bgr, mask_gray)
        if rgba is None:
            summary["parts"].append(
                {
                    "part": part_name,
                    "part_model": part_model,
                    "status": "skipped",
                    "reason": "empty_rgba",
                    "frame": frame,
                }
            )
            continue

        raw_obj = model_obj_path(models_root, part_model)
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        raw_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            if reconstructor is None:
                reconstructor = _make_reconstructor(args)
            reconstructor.reconstruct_part(rgba, str(raw_obj))
            if not raw_obj.exists():
                raise FileNotFoundError(f"InstantMesh did not write raw mesh: {raw_obj}")
            align_info = _align_raw_hunyuan_to_pose_ready(raw_obj, out_obj, obj, part_name, frame, args)
            if not out_obj.exists():
                raise FileNotFoundError(f"InstantMesh alignment did not write pose-ready mesh: {out_obj}")
        except Exception as exc:
            summary["parts"].append(
                {
                    "part": part_name,
                    "part_model": part_model,
                    "status": "failed",
                    "frame": frame,
                    "reason": "instantmesh_or_alignment_failed",
                    "error": str(exc),
                    "raw_model": str(raw_obj),
                    "model": str(out_obj),
                }
            )
            continue

        summary["parts"].append(
            {
                "part": part_name,
                "part_model": part_model,
                "status": "success",
                "frame": frame,
                "model": str(out_obj),
                "raw_model": str(raw_obj),
                "alignment_info": align_info,
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run InstantMesh reconstruction with shared-cache outputs.")
    add_common_args(parser, METHOD)
    add_instantmesh_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_object_pipeline(args, METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/run/recon_instantmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
#   python reconstruction/run/recon_instantmesh.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --gpus 0,1 --num-workers 2 --mode multi_image --coord-dir /shared/recon_coord/instantmesh --reset-coord
#
# Key parameters:
#   --instantmesh-root/--instantmesh-config-path: InstantMesh source tree and config.
#   --instantmesh-diffusion-model/--instantmesh-unet-path/--instantmesh-model-path: checkpoint sources.
#   --instantmesh-diffusion-steps/--instantmesh-view/--instantmesh-foreground-ratio: generation controls.
#   --work-root: shared output/cache root. TSDF/DMesh methods reuse <work-root>/instantmesh.
#   --gpus/--num-workers: local multi-GPU scheduling. Usually use one worker per GPU.
