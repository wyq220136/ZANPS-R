from pathlib import Path
import sys

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

import cv2

from recon_utils import (
    DatasetObject,
    add_common_args,
    ensure_dir,
    find_image,
    list_parts,
    load_depth_m,
    mask_path_for_part_frame,
    method_models_dir,
    method_pose_ready_dir,
    model_obj_path,
    part_model_name,
    run_object_pipeline,
    select_best_frame_for_part,
)


METHOD = "sam3d"


def _import_point_reconstruct():
    repo_root = Path(__file__).resolve().parents[1]
    recon_root = Path(__file__).resolve().parent
    for p in (str(repo_root), str(recon_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from point_reconstruct import get_inference, raw_pose_estimation

    return get_inference, raw_pose_estimation


def _copy_temp_result(temp_out: Path, part_model: str, models_root: Path, pose_ready_root: Path, overwrite: bool) -> bool:
    src = temp_out / part_model / "model.obj"
    if not src.exists():
        return False
    for root in (models_root, pose_ready_root):
        dst_dir = root / part_model
        dst = dst_dir / "model.obj"
        if dst.exists() and not overwrite:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in (temp_out / part_model).iterdir():
            if file.is_file():
                shutil.copy2(file, dst_dir / file.name)
    return True


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace) -> Dict[str, object]:
    get_inference, raw_pose_estimation = _import_point_reconstruct()
    inference = get_inference()

    work_root = Path(args.work_root).resolve()
    models_root = ensure_dir(method_models_dir(work_root, METHOD, args.split, obj.name))
    pose_ready_root = ensure_dir(method_pose_ready_dir(work_root, METHOD, args.split, obj.name))
    parts = list_parts(obj)
    summary = {"method": METHOD, "object": obj.name, "parts": []}

    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        out_obj = model_obj_path(pose_ready_root, part_model)
        if out_obj.exists() and not args.overwrite:
            summary["parts"].append({"part": part_name, "status": "cached", "model": str(out_obj)})
            continue

        frame = select_best_frame_for_part(obj, part_name, args.min_mask_pixels)
        if frame is None:
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "no_visible_frame"})
            continue

        rgb_path = find_image(obj.rgb_dir, frame)
        depth_path = find_image(obj.depth_dir, frame)
        mask_path = mask_path_for_part_frame(obj, part_name, frame)
        if rgb_path is None or depth_path is None or mask_path is None:
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "missing_rgb_depth_mask"})
            continue

        # point_reconstruct.raw_pose_estimation expects frame-major mask_dir.
        with tempfile.TemporaryDirectory(prefix=f"sam3d_{obj.name}_{part_model}_") as td:
            temp_root = Path(td)
            frame_mask_dir = temp_root / "masks" / frame
            frame_mask_dir.mkdir(parents=True, exist_ok=True)
            temp_mask = frame_mask_dir / f"mask_{part_model[-4:]}.png"
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(temp_mask), mask_img)
            temp_out = temp_root / "out"
            try:
                raw_pose_estimation(
                    intrinsic_path=str(obj.k_path),
                    rgb_path=str(rgb_path),
                    index=0,
                    depth_path=str(depth_path),
                    mask_dir=str(temp_root / "masks"),
                    inference=inference,
                    save_dir=str(temp_out),
                    gt_root=None,
                    flat_output=True,
                )
                ok = _copy_temp_result(temp_out, part_model, models_root, pose_ready_root, args.overwrite)
            except Exception as e:
                summary["parts"].append({"part": part_name, "status": "failed", "frame": frame, "error": str(e)})
                continue

        summary["parts"].append(
            {
                "part": part_name,
                "part_model": part_model,
                "status": "success" if ok else "failed",
                "frame": frame,
                "model": str(out_obj),
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run SAM3D point reconstruction with shared-cache outputs.")
    add_common_args(parser, METHOD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_object_pipeline(args, METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/recon_sam3d.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 1
#   python reconstruction/recon_sam3d.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --gpus 0,1,2,3 --num-workers 4 --mode multi_image --coord-dir /shared/recon_coord/sam3d --reset-coord
#
# Key parameters:
#   --data-root: dataset root whose split contains object folders like dataset_train/val/<object>.
#   --work-root: shared output/cache root. Other methods reuse <work-root>/sam3d.
#   --gpus: comma-separated GPU ids assigned round-robin to workers.
#   --num-workers: local worker processes in this image.
#   --overwrite: rebuild existing SAM3D outputs.
#   --min-mask-pixels: minimum part-mask visibility for selecting a reference frame.
