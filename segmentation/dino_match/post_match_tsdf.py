import json
import os
import re
import shutil
from collections import defaultdict

import cv2
import numpy as np

try:
    from reconstruction.tsdf_fusion import PartTSDF
except Exception:
    from tsdf_fusion import PartTSDF


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def _find_intrinsic_path(obj_dir):
    for name in ("intrinsic.txt", "cam_K.txt", "camera_intrinsic.txt", "K.txt"):
        p = os.path.join(obj_dir, name)
        if os.path.exists(p):
            return p
    return None


def _load_intrinsic(path):
    k = np.loadtxt(path, dtype=np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"invalid intrinsic shape {k.shape} from {path}")
    return k.astype(np.float32)


def _find_rgb_path(obj_dir, frame_id):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(obj_dir, "rgb", f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def _find_depth_path(obj_dir, frame_id):
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy", ".exr"):
        p = os.path.join(obj_dir, "depth", f"{frame_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def _load_depth(path):
    if path.lower().endswith(".npy"):
        depth = np.load(path).astype(np.float32)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"cannot read depth: {path}")
        depth = depth.astype(np.float32)
    depth[(~np.isfinite(depth)) | (depth < 0.0)] = 0.0
    return depth


def _load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _pose_from_item(item):
    for key in ("post_tsdf_refined_pose", "pointcloud_pose", "refined_pose", "coarse_pose"):
        pose = item.get(key, None)
        if pose is None:
            continue
        try:
            return np.asarray(pose, dtype=np.float32).reshape(4, 4), key
        except Exception:
            continue
    return None, ""


def _matrix_to_list(mat):
    return np.asarray(mat, dtype=np.float32).reshape(4, 4).tolist()


def run_post_match_tsdf_fusion_for_object(
    obj_dir,
    match_json_relpath,
    reset_tsdf_state=True,
):
    """
    Fuse TSDF only after final CAD-mask assignment is complete.

    Matching stays based on immutable SAM3D priors; this pass writes fused mesh
    paths back into the final match json for downstream pose estimation.
    """
    match_json_path = os.path.join(obj_dir, match_json_relpath)
    if not os.path.exists(match_json_path):
        print(f"[POST-TSDF][SKIP] match json missing: {match_json_path}")
        return None

    intrinsic_path = _find_intrinsic_path(obj_dir)
    if intrinsic_path is None:
        print(f"[POST-TSDF][SKIP] intrinsic missing under: {obj_dir}")
        return None
    k = _load_intrinsic(intrinsic_path)

    with open(match_json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    tsdf_root = os.path.join(obj_dir, "models", "tsdf_state")
    if reset_tsdf_state and os.path.isdir(tsdf_root):
        shutil.rmtree(tsdf_root)
    os.makedirs(tsdf_root, exist_ok=True)

    managers = {}
    integrated_by_cad = defaultdict(int)

    def get_mgr(cad_model_dir):
        key = os.path.abspath(cad_model_dir)
        if key not in managers:
            managers[key] = PartTSDF(cad_model_dir=key, tsdf_root=tsdf_root)
        return key, managers[key]

    frame_ids = sorted(all_results.keys(), key=natural_sort_key)
    for frame_id in frame_ids:
        rgb_path = _find_rgb_path(obj_dir, frame_id)
        depth_path = _find_depth_path(obj_dir, frame_id)
        if rgb_path is None or depth_path is None:
            for item in all_results.get(frame_id, []) or []:
                item["post_tsdf_integrated"] = False
                item["post_tsdf_error"] = "rgb_or_depth_missing"
            continue

        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            for item in all_results.get(frame_id, []) or []:
                item["post_tsdf_integrated"] = False
                item["post_tsdf_error"] = "rgb_read_failed"
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth = _load_depth(depth_path)

        for item in all_results.get(frame_id, []) or []:
            cad_model_dir = item.get("cad_model_dir", "")
            if not cad_model_dir or not os.path.isdir(cad_model_dir):
                item["post_tsdf_integrated"] = False
                item["post_tsdf_error"] = "cad_model_dir_missing"
                continue

            mask_path = item.get("saved_mask_path", "") or item.get("selected_mask_saved_path", "") or item.get("mask_path", "")
            if not mask_path or not os.path.exists(mask_path):
                item["post_tsdf_integrated"] = False
                item["post_tsdf_error"] = "mask_missing"
                continue

            pose, pose_key = _pose_from_item(item)
            if pose is None:
                item["post_tsdf_integrated"] = False
                item["post_tsdf_error"] = "pose_missing"
                continue

            try:
                mask = _load_mask(mask_path)
                cad_key, mgr = get_mgr(cad_model_dir)
                refined_pose = mgr.refine_pose_point_to_plane_icp(depth, mask, k, pose)
                integrated = mgr.integrate(image_rgb, depth, mask, k, refined_pose)
                item["post_tsdf_integrated"] = bool(integrated)
                item["post_tsdf_pose_source"] = pose_key
                item["post_tsdf_refined_pose"] = _matrix_to_list(refined_pose)
                item["refined_pose"] = _matrix_to_list(refined_pose)
                if integrated:
                    integrated_by_cad[cad_key] += 1
            except Exception as e:
                item["post_tsdf_integrated"] = False
                item["post_tsdf_error"] = str(e)

    updated_by_cad = {}
    for cad_key, mgr in managers.items():
        try:
            updated = mgr.update_fused_mesh() if integrated_by_cad.get(cad_key, 0) > 0 else False
            fused_path = mgr.fused_mesh_path if updated and os.path.exists(mgr.fused_mesh_path) else ""
            updated_by_cad[cad_key] = {
                "updated": bool(updated),
                "fused_model_path": fused_path,
                "integrated_count": int(integrated_by_cad.get(cad_key, 0)),
            }
        except Exception as e:
            updated_by_cad[cad_key] = {
                "updated": False,
                "fused_model_path": "",
                "integrated_count": int(integrated_by_cad.get(cad_key, 0)),
                "error": str(e),
            }

    for frame_id in frame_ids:
        for item in all_results.get(frame_id, []) or []:
            cad_model_dir = item.get("cad_model_dir", "")
            cad_key = os.path.abspath(cad_model_dir) if cad_model_dir else ""
            status = updated_by_cad.get(cad_key)
            if status is None:
                item["post_tsdf_model_updated"] = False
                continue
            item["post_tsdf_model_updated"] = bool(status.get("updated", False))
            item["post_tsdf_integrated_count_for_cad"] = int(status.get("integrated_count", 0))
            if status.get("fused_model_path"):
                item["fused_model_path"] = status["fused_model_path"]
            if status.get("error"):
                item["post_tsdf_update_error"] = status["error"]

    with open(match_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    total_integrated = sum(integrated_by_cad.values())
    total_updated = sum(1 for x in updated_by_cad.values() if x.get("updated", False))
    print(
        f"[POST-TSDF] {obj_name}: integrated_observations={total_integrated}, "
        f"cad_updated={total_updated}/{len(updated_by_cad)}, json={match_json_path}"
    )
    return match_json_path
