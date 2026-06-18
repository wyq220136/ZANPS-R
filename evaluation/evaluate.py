import os,sys
import re
import gc
import json
import shutil
import argparse

import numpy as np
import torch
import cv2

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_RECON_ROOT = os.path.join(REPO_ROOT, "reconstruction")
LOCAL_SAM3D_ROOT = os.path.join(REPO_ROOT, "sam-3d-objects")
LOCAL_SAM3D_NOTEBOOK_ROOT = os.path.join(LOCAL_SAM3D_ROOT, "notebook")
for _p in (LOCAL_RECON_ROOT, LOCAL_SAM3D_ROOT, LOCAL_SAM3D_NOTEBOOK_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

sys.path.append("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv")
sys.path.append("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/reconstruction")
from reconstruct import natural_sort_key
from point_reconstruct import (
    raw_pose_estimation,
    estimate_frame_init_poses,
    estimate_frame_init_poses_fast,
    get_inference,
)
from ref_pose.estimater import pose_single_estimation
from ref_pose.dataloader import SingleLoader

import sys
sys.path.append("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/sam-3d-objects")
sys.path.append("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/sam-3d-objects/notebook")
# keep import side effects for sam3d env bootstrap
from inference import Inference  # noqa: F401


def parse_view_and_frame(frame_id: str, obj_name: str):
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix):]
        m = re.fullmatch(r"(\d+)_(\d+)", tail)
        if m:
            return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", frame_id)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return 0, 0


def select_first_frame_per_view(obj_dir: str):
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    mask_root = os.path.join(obj_dir, "gt_mask")
    if not os.path.isdir(mask_root):
        return []

    frame_ids = sorted(
        [d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))],
        key=natural_sort_key,
    )
    grouped = {}
    for fid in frame_ids:
        view_id, frame_idx = parse_view_and_frame(fid, obj_name)
        grouped.setdefault(view_id, []).append((frame_idx, fid))

    refs = []
    for view_id in sorted(grouped.keys()):
        refs.append((view_id, sorted(grouped[view_id], key=lambda x: x[0])[0][1]))
    return refs


def _frame_part_visibility_count(obj_dir: str, frame_id: str, min_mask_pixels: int = 64) -> int:
    mask_dir = os.path.join(obj_dir, "gt_mask", frame_id)
    if not os.path.isdir(mask_dir):
        return 0
    cnt = 0
    for n in os.listdir(mask_dir):
        if not n.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        p = os.path.join(mask_dir, n)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if int(np.count_nonzero(m > 0)) >= int(min_mask_pixels):
            cnt += 1
    return cnt


def select_best_frame_per_view(obj_dir: str, min_mask_pixels: int = 64):
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    mask_root = os.path.join(obj_dir, "gt_mask")
    if not os.path.isdir(mask_root):
        return []
    frame_ids = sorted(
        [d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))],
        key=natural_sort_key,
    )
    grouped = {}
    for fid in frame_ids:
        view_id, frame_idx = parse_view_and_frame(fid, obj_name)
        vis_cnt = _frame_part_visibility_count(obj_dir, fid, min_mask_pixels=min_mask_pixels)
        grouped.setdefault(view_id, []).append((vis_cnt, frame_idx, fid))
    refs = []
    for view_id in sorted(grouped.keys()):
        cands = grouped[view_id]
        # Prefer fullest visible set; fallback to earliest frame index.
        cands.sort(key=lambda x: (-x[0], x[1]))
        refs.append((view_id, cands[0][2]))
    return refs


def reconstruct_reference_views(inference, obj_dir: str, gt_root=None, min_mask_pixels: int = 64):
    mask_root = os.path.join(obj_dir, "gt_mask")
    if not os.path.isdir(mask_root):
        print(f"[SKIP] {obj_dir}: gt_mask not found")
        return

    mask_dirs_sorted = sorted(
        [d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))],
        key=natural_sort_key,
    )
    if not mask_dirs_sorted:
        print(f"[SKIP] {obj_dir}: empty gt_mask")
        return

    refs = select_best_frame_per_view(obj_dir, min_mask_pixels=min_mask_pixels)
    if not refs:
        print(f"[SKIP] {obj_dir}: no valid reference frame found")
        return

    models_root = os.path.join(obj_dir, "models")
    os.makedirs(models_root, exist_ok=True)

    for view_id, frame_id in refs:
        if frame_id not in mask_dirs_sorted:
            continue
        frame_mask_dir = os.path.join(mask_root, frame_id)
        rgb_path = ""
        depth_path = ""
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(obj_dir, "rgb", f"{frame_id}{ext}")
            if os.path.exists(cand):
                rgb_path = cand
                break
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(obj_dir, "depth", f"{frame_id}{ext}")
            if os.path.exists(cand):
                depth_path = cand
                break
        if not rgb_path or not depth_path:
            print(f"[SKIP] {obj_dir}: missing rgb/depth for frame {frame_id}")
            continue

        view_model_dir = os.path.join(models_root, f"view_{view_id}")
        os.makedirs(view_model_dir, exist_ok=True)

        print(f"[RECON] {os.path.basename(obj_dir)} view={view_id}, ref={frame_id} -> {view_model_dir}")
        raw_pose_estimation(
            intrinsic_path=os.path.join(obj_dir, "K.txt"),
            rgb_path=rgb_path,
            index=0,
            depth_path=depth_path,
            mask_dir=frame_mask_dir,
            inference=inference,
            save_dir=view_model_dir,
            gt_root=gt_root,
            flat_output=True,
        )
    print(f"[RECON-DONE] {os.path.basename(obj_dir)} reference SAM3D models ready")


def run_pose_estimation_only(
    obj_dir: str,
    no_nvdiff: bool = False,
    max_parts_per_frame: int | None = 20,
    ablation: bool = False,
):
    rgb_folder = os.path.join(obj_dir, "rgb")
    if not os.path.isdir(rgb_folder):
        return

    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    k_path = os.path.join(obj_dir, "K.txt")
    if not os.path.exists(k_path):
        print(f"[SKIP] {obj_dir}: K.txt missing")
        return
    k = np.loadtxt(k_path)

    loader = None
    refiner_mode = "original" if ablation else "validity_mask"
    output_tag = "ablation" if ablation else ""
    rgb_files = sorted(os.listdir(rgb_folder), key=natural_sort_key)
    for rgb_name in rgb_files:
        rgb_path = os.path.join(rgb_folder, rgb_name)
        frame_id = os.path.splitext(rgb_name)[0]
        view_id, _ = parse_view_and_frame(frame_id, obj_name)

        mesh_dir = os.path.join(obj_dir, "models_test", f"view_{view_id}")
        mask_dir = os.path.join(obj_dir, "gt_mask", frame_id)
        depth_path = os.path.join(obj_dir, "depth", rgb_name)

        if not os.path.isdir(mesh_dir):
            print(f"Warning: Mesh dir {mesh_dir} not found, skipping frame {frame_id}.")
            continue
        if not os.path.isdir(mask_dir):
            print(f"Warning: Mask dir {mask_dir} not found, skipping frame {frame_id}.")
            continue
        if not os.path.exists(depth_path):
            print(f"Warning: Depth {depth_path} not found, skipping frame {frame_id}.")
            continue

        if loader is None:
            loader = SingleLoader(
                rgb_path,
                mesh_dir,
                mask_dir,
                k,
                depth_path,
                use_nvdiffrast=(not no_nvdiff),
                refiner_mode=refiner_mode,
            )
        else:
            loader.reinit(rgb_path, mesh_dir, mask_dir, k, depth_path)

        meshs = [
            os.path.join(mesh_dir, m)
            for m in sorted(os.listdir(mesh_dir), key=natural_sort_key)
            if os.path.exists(os.path.join(mesh_dir, m, "model.obj"))
        ]
        if len(meshs) <= 0:
            continue

        pose_single_estimation(
            loader,
            len(meshs),
            obj_dir,
            use_nvdiffrast=(not no_nvdiff),
            max_parts_per_frame=max_parts_per_frame,
            output_tag=output_tag,
        )
        gc.collect()
        torch.cuda.empty_cache()

    loader = None
    gc.collect()
    torch.cuda.empty_cache()


def _load_dataset_train_val_part_names(obj_dir: str):
    manifest_path = os.path.join(obj_dir, "dataset_train_val_adapter_parts.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        for item in data.get("parts", []):
            if item.get("index") is not None and item.get("name") is not None:
                pairs.append((int(item["index"]), str(item["name"])))
        return [name for _, name in sorted(pairs)]
    except Exception:
        return []


def _resolve_gt_pose_model_path(obj_dir: str, cad_part_id: int) -> str:
    part_names = _load_dataset_train_val_part_names(obj_dir)
    candidates = []
    if 0 <= int(cad_part_id) < len(part_names):
        candidates.append(os.path.join(obj_dir, "models", part_names[int(cad_part_id)], "model.obj"))
    candidates.extend(
        [
            os.path.join(obj_dir, "models", f"link_{int(cad_part_id)}", "model.obj"),
            os.path.join(obj_dir, "models", f"model_{int(cad_part_id):04d}", "model.obj"),
            os.path.join(obj_dir, "models", f"model_{int(cad_part_id)}", "model.obj"),
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _load_json_or_none(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _dataset_train_val_source_object_dir(obj_dir: str) -> str:
    manifest = _load_json_or_none(os.path.join(obj_dir, "dataset_train_val_adapter_parts.json")) or {}
    source_root = str(manifest.get("source_root", "")).strip()
    if source_root:
        cand = os.path.join(source_root, os.path.basename(obj_dir.rstrip("/\\")))
        if os.path.isdir(cand):
            return cand
    source_rel = str(manifest.get("source_rel", "")).strip()
    if source_rel:
        for base in (os.path.dirname(os.path.dirname(obj_dir)), os.getcwd()):
            cand = os.path.join(base, source_rel)
            if os.path.isdir(cand):
                return cand
    return obj_dir


def _scale_from_bbox_json(source_obj_dir: str, target_max_extent: float = 0.6):
    data = _load_json_or_none(os.path.join(source_obj_dir, "bounding_box.json"))
    if not isinstance(data, dict):
        return None
    try:
        mn = np.asarray(data.get("min", data.get("bbox_min")), dtype=np.float64)
        mx = np.asarray(data.get("max", data.get("bbox_max")), dtype=np.float64)
        if mn.shape != (3,) or mx.shape != (3,):
            return None
        max_extent = float(np.max(mx - mn))
        if max_extent <= 1e-12:
            return None
        return float(np.clip(float(target_max_extent) / max_extent, 0.25, 4.0))
    except Exception:
        return None


def _scale_from_meta(obj_dir: str):
    data = _load_json_or_none(os.path.join(obj_dir, "meta.json"))
    if not isinstance(data, dict):
        return None
    try:
        render = data.get("render", {})
        val = render.get("object_scale", data.get("object_scale", None)) if isinstance(render, dict) else data.get("object_scale", None)
        return float(val) if val is not None else None
    except Exception:
        return None


def _resolve_dataset_train_val_model_scale(obj_dir: str) -> float:
    val = _scale_from_bbox_json(_dataset_train_val_source_object_dir(obj_dir), target_max_extent=0.6)
    if val is not None:
        return float(val)
    val = _scale_from_meta(obj_dir)
    if val is not None:
        return float(val)
    return 1.0


def _copy_scaled_obj(src_model: str, dst_model: str, scale: float):
    scale = float(scale)
    if abs(scale - 1.0) <= 1e-12:
        shutil.copy2(src_model, dst_model)
        return
    with open(src_model, "r", encoding="utf-8", errors="ignore") as f_in, open(dst_model, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith("v "):
                toks = line.rstrip("\n").split()
                if len(toks) >= 4:
                    try:
                        vals = [float(toks[1]) * scale, float(toks[2]) * scale, float(toks[3]) * scale]
                        rest = toks[4:]
                        f_out.write(
                            "v {:.8f} {:.8f} {:.8f}{}\n".format(
                                vals[0],
                                vals[1],
                                vals[2],
                                (" " + " ".join(rest)) if rest else "",
                            )
                        )
                        continue
                    except Exception:
                        pass
            f_out.write(line)


def _prepare_pose_inputs_from_match_results(
    obj_dir: str,
    frame_id: str,
    frame_matches,
    pose_model_subdir: str,
    matched_mask_subdir: str,
    pose_model_source: str = "original",
    min_visible_pixels: int = 64,
):
    pose_model_frame_dir = os.path.join(obj_dir, pose_model_subdir, frame_id)
    matched_mask_frame_dir = os.path.join(obj_dir, matched_mask_subdir, frame_id)
    os.makedirs(pose_model_frame_dir, exist_ok=True)
    os.makedirs(matched_mask_frame_dir, exist_ok=True)

    valid_count = 0
    coarse_pose_overrides = {}
    for part_idx, m in enumerate(frame_matches):
        cad_model_dir = m.get("cad_model_dir", "")
        cad_part_id = int(m.get("cad_part_id", part_idx))
        if pose_model_source == "gt":
            src_model = _resolve_gt_pose_model_path(obj_dir, cad_part_id)
        else:
            src_model = os.path.join(cad_model_dir, "model.obj")
        if not os.path.exists(src_model):
            continue

        src_mask = m.get("saved_mask_path", m.get("mask_path", ""))
        if not src_mask or (not os.path.exists(src_mask)):
            continue

        try:
            gt_mask_dir = os.path.join(obj_dir, "gt_mask", frame_id)
            gt_mask_path = ""
            src_base = os.path.splitext(os.path.basename(src_mask))[0] if src_mask else ""
            cand_names = [src_base, f"mask_{cad_part_id}", f"mask_{part_idx}", str(cad_part_id), str(part_idx)]
            for cn in cand_names:
                for ext in (".png", ".jpg", ".jpeg"):
                    p = os.path.join(gt_mask_dir, f"{cn}{ext}")
                    if os.path.exists(p):
                        gt_mask_path = p
                        break
                if gt_mask_path:
                    break
            if gt_mask_path and os.path.exists(gt_mask_path):
                gm = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                if gm is None or int(np.count_nonzero(gm > 0)) < int(min_visible_pixels):
                    continue
        except Exception:
            pass

        dst_mask = os.path.join(matched_mask_frame_dir, f"mask_{part_idx}.png")
        shutil.copy2(src_mask, dst_mask)

        dst_model_dir = os.path.join(pose_model_frame_dir, f"model_{part_idx:04d}")
        os.makedirs(dst_model_dir, exist_ok=True)
        dst_model = os.path.join(dst_model_dir, "model.obj")
        if pose_model_source == "gt":
            _copy_scaled_obj(src_model, dst_model, _resolve_dataset_train_val_model_scale(obj_dir))
        else:
            shutil.copy2(src_model, dst_model)
        if pose_model_source == "original":
            for ref_name in (
                "reference_points.npy",
                "reference_points_obj.npy",
                "reference_points_cam.npy",
                "raw_pose.txt",
                "local_to_object.txt",
                "local_to_reference_camera.txt",
            ):
                src_ref = os.path.join(cad_model_dir, ref_name)
                if os.path.exists(src_ref):
                    shutil.copy2(src_ref, os.path.join(dst_model_dir, ref_name))
            pose_from_match = m.get("refined_pose", None) or m.get("coarse_pose", None)
            if pose_from_match is not None:
                try:
                    p = np.asarray(pose_from_match, dtype=np.float32).reshape(4, 4)
                    np.savetxt(os.path.join(dst_model_dir, "raw_pose.txt"), p, fmt="%.6f")
                    coarse_pose_overrides[int(part_idx)] = p.astype(np.float32)
                except Exception:
                    pass
        valid_count += 1

    return pose_model_frame_dir, matched_mask_frame_dir, valid_count, coarse_pose_overrides


def run_pose_estimation_from_match_results(
    obj_dir: str,
    match_json_relpath: str = os.path.join("match_vis_direct_match", "match_results_sam6d_style.json"),
    pose_model_subdir: str = "pose_input_models",
    matched_mask_subdir: str = "matched_mask",
    no_nvdiff: bool = False,
    max_parts_per_frame: int | None = 20,
    init_mode: str = "fast",
    ablation: bool = False,
    pose_model_source: str = "original",
    edge_gate: bool = False,
    edge_gate_max_angle_deg: float = 90.0,
    edge_gate_near_ratio: float = 0.15,
    min_visible_pixels: int = 64,
):
    if pose_model_source not in ("original", "gt"):
        raise ValueError(f"invalid pose_model_source: {pose_model_source}")
    rgb_folder = os.path.join(obj_dir, "rgb")
    if not os.path.isdir(rgb_folder):
        return

    match_json_path = os.path.join(obj_dir, match_json_relpath)
    if not os.path.exists(match_json_path):
        print(f"[SKIP] {obj_dir}: match results not found: {match_json_path}")
        return

    with open(match_json_path, "r", encoding="utf-8") as f:
        match_results = json.load(f)

    k_path = os.path.join(obj_dir, "K.txt")
    if not os.path.exists(k_path):
        print(f"[SKIP] {obj_dir}: K.txt missing")
        return
    k = np.loadtxt(k_path)

    loader = None
    refiner_mode = "original" if ablation else "validity_mask"
    output_tag = "ablation" if ablation else ""
    frame_ids = sorted(match_results.keys(), key=natural_sort_key)
    for frame_id in frame_ids:
        frame_matches = match_results.get(frame_id, [])
        if not frame_matches:
            continue

        rgb_path = os.path.join(rgb_folder, f"{frame_id}.png")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(rgb_folder, f"{frame_id}.jpg")
        if not os.path.exists(rgb_path):
            continue

        depth_name = os.path.basename(rgb_path)
        depth_path = os.path.join(obj_dir, "depth", depth_name)
        if not os.path.exists(depth_path):
            continue

        mesh_dir, mask_dir, valid_count, coarse_pose_overrides = _prepare_pose_inputs_from_match_results(
            obj_dir=obj_dir,
            frame_id=frame_id,
            frame_matches=frame_matches,
            pose_model_subdir=pose_model_subdir,
            matched_mask_subdir=matched_mask_subdir,
            pose_model_source=pose_model_source,
            min_visible_pixels=min_visible_pixels,
        )
        if valid_count <= 0:
            continue

        init_pose_overrides = dict(coarse_pose_overrides)
        if init_mode == "sam":
            estimated_overrides = estimate_frame_init_poses(
                intrinsic_path=k_path,
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_dir=mask_dir,
                inference=get_inference(),
                reference_model_dir=mesh_dir,
                random_seed=(hash(frame_id) % (2**31 - 1)),
                edge_gate=edge_gate,
                edge_max_angle_deg=edge_gate_max_angle_deg,
                edge_near_ratio=edge_gate_near_ratio,
            )
        else:
            estimated_overrides = estimate_frame_init_poses_fast(
                intrinsic_path=k_path,
                depth_path=depth_path,
                mask_dir=mask_dir,
                reference_model_dir=mesh_dir,
                random_seed=(hash(frame_id) % (2**31 - 1)),
                edge_gate=edge_gate,
                edge_max_angle_deg=edge_gate_max_angle_deg,
                edge_near_ratio=edge_gate_near_ratio,
            )
        for pid, tf in estimated_overrides.items():
            if int(pid) not in init_pose_overrides:
                init_pose_overrides[int(pid)] = tf

        if loader is None:
            loader = SingleLoader(
                rgb_path,
                mesh_dir,
                mask_dir,
                k,
                depth_path,
                use_nvdiffrast=(not no_nvdiff),
                refiner_mode=refiner_mode,
            )
        else:
            loader.reinit(rgb_path, mesh_dir, mask_dir, k, depth_path)

        meshs = [
            os.path.join(mesh_dir, m)
            for m in sorted(os.listdir(mesh_dir), key=natural_sort_key)
            if os.path.exists(os.path.join(mesh_dir, m, "model.obj"))
        ]
        if len(meshs) <= 0:
            continue

        pose_single_estimation(
            loader,
            len(meshs),
            obj_dir,
            use_nvdiffrast=(not no_nvdiff),
            max_parts_per_frame=max_parts_per_frame,
            init_pose_overrides=init_pose_overrides,
            output_tag=output_tag,
        )
        gc.collect()
        torch.cuda.empty_cache()

    loader = None
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Reconstruct/Pose pipeline switch by --build")
    parser.add_argument("--root", type=str, default="/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs", help="Root directory")
    parser.add_argument("--start", type=int, default=0, help="Start index of objects list")
    parser.add_argument("--end", type=int, default=288, help="End index of objects list (exclusive)")
    parser.add_argument("--build", type=int, default=1, help="1: only reconstruction, 0: only pose estimation")
    parser.add_argument("--gt-root", type=str, default=None, help="Optional GT pose folder for mesh-to-object alignment")
    parser.add_argument(
        "--pose-source",
        type=str,
        default="fast",
        choices=["match", "view", "fast"],
        help="Pose input source when build=0. 'match': match_results + SAM init; 'fast': match_results + depth-only init; 'view': use models/view_*",
    )
    parser.add_argument(
        "--match-json-relpath",
        type=str,
        default=os.path.join("match_vis_direct_match", "match_results_sam6d_style.json"),
        help="Relative path under each object to match results json (used when pose-source=match)",
    )
    parser.add_argument("--pose-model-subdir", type=str, default="pose_input_models")
    parser.add_argument(
        "--pose-model-source",
        type=str,
        default="original",
        choices=["original"],
        help="Model used by pose estimation when pose-source is match/fast. Only SAM3D original model is supported.",
    )
    parser.add_argument("--matched-mask-subdir", type=str, default="matched_mask")
    parser.add_argument(
        "--use-nvdiff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NvDiff in pose estimation path. Default is enabled for full FoundationPose accuracy.",
    )
    parser.add_argument(
        "--max-parts-per-frame",
        type=int,
        default=20,
        help="Upper bound of parts processed per frame during pose estimation.",
    )
    parser.add_argument("--min-visible-pixels", type=int, default=64, help="Visibility threshold in gt_mask.")
    parser.add_argument(
        "-ablation",
        "--ablation",
        action="store_true",
        help="Use original FoundationPose refiner without validity-mask decoder for build=0 ablation.",
    )
    args = parser.parse_args()

    root_dir = args.root
    
    obj_list = [os.path.join(root_dir, o) for o in sorted(os.listdir(root_dir), key=natural_sort_key)]
    if args.end <= len(obj_list):
        objs = obj_list[args.start:args.end]
    else:
        objs = obj_list[args.start:]

    print(f"Processing range: [{args.start}:{args.end}], Total objects in this batch: {len(objs)}")

    if args.build == 1:
        print("Mode: reconstruction only")
        inference = get_inference()
        for obj_dir in objs:
            reconstruct_reference_views(inference, obj_dir, gt_root=args.gt_root, min_mask_pixels=args.min_visible_pixels)
        gc.collect()
        torch.cuda.empty_cache()
        return

    print("Mode: pose estimation only")
    if args.ablation:
        print("Ablation: original FoundationPose refiner without validity-mask decoder")
    for obj_dir in objs:
        if args.pose_source in ("match", "fast"):
            run_pose_estimation_from_match_results(
                obj_dir,
                match_json_relpath=args.match_json_relpath,
                pose_model_subdir=args.pose_model_subdir,
                matched_mask_subdir=args.matched_mask_subdir,
                no_nvdiff=(not args.use_nvdiff),
                max_parts_per_frame=args.max_parts_per_frame,
                init_mode=("sam" if args.pose_source == "match" else "fast"),
                ablation=args.ablation,
                pose_model_source=args.pose_model_source,
                min_visible_pixels=args.min_visible_pixels,
            )
        else:
            run_pose_estimation_only(
                obj_dir,
                no_nvdiff=(not args.use_nvdiff),
                max_parts_per_frame=args.max_parts_per_frame,
                ablation=args.ablation,
            )


if __name__ == "__main__":
    main()
