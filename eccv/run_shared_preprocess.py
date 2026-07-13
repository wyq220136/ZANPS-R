import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path


DEFAULT_RGBD_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs"
DEFAULT_PARTNET_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train/test"
DEFAULT_SEGMENT_ANYTHING_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/related_works/segment-anything"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
DEFAULT_SAMPLE_LIST = [
    "Box_100189",
    "Bucket_100438",
    "CoffeeMachine_103074",
    "Dishwasher_12530",
    "Microwave_7263",
    "Printer_103972",
    "Remote_101028",
    "Keyboard_12738",
    "StorageFurniture_45134",
    "StorageFurniture_45779",
    "StorageFurniture_45910",
    "Toaster_103469",
    "Toilet_103234",
    "WashingMachine_103528",
    "Camera_102398",
    "Camera_102874",
    "Microwave_7349",
    "Printer_104016",
]

ECCV_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ECCV_ROOT.parent
SERVER_PROJECT_ROOT = Path("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan")
EVALUATION_ROOT = REPO_ROOT / "evaluation"
RECON_ROOT = REPO_ROOT / "reconstruction"


def _resolve_project_dir(name):
    local = REPO_ROOT / name
    if local.exists():
        return local
    return SERVER_PROJECT_ROOT / name


REF_POSE_ROOT = _resolve_project_dir("ref_pose")
SAM3D_ROOT = _resolve_project_dir("sam-3d-objects")
SAM3D_NOTEBOOK_ROOT = SAM3D_ROOT / "notebook"
_EVAL_MOD = None
_SAM_MOD = None
_CHECK_MOD = None


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def _get_eval_mod():
    global _EVAL_MOD
    if _EVAL_MOD is not None:
        return _EVAL_MOD
    for _p in (
        REPO_ROOT,
        REF_POSE_ROOT.parent,
        REF_POSE_ROOT,
        ECCV_ROOT,
        EVALUATION_ROOT,
        RECON_ROOT,
        SAM3D_ROOT,
        SAM3D_NOTEBOOK_ROOT,
    ):
        p = str(_p)
        if p not in sys.path:
            sys.path.insert(0, p)
    import evaluate as eval_mod  # noqa: E402

    _EVAL_MOD = eval_mod
    return _EVAL_MOD


def _get_sam_mod():
    global _SAM_MOD
    if _SAM_MOD is not None:
        return _SAM_MOD
    for _p in (ECCV_ROOT, ECCV_ROOT / "segmentation"):
        p = str(_p)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from segmentation import sam_utils as sam_mod  # noqa: E402
    except ImportError:
        import sam_utils as sam_mod  # noqa: E402
    _SAM_MOD = sam_mod
    return _SAM_MOD


def _get_check_mod():
    global _CHECK_MOD
    if _CHECK_MOD is not None:
        return _CHECK_MOD
    for _p in (ECCV_ROOT, ECCV_ROOT / "segmentation"):
        p = str(_p)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from segmentation import check as check_mod  # noqa: E402
    except ImportError:
        import check as check_mod  # noqa: E402
    _CHECK_MOD = check_mod
    return _CHECK_MOD


def _load_object_names(root_dir, object_source, explicit_objects):
    if explicit_objects:
        return [x.strip() for x in explicit_objects.split(",") if x.strip()]
    if object_source == "sample":
        return [x for x in DEFAULT_SAMPLE_LIST if os.path.isdir(os.path.join(root_dir, x))]
    return sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))],
        key=natural_sort_key,
    )


def _apply_start_end(object_names, start, end):
    n = len(object_names)
    s = min(max(0, int(start)), n)
    e = n if end is None else min(max(0, int(end)), n)
    if e < s:
        return []
    return object_names[s:e]


def _parse_gpu_ids(raw):
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_worker_counts_per_gpu(raw, gpu_ids, fallback_per_gpu):
    if not gpu_ids:
        return []
    text = str(raw or "").strip()
    if text:
        parts = [x.strip() for x in text.split(",") if x.strip()]
        if len(parts) == 1:
            counts = [int(parts[0])] * len(gpu_ids)
        elif len(parts) == len(gpu_ids):
            counts = [int(x) for x in parts]
        else:
            raise ValueError(
                "--sam-workers-per-gpu must be one integer or have the same "
                f"number of entries as --sam-gpu-ids ({len(gpu_ids)})."
            )
    else:
        counts = [max(1, int(fallback_per_gpu))] * len(gpu_ids)
    if any(c < 0 for c in counts):
        raise ValueError("--sam-workers-per-gpu values must be non-negative.")
    if sum(counts) <= 0:
        raise ValueError("At least one preprocess worker is required across selected GPUs.")
    return counts


def _interleaved_gpu_slots(gpu_ids, worker_counts):
    slots = []
    for idx in range(max(worker_counts)):
        for gpu_id, count in zip(gpu_ids, worker_counts):
            if idx < count:
                slots.append(str(gpu_id))
    return slots


def _round_robin_chunks(items, num_chunks):
    chunks = [[] for _ in range(max(1, int(num_chunks)))]
    for idx, item in enumerate(items):
        chunks[idx % len(chunks)].append(item)
    return [c for c in chunks if c]


def _worker_gpu_slots(gpu_ids, num_workers):
    workers = max(1, int(num_workers))
    if not gpu_ids:
        return [""] * workers
    return [gpu_ids[i % len(gpu_ids)] for i in range(workers)]


def select_objects(args):
    all_objects = _load_object_names(args.root, args.object_source, args.objects)
    return _apply_start_end(all_objects, args.start, args.end)


def parse_view_and_frame(frame_id, obj_name):
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


def _frame_part_visibility_count(obj_dir, frame_id, min_mask_pixels):
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0
    cnt = 0
    for item in _part_masks_for_frame(obj_dir, frame_id):
        mask = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if int(np.count_nonzero(mask > 0)) >= int(min_mask_pixels):
            cnt += 1
    return cnt


def _select_first_frame_per_view(obj_dir):
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    frame_ids = _part_mask_frame_ids(obj_dir)
    if not frame_ids:
        return []
    grouped = {}
    for fid in frame_ids:
        view_id, frame_idx = parse_view_and_frame(fid, obj_name)
        grouped.setdefault(view_id, []).append((frame_idx, fid))
    refs = []
    for view_id in sorted(grouped.keys()):
        refs.append((view_id, sorted(grouped[view_id], key=lambda x: x[0])[0][1]))
    return refs


def _select_best_frame_per_view(obj_dir, min_mask_pixels):
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    frame_ids = _part_mask_frame_ids(obj_dir)
    if not frame_ids:
        return []
    grouped = {}
    for fid in frame_ids:
        view_id, frame_idx = parse_view_and_frame(fid, obj_name)
        vis_cnt = _frame_part_visibility_count(obj_dir, fid, min_mask_pixels=min_mask_pixels)
        grouped.setdefault(view_id, []).append((vis_cnt, frame_idx, fid))
    refs = []
    for view_id in sorted(grouped.keys()):
        cands = grouped[view_id]
        cands.sort(key=lambda x: (-x[0], x[1]))
        refs.append((view_id, cands[0][2]))
    return refs


def _partnet_frame_id(filename):
    return os.path.splitext(filename)[0]


def _object_mask_source_dir(obj_dir):
    for name in ("object_masks", "object_mask"):
        path = os.path.join(obj_dir, name)
        if os.path.isdir(path):
            return path
    return None


def _image_names(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        [n for n in os.listdir(path) if os.path.splitext(n)[1].lower() in IMAGE_EXTS],
        key=natural_sort_key,
    )


def _write_mask_png(src, dst, overwrite=False):
    if os.path.exists(dst) and not overwrite:
        return False
    if src.lower().endswith(".png"):
        return _link_or_copy(src, dst, overwrite=overwrite)
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(f"OpenCV is required to convert non-PNG mask: {src}") from exc
    mask = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    return bool(cv2.imwrite(dst, mask))


def _part_dirs_from_masks(obj_dir):
    masks_dir = os.path.join(obj_dir, "masks")
    if not os.path.isdir(masks_dir):
        return []
    return sorted(
        [d for d in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, d))],
        key=natural_sort_key,
    )


def _part_mask_frame_ids(obj_dir):
    out = set()
    masks_dir = os.path.join(obj_dir, "masks")
    for part_name in _part_dirs_from_masks(obj_dir):
        part_dir = os.path.join(masks_dir, part_name)
        for name in _image_names(part_dir):
            out.add(_partnet_frame_id(name))
    return sorted(out, key=natural_sort_key)


def _part_masks_for_frame(obj_dir, frame_id):
    masks_dir = os.path.join(obj_dir, "masks")
    out = []
    for part_idx, part_name in enumerate(_part_dirs_from_masks(obj_dir)):
        part_dir = os.path.join(masks_dir, part_name)
        path = _find_frame_path(obj_dir, os.path.join("masks", part_name), frame_id)
        if not path:
            continue
        out.append(
            {
                "part_id": int(part_idx),
                "part_name": part_name,
                "mask_name": f"mask_{part_idx}.png",
                "path": path,
                "rel_path": os.path.relpath(path, obj_dir),
                "source_layout": "masks/<part>/<frame>",
            }
        )
    return out


def ensure_partnet_compatibility_for_object(args, obj_name):
    obj_dir = os.path.join(args.root, obj_name)
    if not os.path.isdir(obj_dir):
        raise FileNotFoundError(f"object dir not found: {obj_dir}")
    for required in ("rgb", "depth", "masks", "K.txt"):
        if not os.path.exists(os.path.join(obj_dir, required)):
            raise FileNotFoundError(f"missing required PartNet input: {obj_dir}/{required}")

    part_names = _part_dirs_from_masks(obj_dir)
    if not part_names:
        raise FileNotFoundError(f"no part mask directories under: {obj_dir}/masks")

    manifest = {
        "source": "partnet_in_place",
        "root": os.path.abspath(args.root),
        "object": obj_name,
        "part_mask_layout": "masks/<part_name>/<frame_id>.png",
        "object_mask_layout": "object_masks/<frame_id>.png or object_mask/<frame_id>.png",
        "compatibility_dirs_created": False,
        "parts": [{"index": idx, "name": name} for idx, name in enumerate(part_names)],
    }

    print(
        f"[{_now()}] [PARTNET-COMPAT] {obj_name}: parts={len(part_names)} "
        f"using_raw_masks=1 compatibility_dirs_created=0"
    )
    return manifest


def ensure_partnet_compatibility(args, objects):
    print(f"[{_now()}] [PARTNET-COMPAT] objects={len(objects)} root={args.root}")
    for obj_name in objects:
        ensure_partnet_compatibility_for_object(args, obj_name)


def _atomic_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _link_or_copy(src, dst, overwrite=False):
    if os.path.exists(dst) or os.path.islink(dst):
        if not overwrite:
            return False
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.unlink(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        rel_src = os.path.relpath(src, os.path.dirname(dst))
        os.symlink(rel_src, dst)
    except OSError:
        _atomic_copy(src, dst)
    return True


def _find_frame_path(obj_dir, subdir, frame_id):
    root = os.path.join(obj_dir, subdir)
    for ext in IMAGE_EXTS:
        path = os.path.join(root, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return ""


def _select_reference_frames(obj_dir, policy, min_visible_pixels):
    if policy == "first":
        return _select_first_frame_per_view(obj_dir)
    if policy == "best":
        return _select_best_frame_per_view(obj_dir, min_mask_pixels=min_visible_pixels)
    raise ValueError(f"unknown reference policy: {policy}")


def _has_existing_models(view_model_dir):
    if not os.path.isdir(view_model_dir):
        return False
    for name in os.listdir(view_model_dir):
        model_path = os.path.join(view_model_dir, name, "model.obj")
        if os.path.exists(model_path):
            return True
    return False


def _mirror_reference_masks(obj_dir, refs, reference_mask_subdir, overwrite=False):
    ref_root = os.path.join(obj_dir, reference_mask_subdir)
    mirrored = []
    written = 0
    for view_id, frame_id in refs:
        dst_dir = os.path.join(ref_root, frame_id)
        frame_masks = _part_masks_for_frame(obj_dir, frame_id)
        if not frame_masks:
            print(f"[{_now()}] [REF-SKIP] missing raw part masks: {obj_dir}/masks/*/{frame_id}.*")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        masks = []
        mask_details = []
        for item in frame_masks:
            src = item["path"]
            name = item["mask_name"]
            dst = os.path.join(dst_dir, name)
            written += int(_link_or_copy(src, dst, overwrite=overwrite))
            rel = os.path.join(reference_mask_subdir, frame_id, name)
            masks.append(rel)
            mask_details.append(
                {
                    "path": rel,
                    "part_id": int(item["part_id"]),
                    "part_name": item["part_name"],
                    "source_mask": item["rel_path"],
                }
            )
        mirrored.append(
            {
                "view_id": int(view_id),
                "frame_id": str(frame_id),
                "mask_dir": os.path.join(reference_mask_subdir, frame_id),
                "masks": masks,
                "mask_details": mask_details,
            }
        )
    return mirrored, written


def _clear_png_dir(path, overwrite=False):
    if os.path.isdir(path) and overwrite:
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isfile(p) and name.lower().endswith(IMAGE_EXTS):
                os.remove(p)
    os.makedirs(path, exist_ok=True)


def _load_mask_np(path):
    import cv2
    import numpy as np

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _mask_iou(a, b):
    import numpy as np

    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _find_matching_gt_part(obj_dir, frame_id, mask_np):
    best = None
    for item in _part_masks_for_frame(obj_dir, frame_id):
        try:
            gt = _load_mask_np(item["path"])
        except Exception:
            continue
        score = _mask_iou(mask_np, gt)
        if best is None or score > best["iou"]:
            best = {
                "part_id": int(item["part_id"]),
                "part_name": item["part_name"],
                "mask_name": item["mask_name"],
                "source_mask": item["rel_path"],
                "iou": float(score),
            }
    return best


def _extract_mask_id(name, fallback):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else int(fallback)


def _write_selected_mask(mask_np, path):
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask_u8 = (np.asarray(mask_np) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(path, mask_u8):
        raise RuntimeError(f"failed to write mask: {path}")


def _stage_complete_path(obj_dir, stage_subdir):
    return os.path.join(obj_dir, stage_subdir, ".complete.json")


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_complete_marker(obj_dir, stage_subdir):
    path = _stage_complete_path(obj_dir, stage_subdir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_complete_marker(obj_dir, stage_subdir, stage_name, payload):
    marker = {
        "stage": stage_name,
        "stage_subdir": stage_subdir,
        "object": os.path.basename(obj_dir.rstrip("/\\")),
        "completed_at": _now(),
    }
    marker.update(payload or {})
    _atomic_write_json(_stage_complete_path(obj_dir, stage_subdir), marker)
    return marker


def _rgb_frame_ids(obj_dir):
    rgb_dir = os.path.join(obj_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        return []
    return [
        os.path.splitext(n)[0]
        for n in _image_names(rgb_dir)
    ]


def _count_pngs(path):
    if not os.path.isdir(path):
        return 0
    return len([n for n in os.listdir(path) if n.lower().endswith(IMAGE_EXTS)])


def _remove_stage_dir(obj_dir, stage_subdir):
    path = os.path.join(obj_dir, stage_subdir)
    if os.path.isdir(path):
        shutil.rmtree(path)


def _pred_mask_complete(obj_dir, args):
    marker = _load_complete_marker(obj_dir, args.pred_mask_subdir)
    if not marker:
        return False
    expected = _rgb_frame_ids(obj_dir)
    frames = marker.get("frames", [])
    if sorted(map(str, frames), key=natural_sort_key) != sorted(map(str, expected), key=natural_sort_key):
        return False
    root = os.path.join(obj_dir, args.pred_mask_subdir)
    detail_by_frame = {
        str(item.get("frame_id")): item
        for item in marker.get("frame_details", [])
        if isinstance(item, dict)
    }
    actual_total = 0
    for frame_id in expected:
        frame_dir = os.path.join(root, frame_id)
        if not os.path.isdir(frame_dir):
            return False
        actual_count = _count_pngs(frame_dir)
        actual_total += actual_count
        detail = detail_by_frame.get(str(frame_id))
        if detail is not None and int(detail.get("num_masks", -1)) != actual_count:
            return False
    if "num_masks" in marker and int(marker.get("num_masks", -1)) != actual_total:
        return False
    return True


def _chosen_part_complete(obj_dir, args, refs):
    marker = _load_complete_marker(obj_dir, args.chosen_part_subdir)
    manifest_path = os.path.join(obj_dir, args.chosen_part_subdir, args.selected_json_name)
    if not marker or not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return False
    frames = sorted([str(fid) for _, fid in refs], key=natural_sort_key)
    marked = sorted([str(x) for x in marker.get("reference_frames", [])], key=natural_sort_key)
    if frames != marked:
        return False
    for frame_id in frames:
        if not os.path.isdir(os.path.join(obj_dir, args.chosen_part_subdir, frame_id)):
            return False
    actual_masks = 0
    for ref in manifest.get("references", []):
        if not isinstance(ref, dict):
            return False
        for rel_path in ref.get("masks", []):
            mask_path = rel_path if os.path.isabs(rel_path) else os.path.join(obj_dir, rel_path)
            if not os.path.exists(mask_path):
                return False
            actual_masks += 1
    if "num_masks" in marker and int(marker.get("num_masks", -1)) != actual_masks:
        return False
    return True


def _selected_mesh_complete(obj_dir, args):
    marker = _load_complete_marker(obj_dir, args.selected_mesh_subdir)
    if not marker:
        return False
    mesh_paths = marker.get("mesh_paths", [])
    for p in mesh_paths:
        path = p if os.path.isabs(p) else os.path.join(obj_dir, p)
        if not os.path.exists(path):
            return False
    if "expected_meshes" in marker and len(mesh_paths) < int(marker.get("expected_meshes", 0)):
        return False
    return True


def _load_object_mask_np(obj_dir, frame_id):
    for subdir in ("object_masks", "object_mask"):
        p = _find_frame_path(obj_dir, subdir, frame_id)
        if p:
            try:
                return _load_mask_np(p)
            except Exception:
                return None
    return None


def _generate_frame_sam_candidates(obj_dir, frame_id, args, mask_generator_state):
    sam_mod = _get_sam_mod()
    rgb_path = _find_frame_path(obj_dir, "rgb", frame_id)
    if not rgb_path:
        raise FileNotFoundError(f"missing rgb for frame={frame_id} under {obj_dir}/rgb")
    object_mask_path = ""
    for subdir in ("object_masks", "object_mask"):
        object_mask_path = _find_frame_path(obj_dir, subdir, frame_id)
        if object_mask_path:
            break
    if mask_generator_state.get("model") is None:
        mask_generator_state["model"] = sam_mod.create_mask_generator(
            model_cfg=args.model_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_region_area,
        )
    image_rgb, _, candidates = sam_mod.generate_candidate_masks(
        image_path=rgb_path,
        mask_path=object_mask_path if object_mask_path else None,
        model_cfg=args.model_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        iou_threshold=args.duplicate_iou_threshold,
        mask_generator=mask_generator_state["model"],
    )
    return rgb_path, image_rgb, candidates


def _materialize_pred_masks_for_object(obj_dir, args, mask_generator_state):
    import numpy as np

    if args.overwrite:
        _remove_stage_dir(obj_dir, args.pred_mask_subdir)
    elif _pred_mask_complete(obj_dir, args):
        marker = _load_complete_marker(obj_dir, args.pred_mask_subdir)
        print(f"[{_now()}] [PRED-MASK-EXISTS] {os.path.basename(obj_dir)} frames={len(marker.get('frames', []))}")
        return marker

    pred_root = os.path.join(obj_dir, args.pred_mask_subdir)
    os.makedirs(pred_root, exist_ok=True)
    frame_ids = _rgb_frame_ids(obj_dir)
    frames_meta = []
    total_masks = 0
    for frame_id in frame_ids:
        frame_dir = os.path.join(pred_root, frame_id)
        _clear_png_dir(frame_dir, overwrite=True)
        object_mask = _load_object_mask_np(obj_dir, frame_id)
        _, _, candidates = _generate_frame_sam_candidates(obj_dir, frame_id, args, mask_generator_state)
        saved = 0
        for cand_idx, ann in enumerate(candidates):
            mask_np = (np.asarray(ann["segmentation"]) > 0).astype(np.uint8)
            if object_mask is not None:
                if object_mask.shape != mask_np.shape:
                    import cv2

                    object_mask_use = cv2.resize(
                        object_mask.astype(np.uint8),
                        (mask_np.shape[1], mask_np.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    object_mask_use = object_mask
                mask_np = ((mask_np > 0) & (object_mask_use > 0)).astype(np.uint8)
            area = int(mask_np.sum())
            if area < int(args.min_visible_pixels):
                continue
            _write_selected_mask(mask_np, os.path.join(frame_dir, f"mask_{saved}.png"))
            saved += 1
        total_masks += saved
        frames_meta.append({"frame_id": str(frame_id), "num_masks": int(saved), "num_candidates": int(len(candidates))})
        print(
            f"[{_now()}] [PRED-MASK] {os.path.basename(obj_dir)} "
            f"frame={frame_id} candidates={len(candidates)} saved={saved}"
        )
    marker = _write_complete_marker(
        obj_dir,
        args.pred_mask_subdir,
        "pred_mask",
        {
            "frames": frame_ids,
            "frame_details": frames_meta,
            "num_frames": int(len(frame_ids)),
            "num_masks": int(total_masks),
            "object_mask_filter": True,
            "min_visible_pixels": int(args.min_visible_pixels),
        },
    )
    return marker


_VLM_PROBE_DONE = False


def _probe_vlm_connectivity(args):
    global _VLM_PROBE_DONE
    if _VLM_PROBE_DONE or args.reference_selector != "sam-vlm":
        return
    import numpy as np

    check_mod = _get_check_mod()
    image_np = np.zeros((32, 32, 3), dtype=np.uint8)
    image_np[:, :] = [80, 80, 80]
    mask_np = np.zeros((32, 32), dtype=np.uint8)
    mask_np[8:24, 8:24] = 255
    ok, reason = check_mod.evaluate_segmentation(image_np, mask_np, save_path=None)
    _ = ok
    if str(reason).startswith("Error occurred:"):
        raise RuntimeError(f"VLM checker is not reachable: {reason}")
    _VLM_PROBE_DONE = True
    print(f"[{_now()}] [VLM-PROBE] ok", flush=True)


def _load_pred_mask_candidates(obj_dir, frame_id, args):
    pred_dir = os.path.join(obj_dir, args.pred_mask_subdir, frame_id)
    if not os.path.isdir(pred_dir):
        return []
    out = []
    for idx, name in enumerate(_image_names(pred_dir)):
        path = os.path.join(pred_dir, name)
        try:
            mask = _load_mask_np(path)
        except Exception:
            continue
        out.append(
            {
                "candidate_index": _extract_mask_id(name, idx),
                "mask": mask,
                "area": int(mask.sum()),
                "path": path,
                "name": name,
            }
        )
    out.sort(key=lambda x: int(x["area"]), reverse=True)
    if args.max_sam_candidates_per_ref > 0:
        out = out[: args.max_sam_candidates_per_ref]
    return out


def _select_masks_with_vlm_from_pred_masks(obj_dir, frame_id, args):
    import cv2

    check_mod = _get_check_mod()
    rgb_path = _find_frame_path(obj_dir, "rgb", frame_id)
    if not rgb_path:
        raise FileNotFoundError(f"missing reference rgb for frame={frame_id} under {obj_dir}/rgb")
    image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read reference rgb: {rgb_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    candidates = _load_pred_mask_candidates(obj_dir, frame_id, args)
    selected = []
    seen_part_ids = set()

    def _evaluate_candidate(item):
        if int(item["area"]) < int(args.min_visible_pixels):
            return {**item, "ok": False, "reason": "too_small"}
        ok, reason = check_mod.evaluate_segmentation(
            image_rgb,
            item["mask"].astype("uint8") * 255,
            save_path=None,
        )
        return {**item, "ok": bool(ok), "reason": str(reason)}

    if int(args.vlm_workers) > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=int(args.vlm_workers)) as executor:
            eval_results = list(executor.map(_evaluate_candidate, candidates))
    else:
        eval_results = [_evaluate_candidate(item) for item in candidates]

    for res in eval_results:
        if not bool(res["ok"]):
            print(
                f"[{_now()}] [REF-VLM-REJECT] {os.path.basename(obj_dir)} "
                f"frame={frame_id} cand={res['candidate_index']} area={res['area']} reason={res['reason']}"
            )
            continue
        matched = _find_matching_gt_part(obj_dir, frame_id, res["mask"])
        part_id = int(matched["part_id"]) if matched is not None else int(len(selected))
        if args.unique_reference_parts and part_id in seen_part_ids:
            print(
                f"[{_now()}] [REF-VLM-DUP] {os.path.basename(obj_dir)} "
                f"frame={frame_id} cand={res['candidate_index']} part={part_id}"
            )
            continue
        seen_part_ids.add(part_id)
        selected.append({**res, "part_id": int(part_id), "matched_gt": matched})
        if args.max_selected_parts_per_ref > 0 and len(selected) >= int(args.max_selected_parts_per_ref):
            break
    return selected, len(candidates)


def _generate_reference_sam_candidates(obj_dir, frame_id, args, mask_generator_state):
    sam_mod = _get_sam_mod()
    rgb_path = _find_frame_path(obj_dir, "rgb", frame_id)
    if not rgb_path:
        raise FileNotFoundError(f"missing reference rgb for frame={frame_id} under {obj_dir}/rgb")
    object_mask_path = ""
    for subdir in ("object_masks", "object_mask"):
        object_mask_path = _find_frame_path(obj_dir, subdir, frame_id)
        if object_mask_path:
            break
    if mask_generator_state.get("model") is None:
        mask_generator_state["model"] = sam_mod.create_mask_generator(
            model_cfg=args.model_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_region_area,
        )
    image_rgb, _, candidates = sam_mod.generate_candidate_masks(
        image_path=rgb_path,
        mask_path=object_mask_path if object_mask_path else None,
        model_cfg=args.model_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        iou_threshold=args.duplicate_iou_threshold,
        mask_generator=mask_generator_state["model"],
    )
    return rgb_path, image_rgb, candidates


def _select_masks_with_vlm(obj_dir, frame_id, args, mask_generator_state):
    import numpy as np

    check_mod = _get_check_mod()
    rgb_path, image_rgb, candidates = _generate_reference_sam_candidates(
        obj_dir,
        frame_id,
        args,
        mask_generator_state,
    )
    ranked = sorted(candidates, key=lambda x: int(x.get("area", 0)), reverse=True)
    if args.max_sam_candidates_per_ref > 0:
        ranked = ranked[: args.max_sam_candidates_per_ref]

    selected = []
    seen_part_ids = set()

    def _evaluate_candidate(item):
        cand_idx, ann = item
        mask_np = (np.asarray(ann["segmentation"]) > 0).astype(np.uint8)
        area = int(mask_np.sum())
        if area < int(args.min_visible_pixels):
            return {
                "cand_idx": int(cand_idx),
                "mask": mask_np,
                "area": int(area),
                "ok": False,
                "reason": "too_small",
            }
        ok, reason = check_mod.evaluate_segmentation(
            image_rgb,
            mask_np.astype(np.uint8) * 255,
            save_path=None,
        )
        return {
            "cand_idx": int(cand_idx),
            "mask": mask_np,
            "area": int(area),
            "ok": bool(ok),
            "reason": str(reason),
        }

    eval_items = list(enumerate(ranked))
    if int(args.vlm_workers) > 1 and len(eval_items) > 1:
        with ThreadPoolExecutor(max_workers=int(args.vlm_workers)) as executor:
            eval_results = list(executor.map(_evaluate_candidate, eval_items))
    else:
        eval_results = [_evaluate_candidate(item) for item in eval_items]

    for res in eval_results:
        cand_idx = int(res["cand_idx"])
        mask_np = res["mask"]
        area = int(res["area"])
        if not bool(res["ok"]):
            print(
                f"[{_now()}] [REF-VLM-REJECT] {os.path.basename(obj_dir)} "
                f"frame={frame_id} cand={cand_idx} area={area} reason={res['reason']}"
            )
            continue
        matched = _find_matching_gt_part(obj_dir, frame_id, mask_np)
        part_id = int(matched["part_id"]) if matched is not None else int(len(selected))
        if args.unique_reference_parts and part_id in seen_part_ids:
            print(
                f"[{_now()}] [REF-VLM-DUP] {os.path.basename(obj_dir)} "
                f"frame={frame_id} cand={cand_idx} part={part_id}"
            )
            continue
        seen_part_ids.add(part_id)
        selected.append(
            {
                "mask": mask_np,
                "candidate_index": int(cand_idx),
                "area": int(area),
                "part_id": int(part_id),
                "matched_gt": matched,
                "rgb_path": rgb_path,
            }
        )
        if args.max_selected_parts_per_ref > 0 and len(selected) >= int(args.max_selected_parts_per_ref):
            break
    return selected, len(candidates)


def _materialize_sam_vlm_reference_masks(obj_dir, refs, args, mask_generator_state):
    ref_root = os.path.join(obj_dir, args.chosen_part_subdir)
    manifest_path = os.path.join(obj_dir, args.chosen_part_subdir, args.selected_json_name)
    can_reuse_existing = False
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
            can_reuse_existing = prev_manifest.get("selection_backend") == "sam-vlm"
        except Exception:
            can_reuse_existing = False
    refs_meta = []
    written = 0
    for view_id, frame_id in refs:
        dst_dir = os.path.join(ref_root, frame_id)
        if can_reuse_existing and os.path.isdir(dst_dir) and (not args.overwrite_reference):
            existing = [
                n for n in sorted(os.listdir(dst_dir), key=natural_sort_key)
                if n.lower().endswith(IMAGE_EXTS)
            ]
            if existing:
                refs_meta.append(
                    {
                        "view_id": int(view_id),
                        "frame_id": str(frame_id),
                        "mask_dir": os.path.join(args.chosen_part_subdir, frame_id),
                        "masks": [os.path.join(args.chosen_part_subdir, frame_id, n) for n in existing],
                        "selection_backend": args.reference_selector,
                        "status": "exists",
                    }
                )
                print(
                    f"[{_now()}] [REF-EXISTS] {os.path.basename(obj_dir)} "
                    f"view={view_id} frame={frame_id} masks={len(existing)}"
                )
                continue

        _clear_png_dir(dst_dir, overwrite=True)
        selected, total_candidates = _select_masks_with_vlm_from_pred_masks(
            obj_dir,
            frame_id,
            args,
        )
        masks_meta = []
        for out_idx, item in enumerate(selected):
            part_id = int(item["part_id"])
            out_name = f"mask_{part_id}.png"
            out_path = os.path.join(dst_dir, out_name)
            if os.path.exists(out_path):
                out_name = f"mask_{part_id}_{out_idx:02d}.png"
                out_path = os.path.join(dst_dir, out_name)
            _write_selected_mask(item["mask"], out_path)
            written += 1
            masks_meta.append(
                {
                    "path": os.path.join(args.chosen_part_subdir, frame_id, out_name),
                    "candidate_index": int(item["candidate_index"]),
                    "area": int(item["area"]),
                    "part_id": part_id,
                    "matched_gt": item["matched_gt"],
                    "source_pred_mask": os.path.relpath(item["path"], obj_dir),
                }
            )

        refs_meta.append(
            {
                "view_id": int(view_id),
                "frame_id": str(frame_id),
                "mask_dir": os.path.join(args.chosen_part_subdir, frame_id),
                "masks": [m["path"] for m in masks_meta],
                "mask_details": masks_meta,
                "sam_candidates": int(total_candidates),
                "selection_backend": args.reference_selector,
                "status": "selected" if masks_meta else "empty",
            }
        )
        print(
            f"[{_now()}] [REF-SAM-VLM] {os.path.basename(obj_dir)} "
            f"view={view_id} frame={frame_id} candidates={total_candidates} selected={len(masks_meta)}"
        )
    return refs_meta, written


def _materialize_reference_masks(obj_dir, refs, args, mask_generator_state):
    if args.reference_selector == "gt-mask":
        refs_meta, written = _mirror_reference_masks(
            obj_dir,
            refs,
            args.chosen_part_subdir,
            overwrite=args.overwrite_reference,
        )
        for item in refs_meta:
            item["selection_backend"] = args.reference_selector
        return refs_meta, written
    if args.reference_selector == "sam-vlm":
        return _materialize_sam_vlm_reference_masks(
            obj_dir,
            refs,
            args,
            mask_generator_state,
        )
    raise ValueError(f"unsupported reference selector: {args.reference_selector}")


def _write_reference_manifest(obj_dir, obj_name, args, refs_meta, recon_meta):
    out_dir = os.path.join(obj_dir, args.chosen_part_subdir)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "object": obj_name,
        "dataset_kind": args.dataset_kind,
        "selection_backend": args.reference_selector,
        "reference_policy": args.reference_policy,
        "reference_mask_subdir": args.chosen_part_subdir,
        "pred_mask_subdir": args.pred_mask_subdir,
        "selected_mesh_subdir": args.selected_mesh_subdir,
        "note": (
            "Reference masks are materialized in chosen_part. sam-vlm consumes shared pred_mask "
            "candidates and filters them with the VLM checker; gt-mask uses dataset part masks."
        ),
        "references": refs_meta,
        "reconstruction": recon_meta,
    }
    path = os.path.join(out_dir, args.selected_json_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _reconstruct_selected_refs(inference_state, obj_dir, refs_meta, args):
    eval_mod = _get_eval_mod()
    models_root = os.path.join(obj_dir, args.selected_mesh_subdir)
    os.makedirs(models_root, exist_ok=True)
    gt_root = args.gt_root.strip() or None
    recon_meta = []

    for ref in refs_meta:
        view_id = int(ref["view_id"])
        frame_id = str(ref["frame_id"])
        rgb_path = _find_frame_path(obj_dir, "rgb", frame_id)
        depth_path = _find_frame_path(obj_dir, "depth", frame_id)
        mask_dir = os.path.join(obj_dir, ref.get("mask_dir", os.path.join(args.chosen_part_subdir, frame_id)))
        view_model_dir = os.path.join(models_root, f"view_{view_id}")
        mask_names = _image_names(mask_dir)
        if not rgb_path or not depth_path or not os.path.isdir(mask_dir) or not mask_names:
            print(
                f"[{_now()}] [RECON-SKIP] {os.path.basename(obj_dir)} "
                f"view={view_id} ref={frame_id} missing rgb/depth/mask"
            )
            recon_meta.append(
                {
                    "view_id": int(view_id),
                    "frame_id": str(frame_id),
                    "status": "skipped_missing_input",
                    "mask_dir": mask_dir,
                    "model_dir": view_model_dir,
                }
            )
            continue
        can_reuse_models = (
            args.reference_selector == "gt-mask"
            or str(ref.get("status", "")) == "exists"
        )
        if (
            args.skip_existing_recon
            and can_reuse_models
            and (not args.overwrite_reference)
            and _has_existing_models(view_model_dir)
        ):
            print(
                f"[{_now()}] [RECON-EXISTS] {os.path.basename(obj_dir)} "
                f"view={view_id} ref={frame_id} -> {view_model_dir}"
            )
            recon_meta.append(
                {
                    "view_id": int(view_id),
                    "frame_id": str(frame_id),
                    "status": "exists",
                    "mask_dir": mask_dir,
                    "model_dir": view_model_dir,
                }
            )
            continue

        os.makedirs(view_model_dir, exist_ok=True)
        print(
            f"[{_now()}] [RECON] {os.path.basename(obj_dir)} "
            f"view={view_id} ref={frame_id} -> {view_model_dir}"
        )
        if inference_state["model"] is None:
            inference_state["model"] = eval_mod.get_inference()
        eval_mod.raw_pose_estimation(
            intrinsic_path=os.path.join(obj_dir, "K.txt"),
            rgb_path=rgb_path,
            index=0,
            depth_path=depth_path,
            mask_dir=mask_dir,
            inference=inference_state["model"],
            save_dir=view_model_dir,
            gt_root=gt_root,
            flat_output=True,
        )
        recon_meta.append(
            {
                "view_id": int(view_id),
                "frame_id": str(frame_id),
                "status": "reconstructed",
                "mask_dir": mask_dir,
                "num_masks": int(len(mask_names)),
                "model_dir": view_model_dir,
            }
        )
    return recon_meta


def _preprocess_one_object(obj_name, args, inference_state, mask_generator_state):
    obj_dir = os.path.join(args.root, obj_name)
    if not os.path.isdir(obj_dir):
        raise FileNotFoundError(f"object dir not found: {obj_dir}")
    refs = _select_reference_frames(
        obj_dir,
        policy=args.reference_policy,
        min_visible_pixels=args.min_visible_pixels,
    )
    if not refs:
        print(f"[{_now()}] [REF-SKIP] {obj_name}: no reference frame found")
        return

    pred_marker = _materialize_pred_masks_for_object(obj_dir, args, mask_generator_state)
    _ = pred_marker

    if args.reference_selector == "sam-vlm" and (args.overwrite or not _chosen_part_complete(obj_dir, args, refs)):
        _probe_vlm_connectivity(args)

    refs_meta = None
    manifest_path = os.path.join(obj_dir, args.chosen_part_subdir, args.selected_json_name)
    if (not args.overwrite) and _chosen_part_complete(obj_dir, args, refs):
        with open(manifest_path, "r", encoding="utf-8") as f:
            prev_manifest = json.load(f)
        refs_meta = prev_manifest.get("references", [])
        print(f"[{_now()}] [CHOSEN-PART-EXISTS] {obj_name}: refs={len(refs_meta)}")
    else:
        if args.overwrite:
            _remove_stage_dir(obj_dir, args.chosen_part_subdir)
        refs_meta, mask_written = _materialize_reference_masks(obj_dir, refs, args, mask_generator_state)
        print(
            f"[{_now()}] [REF-SELECT] {obj_name}: refs={len(refs_meta)} "
            f"selector={args.reference_selector} policy={args.reference_policy} masks_updated={mask_written}"
        )
        _write_complete_marker(
            obj_dir,
            args.chosen_part_subdir,
            "chosen_part",
            {
                "reference_frames": [str(fid) for _, fid in refs],
                "num_references": int(len(refs_meta)),
                "num_masks": int(sum(len(ref.get("masks", [])) for ref in refs_meta)),
                "selection_backend": args.reference_selector,
                "pred_mask_subdir": args.pred_mask_subdir,
            },
        )

    if args.skip_recon:
        recon_meta = [
            {
                "view_id": int(ref["view_id"]),
                "frame_id": str(ref["frame_id"]),
                "status": "recon_skipped_by_arg",
                "mask_dir": os.path.join(obj_dir, ref.get("mask_dir", "")),
                "model_dir": os.path.join(obj_dir, args.selected_mesh_subdir, f"view_{int(ref['view_id'])}"),
            }
            for ref in refs_meta
        ]
    elif (not args.overwrite) and _selected_mesh_complete(obj_dir, args):
        marker = _load_complete_marker(obj_dir, args.selected_mesh_subdir)
        recon_meta = marker.get("reconstruction", [])
        print(f"[{_now()}] [SELECTED-MESH-EXISTS] {obj_name}: meshes={len(marker.get('mesh_paths', []))}")
    else:
        if args.overwrite:
            _remove_stage_dir(obj_dir, args.selected_mesh_subdir)
        recon_meta = _reconstruct_selected_refs(inference_state, obj_dir, refs_meta, args)
        mesh_paths = []
        mesh_root = os.path.join(obj_dir, args.selected_mesh_subdir)
        if os.path.isdir(mesh_root):
            for root, _, files in os.walk(mesh_root):
                for name in files:
                    if name == "model.obj":
                        mesh_paths.append(os.path.relpath(os.path.join(root, name), obj_dir))
        expected_meshes = int(sum(len(ref.get("masks", [])) for ref in refs_meta))
        if len(mesh_paths) < expected_meshes:
            raise RuntimeError(
                f"selected_mesh incomplete for {obj_name}: "
                f"expected_meshes={expected_meshes} found={len(mesh_paths)}"
            )
        _write_complete_marker(
            obj_dir,
            args.selected_mesh_subdir,
            "selected_mesh",
            {
                "mesh_paths": sorted(mesh_paths, key=natural_sort_key),
                "num_meshes": int(len(mesh_paths)),
                "expected_meshes": int(expected_meshes),
                "reconstruction": recon_meta,
                "chosen_part_subdir": args.chosen_part_subdir,
            },
        )

    manifest_path = _write_reference_manifest(obj_dir, obj_name, args, refs_meta, recon_meta)
    print(f"[{_now()}] [REF-MANIFEST] {obj_name}: {manifest_path}")


def _preprocess_object_chunk_worker(payload):
    args_dict, obj_names, gpu_id = payload
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    inference_state = {"model": None}
    mask_generator_state = {"model": None}
    done = []
    print(
        f"[{_now()}] [PREPROCESS-WORKER] pid={os.getpid()} gpu={gpu_id or 'default'} "
        f"objects={len(obj_names)}",
        flush=True,
    )
    for obj_name in obj_names:
        _preprocess_one_object(obj_name, args, inference_state, mask_generator_state)
        done.append(obj_name)
    return done


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Shared in-place preprocessing for the ECCV mask-mesh experiments: "
            "prepare dataset layout, select reference part masks, and reconstruct them with SAM3D."
        )
    )
    parser.add_argument("--dataset-kind", type=str, default="partnet", choices=["rgbd", "partnet"])
    parser.add_argument("--root", type=str, default="", help="Dataset root containing object folders.")
    parser.add_argument("--object-source", type=str, default="all", choices=["sample", "all"])
    parser.add_argument("--objects", type=str, default="", help="Optional comma-separated object names.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--gt-root", type=str, default="", help="Optional pose root for SAM3D mesh alignment.")
    parser.add_argument("--min-visible-pixels", type=int, default=64)
    parser.add_argument("--reference-policy", type=str, default="first", choices=["first", "best"])
    parser.add_argument(
        "--reference-selector",
        type=str,
        default="sam-vlm",
        choices=["sam-vlm", "gt-mask"],
        help="Reference part selector backend. sam-vlm runs SAM candidates then VLM filtering; gt-mask uses dataset masks.",
    )
    parser.add_argument(
        "--pred-mask-subdir",
        type=str,
        default="pred_mask",
        help="Stage-a output folder: SAM candidate masks for every RGB frame.",
    )
    parser.add_argument(
        "--chosen-part-subdir",
        type=str,
        default="chosen_part",
        help="Stage-b output folder: VLM-selected reference-frame masks.",
    )
    parser.add_argument(
        "--selected-mesh-subdir",
        type=str,
        default="selected_mesh",
        help="Stage-c output folder: meshes reconstructed from chosen reference parts.",
    )
    parser.add_argument("--reference-mask-subdir", type=str, default="chosen_part", help="Legacy alias; kept for old launch scripts.")
    parser.add_argument("--ref-select-subdir", type=str, default="chosen_part", help="Legacy alias; kept for old launch scripts.")
    parser.add_argument("--selected-json-name", type=str, default="selected_parts.json")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild pred_mask, chosen_part, and selected_mesh stages.")
    parser.add_argument("--overwrite-reference", action="store_true")
    parser.add_argument("--overwrite-compatibility", action="store_true")
    parser.add_argument("--skip-recon", action="store_true")
    parser.add_argument("--skip-existing-recon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=str(Path(DEFAULT_SEGMENT_ANYTHING_ROOT) / "sam_vit_h_4b8939.pth"),
    )
    parser.add_argument("--sam2-checkpoint", type=str, default="")
    parser.add_argument("--sam-model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--sam-gpu-ids", type=str, default="", help="Comma-separated GPU ids used by preprocess workers.")
    parser.add_argument("--sam-procs-per-gpu", type=int, default=1, help="Preprocess/SAM worker slots per GPU.")
    parser.add_argument(
        "--sam-workers-per-gpu",
        type=str,
        default="",
        help="Optional comma-separated preprocess/SAM worker counts aligned with --sam-gpu-ids, e.g. 12,5.",
    )
    parser.add_argument("--points-per-side", type=int, default=48)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.9)
    parser.add_argument("--min-mask-region-area", type=int, default=50)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-sam-candidates-per-ref",
        type=int,
        default=30,
        help="Maximum SAM candidate masks sent to VLM per reference frame. Use <=0 for all.",
    )
    parser.add_argument(
        "--max-selected-parts-per-ref",
        type=int,
        default=0,
        help="Maximum VLM-accepted reference parts per frame. Use <=0 for unlimited.",
    )
    parser.add_argument(
        "--unique-reference-parts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When GT masks exist, keep at most one selected SAM mask per matched part id.",
    )
    parser.add_argument("--vlm-workers", type=int, default=1, help="Concurrent VLM candidate checks per reference frame.")
    parser.add_argument("--sam3d-workers", type=int, default=1, help="Concurrent object-level SAM3D/preprocess workers.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.overwrite:
        args.overwrite_reference = True
        args.overwrite_compatibility = True
        args.skip_existing_recon = False
    if not args.root:
        args.root = DEFAULT_PARTNET_ROOT if args.dataset_kind == "partnet" else DEFAULT_RGBD_ROOT
    args.root = os.path.abspath(args.root)
    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"root not found: {args.root}")

    objects = select_objects(args)
    if not objects:
        print(f"[{_now()}] No objects to preprocess.")
        return

    if args.dataset_kind == "partnet":
        ensure_partnet_compatibility(args, objects)

    inference_state = {"model": None}
    mask_generator_state = {"model": None}
    print(
        f"[{_now()}] [PREPROCESS] dataset={args.dataset_kind} objects={len(objects)} "
        f"root={args.root} selector={args.reference_selector} policy={args.reference_policy}"
    )

    gpu_ids = _parse_gpu_ids(args.sam_gpu_ids)
    explicit_gpu_workers = bool(str(args.sam_workers_per_gpu or "").strip())
    gpu_slots = []
    if gpu_ids:
        counts = _parse_worker_counts_per_gpu(args.sam_workers_per_gpu, gpu_ids, args.sam_procs_per_gpu)
        gpu_slots = _interleaved_gpu_slots(gpu_ids, counts)
    sam_slots = len(gpu_slots) if gpu_slots else 1
    if explicit_gpu_workers and gpu_slots:
        preprocess_workers = sam_slots
    else:
        preprocess_workers = max(1, int(args.sam3d_workers), sam_slots)
    preprocess_workers = min(preprocess_workers, len(objects))
    if preprocess_workers > 1:
        slots = gpu_slots[:preprocess_workers] if gpu_slots else _worker_gpu_slots(gpu_ids, preprocess_workers)
        chunks = _round_robin_chunks(objects, preprocess_workers)
        payloads = [
            (vars(args), chunk, slots[i % len(slots)])
            for i, chunk in enumerate(chunks)
        ]
        print(
            f"[{_now()}] [PREPROCESS-PARALLEL] workers={len(payloads)} "
            f"gpu_ids={','.join(gpu_ids) if gpu_ids else 'default'} "
            f"gpu_slots={','.join(slots) if gpu_slots else 'default'} "
            f"vlm_workers_per_process={args.vlm_workers}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            for done in executor.map(_preprocess_object_chunk_worker, payloads):
                print(f"[{_now()}] [PREPROCESS-WORKER-DONE] objects={len(done)}", flush=True)
    else:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            print(f"[{_now()}] [PREPROCESS-GPU] gpu={gpu_ids[0]}", flush=True)
        for obj_name in objects:
            _preprocess_one_object(obj_name, args, inference_state, mask_generator_state)
    print(f"[{_now()}] [PREPROCESS-DONE]")


if __name__ == "__main__":
    main()
