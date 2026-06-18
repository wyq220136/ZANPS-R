import json
import os
import shutil
import subprocess
import sys
import time
import uuid

import torch.multiprocessing as mp

import run as base_run


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE_ROOT = os.path.join(SCRIPT_DIR, "dataset_train", "val")
DEFAULT_ROOT = os.path.join(SCRIPT_DIR, "dataset_train_val_work")
DEFAULT_COORD_SUBDIR = os.path.join("_pipeline_coord", "run_dataset_train_val")

_RUN_DEFAULT_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs"
_RUN_DEFAULT_GT_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/gt_pose_from_ann"
_RUN_DEFAULT_COORD_DIR = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/tmp"
_RUN_DEFAULT_ROOT_ALIASES = {
    os.path.abspath(_RUN_DEFAULT_ROOT),
    os.path.abspath("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs"),
    os.path.abspath("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_inter/objs"),
    os.path.abspath("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_inter/objs"),
}
_RUN_DEFAULT_GT_ROOT_ALIASES = {
    os.path.abspath(_RUN_DEFAULT_GT_ROOT),
    os.path.abspath("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/gt_pose_from_ann"),
    os.path.abspath("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_inter/gt_pose_from_ann"),
    os.path.abspath("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_inter/gt_pose_from_ann"),
}
_RUN_DEFAULT_COORD_DIR_ALIASES = {
    os.path.abspath(_RUN_DEFAULT_COORD_DIR),
    os.path.abspath("/inspire/qb-dev/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/tmp"),
}
PREPARE_VERSION = 2


def _argv_has(flag_name):
    prefix = flag_name + "="
    return any(arg == flag_name or arg.startswith(prefix) for arg in sys.argv[1:])


def _extract_adapter_args():
    adapter = {
        "prepare_workdir": False,
        "prepare_only": False,
        "skip_prepare_check": False,
        "prepare_wait_timeout_sec": 0.0,
        "eval_after_run": False,
        "skip_final_eval": False,
        "pose_model_source": "",
    }
    cleaned = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--prepare-workdir":
            adapter["prepare_workdir"] = True
            i += 1
            continue
        if arg == "--prepare-only":
            adapter["prepare_workdir"] = True
            adapter["prepare_only"] = True
            i += 1
            continue
        if arg in ("--skip-prepare-check", "--skip_prepare_check"):
            adapter["skip_prepare_check"] = True
            i += 1
            continue
        if arg == "--prepare-wait-timeout-sec":
            if i + 1 >= len(sys.argv):
                raise ValueError("--prepare-wait-timeout-sec requires a value")
            adapter["prepare_wait_timeout_sec"] = float(sys.argv[i + 1])
            i += 2
            continue
        if arg.startswith("--prepare-wait-timeout-sec="):
            adapter["prepare_wait_timeout_sec"] = float(arg.split("=", 1)[1])
            i += 1
            continue
        if arg == "--eval-after-run":
            adapter["eval_after_run"] = True
            i += 1
            continue
        if arg == "--skip-final-eval":
            adapter["skip_final_eval"] = True
            i += 1
            continue
        if arg == "--dataset-train-val-pose-model-source":
            if i + 1 >= len(sys.argv):
                raise ValueError("--dataset-train-val-pose-model-source requires a value")
            adapter["pose_model_source"] = sys.argv[i + 1]
            i += 2
            continue
        if arg.startswith("--dataset-train-val-pose-model-source="):
            adapter["pose_model_source"] = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--use-gt-mesh":
            adapter["pose_model_source"] = "gt_mesh"
            i += 1
            continue
        if arg == "--use-recon-mesh":
            adapter["pose_model_source"] = "recon_mesh"
            i += 1
            continue
        if arg == "--stage":
            if i + 1 >= len(sys.argv):
                raise ValueError("--stage requires a value")
            stage = sys.argv[i + 1]
            if stage == "direct_match_pose_eval":
                adapter["eval_after_run"] = True
                cleaned.extend(["--stage", "direct_match_pose"])
                i += 2
                continue
        if arg.startswith("--stage="):
            stage = arg.split("=", 1)[1]
            if stage == "direct_match_pose_eval":
                adapter["eval_after_run"] = True
                cleaned.append("--stage=direct_match_pose")
                i += 1
                continue
        cleaned.append(arg)
        i += 1
    sys.argv = cleaned
    return adapter


def _copy_if_needed(src, dst):
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        try:
            if os.path.getmtime(dst) >= os.path.getmtime(src) and os.path.getsize(dst) == os.path.getsize(src):
                return False
        except OSError:
            return False
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
    return True


def _copytree_if_needed(src, dst):
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=os.path.isdir(src))
    except OSError:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _ensure_adapter_dir(path):
    if os.path.islink(path) or os.path.isfile(path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    os.makedirs(path, exist_ok=True)


def _prefixed_frame_id(obj_name, frame_name):
    return f"{obj_name}_0_{os.path.splitext(frame_name)[0]}"


def _prepare_frame_media(src_obj_dir, work_obj_dir, obj_name):
    for dirname in ("rgb", "depth"):
        src_dir = os.path.join(src_obj_dir, dirname)
        dst_dir = os.path.join(work_obj_dir, dirname)
        if not os.path.isdir(src_dir):
            continue
        _ensure_adapter_dir(dst_dir)
        for name in sorted(os.listdir(src_dir), key=base_run.eval_mod.natural_sort_key):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ext = os.path.splitext(name)[1]
            dst_name = f"{_prefixed_frame_id(obj_name, name)}{ext}"
            _link_or_copy(os.path.join(src_dir, name), os.path.join(dst_dir, dst_name))


def _list_object_names(args):
    names = base_run._load_object_names(
        args.dataset_train_val_source_root,
        args.object_source,
        args.objects,
    )
    return base_run._apply_start_end(names, args.start, args.end)


def _prepare_gt_mask(src_obj_dir, work_obj_dir, obj_name):
    masks_dir = os.path.join(src_obj_dir, "masks")
    if not os.path.isdir(masks_dir):
        return {"gt_mask_files": 0, "parts": []}

    part_names = [
        name
        for name in os.listdir(masks_dir)
        if os.path.isdir(os.path.join(masks_dir, name))
    ]
    part_names = sorted(part_names, key=base_run.eval_mod.natural_sort_key)

    copied = 0
    for part_idx, part_name in enumerate(part_names):
        part_dir = os.path.join(masks_dir, part_name)
        frame_names = [
            name
            for name in os.listdir(part_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for frame_name in sorted(frame_names, key=base_run.eval_mod.natural_sort_key):
            frame_id = _prefixed_frame_id(obj_name, frame_name)
            src = os.path.join(part_dir, frame_name)
            dst = os.path.join(work_obj_dir, "gt_mask", frame_id, f"mask_{part_idx}.png")
            copied += int(_copy_if_needed(src, dst))

    manifest_path = os.path.join(work_obj_dir, "dataset_train_val_adapter_parts.json")
    manifest = {
        "source": "dataset_train/val",
        "gt_mask_layout": "gt_mask/<frame_id>/mask_<part_index>.png",
        "parts": [{"index": idx, "name": name} for idx, name in enumerate(part_names)],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {"gt_mask_files": copied, "parts": part_names}


def _prepare_gt_pose(src_obj_dir, args, obj_name, part_names):
    cam_params_dir = os.path.join(src_obj_dir, "cam_params")
    if not os.path.isdir(cam_params_dir):
        return 0

    gt_root = args.gt_root
    os.makedirs(gt_root, exist_ok=True)

    frame_to_pose_files = {}
    copied = 0
    for part_idx, part_name in enumerate(part_names):
        part_pose_dir = os.path.join(cam_params_dir, part_name)
        if not os.path.isdir(part_pose_dir):
            continue
        frame_names = [
            name
            for name in os.listdir(part_pose_dir)
            if name.lower().endswith(".txt")
        ]
        for frame_name in sorted(frame_names, key=base_run.eval_mod.natural_sort_key):
            frame_id = _prefixed_frame_id(obj_name, frame_name)
            pose_name = f"{frame_id}__link_{part_idx}.txt"
            src = os.path.join(part_pose_dir, frame_name)
            dst = os.path.join(gt_root, pose_name)
            copied += int(_copy_if_needed(src, dst))
            frame_to_pose_files.setdefault(frame_id, {})[part_idx] = pose_name

    for frame_id, pose_by_part in frame_to_pose_files.items():
        parts_path = os.path.join(gt_root, f"{frame_id}__parts.txt")
        pose_list = [
            pose_by_part[idx]
            for idx in sorted(pose_by_part.keys())
        ]
        content = "\n".join(pose_list) + ("\n" if pose_list else "")
        old = None
        if os.path.exists(parts_path):
            with open(parts_path, "r", encoding="utf-8") as f:
                old = f.read()
        if old != content:
            with open(parts_path, "w", encoding="utf-8") as f:
                f.write(content)
            copied += 1

    return copied


def _prepare_object_mask(src_obj_dir, work_obj_dir, obj_name):
    object_mask_dir = os.path.join(src_obj_dir, "object_mask")
    if not os.path.isdir(object_mask_dir):
        return 0

    copied = 0
    for name in sorted(os.listdir(object_mask_dir), key=base_run.eval_mod.natural_sort_key):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        src = os.path.join(object_mask_dir, name)
        ext = os.path.splitext(name)[1]
        dst = os.path.join(work_obj_dir, "mask", f"{_prefixed_frame_id(obj_name, name)}{ext}")
        copied += int(_copy_if_needed(src, dst))
    return copied


def _prepare_work_object(src_obj_dir, work_obj_dir, obj_name):
    os.makedirs(work_obj_dir, exist_ok=True)

    _prepare_frame_media(src_obj_dir, work_obj_dir, obj_name)

    src_cam_params = os.path.join(src_obj_dir, "cam_params")
    dst_cam_params = os.path.join(work_obj_dir, "cam_params")
    if os.path.exists(src_cam_params):
        _link_or_copy(src_cam_params, dst_cam_params)

    for name in ("K.txt", "meta.json"):
        src = os.path.join(src_obj_dir, name)
        dst = os.path.join(work_obj_dir, name)
        if os.path.exists(src):
            _copy_if_needed(src, dst)

    # Keep models writable in the work stack because recon writes models/view_*.
    _copytree_if_needed(os.path.join(src_obj_dir, "models"), os.path.join(work_obj_dir, "models"))


def _prepare_dataset_train_val(args, object_names):
    prepared = []
    for obj_name in object_names:
        src_obj_dir = os.path.join(args.dataset_train_val_source_root, obj_name)
        work_obj_dir = os.path.join(args.root, obj_name)
        if not os.path.isdir(src_obj_dir):
            continue
        _prepare_work_object(src_obj_dir, work_obj_dir, obj_name)
        _ensure_adapter_dir(os.path.join(work_obj_dir, "gt_mask"))
        _ensure_adapter_dir(os.path.join(work_obj_dir, "mask"))
        gt_info = _prepare_gt_mask(src_obj_dir, work_obj_dir, obj_name)
        mask_files = _prepare_object_mask(src_obj_dir, work_obj_dir, obj_name)
        gt_pose_files = _prepare_gt_pose(src_obj_dir, args, obj_name, gt_info["parts"])
        prepared.append((obj_name, len(gt_info["parts"]), gt_info["gt_mask_files"], mask_files, gt_pose_files))

    if prepared:
        print(f"[{base_run._now()}] Prepared dataset_train/val compatibility files for {len(prepared)} objects.")
        for obj_name, part_count, gt_files, mask_files, gt_pose_files in prepared:
            print(
                f"[{base_run._now()}]   {obj_name}: parts={part_count}, "
                f"gt_mask_updated={gt_files}, mask_updated={mask_files}, "
                f"gt_pose_updated={gt_pose_files}"
            )


def _prepare_state_dir(args):
    return os.path.join(args.root, "_dataset_train_val_prepare")


def _prepare_manifest_path(args):
    return os.path.join(_prepare_state_dir(args), "prepared_objects.json")


def _prepare_lock_path(args):
    return os.path.join(_prepare_state_dir(args), "prepare.lock")


def _load_prepare_manifest(args):
    path = _prepare_manifest_path(args)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_prepare_manifest(args, manifest):
    path = _prepare_manifest_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _prepared_object_set(args):
    manifest = _load_prepare_manifest(args)
    if int(manifest.get("version", -1)) != PREPARE_VERSION:
        return set()
    if os.path.abspath(str(manifest.get("source_root", ""))) != os.path.abspath(args.dataset_train_val_source_root):
        return set()
    objects = manifest.get("objects", {})
    if not isinstance(objects, dict):
        return set()
    return {
        name
        for name, info in objects.items()
        if isinstance(info, dict) and int(info.get("version", -1)) == PREPARE_VERSION
    }


def _objects_are_prepared(args, object_names):
    prepared = _prepared_object_set(args)
    return all(obj_name in prepared for obj_name in object_names)


def _mark_objects_prepared(args, object_names):
    manifest = _load_prepare_manifest(args)
    if int(manifest.get("version", -1)) != PREPARE_VERSION:
        manifest = {}
    manifest["version"] = PREPARE_VERSION
    manifest["source_root"] = os.path.abspath(args.dataset_train_val_source_root)
    manifest["work_root"] = os.path.abspath(args.root)
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    manifest.setdefault("objects", {})
    for obj_name in object_names:
        manifest["objects"][obj_name] = {
            "version": PREPARE_VERSION,
            "prepared_at": manifest["updated_at"],
        }
    _write_prepare_manifest(args, manifest)


def _try_acquire_prepare_lock(args):
    os.makedirs(_prepare_state_dir(args), exist_ok=True)
    lock_path = _prepare_lock_path(args)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "time": base_run._now()}, ensure_ascii=False))
    return lock_path


def _release_prepare_lock(lock_path):
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        pass


def _wait_for_prepare(args, object_names):
    start = time.time()
    last_log = 0.0
    while not _objects_are_prepared(args, object_names):
        now = time.time()
        if args.prepare_wait_timeout_sec > 0 and (now - start) > args.prepare_wait_timeout_sec:
            raise TimeoutError(
                f"Timed out waiting for dataset_train_val_work prepare marker: "
                f"{_prepare_manifest_path(args)}"
            )
        if now - last_log >= 30.0:
            prepared = len(_prepared_object_set(args).intersection(object_names))
            print(
                f"[{base_run._now()}] Waiting for workdir prepare: "
                f"{prepared}/{len(object_names)} objects ready. "
                f"marker={_prepare_manifest_path(args)}",
                flush=True,
            )
            last_log = now
        time.sleep(5.0)


def _handle_prepare_phase(args, object_names):
    if args.skip_prepare_check:
        print(f"[{base_run._now()}] Skip workdir prepare marker check by request.")
        return
    if not object_names:
        return
    os.makedirs(args.root, exist_ok=True)
    os.makedirs(_prepare_state_dir(args), exist_ok=True)

    if _objects_are_prepared(args, object_names):
        print(f"[{base_run._now()}] Workdir prepare already done; skip copy/adapt stage.")
        return

    if not args.prepare_workdir:
        _wait_for_prepare(args, object_names)
        print(f"[{base_run._now()}] Workdir prepare marker is ready; continue.")
        return

    lock_path = _try_acquire_prepare_lock(args)
    if lock_path is None:
        _wait_for_prepare(args, object_names)
        print(f"[{base_run._now()}] Workdir prepare marker is ready; continue.")
        return

    try:
        if _objects_are_prepared(args, object_names):
            print(f"[{base_run._now()}] Workdir prepare already done; skip copy/adapt stage.")
            return
        missing = [obj for obj in object_names if obj not in _prepared_object_set(args)]
        print(
            f"[{base_run._now()}] Preparing dataset_train_val workdir: "
            f"{len(missing)}/{len(object_names)} objects need prepare."
        )
        _prepare_dataset_train_val(args, missing)
        _mark_objects_prepared(args, missing)
        print(f"[{base_run._now()}] Workdir prepare done: {_prepare_manifest_path(args)}")
    finally:
        _release_prepare_lock(lock_path)


def parse_args():
    adapter_args = _extract_adapter_args()
    args = base_run.parse_args()

    args.repo_root = os.path.abspath(args.repo_root)
    args.dataset_train_val_source_root = DEFAULT_SOURCE_ROOT
    args.prepare_workdir = bool(adapter_args["prepare_workdir"])
    args.prepare_only = bool(adapter_args["prepare_only"])
    args.skip_prepare_check = bool(adapter_args["skip_prepare_check"])
    args.prepare_wait_timeout_sec = float(adapter_args["prepare_wait_timeout_sec"])
    args.eval_after_run = bool(adapter_args["eval_after_run"])
    args.skip_final_eval = bool(adapter_args["skip_final_eval"])
    if adapter_args["pose_model_source"]:
        src = str(adapter_args["pose_model_source"]).strip().lower()
        if src not in ("recon_mesh", "gt_mesh"):
            raise ValueError("--dataset-train-val-pose-model-source must be recon_mesh or gt_mesh")
        args.pose_model_source = "gt" if src == "gt_mesh" else "original"

    if os.path.abspath(args.root) in _RUN_DEFAULT_ROOT_ALIASES:
        args.root = DEFAULT_ROOT
    args.root = os.path.abspath(args.root)

    if (not _argv_has("--coord-dir")) and os.path.abspath(args.coord_dir) in _RUN_DEFAULT_COORD_DIR_ALIASES:
        args.coord_dir = os.path.join(args.root, DEFAULT_COORD_SUBDIR)

    local_gt_root = os.path.join(args.root, "gt_pose_from_ann")
    if os.path.abspath(args.gt_root) in _RUN_DEFAULT_GT_ROOT_ALIASES:
        args.gt_root = local_gt_root

    if not (_argv_has("--pose-eval-from-ann") or _argv_has("--no-pose-eval-from-ann")):
        args.pose_eval_from_ann = os.path.isdir(args.gt_root)

    if not _argv_has("--pred-mask-subdir"):
        args.pred_mask_subdir = "pred_mask_direct_match_dataset_train_val"
    if not _argv_has("--match-out-subdir"):
        args.match_out_subdir = "match_vis_direct_match_dataset_train_val"
    if not _argv_has("--matched-mask-subdir"):
        args.matched_mask_subdir = "matched_pred_mask_direct_match_dataset_train_val"
    if not _argv_has("--adaptive-reranked-mask-subdir"):
        args.adaptive_reranked_mask_subdir = "matched_pred_mask_direct_match_adaptive_dataset_train_val"
    if not _argv_has("--eval-matched-mask-subdir"):
        args.eval_matched_mask_subdir = "matched_mask_dataset_train_val"
    if not _argv_has("--pose-model-subdir"):
        args.pose_model_subdir = "pose_input_models_dataset_train_val"
    if not _argv_has("--pose-eval-json-name"):
        args.pose_eval_json_name = "pose_eval_from_ann_dataset_train_val.json"

    if (
        args.stage in ("direct_match", "direct_match_pose")
        and not (
            _argv_has("--direct-match-use-gt-mask")
            or _argv_has("--no-direct-match-use-gt-mask")
        )
    ):
        args.direct_match_use_gt_mask = True

    if args.stage == "direct_match_pose" and not (
        _argv_has("--skip-adaptive-weight")
        or _argv_has("--skip-adaptive")
        or _argv_has("--skip_adaptive")
    ):
        args.skip_adaptive_weight = True

    if (
        args.stage in ("direct_match", "pose_est", "direct_match_pose")
        and not args.prepare_workdir
        and not _argv_has("--skip-prepare-check")
        and not _argv_has("--skip_prepare_check")
    ):
        args.skip_prepare_check = True

    return args


def _run_final_eval(args):
    eval_script = os.path.join(args.repo_root, "scripts", "eval_run_dataset_train_val_pose_auc_ar.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"final eval script not found: {eval_script}")
    cmd = [
        sys.executable,
        eval_script,
        "--work-root",
        args.root,
        "--source-root",
        args.dataset_train_val_source_root,
        "--start",
        str(args.start),
        "--end",
        str(args.end if args.end is not None else -1),
    ]
    if args.objects.strip():
        cmd.extend(["--objects", args.objects])
    if args.gt_root.strip():
        cmd.extend(["--gt-pose-root", args.gt_root])
    if args.ablation:
        cmd.extend(["--output-tag", "ablation"])
    print(f"[{base_run._now()}] [FINAL_EVAL] cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=args.repo_root)


def main():
    args = parse_args()
    if not os.path.isdir(args.dataset_train_val_source_root):
        raise FileNotFoundError(f"dataset_train val root not found: {args.dataset_train_val_source_root}")
    os.makedirs(args.root, exist_ok=True)

    object_names = _list_object_names(args)
    _handle_prepare_phase(args, object_names)
    if args.prepare_only:
        print(f"[{base_run._now()}] prepare-only requested; exit before pipeline.")
        return

    # The work root also contains shared folders such as gt_pose_from_ann and
    # _pipeline_coord. Pin the downstream run.py object list to the dataset
    # selection already resolved from dataset_train/val, otherwise
    # --object-source all can accidentally pick those shared directories.
    args.objects = ",".join(object_names)
    args.start = 0
    args.end = None

    base_run.parse_args = lambda: args
    base_run.main()
    if args.eval_after_run and not args.skip_final_eval:
        _run_final_eval(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()



# CUDA_VISIBLE_DEVICES=0 python run_partnet.py --mode multi_image --object-source all --num-workers 6 --reset-coord --stage recon --start 0 --end 9999 --edge-gate
# CUDA_VISIBLE_DEVICES=7 python run_partnet.py --mode multi_image --object-source all --num-workers 6 --stage recon --start 0 --end 9999 --edge-gate



# CUDA_VISIBLE_DEVICES=6 python run_partnet.py --mode multi_image --object-source all --num-workers 14 --stage pose_est --start 0 --end 134 --edge-gate --coord-dir /inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_val_work/_pipeline_coord/direct_match_v1 --ablation
# CUDA_VISIBLE_DEVICES=7 python run_partnet.py --mode multi_image --object-source all --num-workers 6 --stage all --start 0 --end 9999 --edge-gate --coord-dir /inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_val_work/_pipeline_coord/direct_match_v1 --skip-adaptive-weight --ablation
