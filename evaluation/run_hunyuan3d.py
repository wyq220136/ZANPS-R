import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import torch.multiprocessing as mp
from multiprocessing import Process

import evaluate as eval_mod
from reconstruction.reconstruct_hunyuan3d import (
    HunyuanReconstructor,
    reconstruct_reference_views_hunyuan3d,
)

sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv")
_SEGMENTATION_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation")
if _SEGMENTATION_ROOT not in sys.path:
    sys.path.append(_SEGMENTATION_ROOT)
from dino_match.adaptive_weight import run_pointcloud_rerank_for_object


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
    "Printer_104016"
]


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _load_object_names(root_dir, object_source, explicit_objects):
    if explicit_objects:
        keep = [x.strip() for x in explicit_objects.split(",") if x.strip()]
        return keep
    if object_source == "sample":
        return [x for x in DEFAULT_SAMPLE_LIST if os.path.isdir(os.path.join(root_dir, x))]
    return sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))],
        key=eval_mod.natural_sort_key,
    )


def _apply_start_end(object_names, start, end):
    n = len(object_names)
    s = max(0, int(start))
    if end is None:
        e = n
    else:
        e = max(0, int(end))
    s = min(s, n)
    e = min(e, n)
    if e < s:
        return []
    return object_names[s:e]


class FileCoordinator:
    def __init__(self, coord_dir, object_names, stale_lock_sec=12 * 3600):
        self.coord_dir = _ensure_dir(coord_dir)
        self.locks_dir = _ensure_dir(os.path.join(coord_dir, "locks"))
        self.done_dir = _ensure_dir(os.path.join(coord_dir, "done"))
        self.fail_dir = _ensure_dir(os.path.join(coord_dir, "fail"))
        self.crash_dir = _ensure_dir(os.path.join(coord_dir, "crash"))
        self.manifest = os.path.join(coord_dir, "objects.json")
        self.object_names = list(object_names)
        self.stale_lock_sec = int(stale_lock_sec)
        self._init_manifest()

    def _init_manifest(self):
        if not os.path.exists(self.manifest):
            with open(self.manifest, "w", encoding="utf-8") as f:
                json.dump(self.object_names, f, ensure_ascii=False, indent=2)
            return
        with open(self.manifest, "r", encoding="utf-8") as f:
            m = json.load(f)
        self.object_names = [x for x in m if x in set(self.object_names)] or self.object_names

    def _done_path(self, obj):
        return os.path.join(self.done_dir, f"{obj}.done")

    def _fail_path(self, obj):
        return os.path.join(self.fail_dir, f"{obj}.fail")

    def _lock_path(self, obj):
        return os.path.join(self.locks_dir, f"{obj}.lock")

    def is_finished(self, obj):
        return os.path.exists(self._done_path(obj)) or os.path.exists(self._fail_path(obj))

    def all_finished(self):
        for obj in self.object_names:
            if not self.is_finished(obj):
                return False
        return True

    def claim_one(self, worker_id):
        if not self.object_names:
            return None, None
        start = (hash(worker_id) + int(time.time())) % len(self.object_names)
        ordered = self.object_names[start:] + self.object_names[:start]
        for obj in ordered:
            if self.is_finished(obj):
                continue
            lock_path = self._lock_path(obj)
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"worker": worker_id, "time": _now()}, ensure_ascii=False))
                return obj, lock_path
            except FileExistsError:
                try:
                    mtime = os.path.getmtime(lock_path)
                    if (time.time() - mtime) > self.stale_lock_sec:
                        os.remove(lock_path)
                except Exception:
                    pass
                continue
        return None, None

    def release(self, lock_path):
        if not lock_path:
            return
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass

    def mark_done(self, obj, worker_id):
        with open(self._done_path(obj), "w", encoding="utf-8") as f:
            f.write(f"{_now()} {worker_id}\n")

    def mark_fail(self, obj, worker_id, err):
        with open(self._fail_path(obj), "w", encoding="utf-8") as f:
            f.write(f"{_now()} {worker_id}\n{err}\n")

    def mark_crash(self, worker_id, err):
        safe_worker = re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_id)
        path = os.path.join(self.crash_dir, f"{safe_worker}.crash")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{_now()} {worker_id}\n{err}\n")


def _run_direct_match_for_object(args, obj_name):
    cmd = [
        sys.executable,
        os.path.join("segmentation", "direct_match.py"),
        "--data-root",
        args.root,
        "--object-source",
        "all",
        "--objects",
        obj_name,
        "--start",
        "0",
        "--end",
        "1",
        "--num-workers",
        str(args.direct_match_num_workers),
        "--match-workers",
        str(args.direct_match_match_workers),
        "--match-out-subdir",
        args.match_out_subdir,
        "--matched-mask-subdir",
        args.matched_mask_subdir,
        "--pred-mask-subdir",
        args.pred_mask_subdir,
        "--sam6d-pos-weight",
        str(args.sam6d_pos_weight),
        "--sam6d-neg-weight",
        str(args.sam6d_neg_weight),
        "--sam6d-normal-weight",
        str(args.sam6d_normal_weight),
        "--sam6d-edge-weight",
        str(args.sam6d_edge_weight),
        "--min-visible-pixels",
        str(args.direct_match_min_visible_pixels),
    ]
    if args.skip_adaptive_weight:
        cmd.append("--skip-adaptive-weight")
    else:
        cmd.append("--defer-adaptive-weight")
    if args.direct_match_use_gt_mask:
        cmd.append("--use-gt-mask-for-match")
    if args.overwrite_segmentation:
        cmd.append("--overwrite-segmentation")
    print(f"[{_now()}] [DIRECT_MATCH] {obj_name} cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=args.repo_root)


def _process_object(args, obj_name, inference):
    obj_dir = os.path.join(args.root, obj_name)
    if not os.path.isdir(obj_dir):
        raise FileNotFoundError(f"Object folder not found: {obj_dir}")

    match_json_name = args.match_json_name
    run_recon = args.stage in ("all", "recon")
    run_match = args.stage in ("all", "direct_match", "direct_match_pose")
    run_pose = args.stage in ("all", "pose_est", "direct_match_pose")

    if run_recon:
        print(f"[{_now()}] [BUILD=1] {obj_name}")
        reconstruct_reference_views_hunyuan3d(
            inference,
            obj_dir,
            min_mask_pixels=args.min_visible_pixels,
        )

    if run_match:
        _run_direct_match_for_object(args, obj_name)
        if not args.skip_adaptive_weight:
            print(f"[{_now()}] [ADAPTIVE] {obj_name}")
            run_pointcloud_rerank_for_object(
                obj_dir=obj_dir,
                match_out_dir=os.path.join(obj_dir, args.match_out_subdir),
                reranked_mask_subdir=args.adaptive_reranked_mask_subdir,
                match_result_json_name=args.match_json_name,
                reranked_json_name=args.adaptive_reranked_json_name,
                topk_per_cad=args.direct_match_topk_per_cad,
                max_points=args.adaptive_max_points,
                random_seed=args.adaptive_random_seed,
                sam6d_weight=args.adaptive_sam6d_weight,
                pointcloud_weight=args.adaptive_pointcloud_weight,
                render_weight=args.adaptive_render_weight,
                render_model_name=args.adaptive_render_model_name,
            )
            match_json_name = args.adaptive_reranked_json_name

    if run_pose and args.pose_est:
        match_json_relpath = os.path.join(args.match_out_subdir, match_json_name)
        match_json_path = os.path.join(obj_dir, match_json_relpath)
        if not os.path.exists(match_json_path):
            raise FileNotFoundError(
                f"Match result not found for pose estimation: {match_json_path}"
            )
        print(f"[{_now()}] [BUILD=0] {obj_name}")
        eval_mod.run_pose_estimation_from_match_results(
            obj_dir=obj_dir,
            match_json_relpath=match_json_relpath,
            pose_model_subdir=args.pose_model_subdir,
            pose_model_source=args.pose_model_source,
            matched_mask_subdir=args.eval_matched_mask_subdir,
            no_nvdiff=(not args.use_nvdiff),
            max_parts_per_frame=args.max_parts_per_frame,
            init_mode=args.pose_init_mode,
            ablation=args.ablation,
            edge_gate=args.edge_gate,
            edge_gate_max_angle_deg=args.edge_gate_max_angle_deg,
            edge_gate_near_ratio=args.edge_gate_near_ratio,
            min_visible_pixels=args.min_visible_pixels,
        )


def _worker_loop(worker_idx, args, coord: FileCoordinator):
    worker_id = f"{socket.gethostname()}_pid{os.getpid()}_w{worker_idx}"
    try:
        print(f"[{_now()}] Worker started: {worker_id}", flush=True)
        inference = None
        if args.stage in ("all", "recon"):
            inference = HunyuanReconstructor(
                model_path=args.hunyuan_model_path,
                subfolder=args.hunyuan_subfolder,
                num_inference_steps=args.hunyuan_steps,
                octree_resolution=args.hunyuan_octree_resolution,
                guidance_scale=args.hunyuan_guidance_scale,
            )
        while True:
            obj_name, lock_path = coord.claim_one(worker_id)
            if obj_name is None:
                if coord.all_finished():
                    print(f"[{_now()}] Worker exit (all finished): {worker_id}", flush=True)
                    return
                time.sleep(args.poll_interval_sec)
                continue
            try:
                print(f"[{_now()}] [CLAIM] {worker_id} -> {obj_name}", flush=True)
                _process_object(args, obj_name, inference)
                coord.mark_done(obj_name, worker_id)
                print(f"[{_now()}] [DONE] {worker_id} -> {obj_name}", flush=True)
            except Exception as e:
                coord.mark_fail(obj_name, worker_id, traceback.format_exc())
                print(f"[{_now()}] [FAIL] {worker_id} -> {obj_name}: {e}", flush=True)
            finally:
                coord.release(lock_path)
    except BaseException:
        err = traceback.format_exc()
        try:
            coord.mark_crash(worker_id, err)
        except Exception:
            pass
        print(f"[{_now()}] [WORKER_CRASH] {worker_id}\n{err}", flush=True)
        raise


def _reset_coord(coord_dir):
    if not os.path.isdir(coord_dir):
        return
    for sub in ("locks", "done", "fail", "crash"):
        p = os.path.join(coord_dir, sub)
        if not os.path.isdir(p):
            continue
        for name in os.listdir(p):
            fp = os.path.join(p, name)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Split pipeline: recon / direct_match / pose_est (or all).")
    parser.add_argument("--repo-root", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--root", type=str, default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/objs", help="Root directory containing object folders.")
    parser.add_argument("--object-source", type=str, default="all", choices=["sample", "all"])
    parser.add_argument("--objects", type=str, default="", help="Optional comma-separated object names.")
    parser.add_argument("--start", type=int, default=0, help="Start index of objects list (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="End index of objects list (exclusive).")
    parser.add_argument("--pose-est", action=argparse.BooleanOptionalAction, default=True, help="Whether to run evaluate(build=0).")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "recon", "direct_match", "pose_est", "direct_match_pose"],
        help="Select a single stage, run direct_match+pose_est, or run all stages.",
    )

    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi_image"], help="single: one machine; multi_image: shared storage+network, run this script on each image.")
    parser.add_argument("--num-workers", type=int, default=1, help="Local worker process count in this image.")
    parser.add_argument("--coord-dir", type=str, default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/eccv/tmp", help="Shared coordination dir. Defaults to <root>/_pipeline_coord/default_run.")
    parser.add_argument("--reset-coord", action="store_true", help="Reset done/fail/locks in coord dir before run.")
    parser.add_argument("--poll-interval-sec", type=float, default=3.0)
    parser.add_argument("--stale-lock-sec", type=int, default=12 * 3600)

    parser.add_argument("--gt-root", type=str, default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/test_intra/gt_pose_from_ann")
    parser.add_argument("--pose-init-mode", type=str, default="fast", choices=["fast", "sam"])
    parser.add_argument("--use-nvdiff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-parts-per-frame", type=int, default=20)
    parser.add_argument(
        "-ablation",
        "--ablation",
        action="store_true",
        help="Use original FoundationPose refiner without validity-mask decoder during evaluate(build=0).",
    )

    parser.add_argument("--pred-mask-subdir", type=str, default="pred_mask_direct_match")
    parser.add_argument("--match-out-subdir", type=str, default="match_vis_direct_match")
    parser.add_argument("--matched-mask-subdir", type=str, default="matched_pred_mask_direct_match")
    parser.add_argument("--match-json-name", type=str, default="match_results_sam6d_style.json")
    parser.add_argument("--eval-matched-mask-subdir", type=str, default="matched_mask")
    parser.add_argument("--pose-model-subdir", type=str, default="pose_input_models")
    parser.add_argument(
        "--pose-model-source",
        type=str,
        default="original",
        choices=["original", "gt"],
        help="Model used by pose estimation. original uses matched/reconstructed CAD dirs; gt uses dataset GT part meshes.",
    )
    parser.add_argument("--overwrite-segmentation", action="store_true")
    parser.add_argument("--direct-match-num-workers", type=int, default=1, help="SAM stage worker count for each object run.")
    parser.add_argument("--direct-match-match-workers", type=int, default=1, help="Object-level match workers in direct_match (set 1 here per-object).")
    parser.add_argument(
        "--direct-match-use-gt-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass gt_mask directly to direct_match as the CAD-mask matching candidates.",
    )
    parser.add_argument("--sam6d-pos-weight", type=float, default=0.35)
    parser.add_argument("--sam6d-neg-weight", type=float, default=0.35)
    parser.add_argument("--sam6d-normal-weight", type=float, default=0.15)
    parser.add_argument("--sam6d-edge-weight", type=float, default=0.15)
    parser.add_argument("--direct-match-min-visible-pixels", type=int, default=30, help="GT visibility threshold used inside direct_match CAD filtering.")
    parser.add_argument(
        "--skip-adaptive-weight",
        "--skip-adaptive",
        "--skip_adaptive",
        dest="skip_adaptive_weight",
        action="store_true",
        help="Skip point-cloud adaptive rerank after direct_match.",
    )
    parser.add_argument("--direct-match-topk-per-cad", type=int, default=3, help="Top-K SAM6D candidates per CAD for adaptive rerank.")
    parser.add_argument("--adaptive-reranked-mask-subdir", type=str, default="matched_pred_mask_direct_match_adaptive")
    parser.add_argument("--adaptive-reranked-json-name", type=str, default="match_results_adaptive_weight.json")
    parser.add_argument("--adaptive-max-points", type=int, default=2000)
    parser.add_argument("--adaptive-random-seed", type=int, default=2025)
    parser.add_argument("--adaptive-sam6d-weight", type=float, default=0.6)
    parser.add_argument("--adaptive-pointcloud-weight", type=float, default=0.35)
    parser.add_argument("--adaptive-render-weight", type=float, default=0.05)
    parser.add_argument("--adaptive-render-model-name", type=str, default="dinov2_vitl14")
    parser.add_argument("--min-visible-pixels", type=int, default=64, help="Visibility threshold in gt_mask for reference selection and pose-stage filtering.")
    parser.add_argument("--hunyuan-model-path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--hunyuan-subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--hunyuan-steps", type=int, default=5, help="Hunyuan3D shape generation steps.")
    parser.add_argument("--hunyuan-octree-resolution", type=int, default=256, help="Hunyuan3D octree resolution.")
    parser.add_argument("--hunyuan-guidance-scale", type=float, default=5.5, help="Hunyuan3D guidance scale.")
    parser.add_argument(
        "--edge-gate",
        action="store_true",
        help="Enable connector-vector angle gating for point-alignment init pose.",
    )
    parser.add_argument(
        "--edge-gate-max-angle-deg",
        type=float,
        default=90.0,
        help="Reject init poses whose connector-vector angle is >= this value.",
    )
    parser.add_argument(
        "--edge-gate-near-ratio",
        type=float,
        default=0.15,
        help="Near-to-body point ratio used to compute connector vector.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo_root = os.path.abspath(args.repo_root)
    args.root = os.path.abspath(args.root)
    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"root not found: {args.root}")

    all_objects = _load_object_names(args.root, args.object_source, args.objects)
    objects = _apply_start_end(all_objects, args.start, args.end)
    if not objects:
        print("No objects to process.")
        return

    coord_dir = args.coord_dir.strip() or os.path.join(args.root, "_pipeline_coord", "default_run")
    coord_dir = os.path.abspath(coord_dir)
    if args.reset_coord:
        _reset_coord(coord_dir)
    coord = FileCoordinator(coord_dir=coord_dir, object_names=objects, stale_lock_sec=args.stale_lock_sec)

    print(f"[{_now()}] mode={args.mode}, local_workers={args.num_workers}, coord_dir={coord_dir}")
    end_print = args.end if args.end is not None else len(all_objects)
    print(f"[{_now()}] objects={len(objects)} (slice [{args.start}:{end_print}] from total={len(all_objects)})")

    workers = []
    for i in range(max(1, int(args.num_workers))):
        p = Process(target=_worker_loop, args=(i, args, coord), daemon=False)
        p.start()
        workers.append(p)

    exit_code = 0
    for p in workers:
        p.join()
        if p.exitcode != 0:
            print(f"[{_now()}] Worker process exited abnormally: pid={p.pid} exitcode={p.exitcode}")
            exit_code = 1
    if exit_code != 0:
        raise RuntimeError("One or more worker processes exited abnormally.")

    print(f"[{_now()}] All local workers completed.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()



# python run.py --object-source sample --num-workers 2 --pose-est
# python run.py --mode multi_image --object-source all --num-workers 2 --pose-est
# CUDA_VISIBLE_DEVICES=0 python run.py --mode multi_image --object-source all --num-workers 4 --no-pose-est --reset-coord
# CUDA_VISIBLE_DEVICES=1 python run.py --mode multi_image --object-source all --num-workers 4 --no-pose-est
# CUDA_VISIBLE_DEVICES=0 python run.py --mode multi_image --object-source all --num-workers 6 --reset-coord --stage recon --start 0 --end 130 --edge-gate
# CUDA_VISIBLE_DEVICES=7 python run.py --mode multi_image --object-source all --num-workers 6 --stage recon --start 0 --end 130 --edge-gate
