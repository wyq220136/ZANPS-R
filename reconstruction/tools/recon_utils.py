import argparse
import faulthandler
import json
import os
import re
import shutil
import signal
import socket
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg")

SAPIENCAM_TO_CVCAM = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def natural_sort_key(s: object):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_part_id(name: str, fallback: int = 0) -> int:
    m = re.search(r"(\d+)", str(name))
    if m:
        return int(m.group(1))
    return int(fallback)


def part_model_name(part_name: str, fallback_idx: int = 0) -> str:
    return f"model_{parse_part_id(part_name, fallback_idx):04d}"


@dataclass
class DatasetObject:
    data_root: Path
    split: str
    name: str

    @property
    def root(self) -> Path:
        return self.data_root / self.split / self.name

    @property
    def rgb_dir(self) -> Path:
        return self.root / "rgb"

    @property
    def depth_dir(self) -> Path:
        return self.root / "depth"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def cam_params_dir(self) -> Path:
        return self.root / "cam_params"

    @property
    def gt_models_dir(self) -> Path:
        upper = self.root / "Models"
        return upper if upper.exists() else self.root / "models"

    @property
    def object_mask_dir(self) -> Path:
        for name in ("objectmask", "object_mask"):
            p = self.root / name
            if p.exists():
                return p
        return self.root / "object_mask"

    @property
    def k_path(self) -> Path:
        return self.root / "K.txt"


def list_objects(data_root: Path, split: str, object_source: str, objects: str = "") -> List[str]:
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"split root not found: {split_root}")
    if objects.strip():
        return [x.strip() for x in objects.split(",") if x.strip()]
    out = [p.name for p in split_root.iterdir() if p.is_dir()]
    return sorted(out, key=natural_sort_key)


def slice_objects(names: Sequence[str], start: int, end: Optional[int]) -> List[str]:
    n = len(names)
    s = min(max(0, int(start)), n)
    e = n if end is None else min(max(0, int(end)), n)
    if e < s:
        return []
    return list(names[s:e])


def list_parts(obj: DatasetObject) -> List[str]:
    if not obj.masks_dir.is_dir():
        raise FileNotFoundError(f"masks folder not found: {obj.masks_dir}")
    return sorted([p.name for p in obj.masks_dir.iterdir() if p.is_dir()], key=natural_sort_key)


def list_frames(obj: DatasetObject) -> List[str]:
    frames = {p.stem for p in obj.depth_dir.iterdir() if p.is_file()} if obj.depth_dir.is_dir() else set()
    if obj.rgb_dir.is_dir():
        frames &= {p.stem for p in obj.rgb_dir.iterdir() if p.is_file()}
    if obj.object_mask_dir.is_dir():
        frames &= {p.stem for p in obj.object_mask_dir.iterdir() if p.is_file()}
    return sorted(frames, key=natural_sort_key)


def find_image(folder: Path, frame: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = folder / f"{frame}{ext}"
        if p.exists():
            return p
    return None


def mask_path_for_part_frame(obj: DatasetObject, part_name: str, frame: str) -> Optional[Path]:
    part_dir = obj.masks_dir / part_name
    for ext in IMAGE_EXTS:
        p = part_dir / f"{frame}{ext}"
        if p.exists():
            return p
    return None


def load_k(obj: DatasetObject) -> np.ndarray:
    if not obj.k_path.exists():
        raise FileNotFoundError(f"K.txt not found: {obj.k_path}")
    k = np.loadtxt(obj.k_path).astype(np.float32)
    return k.reshape(3, 3)


def load_depth_m(path: Path, depth_scale: float = 1000.0) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"failed to read depth: {path}")
    depth = depth.astype(np.float32)
    if depth.size > 0 and float(np.nanmax(depth)) > 50.0:
        depth = depth / float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_mask(path: Path, shape_hw: Optional[Tuple[int, int]] = None, threshold: int = 127) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if shape_hw is not None and mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > int(threshold)


def load_rgb(path: Path) -> np.ndarray:
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"failed to read rgb: {path}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def load_pose(path: Path, convention: str = "cv") -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"invalid pose shape {pose.shape}: {path}")
    if convention == "sapien":
        pose = SAPIENCAM_TO_CVCAM @ pose
    return pose.astype(np.float32)


def pose_path_for_part_frame(obj: DatasetObject, part_name: str, frame: str) -> Optional[Path]:
    part_dir = obj.cam_params_dir / part_name
    p = part_dir / f"{frame}.txt"
    return p if p.exists() else None


def select_best_frame_for_part(obj: DatasetObject, part_name: str, min_mask_pixels: int) -> Optional[str]:
    best = None
    for frame in list_frames(obj):
        mp = mask_path_for_part_frame(obj, part_name, frame)
        if mp is None:
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        count = int(np.count_nonzero(m > 0))
        if count < int(min_mask_pixels):
            continue
        if best is None or count > best[0]:
            best = (count, frame)
    return None if best is None else best[1]


def method_object_dir(work_root: Path, method: str, split: str, object_name: str) -> Path:
    return work_root / method / split / object_name


def method_models_dir(work_root: Path, method: str, split: str, object_name: str) -> Path:
    return method_object_dir(work_root, method, split, object_name) / "models" / "view_0"


def method_pose_ready_dir(work_root: Path, method: str, split: str, object_name: str) -> Path:
    return method_object_dir(work_root, method, split, object_name) / "pose_ready_models" / "view_0"


def model_obj_path(root: Path, part_model: str) -> Path:
    return root / part_model / "model.obj"


def method_has_all_parts(work_root: Path, method: str, split: str, object_name: str, parts: Sequence[str]) -> bool:
    pose_root = method_pose_ready_dir(work_root, method, split, object_name)
    return all(model_obj_path(pose_root, part_model_name(p, i)).exists() for i, p in enumerate(parts))


def copy_model_tree(src: Path, dst: Path, overwrite: bool = False) -> int:
    count = 0
    if not src.exists():
        return 0
    for obj_file in src.rglob("model.obj"):
        rel = obj_file.parent.relative_to(src)
        out_dir = dst / rel
        out_file = out_dir / "model.obj"
        if out_file.exists() and not overwrite:
            count += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(obj_file, out_file)
        for aux in obj_file.parent.glob("model.*"):
            if aux.name == "model.obj":
                continue
            try:
                shutil.copy2(aux, out_dir / aux.name)
            except Exception:
                pass
        count += 1
    return count


class FileCoordinator:
    def __init__(self, coord_dir: Path, object_names: Sequence[str], stale_lock_sec: int = 12 * 3600):
        self.coord_dir = ensure_dir(coord_dir)
        self.locks_dir = ensure_dir(self.coord_dir / "locks")
        self.done_dir = ensure_dir(self.coord_dir / "done")
        self.fail_dir = ensure_dir(self.coord_dir / "fail")
        self.crash_dir = ensure_dir(self.coord_dir / "crash")
        self.manifest = self.coord_dir / "objects.json"
        self.object_names = list(object_names)
        self.stale_lock_sec = int(stale_lock_sec)
        if not self.manifest.exists():
            write_json(self.manifest, self.object_names)

    def _path(self, sub: str, obj: str, suffix: str) -> Path:
        return self.coord_dir / sub / f"{obj}{suffix}"

    def is_finished(self, obj: str) -> bool:
        return self._path("done", obj, ".done").exists() or self._path("fail", obj, ".fail").exists()

    def all_finished(self) -> bool:
        return all(self.is_finished(x) for x in self.object_names)

    def claim_one(self, worker_id: str) -> Tuple[Optional[str], Optional[Path]]:
        ordered = list(self.object_names)
        if ordered:
            start = (hash(worker_id) + int(time.time())) % len(ordered)
            ordered = ordered[start:] + ordered[:start]
        for obj in ordered:
            if self.is_finished(obj):
                continue
            lock_path = self._path("locks", obj, ".lock")
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"worker": worker_id, "time": now()}, ensure_ascii=False))
                return obj, lock_path
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > self.stale_lock_sec:
                        lock_path.unlink()
                except Exception:
                    pass
        return None, None

    def release(self, lock_path: Optional[Path]) -> None:
        if lock_path is not None:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass

    def mark_done(self, obj: str, worker_id: str) -> None:
        self._path("done", obj, ".done").write_text(f"{now()} {worker_id}\n", encoding="utf-8")

    def mark_fail(self, obj: str, worker_id: str, err: str) -> None:
        self._path("fail", obj, ".fail").write_text(f"{now()} {worker_id}\n{err}\n", encoding="utf-8")

    def mark_crash(self, worker_id: str, err: str) -> None:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_id)
        (self.crash_dir / f"{safe}.crash").write_text(f"{now()} {worker_id}\n{err}\n", encoding="utf-8")


def reset_coord(coord_dir: Path) -> None:
    if not coord_dir.is_dir():
        return
    for sub in ("locks", "done", "fail", "crash"):
        d = coord_dir / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)


def _read_failure_reports(coord_dir: Path, limit: int = 3) -> List[str]:
    reports: List[str] = []
    for sub, suffix in (("fail", "*.fail"), ("crash", "*.crash")):
        d = coord_dir / sub
        if not d.is_dir():
            continue
        paths = sorted(d.glob(suffix), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[: max(0, int(limit))]:
            try:
                text = p.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                text = f"<failed to read {p}: {e}>"
            reports.append(f"{p}:\n{text}")
    return reports


def _read_worker_logs(coord_dir: Path, limit: int = 3) -> List[str]:
    d = coord_dir / "worker_logs"
    if not d.is_dir():
        return []
    reports: List[str] = []
    paths = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[: max(0, int(limit))]:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            text = "\n".join(lines[-80:])
        except Exception as e:
            text = f"<failed to read {p}: {e}>"
        reports.append(f"{p}:\n{text}")
    return reports


def _read_active_locks(coord_dir: Path, limit: int = 5) -> List[str]:
    d = coord_dir / "locks"
    if not d.is_dir():
        return []
    reports: List[str] = []
    paths = sorted(d.glob("*.lock"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[: max(0, int(limit))]:
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            text = f"<failed to read {p}: {e}>"
        age = time.time() - p.stat().st_mtime
        reports.append(f"{p} age_sec={age:.1f}:\n{text}")
    return reports


def _format_process_exit(p: Process) -> str:
    code = p.exitcode
    if code is None:
        return f"pid={p.pid} still running"
    if code < 0:
        signum = -code
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = f"SIG{signum}"
        hint = "possible OOM kill" if sig_name == "SIGKILL" else "native crash/abort possible"
        return f"pid={p.pid} exitcode={code} signal={sig_name} ({hint})"
    return f"pid={p.pid} exitcode={code}"


def _parse_gpu_ids(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_workers_per_gpu(raw: str, gpus: List[str]) -> List[int]:
    if not gpus:
        return []
    text = str(raw or "").strip()
    if not text:
        return []
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if len(parts) == 1:
        counts = [int(parts[0])] * len(gpus)
    elif len(parts) == len(gpus):
        counts = [int(x) for x in parts]
    else:
        raise ValueError(
            "--workers-per-gpu must be one integer or have the same "
            f"number of entries as --gpus ({len(gpus)})."
        )
    if any(c < 0 for c in counts):
        raise ValueError("--workers-per-gpu values must be non-negative.")
    if sum(counts) <= 0:
        raise ValueError("At least one worker is required across selected GPUs.")
    return counts


def _interleaved_gpu_slots(gpus: List[str], worker_counts: List[int]) -> List[str]:
    slots: List[str] = []
    for idx in range(max(worker_counts)):
        for gpu_id, count in zip(gpus, worker_counts):
            if idx < count:
                slots.append(str(gpu_id))
    return slots


def add_common_args(parser: argparse.ArgumentParser, method: str) -> None:
    parser.add_argument("--data-root", type=str, default="dataset_train", help="Root containing split/object dirs.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split, e.g. val.")
    parser.add_argument("--work-root", type=str, default="reconstruction_runs", help="Shared output/cache root.")
    parser.add_argument("--object-source", choices=["all"], default="all")
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--mode", choices=["single", "multi_image"], default="single")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--coord-dir", type=str, default="", help=f"Shared coord dir. Default: <work-root>/_coord/{method}.")
    parser.add_argument("--reset-coord", action="store_true")
    parser.add_argument("--stale-lock-sec", type=int, default=12 * 3600)
    parser.add_argument("--poll-interval-sec", type=float, default=3.0)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids assigned round-robin to workers.")
    parser.add_argument(
        "--workers-per-gpu",
        type=str,
        default="",
        help="Optional comma-separated worker counts aligned with --gpus, e.g. 12,5.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--pose-convention", choices=["cv", "sapien"], default="sapien")


def run_object_pipeline(args, method: str, worker_fn: Callable[[DatasetObject, argparse.Namespace], Dict[str, object]]) -> None:
    data_root = Path(args.data_root).resolve()
    work_root = Path(args.work_root).resolve()
    all_objects = list_objects(data_root, args.split, args.object_source, args.objects)
    objects = slice_objects(all_objects, args.start, args.end)
    if not objects:
        print("No objects to process.")
        return

    coord_dir = Path(args.coord_dir).resolve() if args.coord_dir.strip() else work_root / "_coord" / method
    if args.reset_coord:
        reset_coord(coord_dir)
    coord = FileCoordinator(coord_dir, objects, stale_lock_sec=args.stale_lock_sec)
    gpus = _parse_gpu_ids(args.gpus)
    workers_per_gpu_raw = str(getattr(args, "workers_per_gpu", "") or "").strip()
    if workers_per_gpu_raw and not gpus:
        raise ValueError("--workers-per-gpu requires --gpus to be set.")
    per_gpu_counts = _parse_workers_per_gpu(workers_per_gpu_raw, gpus)
    if per_gpu_counts:
        worker_gpu_slots = _interleaved_gpu_slots(gpus, per_gpu_counts)
        num_workers = len(worker_gpu_slots)
    else:
        num_workers = max(1, int(args.num_workers))
        worker_gpu_slots = [gpus[i % len(gpus)] for i in range(num_workers)] if gpus else []

    print(
        f"[{now()}] method={method} split={args.split} objects={len(objects)} "
        f"workers={num_workers} gpus={','.join(gpus) if gpus else 'default'} "
        f"worker_gpu_slots={','.join(worker_gpu_slots) if worker_gpu_slots else 'default'} "
        f"coord={coord_dir}"
    )

    def loop(worker_idx: int):
        if worker_gpu_slots:
            os.environ["CUDA_VISIBLE_DEVICES"] = worker_gpu_slots[worker_idx % len(worker_gpu_slots)]
        worker_id = f"{socket.gethostname()}_pid{os.getpid()}_w{worker_idx}"
        worker_log_dir = ensure_dir(coord_dir / "worker_logs")
        worker_log_path = worker_log_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', worker_id)}.log"
        worker_log = worker_log_path.open("a", encoding="utf-8")
        faulthandler.enable(file=worker_log, all_threads=True)
        worker_log.write(f"[{now()}] worker_start id={worker_id} pid={os.getpid()}\n")
        worker_log.flush()
        try:
            while True:
                obj_name, lock_path = coord.claim_one(worker_id)
                if obj_name is None:
                    if coord.all_finished():
                        return
                    time.sleep(float(args.poll_interval_sec))
                    continue
                try:
                    print(f"[{now()}] [CLAIM] {worker_id} -> {obj_name}", flush=True)
                    worker_log.write(f"[{now()}] claim {obj_name}\n")
                    worker_log.flush()
                    ds_obj = DatasetObject(data_root=data_root, split=args.split, name=obj_name)
                    summary = worker_fn(ds_obj, args)
                    out_dir = method_object_dir(work_root, method, args.split, obj_name)
                    write_json(out_dir / "summary.json", summary)
                    coord.mark_done(obj_name, worker_id)
                    print(f"[{now()}] [DONE] {worker_id} -> {obj_name}", flush=True)
                    worker_log.write(f"[{now()}] done {obj_name}\n")
                    worker_log.flush()
                except Exception:
                    err = traceback.format_exc()
                    coord.mark_fail(obj_name, worker_id, err)
                    print(f"[{now()}] [FAIL] {worker_id} -> {obj_name}\n{err}", flush=True)
                    worker_log.write(f"[{now()}] fail {obj_name}\n{err}\n")
                    worker_log.flush()
                finally:
                    coord.release(lock_path)
        except BaseException:
            err = traceback.format_exc()
            coord.mark_crash(worker_id, err)
            worker_log.write(f"[{now()}] crash worker\n{err}\n")
            worker_log.flush()
            raise
        finally:
            try:
                worker_log.write(f"[{now()}] worker_exit id={worker_id}\n")
                worker_log.flush()
                worker_log.close()
            except Exception:
                pass

    if num_workers <= 1:
        loop(0)
        reports = _read_failure_reports(coord_dir)
        if reports:
            detail = "\n\n".join(reports)
            raise RuntimeError(f"One or more workers failed.\n\n{detail}")
        return

    workers: List[Process] = []
    for i in range(num_workers):
        p = Process(target=loop, args=(i,), daemon=False)
        p.start()
        workers.append(p)
    exit_code = 0
    exit_reports: List[str] = []
    for p in workers:
        p.join()
        if p.exitcode != 0:
            exit_code = 1
            exit_reports.append(_format_process_exit(p))
    reports = _read_failure_reports(coord_dir)
    if exit_code != 0 or reports:
        detail_parts = []
        if exit_reports:
            detail_parts.append("Worker exits:\n" + "\n".join(exit_reports))
        detail_parts.extend(reports)
        worker_logs = _read_worker_logs(coord_dir)
        if worker_logs:
            detail_parts.append("Worker logs:\n" + "\n\n".join(worker_logs))
        locks = _read_active_locks(coord_dir)
        if locks:
            detail_parts.append("Remaining locks:\n" + "\n\n".join(locks))
        detail = "\n\n".join(detail_parts)
        if detail:
            raise RuntimeError(f"One or more workers failed.\n\n{detail}")
        raise RuntimeError(f"One or more workers failed. Check coordinator logs under: {coord_dir}")


def frames_for_part(obj: DatasetObject, part_name: str, max_frames: int = 0, frame_stride: int = 1) -> List[str]:
    frames = []
    for frame in list_frames(obj):
        if mask_path_for_part_frame(obj, part_name, frame) is not None:
            frames.append(frame)
    frames = frames[:: max(1, int(frame_stride))]
    if max_frames and int(max_frames) > 0:
        frames = frames[: int(max_frames)]
    return frames


def backproject(depth_m: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    valid = mask & (depth_m > 1e-6)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - k[0, 2]) * z / max(float(k[0, 0]), 1e-8)
    y = (ys.astype(np.float32) - k[1, 2]) * z / max(float(k[1, 1]), 1e-8)
    return np.stack([x, y, z], axis=1).astype(np.float32)
