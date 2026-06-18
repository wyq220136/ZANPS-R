import argparse
import concurrent.futures
import importlib
import multiprocessing as mp
import os, sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
build_scene_main = importlib.import_module("scene_builder.main")
from scene_builder.instance_parts import InstanceInfo, discover_instances, select_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Models against dataset output and rebuild selected instances whose "
            "output directory has no meta.json."
        )
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/Models",
        help="Root containing PartModels folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train",
        help="Dataset output root with direct object folders: <output-root>/<object>/.",
    )
    parser.add_argument("--instance", type=str, default=None, help="Only check/repair one instance name.")
    parser.add_argument(
        "--start-instance",
        type=str,
        default=None,
        help="Only check/repair source instances whose discovered order is at or after this instance.",
    )
    parser.add_argument(
        "--end-instance",
        type=str,
        default=None,
        help="Only check/repair source instances whose discovered order is at or before this instance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print source instances without meta.json; do not rebuild anything.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Do not force-delete an existing broken instance directory before repairing.",
    )
    parser.add_argument("--views", type=int, default=50, help="Views per instance.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-deg", type=float, default=35.0)
    parser.add_argument("--radius-scale", type=float, default=1.25)
    parser.add_argument("--target-max-extent", type=float, default=0.45)
    parser.add_argument("--min-object-scale", type=float, default=0.25)
    parser.add_argument("--max-object-scale", type=float, default=1.5)
    parser.add_argument(
        "--min-object-coverage",
        type=float,
        default=0.03,
        help="Relaxed lower bound for repair mode; build_scene keeps the stricter default.",
    )
    parser.add_argument(
        "--max-object-coverage",
        type=float,
        default=0.55,
        help="Relaxed upper bound for repair mode; build_scene keeps the stricter default.",
    )
    parser.add_argument(
        "--min-part-mask-pixels",
        type=int,
        default=20,
        help="Relaxed minimum visible pixels for repair mode.",
    )
    parser.add_argument(
        "--min-part-mask-coverage",
        type=float,
        default=0.000020,
        help="Relaxed per-image part coverage floor for repair mode.",
    )
    parser.add_argument(
        "--require-all-part-visible",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repair mode default: require only one visible movable part. Use --require-all-part-visible to restore stricter checking.",
    )
    parser.add_argument("--view-candidate-multiplier", type=int, default=24)
    parser.add_argument(
        "--part-occlusion-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair mode default: reject views whose movable part is mostly hidden in a part-only projection check.",
    )
    parser.add_argument(
        "--min-part-visible-ratio",
        type=float,
        default=0.08,
        help=(
            "Minimum full-scene visible pixels divided by part-only projected pixels. "
            "Repair mode keeps this relaxed because thin faucet handles/switches can "
            "project much larger in part-only renders than their visible full-scene mask."
        ),
    )
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--exclude-part-keywords",
        type=str,
        default="base_body,frame,body",
        help=(
            "Comma-separated case-insensitive keywords for part/link names to exclude "
            "from exported models, masks, and cam_params. Empty string disables this filter."
        ),
    )
    parser.add_argument(
        "--render-ground",
        action="store_true",
        help="Render ground plane. Default is off to avoid object-ground intersection/occlusion.",
    )
    rt_group = parser.add_mutually_exclusive_group()
    rt_group.add_argument("--rt", dest="rt", action="store_true", default=True)
    rt_group.add_argument("--no-rt", dest="rt", action="store_false")
    rt_camera_group = parser.add_mutually_exclusive_group()
    rt_camera_group.add_argument("--rt-camera-shader", dest="rt_camera_shader", action="store_true", default=False)
    rt_camera_group.add_argument("--no-rt-camera-shader", dest="rt_camera_shader", action="store_false")
    parser.add_argument("--rt-spp", type=int, default=64)
    parser.add_argument("--rt-path-depth", type=int, default=8)
    parser.add_argument("--rt-denoiser", type=str, default="optix", choices=["optix", "oidn", "none"])
    parser.add_argument(
        "--background-mode",
        type=str,
        default="composite",
        choices=["plain", "composite", "scene"],
    )
    parser.add_argument("--background-variants", type=int, default=4)
    parser.add_argument("--background-seed", type=int, default=2026)
    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument("--joint-motion", dest="joint_motion", action="store_true", default=True)
    joint_group.add_argument("--no-joint-motion", dest="joint_motion", action="store_false")
    parser.add_argument("--joint-motion-fraction", type=float, default=0.005)
    parser.add_argument("--joint-motion-max-delta", type=float, default=2.0)
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of repair worker processes. Use 1 for single-process; <=0 means auto.",
    )
    parser.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=20,
        help="Restart each worker process after this many instances. Use <=0 to disable worker recycling.",
    )
    parser.add_argument(
        "--instance-timeout",
        type=int,
        default=1800,
        help="Maximum seconds allowed for one instance in single-worker mode. Use <=0 to disable.",
    )
    return parser.parse_args()


def list_output_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    if not output_root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {output_root}")
    return sorted(p for p in output_root.iterdir() if p.is_dir())


def split_instances_by_meta(
    selected_instances: Sequence[InstanceInfo],
    output_root: Path,
) -> Tuple[List[InstanceInfo], List[InstanceInfo]]:
    missing: List[InstanceInfo] = []
    complete: List[InstanceInfo] = []
    for inst in selected_instances:
        meta_path = output_root / inst.instance_name / "meta.json"
        if meta_path.exists():
            complete.append(inst)
        else:
            missing.append(inst)
    return missing, complete


def find_unknown_output_dirs(
    output_dirs: Sequence[Path],
    source_by_name: Dict[str, InstanceInfo],
) -> List[Path]:
    return [object_dir for object_dir in output_dirs if object_dir.name not in source_by_name]


def make_repair_args(args: argparse.Namespace) -> argparse.Namespace:
    repair_args = argparse.Namespace(**vars(args))
    repair_args.output_root = str(Path(args.output_root))
    repair_args.overwrite = bool(args.overwrite)
    return repair_args


def handle_result(
    result: Dict[str, object],
    inst_name: str,
    skipped: List[Tuple[str, str]],
    failed: List[Tuple[str, str]],
) -> bool:
    status = str(result.get("status", "failed"))
    if status == "success":
        return True
    if status == "skipped":
        err = str(result.get("error", "skipped"))
        skipped.append((inst_name, err))
        return False
    err = str(result.get("error", "unknown_error"))
    failed.append((inst_name, err))
    return False


def repair_instances(
    targets: Sequence[InstanceInfo],
    repair_args: argparse.Namespace,
) -> Tuple[int, List[Tuple[str, str]], List[Tuple[str, str]]]:
    num_workers = build_scene_main.resolve_num_processes(repair_args.processes, len(targets))
    args_dict = vars(repair_args).copy()
    ok = 0
    skipped: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    print(f"[Info] worker_processes={num_workers}")
    if num_workers <= 1:
        for idx, inst in enumerate(targets, start=1):
            print(f"[Repair {idx}/{len(targets)}] {inst.instance_name}")
            if int(repair_args.instance_timeout) > 0:
                result = build_scene_main.run_one_instance_isolated(
                    inst,
                    args_dict,
                    int(repair_args.instance_timeout),
                )
            else:
                result = build_scene_main._run_one_instance_worker(inst, args_dict)
            if handle_result(result, inst.instance_name, skipped, failed):
                ok += 1
                print(f"[OK] {inst.instance_name}")
            elif skipped and skipped[-1][0] == inst.instance_name:
                print(f"[Skip] {inst.instance_name}: {skipped[-1][1]}")
            else:
                print(f"[Error] {inst.instance_name}: {failed[-1][1]}")
        return ok, skipped, failed

    tasks_per_worker = int(repair_args.max_tasks_per_worker)
    if tasks_per_worker <= 0:
        tasks_per_worker = len(targets)
    batch_size = max(1, tasks_per_worker)
    finished = 0
    total = len(targets)
    ctx = mp.get_context("spawn")
    print(f"[Info] worker_recycle_every={tasks_per_worker} tasks/worker, batch_size={batch_size}")
    for batch_start in range(0, len(targets), batch_size):
        batch = targets[batch_start : batch_start + batch_size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            future_to_instance = {
                executor.submit(build_scene_main._run_one_instance_worker, inst, args_dict): inst.instance_name
                for inst in batch
            }
            for future in concurrent.futures.as_completed(future_to_instance):
                finished += 1
                inst_name = future_to_instance[future]
                try:
                    result = future.result()
                except Exception as e:
                    failed.append((inst_name, str(e)))
                    print(f"[Error {finished}/{total}] {inst_name}: {e}")
                    continue
                if handle_result(result, inst_name, skipped, failed):
                    ok += 1
                    print(f"[OK {finished}/{total}] {inst_name}")
                elif skipped and skipped[-1][0] == inst_name:
                    print(f"[Skip {finished}/{total}] {inst_name}: {skipped[-1][1]}")
                else:
                    print(f"[Error {finished}/{total}] {inst_name}: {failed[-1][1]}")

    return ok, skipped, failed


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root)
    output_root = Path(args.output_root)

    if not models_root.exists():
        raise FileNotFoundError(f"models root not found: {models_root}")

    all_instances = discover_instances(models_root)
    if not all_instances:
        raise RuntimeError("No valid instances found under models root.")

    selected_instances = select_instances(
        all_instances,
        args.instance,
        args.start_instance,
        args.end_instance,
    )
    source_by_name = {inst.instance_name: inst for inst in all_instances}
    output_dirs = list_output_dirs(output_root)
    missing_instances, complete_instances = split_instances_by_meta(selected_instances, output_root)
    unknown_dirs = find_unknown_output_dirs(output_dirs, source_by_name)

    print(f"[Info] models_root={models_root}")
    print(f"[Info] output_root={output_root}")
    print(f"[Info] discovered_source_instances={len(all_instances)}")
    print(f"[Info] selected_source_instances={len(selected_instances)}")
    print(f"[Info] existing_output_dirs={len(output_dirs)}")
    print(f"[Info] complete_meta={len(complete_instances)}")
    print(f"[Info] missing_or_incomplete_meta={len(missing_instances)}")
    print(f"[Info] unknown_output_dirs={len(unknown_dirs)}")

    if missing_instances:
        print("[Missing or incomplete source instances]")
        for inst in missing_instances:
            print(f"  - {inst.instance_name}")
    if unknown_dirs:
        print("[Unknown output dirs skipped]")
        for object_dir in unknown_dirs:
            print(f"  - {object_dir.name}")

    if args.dry_run or not missing_instances:
        print(
            f"[Done] checked={len(selected_instances)}, "
            f"repaired=0, skipped=0, failed=0"
        )
        return

    repair_args = make_repair_args(args)
    ok, skipped, failed = repair_instances(missing_instances, repair_args)

    print(f"[Done] checked={len(selected_instances)}, repaired={ok}, skipped={len(skipped)}, failed={len(failed)}")
    if skipped:
        print("[Skipped Details]")
        for name, err in skipped:
            print(f"  - {name}: {err}")
    if failed:
        print("[Failed Details]")
        for name, err in failed:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
