import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sapien_render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check dataset_train/<object>/meta.json and rebuild instances with missing meta "
            "by reusing scripts/sapien_render.py helpers."
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
        default="dataset_train",
        help="Dataset output root with direct object folders: <output-root>/<object>/.",
    )
    parser.add_argument("--instance", type=str, default=None, help="Only check/repair one instance name.")
    parser.add_argument(
        "--start-instance",
        type=str,
        default=None,
        help="Only check/repair output dirs whose discovered instance order is at or after this instance.",
    )
    parser.add_argument(
        "--end-instance",
        type=str,
        default=None,
        help="Only check/repair output dirs whose discovered instance order is at or before this instance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print missing meta.json entries; do not rebuild anything.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Do not delete/rebuild an existing broken instance directory before repairing.",
    )
    parser.add_argument("--views", type=int, default=50, help="Views per instance.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-deg", type=float, default=35.0)
    parser.add_argument("--radius-scale", type=float, default=1.15)
    parser.add_argument("--target-max-extent", type=float, default=0.6)
    parser.add_argument("--min-object-coverage", type=float, default=0.08)
    parser.add_argument("--max-object-coverage", type=float, default=0.65)
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
    rt_group.add_argument(
        "--rt",
        dest="rt",
        action="store_true",
        default=True,
        help="Use ray-tracing renderer settings. Enabled by default.",
    )
    rt_group.add_argument(
        "--no-rt",
        dest="rt",
        action="store_false",
        help="Disable ray-tracing renderer settings.",
    )
    rt_camera_group = parser.add_mutually_exclusive_group()
    rt_camera_group.add_argument(
        "--rt-camera-shader",
        dest="rt_camera_shader",
        action="store_true",
        default=False,
        help="Use the ray-tracing camera shader for saved RGB.",
    )
    rt_camera_group.add_argument(
        "--no-rt-camera-shader",
        dest="rt_camera_shader",
        action="store_false",
        help="Keep the default camera shader while retaining other rt settings.",
    )
    parser.add_argument("--rt-spp", type=int, default=64, help="Ray-tracing samples per pixel.")
    parser.add_argument("--rt-path-depth", type=int, default=8, help="Ray-tracing path depth.")
    parser.add_argument(
        "--rt-denoiser",
        type=str,
        default="optix",
        choices=["optix", "oidn", "none"],
        help="Ray-tracing denoiser. Use 'none' to disable denoising.",
    )
    parser.add_argument(
        "--background-mode",
        type=str,
        default="composite",
        choices=["plain", "composite"],
    )
    parser.add_argument("--background-variants", type=int, default=4)
    parser.add_argument("--background-seed", type=int, default=2026)
    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument(
        "--joint-motion",
        dest="joint_motion",
        action="store_true",
        default=True,
        help="Apply small deterministic joint motion per source view. Enabled by default.",
    )
    joint_group.add_argument(
        "--no-joint-motion",
        dest="joint_motion",
        action="store_false",
        help="Keep all articulation joints at their loaded default positions.",
    )
    parser.add_argument("--joint-motion-fraction", type=float, default=0.01)
    parser.add_argument("--joint-motion-max-delta", type=float, default=3.0)
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of repair worker processes. Use 1 for single-process; <=0 means auto.",
    )
    return parser.parse_args()


def list_output_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        raise FileNotFoundError(f"output root not found: {output_root}")
    if not output_root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {output_root}")
    return sorted(p for p in output_root.iterdir() if p.is_dir())


def select_instance_names(args: argparse.Namespace, all_instances: Sequence[sapien_render.InstanceInfo]) -> set:
    selected = sapien_render.select_instances(
        list(all_instances),
        args.instance,
        args.start_instance,
        args.end_instance,
    )
    return {inst.instance_name for inst in selected}


def find_missing_meta_dirs(
    output_dirs: Sequence[Path],
    source_by_name: Dict[str, sapien_render.InstanceInfo],
    selected_names: set,
) -> Tuple[List[Path], List[Path], List[Path]]:
    missing: List[Path] = []
    complete: List[Path] = []
    unknown: List[Path] = []

    for object_dir in output_dirs:
        name = object_dir.name
        if name not in source_by_name:
            if len(selected_names) == len(source_by_name):
                unknown.append(object_dir)
            continue
        if name not in selected_names:
            continue
        if (object_dir / "meta.json").exists():
            complete.append(object_dir)
        else:
            missing.append(object_dir)
    return missing, complete, unknown


def make_repair_args(args: argparse.Namespace) -> argparse.Namespace:
    repair_args = argparse.Namespace(**vars(args))
    repair_args.output_root = str(Path(args.output_root))
    repair_args.overwrite = bool(args.overwrite)
    return repair_args


def repair_instances(
    targets: Sequence[sapien_render.InstanceInfo],
    repair_args: argparse.Namespace,
) -> Tuple[int, List[Tuple[str, str]], List[Tuple[str, str]]]:
    num_workers = sapien_render.resolve_num_processes(repair_args.processes, len(targets))
    args_dict = vars(repair_args).copy()
    ok = 0
    skipped: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    print(f"[Info] worker_processes={num_workers}")
    if num_workers <= 1:
        for idx, inst in enumerate(targets, start=1):
            print(f"[Repair {idx}/{len(targets)}] {inst.instance_name}")
            result = sapien_render._run_one_instance_worker(inst, args_dict)
            status = str(result.get("status", "failed"))
            if status == "success":
                ok += 1
                print(f"[OK] {inst.instance_name}")
            elif status == "skipped":
                err = str(result.get("error", "skipped"))
                skipped.append((inst.instance_name, err))
                print(f"[Skip] {inst.instance_name}: {err}")
            else:
                err = str(result.get("error", "unknown_error"))
                failed.append((inst.instance_name, err))
                print(f"[Error] {inst.instance_name}: {err}")
        return ok, skipped, failed

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_instance = {
            executor.submit(sapien_render._run_one_instance_worker, inst, args_dict): inst.instance_name
            for inst in targets
        }
        total = len(future_to_instance)
        for finished, future in enumerate(concurrent.futures.as_completed(future_to_instance), start=1):
            inst_name = future_to_instance[future]
            try:
                result = future.result()
            except Exception as e:
                failed.append((inst_name, str(e)))
                print(f"[Error {finished}/{total}] {inst_name}: {e}")
                continue

            status = str(result.get("status", "failed"))
            if status == "success":
                ok += 1
                print(f"[OK {finished}/{total}] {inst_name}")
            elif status == "skipped":
                err = str(result.get("error", "skipped"))
                skipped.append((inst_name, err))
                print(f"[Skip {finished}/{total}] {inst_name}: {err}")
            else:
                err = str(result.get("error", "unknown_error"))
                failed.append((inst_name, err))
                print(f"[Error {finished}/{total}] {inst_name}: {err}")

    return ok, skipped, failed


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root)
    output_root = Path(args.output_root)

    if not models_root.exists():
        raise FileNotFoundError(f"models root not found: {models_root}")

    all_instances = sapien_render.discover_instances(models_root)
    if not all_instances:
        raise RuntimeError("No valid instances found under models root.")
    source_by_name = {inst.instance_name: inst for inst in all_instances}
    selected_names = select_instance_names(args, all_instances)
    output_dirs = list_output_dirs(output_root)
    missing_dirs, complete_dirs, unknown_dirs = find_missing_meta_dirs(
        output_dirs,
        source_by_name,
        selected_names,
    )

    print(f"[Info] output_root={output_root}")
    print(f"[Info] discovered_source_instances={len(all_instances)}")
    print(f"[Info] selected_source_instances={len(selected_names)}")
    print(f"[Info] existing_output_dirs={len(output_dirs)}")
    print(f"[Info] complete_meta={len(complete_dirs)}")
    print(f"[Info] missing_meta={len(missing_dirs)}")
    print(f"[Info] unknown_output_dirs={len(unknown_dirs)}")

    if missing_dirs:
        print("[Missing meta.json]")
        for object_dir in missing_dirs:
            print(f"  - {object_dir.name}")
    if unknown_dirs:
        print("[Unknown output dirs skipped]")
        for object_dir in unknown_dirs:
            print(f"  - {object_dir.name}")

    if args.dry_run or not missing_dirs:
        print(
            f"[Done] checked={len(complete_dirs) + len(missing_dirs)}, "
            f"repaired=0, skipped=0, failed=0"
        )
        return

    missing_names = {p.name for p in missing_dirs}
    targets = [inst for inst in all_instances if inst.instance_name in missing_names]
    repair_args = make_repair_args(args)
    ok, skipped, failed = repair_instances(targets, repair_args)

    print(f"[Done] checked={len(complete_dirs) + len(missing_dirs)}, repaired={ok}, skipped={len(skipped)}, failed={len(failed)}")
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
