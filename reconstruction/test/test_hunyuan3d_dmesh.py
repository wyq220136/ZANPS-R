import argparse
import subprocess
import sys
from pathlib import Path


TEST_OBJECTS = [
    "bottle_3517",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _object_arg(objects: list[str]) -> str:
    return ",".join(x.strip() for x in objects if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Smoke test Hunyuan3D + DMesh on selected dataset_train/val objects.")
    parser.add_argument("--data-root", type=str, default="dataset_train")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--work-root", type=str, default="reconstruction_runs_test")
    parser.add_argument("--objects", type=str, default="", help="Override TEST_OBJECTS with comma-separated names.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-build-base", action="store_true", help="Require existing Hunyuan3D cache instead of building it.")
    parser.add_argument("--dmesh-root", type=str, default="")
    parser.add_argument("--dmesh-device", type=str, default="cuda:0")
    parser.add_argument("--dmesh-steps", type=int, default=50)
    parser.add_argument("--dmesh-save-step", type=int, default=10)
    parser.add_argument("--dmesh-refresh-points-step", type=int, default=25)
    parser.add_argument("--copy-base-as-placeholder", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    objects = args.objects.strip() or _object_arg(TEST_OBJECTS)
    if not objects:
        raise ValueError("No test objects selected. Edit TEST_OBJECTS at the top of this file.")

    cmd = [
        sys.executable,
        str(repo / "reconstruction" / "run" / "recon_hunyuan3d_dmesh.py"),
        "--data-root",
        args.data_root,
        "--split",
        args.split,
        "--work-root",
        args.work_root,
        "--objects",
        objects,
        "--num-workers",
        str(args.num_workers),
        "--gpus",
        args.gpus,
        "--dmesh-device",
        args.dmesh_device,
        "--dmesh-steps",
        str(args.dmesh_steps),
        "--dmesh-save-step",
        str(args.dmesh_save_step),
        "--dmesh-refresh-points-step",
        str(args.dmesh_refresh_points_step),
        "--reset-coord",
    ]
    if args.dmesh_root:
        cmd.extend(["--dmesh-root", args.dmesh_root])
    if args.overwrite:
        cmd.append("--overwrite")
    if not args.no_build_base:
        cmd.append("--build-base-if-missing")
    if args.copy_base_as_placeholder:
        cmd.append("--copy-base-as-placeholder")
    cmd.extend(args.extra_args)
    print("[TEST] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))


if __name__ == "__main__":
    main()
