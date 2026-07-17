from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_WORK_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/reconstruction_runs"
DEFAULT_SPLIT = "test"
ROUND_DIR_PATTERNS = ("round_*", "round[0-9]*")
INTERMEDIATE_FILES = ("remesh_preprocessed.obj",)
OUTPUT_ROOTS = ("pose_ready_models", "models")


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _method_dirs(work_root: Path, methods: list[str]) -> list[Path]:
    if methods:
        return [work_root / method for method in methods]
    return sorted(
        [p for p in work_root.iterdir() if p.is_dir() and "dmesh" in p.name],
        key=lambda p: p.name,
    )


def _object_dirs(split_dir: Path, objects: list[str]) -> list[Path]:
    if objects:
        return [split_dir / obj for obj in objects]
    return sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def _candidate_paths(method_dir: Path, split: str, objects: list[str]) -> list[Path]:
    split_dir = method_dir / split
    if not split_dir.is_dir():
        return []

    candidates: list[Path] = []
    for obj_dir in _object_dirs(split_dir, objects):
        if not obj_dir.is_dir():
            continue
        for root_name in OUTPUT_ROOTS:
            view_root = obj_dir / root_name / "view_0"
            if not view_root.is_dir():
                continue
            for part_dir in sorted([p for p in view_root.iterdir() if p.is_dir()], key=lambda p: p.name):
                for pattern in ROUND_DIR_PATTERNS:
                    candidates.extend(p for p in part_dir.glob(pattern) if p.is_dir())
                for name in INTERMEDIATE_FILES:
                    path = part_dir / name
                    if path.is_file():
                        candidates.append(path)
    return sorted(set(candidates), key=lambda p: str(p))


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def _classify_path(path: Path) -> str:
    if path.is_dir() and path.name.startswith("round"):
        return "round_dir"
    if path.is_file() and path.name in INTERMEDIATE_FILES:
        return path.name
    return "unknown"


def _validate_candidates(work_root: Path, candidates: list[Path]) -> list[str]:
    errors: list[str] = []
    resolved_root = work_root.resolve()
    for path in candidates:
        resolved = path.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            errors.append(f"outside work root: {path}")
            continue
        if path.name == "model.obj":
            errors.append(f"would delete final model.obj: {path}")
        kind = _classify_path(path)
        if kind == "unknown":
            errors.append(f"unexpected cleanup candidate: {path}")
        if path.is_dir() and not path.name.startswith("round"):
            errors.append(f"unexpected directory candidate: {path}")
        if path.is_file() and path.name not in INTERMEDIATE_FILES:
            errors.append(f"unexpected file candidate: {path}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Remove old DLMesh intermediate outputs without touching final model.obj files."
    )
    parser.add_argument("--work-root", type=str, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method names. Default: every method directory containing 'dmesh'.",
    )
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names. Default: all objects.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and validate cleanup candidates without deleting. This is also the default when --delete is absent.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate candidate safety and print summary counts. Implied by dry-run and delete.",
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete files. Without this, only prints a dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = Path(args.work_root).resolve()
    if not work_root.is_dir():
        raise FileNotFoundError(f"work root not found: {work_root}")

    methods = _parse_csv(args.methods)
    objects = _parse_csv(args.objects)
    candidates: list[Path] = []
    for method_dir in _method_dirs(work_root, methods):
        candidates.extend(_candidate_paths(method_dir, args.split, objects))

    dry_run = bool(args.dry_run or not args.delete)
    action = "DELETE" if args.delete else "DRY-RUN"
    errors = _validate_candidates(work_root, candidates)
    kind_counts = Counter(_classify_path(path) for path in candidates)
    method_counts = Counter(path.relative_to(work_root).parts[0] for path in candidates)

    print(f"[{action}] work_root={work_root} split={args.split} candidates={len(candidates)}")
    if kind_counts:
        print("[SUMMARY] by kind:")
        for kind, count in sorted(kind_counts.items()):
            print(f"  {kind}: {count}")
    if method_counts:
        print("[SUMMARY] by method:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method}: {count}")
    if errors:
        print("[VALIDATION-FAILED]")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(2)
    print("[VALIDATION-OK] no final model.obj or unexpected paths are selected")

    for path in candidates:
        print(path)
        if args.delete:
            _remove_path(path)
    if dry_run:
        print("Dry-run only. Re-run with --delete to remove these paths.")


if __name__ == "__main__":
    main()
