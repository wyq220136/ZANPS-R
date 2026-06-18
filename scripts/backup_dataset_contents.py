"""
Copy-backup a dataset directory before destructive restructuring.

This script copies every item directly under SRC_DIR into DST_DIR. It is meant
for preserving the raw flat output produced by scripts/sapien_render.py, e.g.
dataset_train/bottle_3398, before dataset_train is moved into train/val/test
subdirectories by the split script.
"""

import argparse
import shutil
from pathlib import Path


SRC_DIR = "dataset_train"
DST_DIR = "dataset_train_raw_backup"
OVERWRITE_EXISTING = False


def _copy_item(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"destination already exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_dataset_contents(src_dir: Path, dst_dir: Path, overwrite: bool = False, dry_run: bool = False) -> int:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {src_dir}")
    if src_dir == dst_dir:
        raise ValueError("SRC_DIR and DST_DIR must be different directories.")
    try:
        dst_dir.relative_to(src_dir)
        raise ValueError(f"DST_DIR must not be inside SRC_DIR: {dst_dir}")
    except ValueError as e:
        if "must not" in str(e):
            raise

    items = sorted(src_dir.iterdir(), key=lambda p: p.name.lower())
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in items:
        dst = dst_dir / src.name
        print(f"[COPY] {src} -> {dst}")
        if not dry_run:
            _copy_item(src, dst, overwrite=overwrite)
        copied += 1
    print(f"[DONE] copied_items={copied} src={src_dir} dst={dst_dir} dry_run={dry_run}")
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Copy all direct contents of a dataset folder into a backup folder.")
    parser.add_argument("--src-dir", type=str, default=SRC_DIR)
    parser.add_argument("--dst-dir", type=str, default=DST_DIR)
    parser.add_argument("--overwrite-existing", action="store_true", default=OVERWRITE_EXISTING)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_dataset_contents(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        overwrite=bool(args.overwrite_existing),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
