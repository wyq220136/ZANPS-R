"""
Split the raw flat dataset produced by scripts/sapien_render.py.

The input is expected to be dataset_train/<object_name>/ directories directly
under DATASET_ROOT. The script stratifies objects by category prefix, e.g.
bottle_3398 and bottle_1209 are both bottle, then moves every object directory
into DATASET_ROOT/train, DATASET_ROOT/val, or DATASET_ROOT/test.
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


DATASET_ROOT = "dataset_train"
SPLIT_RATIOS = (7, 1, 2)
SPLIT_NAMES = ("train", "val", "test")
RANDOM_SEED = 42
SHUFFLE_WITHIN_CATEGORY = True
OVERWRITE_EXISTING = False


def natural_sort_key(s: object):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"([0-9]+)", str(s))]


def infer_category(obj_name: str) -> str:
    return obj_name.split("_", 1)[0] if "_" in obj_name else obj_name


def parse_ratios(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in re.split(r"[:,]", text) if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"expected three split ratios, got: {text!r}")
    ratios = tuple(int(p) for p in parts)
    if any(x < 0 for x in ratios) or sum(ratios) <= 0:
        raise ValueError(f"invalid split ratios: {ratios}")
    return ratios


def split_counts(n: int, ratios: tuple[int, int, int]) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    total = float(sum(ratios))
    raw = [n * r / total for r in ratios]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], ratios[i]), reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts[0], counts[1], counts[2]


def _collect_flat_object_dirs(dataset_root: Path, split_names: tuple[str, str, str]) -> list[Path]:
    entries = sorted(dataset_root.iterdir(), key=lambda p: natural_sort_key(p.name))
    unexpected_files = [p for p in entries if p.is_file()]
    if unexpected_files:
        preview = ", ".join(p.name for p in unexpected_files[:5])
        raise RuntimeError(
            f"dataset root contains top-level files; expected only object directories before splitting: {preview}"
        )

    split_dirs = {dataset_root / name for name in split_names}
    existing_split_dirs = [p for p in split_dirs if p.exists()]
    if existing_split_dirs:
        names = ", ".join(p.name for p in existing_split_dirs)
        raise RuntimeError(
            f"split directories already exist ({names}). Start from the raw flat dataset or remove them first."
        )

    obj_dirs = [p for p in entries if p.is_dir()]
    if not obj_dirs:
        raise RuntimeError(f"no object directories found under {dataset_root}")
    return obj_dirs


def build_split_plan(
    obj_dirs: list[Path],
    ratios: tuple[int, int, int],
    split_names: tuple[str, str, str],
    seed: int,
    shuffle: bool,
):
    grouped = defaultdict(list)
    for p in obj_dirs:
        grouped[infer_category(p.name)].append(p)

    rng = random.Random(int(seed))
    plan = []
    category_summary = []
    for category in sorted(grouped.keys(), key=natural_sort_key):
        dirs = sorted(grouped[category], key=lambda p: natural_sort_key(p.name))
        if shuffle:
            rng.shuffle(dirs)
        train_n, val_n, test_n = split_counts(len(dirs), ratios)
        chunks = (
            dirs[:train_n],
            dirs[train_n : train_n + val_n],
            dirs[train_n + val_n : train_n + val_n + test_n],
        )
        category_summary.append((category, len(dirs), train_n, val_n, test_n))
        for split_name, split_dirs in zip(split_names, chunks):
            for src in split_dirs:
                plan.append((category, split_name, src))
    return plan, category_summary


def split_dataset(
    dataset_root: Path,
    ratios: tuple[int, int, int],
    split_names: tuple[str, str, str],
    seed: int,
    shuffle: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if len(split_names) != 3 or len(set(split_names)) != 3:
        raise ValueError(f"split_names must contain three unique names: {split_names}")

    obj_dirs = _collect_flat_object_dirs(dataset_root, split_names)
    plan, category_summary = build_split_plan(obj_dirs, ratios, split_names, seed, shuffle)

    for category, total, train_n, val_n, test_n in category_summary:
        print(
            f"[CATEGORY] {category}: total={total} "
            f"{split_names[0]}={train_n} {split_names[1]}={val_n} {split_names[2]}={test_n}"
        )

    for _, split_name, src in plan:
        dst = dataset_root / split_name / src.name
        if dst.exists() and not overwrite:
            raise FileExistsError(f"destination already exists: {dst}")

    if not dry_run:
        for split_name in split_names:
            (dataset_root / split_name).mkdir(parents=True, exist_ok=True)

    for category, split_name, src in plan:
        dst = dataset_root / split_name / src.name
        print(f"[MOVE] {src.name} ({category}) -> {split_name}")
        if dry_run:
            continue
        if dst.exists() and overwrite:
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    if not dry_run:
        remaining = sorted([p.name for p in dataset_root.iterdir() if p.name not in split_names])
        if remaining:
            raise RuntimeError(f"split finished but unexpected top-level items remain: {remaining}")
    print(f"[DONE] moved={len(plan)} root={dataset_root} ratios={ratios} dry_run={dry_run}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Move a flat sapien_render dataset into train/val/test by category.")
    parser.add_argument("--dataset-root", type=str, default=DATASET_ROOT)
    parser.add_argument("--ratios", type=str, default=":".join(str(x) for x in SPLIT_RATIOS))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-shuffle", action="store_true", default=not SHUFFLE_WITHIN_CATEGORY)
    parser.add_argument("--overwrite-existing", action="store_true", default=OVERWRITE_EXISTING)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(
        dataset_root=Path(args.dataset_root),
        ratios=parse_ratios(args.ratios),
        split_names=SPLIT_NAMES,
        seed=int(args.seed),
        shuffle=not bool(args.no_shuffle),
        overwrite=bool(args.overwrite_existing),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
