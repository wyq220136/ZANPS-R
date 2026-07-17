#!/usr/bin/env python3
"""Sample object names from a PartNet-style split without touching the dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path


DEFAULT_DATA_ROOT = Path(
    "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
)


def natural_sort_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", text)]


def category_from_object_name(name: str) -> str:
    match = re.match(r"^(.+?)_\d+$", name)
    if match:
        return match.group(1)
    return name.split("_", 1)[0]


def group_by_category(objects):
    groups = OrderedDict()
    for name in sorted(objects, key=natural_sort_key):
        category = category_from_object_name(name)
        groups.setdefault(category, []).append(name)
    return OrderedDict(sorted(groups.items(), key=lambda item: natural_sort_key(item[0])))


def evenly_spaced(items, count: int):
    items = list(items)
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    max_idx = len(items) - 1
    indices = []
    used = set()
    for i in range(count):
        idx = round(i * max_idx / (count - 1))
        while idx in used and idx + 1 < len(items):
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        used.add(idx)
        indices.append(idx)
    return [items[i] for i in sorted(indices)]


def allocate_category_quotas(groups, num: int):
    categories = list(groups.keys())
    quotas = {cat: 0 for cat in categories}
    remaining = int(num)
    active = [cat for cat in categories if groups[cat]]

    while remaining > 0 and active:
        per_cat = max(1, remaining // len(active))
        progressed = False
        next_active = []
        for cat in active:
            capacity = len(groups[cat]) - quotas[cat]
            if capacity <= 0:
                continue
            take = min(capacity, per_cat, remaining)
            if take > 0:
                quotas[cat] += take
                remaining -= take
                progressed = True
            if len(groups[cat]) > quotas[cat]:
                next_active.append(cat)
            if remaining <= 0:
                break
        if not progressed:
            break
        active = next_active
    return quotas


def sample_by_category(objects, num: int):
    groups = group_by_category(objects)
    quotas = allocate_category_quotas(groups, num)
    sampled_by_category = OrderedDict()
    sampled = []
    for category, items in groups.items():
        chosen = evenly_spaced(items, quotas.get(category, 0))
        if chosen:
            sampled_by_category[category] = chosen
            sampled.extend(chosen)
    sampled = sorted(sampled[:num], key=natural_sort_key)
    return sampled, sampled_by_category, groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sample object names into a JSON file.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num", type=int, required=True, help="Number of objects to sample.")
    parser.add_argument("--seed", type=int, default=2026, help="Kept for CLI compatibility; category sampling is deterministic.")
    parser.add_argument("--output", type=Path, default=Path("sampled_objects.json"))
    parser.add_argument("--object-source", choices=["all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_root = (args.data_root / args.split).resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(f"split root not found: {split_root}")

    objects = sorted([p.name for p in split_root.iterdir() if p.is_dir()], key=natural_sort_key)
    if not objects:
        raise RuntimeError(f"no object directories found under: {split_root}")
    if int(args.num) <= 0:
        raise ValueError("--num must be positive")
    if int(args.num) > len(objects):
        raise ValueError(f"--num={args.num} exceeds object count={len(objects)}")

    sampled, sampled_by_category, groups = sample_by_category(objects, int(args.num))

    out_path = args.output
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    if not out_path.parent.is_dir():
        raise FileNotFoundError(f"output parent directory not found: {out_path.parent}")
    payload = {
        "data_root": str(args.data_root),
        "split": args.split,
        "strategy": "category_balanced_evenly_spaced",
        "seed": int(args.seed),
        "num_requested": int(args.num),
        "num_available": len(objects),
        "num_categories_available": len(groups),
        "category_counts_available": {cat: len(items) for cat, items in groups.items()},
        "category_counts_sampled": {cat: len(items) for cat, items in sampled_by_category.items()},
        "objects_by_category": sampled_by_category,
        "objects": sampled,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[sample] wrote {len(sampled)} objects -> {out_path}")


if __name__ == "__main__":
    main()
