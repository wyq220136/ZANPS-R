#!/usr/bin/env python3
"""Merge sample-test pose-estimation shard outputs into one method folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge pose shards for one method.")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_root = args.pose_root / args.method
    shard_method_root = args.shard_root / args.method
    all_rows = []
    summaries = []
    if shard_method_root.exists():
        for shard_dir in sorted([p for p in shard_method_root.iterdir() if p.is_dir()]):
            src_method = shard_dir / args.method
            all_rows.extend(_read_jsonl(src_method / "poses.jsonl"))
            summary_path = src_method / "summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
    method_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(method_root / "poses.jsonl", all_rows)
    objects_dir = method_root / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    if shard_method_root.exists():
        for shard_dir in sorted([p for p in shard_method_root.iterdir() if p.is_dir()]):
            src_objects = shard_dir / args.method / "objects"
            if not src_objects.exists():
                continue
            for item in src_objects.glob("*.jsonl"):
                shutil.copy2(item, objects_dir / item.name)
    summary = {
        "method": args.method,
        "rows": len(all_rows),
        "shards": summaries,
    }
    with (method_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[merge-pose] method={args.method} rows={len(all_rows)} -> {method_root}")


if __name__ == "__main__":
    main()
