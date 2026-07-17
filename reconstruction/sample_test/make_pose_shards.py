#!/usr/bin/env python3
"""Create per-GPU object shards for sample-test pose estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_csv(raw: str):
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _counts(raw: str, n: int):
    vals = _split_csv(raw)
    if not vals:
        return [1] * n
    if len(vals) == 1:
        return [max(1, int(vals[0]))] * n
    if len(vals) != n:
        raise ValueError(f"workers-per-gpu length mismatch: expected {n}, got {len(vals)}")
    return [max(1, int(v)) for v in vals]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Make pose-est shards.")
    parser.add_argument("--objects", type=str, required=True)
    parser.add_argument("--gpu-ids", type=str, required=True)
    parser.add_argument("--workers-per-gpu", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objects = _split_csv(args.objects)
    gpus = _split_csv(args.gpu_ids)
    if not objects:
        raise RuntimeError("no objects provided")
    if not gpus:
        gpus = [""]
    counts = _counts(args.workers_per_gpu, len(gpus))
    total_slots = sum(counts)
    slots = []
    for gpu, count in zip(gpus, counts):
        slots.extend([gpu] * count)
    slot_objects = [[] for _ in slots]
    for idx, obj in enumerate(objects):
        slot_objects[idx % max(1, total_slots)].append(obj)

    gpu_to_objects = {gpu: [] for gpu in gpus}
    for gpu, objs in zip(slots, slot_objects):
        gpu_to_objects[gpu].extend(objs)
    shards = []
    for gpu, count in zip(gpus, counts):
        objs = gpu_to_objects[gpu]
        if not objs:
            continue
        shards.append(
            {
                "gpu_id": gpu,
                "workers": int(count),
                "objects": objs,
                "objects_csv": ",".join(objs),
            }
        )
    payload = {
        "objects": objects,
        "gpu_ids": gpus,
        "workers_per_gpu": counts,
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
