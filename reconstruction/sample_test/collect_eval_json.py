#!/usr/bin/env python3
"""Collect reconstruction and pose evaluation outputs into final JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _float_or_none(value):
    try:
        v = float(value)
    except Exception:
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                fv = _float_or_none(v)
                clean[k] = fv if fv is not None else v
            rows.append(clean)
        return rows


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect sample-test eval outputs.")
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--objects-json", type=Path, required=True)
    parser.add_argument("--recon-eval-root", type=Path, required=True)
    parser.add_argument("--pose-eval-root", type=Path, required=True)
    parser.add_argument("--recon-output", type=Path, required=True)
    parser.add_argument("--pose-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    objects_payload = _read_json(args.objects_json) or {}

    recon_payload = {
        "objects_json": str(args.objects_json),
        "objects": objects_payload.get("objects", []),
        "methods": methods,
        "summary_table": _read_csv(args.recon_eval_root / "summary_table.csv"),
        "per_part_metrics": _read_csv(args.recon_eval_root / "per_part_metrics.csv"),
    }
    args.recon_output.parent.mkdir(parents=True, exist_ok=True)
    with args.recon_output.open("w", encoding="utf-8") as f:
        json.dump(recon_payload, f, ensure_ascii=False, indent=2)

    pose_methods = {}
    for method in methods:
        pose_methods[method] = _read_json(args.pose_eval_root / f"{method}.json")
    pose_payload = {
        "objects_json": str(args.objects_json),
        "objects": objects_payload.get("objects", []),
        "methods": methods,
        "summary_all_methods": _read_json(args.pose_eval_root / "summary_all_methods.json"),
        "method_results": pose_methods,
    }
    args.pose_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pose_output.open("w", encoding="utf-8") as f:
        json.dump(pose_payload, f, ensure_ascii=False, indent=2)

    print(f"[collect] reconstruction json -> {args.recon_output}")
    print(f"[collect] pose json -> {args.pose_output}")


if __name__ == "__main__":
    main()
