from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
ECCV_ROOT = SCRIPT_DIR.parent
REPO_ROOT = ECCV_ROOT.parent
for _p in (REPO_ROOT, ECCV_ROOT, SCRIPT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import run_shared_preprocess as shared  # noqa: E402


DEFAULT_PARTNET_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train/test"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _object_class(obj_name: str) -> str:
    stem, sep, tail = str(obj_name).rpartition("_")
    return stem if sep and tail.isdigit() and stem else str(obj_name)


def _list_candidate_objects(root: Path, explicit_objects: str = "", start: int = 0, end: int | None = None) -> list[str]:
    if explicit_objects.strip():
        names = [x.strip() for x in explicit_objects.split(",") if x.strip()]
    else:
        names = sorted(
            [p.name for p in root.iterdir() if p.is_dir()],
            key=shared.natural_sort_key,
        )
    names = [x for x in names if (root / x).is_dir()]
    lo = max(0, int(start))
    hi = len(names) if end is None else min(len(names), max(0, int(end)))
    return names[lo:hi]


def _largest_remainder_allocation(class_to_objects: dict[str, list[str]], sample_num: int) -> dict[str, int]:
    classes = sorted(class_to_objects.keys(), key=shared.natural_sort_key)
    total = sum(len(class_to_objects[c]) for c in classes)
    if total <= 0 or sample_num <= 0:
        return {c: len(class_to_objects[c]) for c in classes}
    target = min(int(sample_num), total)
    if target >= len(classes):
        alloc = {c: 1 for c in classes}
        remaining = target - len(classes)
    else:
        # 样本数小于类别数时无法覆盖所有类别，只能按类别规模加权抽取若干类别。
        alloc = {c: 0 for c in classes}
        remaining = target
    quotas = {
        c: target * (len(class_to_objects[c]) / float(total))
        for c in classes
    }
    order = sorted(
        classes,
        key=lambda c: (quotas[c] - int(quotas[c]), len(class_to_objects[c]), c),
        reverse=True,
    )
    idx = 0
    while remaining > 0 and order:
        c = order[idx % len(order)]
        if alloc[c] < len(class_to_objects[c]):
            alloc[c] += 1
            remaining -= 1
        idx += 1
        if idx > len(order) * (target + len(order) + 1):
            break
    return alloc


def _balanced_sample_objects(root: Path, object_names: list[str], sample_num: int, seed: int) -> tuple[list[str], dict]:
    class_to_objects: dict[str, list[str]] = defaultdict(list)
    for obj in object_names:
        class_to_objects[_object_class(obj)].append(obj)
    for objs in class_to_objects.values():
        objs.sort(key=shared.natural_sort_key)

    alloc = _largest_remainder_allocation(class_to_objects, int(sample_num))
    rng = random.Random(int(seed))
    sampled_by_class = {}
    sampled = []
    for cls in sorted(class_to_objects.keys(), key=shared.natural_sort_key):
        objs = list(class_to_objects[cls])
        rng.shuffle(objs)
        take = min(int(alloc.get(cls, 0)), len(objs))
        chosen = sorted(objs[:take], key=shared.natural_sort_key)
        sampled_by_class[cls] = {
            "available": len(class_to_objects[cls]),
            "sampled": len(chosen),
            "objects": chosen,
        }
        sampled.extend(chosen)
    sampled = sorted(sampled, key=shared.natural_sort_key)
    meta = {
        "root": str(root),
        "seed": int(seed),
        "requested_sample_num": int(sample_num),
        "available_objects": len(object_names),
        "available_classes": len(class_to_objects),
        "sampled_objects": len(sampled),
        "sampled_classes": sum(1 for v in sampled_by_class.values() if int(v["sampled"]) > 0),
        "objects": sampled,
        "classes": sampled_by_class,
    }
    return sampled, meta


def _load_rgb(obj_dir: Path, frame_id: str) -> np.ndarray:
    rgb_path = shared._find_frame_path(str(obj_dir), "rgb", frame_id)
    if not rgb_path:
        raise FileNotFoundError(f"missing rgb for frame={frame_id} under {obj_dir / 'rgb'}")
    image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read rgb: {rgb_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _make_shared_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pred_mask_subdir=args.pred_mask_subdir,
        chosen_part_subdir=args.out_subdir,
        selected_json_name=args.selected_json_name,
        dataset_kind="partnet",
        reference_selector="sam-vlm",
        reference_policy=args.reference_policy,
        selected_mesh_subdir="",
        min_visible_pixels=int(args.min_visible_pixels),
        max_sam_candidates_per_ref=int(args.max_candidates_per_ref),
        max_selected_parts_per_ref=int(args.max_selected_per_ref),
        vlm_workers=int(args.vlm_workers),
        unique_reference_parts=bool(args.unique_reference_parts),
        overwrite_reference=bool(args.overwrite),
    )


def _evaluate_frame(obj_dir: Path, frame_id: str, args: argparse.Namespace) -> tuple[dict, int]:
    shared_args = _make_shared_args(args)
    candidates = shared._load_pred_mask_candidates(str(obj_dir), frame_id, shared_args)
    image_rgb = _load_rgb(obj_dir, frame_id)
    check_mod = shared._get_check_mod()
    selected = []
    eval_rows = []
    seen_part_ids = set()

    def _eval_one(item):
        if int(item["area"]) < int(args.min_visible_pixels):
            return {**item, "ok": False, "reason": "too_small"}
        ok, reason = check_mod.evaluate_segmentation(
            image_rgb,
            item["mask"].astype("uint8") * 255,
            save_path=None,
        )
        return {**item, "ok": bool(ok), "reason": str(reason)}

    if int(args.vlm_workers) > 1 and len(candidates) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(args.vlm_workers)) as executor:
            eval_results = list(executor.map(_eval_one, candidates))
    else:
        eval_results = [_eval_one(item) for item in candidates]

    for res in eval_results:
        matched = shared._find_matching_gt_part(str(obj_dir), frame_id, res["mask"])
        part_id = int(matched["part_id"]) if matched is not None else int(len(selected))
        row = {
            "candidate_index": int(res["candidate_index"]),
            "name": str(res["name"]),
            "area": int(res["area"]),
            "source_pred_mask": os.path.relpath(str(res["path"]), str(obj_dir)),
            "ok": bool(res["ok"]),
            "reason": str(res["reason"]),
            "matched_gt": matched,
            "selected": False,
            "part_id": int(part_id),
        }
        if not bool(res["ok"]):
            eval_rows.append(row)
            continue
        if args.unique_reference_parts and part_id in seen_part_ids:
            row["reason"] = "duplicate_gt_part"
            eval_rows.append(row)
            continue
        seen_part_ids.add(part_id)
        row["selected"] = True
        selected.append({**res, "part_id": int(part_id), "matched_gt": matched})
        eval_rows.append(row)
        if int(args.max_selected_per_ref) > 0 and len(selected) >= int(args.max_selected_per_ref):
            break

    return {
        "frame_id": str(frame_id),
        "candidates": len(candidates),
        "selected": selected,
        "eval_rows": eval_rows,
    }, len(candidates)


def _write_object_outputs(
    obj_dir: Path,
    obj_name: str,
    refs: list[tuple[int, str]],
    frame_results: list[dict],
    args: argparse.Namespace,
) -> tuple[str, str]:
    out_root = obj_dir / args.out_subdir
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    refs_meta = []
    for view_id, frame_id in refs:
        result = next((x for x in frame_results if x["frame_id"] == frame_id), None)
        if result is None:
            continue
        frame_out = out_root / frame_id
        frame_out.mkdir(parents=True, exist_ok=True)
        masks_meta = []
        for out_idx, item in enumerate(result["selected"]):
            part_id = int(item["part_id"])
            out_name = f"mask_{part_id}.png"
            out_path = frame_out / out_name
            if out_path.exists():
                out_name = f"mask_{part_id}_{out_idx:02d}.png"
                out_path = frame_out / out_name
            shared._write_selected_mask(item["mask"], str(out_path))
            masks_meta.append(
                {
                    "path": os.path.join(args.out_subdir, frame_id, out_name),
                    "candidate_index": int(item["candidate_index"]),
                    "area": int(item["area"]),
                    "part_id": int(part_id),
                    "matched_gt": item["matched_gt"],
                    "source_pred_mask": os.path.relpath(str(item["path"]), str(obj_dir)),
                }
            )
        refs_meta.append(
            {
                "view_id": int(view_id),
                "frame_id": str(frame_id),
                "mask_dir": os.path.join(args.out_subdir, frame_id),
                "masks": [m["path"] for m in masks_meta],
                "mask_details": masks_meta,
                "sam_candidates": int(result["candidates"]),
                "selection_backend": "sam-vlm",
                "status": "selected" if masks_meta else "empty",
            }
        )

    manifest = {
        "object": obj_name,
        "dataset_kind": "partnet",
        "selection_backend": "sam-vlm",
        "reference_policy": args.reference_policy,
        "reference_mask_subdir": args.out_subdir,
        "pred_mask_subdir": args.pred_mask_subdir,
        "note": "Sampled check-only run: consumes existing pred_mask candidates and filters them with eccv/segmentation/check.py.",
        "references": refs_meta,
        "reconstruction": [],
    }
    selected_path = out_root / args.selected_json_name
    with selected_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report_path = out_root / "check_part_report.json"
    report = {
        "object": obj_name,
        "out_subdir": args.out_subdir,
        "frames": [
            {
                "frame_id": str(x["frame_id"]),
                "candidates": int(x["candidates"]),
                "selected": int(len(x["selected"])),
                "evaluations": x["eval_rows"],
            }
            for x in frame_results
        ],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(selected_path), str(report_path)


def _process_object(obj_name: str, args: argparse.Namespace) -> dict:
    root = Path(args.root)
    obj_dir = root / obj_name
    refs = shared._select_reference_frames(
        str(obj_dir),
        policy=args.reference_policy,
        min_visible_pixels=int(args.min_visible_pixels),
    )
    if not refs:
        return {"object": obj_name, "status": "skipped", "reason": "no_reference_frame"}

    out_root = obj_dir / args.out_subdir
    selected_path = out_root / args.selected_json_name
    report_path = out_root / "check_part_report.json"
    if selected_path.exists() and report_path.exists() and not args.overwrite:
        return {
            "object": obj_name,
            "status": "exists",
            "references": [{"view_id": int(v), "frame_id": str(f)} for v, f in refs],
            "selected_json": str(selected_path),
            "report_json": str(report_path),
        }

    if args.dry_run:
        frame_infos = []
        shared_args = _make_shared_args(args)
        for view_id, frame_id in refs:
            candidates = shared._load_pred_mask_candidates(str(obj_dir), frame_id, shared_args)
            frame_infos.append({"view_id": int(view_id), "frame_id": str(frame_id), "candidates": len(candidates)})
        return {"object": obj_name, "status": "dry_run", "references": frame_infos}

    frame_results = []
    for view_id, frame_id in refs:
        result, _ = _evaluate_frame(obj_dir, frame_id, args)
        frame_results.append(result)
        print(
            f"[{_now()}] [CHECK-PART] {obj_name} view={view_id} frame={frame_id} "
            f"candidates={result['candidates']} selected={len(result['selected'])}",
            flush=True,
        )

    selected_json, report_json = _write_object_outputs(obj_dir, obj_name, refs, frame_results, args)
    return {
        "object": obj_name,
        "status": "done",
        "references": [{"view_id": int(v), "frame_id": str(f)} for v, f in refs],
        "selected_json": selected_json,
        "report_json": report_json,
        "frames": [
            {
                "frame_id": str(x["frame_id"]),
                "candidates": int(x["candidates"]),
                "selected": int(len(x["selected"])),
            }
            for x in frame_results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Sample PartNet objects and run only the VLM check stage on existing pred_mask outputs.")
    p.add_argument("--root", type=str, default=DEFAULT_PARTNET_ROOT)
    p.add_argument("--objects", type=str, default="", help="Optional comma-separated object names; disables full-root sampling pool.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--sample-num", type=int, default=20, help="<=0 means run all candidate objects.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pred-mask-subdir", type=str, default="pred_mask")
    p.add_argument("--out-subdir", type=str, default="chosen_part_check_sample")
    p.add_argument("--selected-json-name", type=str, default="selected_parts.json")
    p.add_argument("--sample-json", type=str, default="", help="Default: eccv/check_part_samples/sampled_objects.json")
    p.add_argument("--summary-json", type=str, default="", help="Default: eccv/check_part_samples/check_part_summary.json")
    p.add_argument("--reference-policy", type=str, default="first", choices=["first", "best"])
    p.add_argument("--min-visible-pixels", type=int, default=64)
    p.add_argument("--max-candidates-per-ref", type=int, default=30)
    p.add_argument("--max-selected-per-ref", type=int, default=0)
    p.add_argument("--vlm-workers", type=int, default=1)
    p.add_argument("--unique-reference-parts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    all_objects = _list_candidate_objects(root, args.objects, args.start, args.end)
    sampled_objects, sample_meta = _balanced_sample_objects(root, all_objects, args.sample_num, args.seed)

    sample_dir = ECCV_ROOT / "check_part_samples"
    sample_json = Path(args.sample_json).resolve() if args.sample_json else sample_dir / "sampled_objects.json"
    summary_json = Path(args.summary_json).resolve() if args.summary_json else sample_dir / "check_part_summary.json"
    sample_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    sample_meta.update(
        {
            "pred_mask_subdir": args.pred_mask_subdir,
            "out_subdir": args.out_subdir,
            "reference_policy": args.reference_policy,
            "max_candidates_per_ref": int(args.max_candidates_per_ref),
            "max_selected_per_ref": int(args.max_selected_per_ref),
            "dry_run": bool(args.dry_run),
        }
    )
    with sample_json.open("w", encoding="utf-8") as f:
        json.dump(sample_meta, f, ensure_ascii=False, indent=2)
    print(f"[{_now()}] [SAMPLE] objects={len(sampled_objects)} json={sample_json}", flush=True)

    results = []
    for idx, obj_name in enumerate(sampled_objects, 1):
        print(f"[{_now()}] [OBJECT] {idx}/{len(sampled_objects)} {obj_name}", flush=True)
        try:
            results.append(_process_object(obj_name, args))
        except Exception as exc:
            results.append({"object": obj_name, "status": "failed", "error": repr(exc)})
            print(f"[{_now()}] [ERROR] {obj_name}: {exc!r}", flush=True)

    summary = {
        "sample_json": str(sample_json),
        "root": str(root),
        "sample_num": int(args.sample_num),
        "seed": int(args.seed),
        "objects": sampled_objects,
        "results": results,
        "status_counts": {
            status: sum(1 for r in results if r.get("status") == status)
            for status in sorted({str(r.get("status")) for r in results})
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{_now()}] [DONE] summary={summary_json}", flush=True)


if __name__ == "__main__":
    main()
