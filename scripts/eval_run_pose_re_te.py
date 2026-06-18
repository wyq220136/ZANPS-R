import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pose_eval_from_ann as pose_eval  # noqa: E402


def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(s))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate run.py pose-est outputs against annotation GT poses with RE/TE, "
            "and write one2any-style total/category summaries."
        )
    )
    parser.add_argument("--data-root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--splits", type=str, default="test_intra,test_inter")
    parser.add_argument("--split", type=str, default="", help="Single split alias; overrides --splits.")
    parser.add_argument("--objects", type=str, default="", help="Comma-separated object names; empty means all.")
    parser.add_argument("--object", type=str, default="", help="Single object alias.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end; <=0 means all remaining.")
    parser.add_argument(
        "--match-json",
        type=str,
        default=os.path.join("match_vis_direct_match", "match_results_sam6d_style.json"),
    )
    parser.add_argument("--gt-pose-root", type=str, default="", help="Default: <split-root>/gt_pose_from_ann.")
    parser.add_argument("--output-tag", type=str, default="", help="Pose output suffix, e.g. ablation -> ob_in_cam2_ablation.")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--te-unit-scale", type=float, default=100.0, help="Default converts meters to cm.")
    parser.add_argument("--object-eval-json", type=str, default="pose_eval_from_ann_run_re_te.json")
    parser.add_argument("--output-summary", type=str, default="run_pose_re_te_summary.json")
    parser.add_argument("--output-detail", type=str, default="run_pose_re_te_detail.json")
    return parser.parse_args()


def object_category(obj_name: str) -> str:
    return str(obj_name).split("_", 1)[0]


def resolve_split_roots(args: argparse.Namespace) -> List[Path]:
    root = Path(args.data_root)
    if (root / "objs").is_dir():
        return [root]
    splits = [args.split] if args.split.strip() else [x.strip() for x in args.splits.split(",") if x.strip()]
    return [root / split for split in splits]


def collect_objects(split_root: Path, args: argparse.Namespace) -> List[str]:
    raw = args.object.strip() or args.objects.strip()
    objs_root = split_root / "objs"
    if raw:
        names = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        names = sorted(
            [p.name for p in objs_root.iterdir() if p.is_dir() and not p.name.startswith("_")],
            key=natural_sort_key,
        )
    start = max(0, int(args.start))
    end = int(args.end)
    return names[start:] if end <= 0 else names[start:end]


def accuracy(records: List[dict], re_thres: float, te_thres_m: float) -> Optional[float]:
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return None
    hits = [
        r
        for r in ok
        if float(r.get("R_e_deg", np.inf)) <= float(re_thres)
        and float(r.get("T_e", np.inf)) <= float(te_thres_m)
    ]
    return float(len(hits) / len(ok))


def summarize_records(records: List[dict], te_unit_scale: float) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    re_vals = [float(r["R_e_deg"]) for r in ok if np.isfinite(float(r["R_e_deg"]))]
    te_vals = [float(r["T_e"]) for r in ok if np.isfinite(float(r["T_e"]))]
    te_scaled = [v * float(te_unit_scale) for v in te_vals]
    return {
        "num_records": int(len(records)),
        "num_ok": int(len(ok)),
        "mean_re": float(np.mean(re_vals)) if re_vals else None,
        "median_re": float(np.median(re_vals)) if re_vals else None,
        "mean_te": float(np.mean(te_vals)) if te_vals else None,
        "median_te": float(np.median(te_vals)) if te_vals else None,
        "mean_te_scaled": float(np.mean(te_scaled)) if te_scaled else None,
        "median_te_scaled": float(np.median(te_scaled)) if te_scaled else None,
        "acc_5deg_2cm": accuracy(records, 5.0, 0.02),
        "acc_5deg_5cm": accuracy(records, 5.0, 0.05),
        "acc_10deg_2cm": accuracy(records, 10.0, 0.02),
        "acc_10deg_5cm": accuracy(records, 10.0, 0.05),
        "acc_10deg_10cm": accuracy(records, 10.0, 0.10),
    }


def make_group_summary(key: str, name: str, records: List[dict], te_unit_scale: float, extra: Optional[dict] = None) -> dict:
    out = {key: name, **summarize_records(records, te_unit_scale)}
    if extra:
        out.update(extra)
    return out


def summarize_split(split_root: Path, args: argparse.Namespace) -> tuple[dict, List[dict]]:
    objs_root = split_root / "objs"
    if not objs_root.is_dir():
        raise FileNotFoundError(f"objs root not found: {objs_root}")
    gt_root = Path(args.gt_pose_root) if args.gt_pose_root.strip() else split_root / "gt_pose_from_ann"
    if not gt_root.is_dir():
        raise FileNotFoundError(f"gt pose root not found: {gt_root}")

    objects = collect_objects(split_root, args)
    all_records: List[dict] = []
    per_object = []
    for obj_name in objects:
        obj_dir = objs_root / obj_name
        result = pose_eval.evaluate_object_pose_from_ann(
            obj_dir=str(obj_dir),
            match_json_relpath=args.match_json,
            gt_root=str(gt_root),
            split_root=str(split_root),
            output_json_name=args.object_eval_json,
            output_tag=args.output_tag,
            iou_threshold=args.iou_threshold,
        )
        records = result.get("records", [])
        for rec in records:
            rec["split"] = split_root.name
            rec["category"] = object_category(obj_name)
            if rec.get("status") == "ok":
                rec["T_e_scaled"] = float(rec["T_e"]) * float(args.te_unit_scale)
        all_records.extend(records)
        per_object.append(
            make_group_summary(
                "object",
                obj_name,
                records,
                args.te_unit_scale,
                {"split": split_root.name, "category": object_category(obj_name), "status": result.get("status", "unknown")},
            )
        )

    per_category = []
    for cat in sorted({object_category(o) for o in objects}, key=natural_sort_key):
        cat_records = [r for r in all_records if r.get("category") == cat]
        cat_objects = sorted([o for o in objects if object_category(o) == cat], key=natural_sort_key)
        per_category.append(
            make_group_summary(
                "category",
                cat,
                cat_records,
                args.te_unit_scale,
                {"split": split_root.name, "objects": cat_objects, "num_objects": int(len(cat_objects))},
            )
        )

    summary = {
        "split": split_root.name,
        "split_root": str(split_root),
        "match_json": args.match_json,
        "gt_pose_root": str(gt_root),
        "pose_output_tag": args.output_tag,
        "te_unit_scale": float(args.te_unit_scale),
        "objects": objects,
        "num_objects": int(len(objects)),
        "total": summarize_records(all_records, args.te_unit_scale),
        "per_category": per_category,
        "per_object": per_object,
    }
    return summary, all_records


def main() -> None:
    args = parse_args()
    split_summaries = []
    all_records: List[dict] = []
    for split_root in resolve_split_roots(args):
        if not split_root.exists():
            print(f"[SKIP] split root not found: {split_root}", flush=True)
            continue
        summary, records = summarize_split(split_root, args)
        split_summaries.append(summary)
        all_records.extend(records)
        with (split_root / args.output_summary).open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with (split_root / args.output_detail).open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[OK] {split_root.name}: ok={summary['total']['num_ok']} records={summary['total']['num_records']}", flush=True)

    root = Path(args.data_root)
    combined = {
        "root": str(root),
        "splits": split_summaries,
        "te_unit_scale": float(args.te_unit_scale),
        "total": summarize_records(all_records, args.te_unit_scale),
        "per_category": [],
    }
    for cat in sorted({r.get("category", "") for r in all_records if r.get("category", "")}, key=natural_sort_key):
        combined["per_category"].append(
            make_group_summary("category", cat, [r for r in all_records if r.get("category") == cat], args.te_unit_scale)
        )
    with (root / args.output_summary).open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    with (root / args.output_detail).open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"[Done] wrote {root / args.output_summary}")


if __name__ == "__main__":
    main()
