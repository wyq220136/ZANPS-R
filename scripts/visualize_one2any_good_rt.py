import argparse
import json
import os
from typing import Dict, List

import numpy as np

import visualize_one2any_bad_rt as vis_bad


def select_good_records(records: List[dict], args) -> List[dict]:
    candidates = []
    per_object_count: Dict[str, int] = {}
    for rec in records:
        if args.require_ok and str(rec.get("status", "")).lower() not in ("ok", ""):
            continue

        re_deg = vis_bad.metric_float(rec, "re", "R_e_deg", "R_e", "rot_error")
        te = vis_bad.metric_float(rec, "te", "T_e", "T_e_m", "trans_error")
        te_scaled = vis_bad.metric_float(rec, "te_scaled", "T_e_scaled", "T_e_cm")
        if te_scaled is None and te is not None:
            te_scaled = te * float(args.te_unit_scale)

        if re_deg is None or te is None:
            continue
        if args.max_re is not None and re_deg > args.max_re:
            continue
        if args.max_te is not None and te > args.max_te:
            continue
        if args.max_te_scaled is not None and (te_scaled is None or te_scaled > args.max_te_scaled):
            continue

        rec = dict(rec)
        rec["_score"] = vis_bad.rank_score(rec, args)
        rec["_re"] = re_deg
        rec["_te"] = te
        rec["_te_scaled"] = te_scaled
        candidates.append(rec)

    candidates.sort(
        key=lambda r: (
            r["_score"],
            r.get("_re") if r.get("_re") is not None else np.inf,
            r.get("_te_scaled") if r.get("_te_scaled") is not None else np.inf,
        )
    )

    selected = []
    for rec in candidates:
        obj = str(rec.get("object", ""))
        if args.max_per_object > 0 and per_object_count.get(obj, 0) >= args.max_per_object:
            continue
        selected.append(rec)
        per_object_count[obj] = per_object_count.get(obj, 0) + 1
        if len(selected) >= args.top_k:
            break
    return selected


def run(args):
    detail = vis_bad.load_json(args.detail_json)
    records = list(vis_bad.iter_detail_records(detail))
    selected = select_good_records(records, args)
    os.makedirs(args.output_dir, exist_ok=True)

    bbox_cache: Dict[str, np.ndarray] = {}
    outputs = []
    ok = 0
    for rec in selected:
        success, info = vis_bad.visualize_record(rec, args, bbox_cache)
        outputs.append(info)
        ok += int(success)
        if success:
            print(f"[OK] {info['output_path']}")
        else:
            print(f"[SKIP] {info['split']}/{info['object']}/{info['frame_id']}/{info['part_key']}: {info['status']}")

    summary = {
        "detail_json": args.detail_json,
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "num_records_in_json": len(records),
        "num_selected": len(selected),
        "num_visualized": ok,
        "selection": {
            "top_k": args.top_k,
            "sort_by": args.sort_by,
            "re_ref": args.re_ref,
            "te_scaled_ref": args.te_scaled_ref,
            "max_re": args.max_re,
            "max_te": args.max_te,
            "max_te_scaled": args.max_te_scaled,
            "max_per_object": args.max_per_object,
        },
        "records": outputs,
    }
    summary_path = os.path.join(args.output_dir, "selected_records.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] visualized={ok}/{len(selected)} summary={summary_path}")


def get_args():
    parser = argparse.ArgumentParser(
        "Select low-Re/Te One2Any cases and overlay GT/One2Any bbox+axis on RGB."
    )
    parser.add_argument("--detail-json", type=str, default="one2any_eval_rt_all_splits_detail.json")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="one2any_good_rt_vis")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-per-object", type=int, default=3, help="0 means no per-object limit.")
    parser.add_argument("--sort-by", choices=["max", "sum", "re", "te", "te_scaled"], default="max")
    parser.add_argument("--re-ref", type=float, default=10.0, help="Re normalization for max/sum ranking.")
    parser.add_argument("--te-scaled-ref", type=float, default=5.0, help="Te_scaled normalization for max/sum ranking.")
    parser.add_argument("--te-unit-scale", type=float, default=100.0, help="Convert Te meters to Te_scaled centimeters.")
    parser.add_argument("--max-re", type=float, default=None)
    parser.add_argument("--max-te", type=float, default=None)
    parser.add_argument("--max-te-scaled", type=float, default=None)
    parser.add_argument("--require-ok", action="store_true", default=True)
    parser.add_argument("--include-non-ok", dest="require_ok", action="store_false")
    parser.add_argument("--pred-subdir", type=str, default="one2any_results")
    parser.add_argument("--gt-pose-dir", type=str, default="gt_pose_from_ann")
    parser.add_argument("--mapping-name", type=str, default="part_mapping_first_frame.json")
    parser.add_argument("--bbox-thickness", type=int, default=2)
    parser.add_argument("--axis-thickness", type=int, default=2)
    parser.add_argument("--axis-ratio", type=float, default=0.35)
    parser.add_argument("--min-axis-scale", type=float, default=0.0)
    parser.add_argument("--max-axis-scale", type=float, default=0.10)
    parser.add_argument(
        "--axis-scale-mode",
        choices=["extent", "diag"],
        default="extent",
        help="extent keeps each axis proportional to that bbox dimension; diag reproduces annotate.py-style single length.",
    )
    parser.add_argument(
        "--axis-origin",
        choices=["pose", "bbox-center"],
        default="pose",
        help="pose draws from the pose origin; bbox-center keeps the visual axis centered in the drawn bbox.",
    )
    parser.add_argument("--draw-labels", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
