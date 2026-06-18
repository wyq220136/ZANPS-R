import argparse
import json
import os
import subprocess
import sys
from typing import Dict


def run_cmd(cmd, cwd: str):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_total(summary: dict) -> Dict[str, float]:
    t = summary.get("total", {}) if isinstance(summary, dict) else {}
    return {
        "num_ok": t.get("num_ok", None),
        "mean_re": t.get("mean_re", None),
        "mean_te": t.get("mean_te", None),
        "mean_te_scaled": t.get("mean_te_scaled", None),
        "acc_5deg_2cm": t.get("acc_5deg_2cm", None),
        "acc_5deg_5cm": t.get("acc_5deg_5cm", None),
        "acc_10deg_2cm": t.get("acc_10deg_2cm", None),
        "acc_10deg_5cm": t.get("acc_10deg_5cm", None),
        "acc_10deg_10cm": t.get("acc_10deg_10cm", None),
    }


def main():
    p = argparse.ArgumentParser("Run overall (test_intra + test_inter) pose eval for UNOPose / One2Any / Any6D")
    p.add_argument("--root", type=str, required=True, help="Dataset root containing test_intra and test_inter.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--objects", type=str, default="", help="Optional comma-separated object filter.")
    p.add_argument("--te-unit-scale", type=float, default=100.0)
    p.add_argument("--out", type=str, default="results/overall_pose_eval.json")
    p.add_argument("--skip-unopose", action="store_true")
    p.add_argument("--skip-one2any", action="store_true")
    p.add_argument("--skip-any6d", action="store_true")
    p.add_argument("--unopose-pred-subdir", type=str, default="unopose_results")
    p.add_argument("--one2any-pred-subdir", type=str, default="one2any_results_dataset_train_val")
    p.add_argument("--any6d-pred-subdir", type=str, default="any6d_results")
    p.add_argument("--unopose-pose-frame", type=str, default="standard", choices=["standard", "raw"])
    p.add_argument("--any6d-pose-frame", type=str, default="standard", choices=["standard", "raw-anchor"])
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = os.path.abspath(args.root)

    results = {"root": root, "methods": {}}

    if not args.skip_unopose:
        cmd = [
            args.python,
            os.path.join(repo_root, "UNOPose", "eval_partnet_add_adds.py"),
            "--root",
            root,
            "--all-splits",
            "--pred-subdir",
            args.unopose_pred_subdir,
            "--pose-frame",
            args.unopose_pose_frame,
            "--te-unit-scale",
            str(args.te_unit_scale),
        ]
        if args.objects.strip():
            cmd.extend(["--objects", args.objects.strip()])
        run_cmd(cmd, repo_root)
        unopose_summary = load_json(os.path.join(root, "unopose_eval_rt_all_splits_summary.json"))
        results["methods"]["unopose"] = {
            "summary_path": os.path.join(root, "unopose_eval_rt_all_splits_summary.json"),
            "total": pick_total(unopose_summary),
        }

    if not args.skip_one2any:
        cmd = [
            args.python,
            os.path.join(repo_root, "One2Any", "eval_partnet_add_adds.py"),
            "--root",
            root,
            "--all-splits",
            "--pred-subdir",
            args.one2any_pred_subdir,
            "--te-unit-scale",
            str(args.te_unit_scale),
        ]
        if args.objects.strip():
            cmd.extend(["--objects", args.objects.strip()])
        run_cmd(cmd, repo_root)
        one2any_summary = load_json(os.path.join(root, "one2any_eval_rt_all_splits_summary.json"))
        results["methods"]["one2any"] = {
            "summary_path": os.path.join(root, "one2any_eval_rt_all_splits_summary.json"),
            "total": pick_total(one2any_summary),
        }

    if not args.skip_any6d:
        cmd = [
            args.python,
            os.path.join(repo_root, "Any6D", "eval_partnet_add_adds.py"),
            "--root",
            root,
            "--all-splits",
            "--splits",
            "test_intra,test_inter",
            "--pred-subdir",
            args.any6d_pred_subdir,
            "--pose-frame",
            args.any6d_pose_frame,
            "--te-unit-scale",
            str(args.te_unit_scale),
        ]
        if args.objects.strip():
            cmd.extend(["--objects", args.objects.strip()])
        run_cmd(cmd, repo_root)
        any6d_summary = load_json(os.path.join(root, "any6d_eval_add_adds_all_splits_summary.json"))
        results["methods"]["any6d"] = {
            "summary_path": os.path.join(root, "any6d_eval_add_adds_all_splits_summary.json"),
            "total": pick_total(any6d_summary),
        }

    out_path = os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote: {out_path}")
    for name, data in results["methods"].items():
        t = data["total"]
        print(
            f"[{name}] num_ok={t.get('num_ok')} mean_re={t.get('mean_re')} "
            f"mean_te={t.get('mean_te')} mean_te_scaled={t.get('mean_te_scaled')}"
        )


if __name__ == "__main__":
    main()

