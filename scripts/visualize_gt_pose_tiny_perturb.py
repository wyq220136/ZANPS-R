import argparse
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import visualize_one2any_bad_rt as bad_rt


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def natural_sort_key(value: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", str(value))]


def object_name_from_frame(frame_id: str) -> str:
    return str(frame_id).rsplit("_", 2)[0]


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def metric_float(record: dict, *names: str) -> Optional[float]:
    for name in names:
        if name not in record or record.get(name) is None:
            continue
        try:
            value = float(record[name])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bad_object_score(item: dict, args) -> float:
    mean_re = metric_float(item, "mean_re")
    mean_te_scaled = metric_float(item, "mean_te_scaled")
    mean_te = metric_float(item, "mean_te")
    acc = metric_float(item, args.bad_object_acc_key)
    if mean_te_scaled is None and mean_te is not None:
        mean_te_scaled = mean_te * float(args.te_unit_scale)

    if args.bad_object_sort_by == "mean_re":
        return -np.inf if mean_re is None else mean_re
    if args.bad_object_sort_by == "mean_te_scaled":
        return -np.inf if mean_te_scaled is None else mean_te_scaled
    if args.bad_object_sort_by == "low_acc":
        return -np.inf if acc is None else -acc
    if args.bad_object_sort_by == "sum":
        re_norm = 0.0 if mean_re is None else mean_re / float(args.re_ref)
        te_norm = 0.0 if mean_te_scaled is None else mean_te_scaled / float(args.te_scaled_ref)
        acc_bad = 0.0 if acc is None else (1.0 - acc)
        return float(re_norm + te_norm + acc_bad)

    re_norm = 0.0 if mean_re is None else mean_re / float(args.re_ref)
    te_norm = 0.0 if mean_te_scaled is None else mean_te_scaled / float(args.te_scaled_ref)
    acc_bad = 0.0 if acc is None else (1.0 - acc)
    return float(max(re_norm, te_norm, acc_bad))


def select_bad_objects_from_summary(args) -> Tuple[List[str], str]:
    if not args.summary_json or not os.path.exists(args.summary_json):
        return [], ""
    data = load_json(args.summary_json)
    per_object = data.get("per_object", [])
    if not isinstance(per_object, list):
        return [], str(data.get("split", ""))

    candidates = []
    for item in per_object:
        if not isinstance(item, dict):
            continue
        obj = str(item.get("object", "")).strip()
        if not obj:
            continue
        if args.require_ok_object and str(item.get("status", "")).lower() not in ("ok", ""):
            continue
        num_ok = metric_float(item, "num_ok")
        if num_ok is not None and num_ok <= 0:
            continue
        item = dict(item)
        item["_bad_score"] = bad_object_score(item, args)
        candidates.append(item)

    candidates.sort(
        key=lambda x: (
            x.get("_bad_score", -np.inf),
            metric_float(x, "mean_re") or -np.inf,
            metric_float(x, "mean_te_scaled") or -np.inf,
        ),
        reverse=True,
    )
    top_k = max(0, int(args.bad_object_top_k))
    if top_k > 0:
        candidates = candidates[:top_k]
    return [str(x["object"]) for x in candidates], str(data.get("split", ""))


def select_bad_frame_link_targets(args) -> Dict[Tuple[str, str], Optional[set]]:
    if args.selection_mode != "bad":
        return {}
    if not args.detail_json or not os.path.exists(args.detail_json):
        return {}
    detail = bad_rt.load_json(args.detail_json)
    records = list(bad_rt.iter_detail_records(detail))
    selected = bad_rt.select_bad_records(records, args)
    targets: Dict[Tuple[str, str], Optional[set]] = {}
    for rec in selected:
        split = str(rec.get("split", ""))
        frame_id = str(rec.get("frame_id", ""))
        link_name = str(rec.get("link_name", ""))
        part_key = str(rec.get("part_key", ""))
        if not split or not frame_id:
            continue
        key = (split, frame_id)
        if key not in targets:
            targets[key] = set()
        if link_name:
            targets[key].add(link_name)
        elif part_key:
            idx = re.search(r"(\d+)$", part_key)
            if idx:
                targets[key].add(f"link_{idx.group(1)}")
    return targets


def resolve_split_roots(dataset_root: str, splits: str) -> List[str]:
    root = os.path.abspath(dataset_root)
    if os.path.isdir(os.path.join(root, "bbox")) and os.path.isdir(os.path.join(root, "gt_pose_from_ann")):
        return [root]
    out = []
    for split in parse_csv(splits):
        split_root = os.path.join(root, split)
        if os.path.isdir(split_root):
            out.append(split_root)
    return out


def find_image_path(split_root: str, frame_id: str) -> str:
    obj_name = object_name_from_frame(frame_id)
    folders = [
        os.path.join(split_root, "rgb"),
        os.path.join(split_root, "objs", obj_name, "rgb"),
    ]
    for folder in folders:
        for ext in IMAGE_EXTS:
            path = os.path.join(folder, f"{frame_id}{ext}")
            if os.path.exists(path):
                return path
    return ""


def collect_frame_ids(split_root: str, objects: List[str], max_frames: int, target_frames: Optional[set] = None) -> List[str]:
    if target_frames is not None:
        frame_ids = sorted(target_frames, key=natural_sort_key)
        if max_frames > 0:
            frame_ids = frame_ids[:max_frames]
        return frame_ids

    frame_ids = []
    if objects:
        for obj in objects:
            rgb_dir = os.path.join(split_root, "objs", obj, "rgb")
            gt_mask_dir = os.path.join(split_root, "objs", obj, "gt_mask")
            if os.path.isdir(rgb_dir):
                for name in os.listdir(rgb_dir):
                    stem, ext = os.path.splitext(name)
                    if ext.lower() in IMAGE_EXTS:
                        frame_ids.append(stem)
            elif os.path.isdir(gt_mask_dir):
                frame_ids.extend(
                    [d for d in os.listdir(gt_mask_dir) if os.path.isdir(os.path.join(gt_mask_dir, d))]
                )
            else:
                frame_ids.extend(frame_from_bbox for frame_from_bbox in collect_frame_ids_from_bbox(split_root, obj))
    else:
        rgb_dir = os.path.join(split_root, "rgb")
        if os.path.isdir(rgb_dir):
            for name in os.listdir(rgb_dir):
                stem, ext = os.path.splitext(name)
                if ext.lower() in IMAGE_EXTS:
                    frame_ids.append(stem)
        else:
            frame_ids.extend(collect_frame_ids_from_bbox(split_root, ""))

    seen = set()
    unique = []
    for frame_id in sorted(frame_ids, key=natural_sort_key):
        if frame_id in seen:
            continue
        seen.add(frame_id)
        unique.append(frame_id)
        if max_frames > 0 and len(unique) >= max_frames:
            break
    return unique


def collect_frame_ids_from_bbox(split_root: str, obj_name: str) -> List[str]:
    bbox_dir = os.path.join(split_root, "bbox")
    if not os.path.isdir(bbox_dir):
        return []
    out = []
    for name in os.listdir(bbox_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".pkl":
            continue
        if obj_name and not stem.startswith(f"{obj_name}_"):
            continue
        out.append(stem)
    return out


def load_pose_txt(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4):
            return None
        return pose
    except Exception:
        return None


def build_rt_4x4(pose_rts_param: dict) -> np.ndarray:
    rot = np.asarray(pose_rts_param["R"], dtype=np.float64).reshape(3, 3)
    trans = np.asarray(pose_rts_param["T"], dtype=np.float64).reshape(3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def load_frame_annotation(split_root: str, frame_id: str) -> Dict[str, dict]:
    bbox_path = os.path.join(split_root, "bbox", f"{frame_id}.pkl")
    if not os.path.exists(bbox_path):
        return {}
    try:
        with open(bbox_path, "rb") as f:
            data = pickle.load(f)
        bbox_pose_dict = data.get("bbox_pose_dict", {})
        return bbox_pose_dict if isinstance(bbox_pose_dict, dict) else {}
    except Exception:
        return {}


def load_intrinsic(split_root: str, frame_id: str) -> Optional[np.ndarray]:
    meta_path = os.path.join(split_root, "metafile", f"{frame_id}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return np.asarray(meta["camera_intrinsic"], dtype=np.float64).reshape(3, 3)
        except Exception:
            pass

    obj_name = object_name_from_frame(frame_id)
    k_path = os.path.join(split_root, "objs", obj_name, "K.txt")
    if os.path.exists(k_path):
        try:
            return np.loadtxt(k_path, dtype=np.float64).reshape(3, 3)
        except Exception:
            pass
    return None


def project_points(k: np.ndarray, pts_cam: np.ndarray):
    z = pts_cam[:, 2]
    valid = z > 1e-8
    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (pts_cam[:, 0] * k[0, 0] / np.maximum(z, 1e-8)) + k[0, 2]
    uv[:, 1] = (pts_cam[:, 1] * k[1, 1] / np.maximum(z, 1e-8)) + k[1, 2]
    return uv, valid


def draw_line_if_valid(img: np.ndarray, p1, p2, color, thickness=2):
    if p1 is None or p2 is None:
        return
    cv2.line(img, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_bbox3d(img: np.ndarray, uv: np.ndarray, valid: np.ndarray, color, thickness=2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    def to_pt(i):
        if not bool(valid[i]):
            return None
        return int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))

    for a, b in edges:
        draw_line_if_valid(img, to_pt(a), to_pt(b), color=color, thickness=thickness)


def draw_pose_axis(
    img: np.ndarray,
    k: np.ndarray,
    pose_cam: np.ndarray,
    axis_len: float,
    colors: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
    thickness: int = 2,
):
    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    x_end = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64)
    y_end = np.array([[0.0, axis_len, 0.0]], dtype=np.float64)
    z_end = np.array([[0.0, 0.0, axis_len]], dtype=np.float64)

    r = pose_cam[:3, :3]
    t = pose_cam[:3, 3].reshape(1, 3)
    pts_cam = np.concatenate([origin, x_end, y_end, z_end], axis=0)
    pts_cam = (r @ pts_cam.T).T + t
    uv, valid = project_points(k, pts_cam)

    def pt(i):
        if not bool(valid[i]):
            return None
        return int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))

    o = pt(0)
    draw_line_if_valid(img, o, pt(1), colors[0], thickness=thickness)
    draw_line_if_valid(img, o, pt(2), colors[1], thickness=thickness)
    draw_line_if_valid(img, o, pt(3), colors[2], thickness=thickness)


def local_bbox_from_annotation(info: dict) -> Optional[np.ndarray]:
    bbox_world = np.asarray(info.get("bbox", []), dtype=np.float64)
    if bbox_world.shape != (8, 3):
        return None
    pose_rts = info.get("pose_RTS_param", None)
    if pose_rts is None:
        return None
    pose_world = build_rt_4x4(pose_rts)
    return (pose_world[:3, :3].T @ (bbox_world - pose_world[:3, 3]).T).T


def draw_pose_with_annotation_bbox(
    img: np.ndarray,
    k: np.ndarray,
    pose_cam: np.ndarray,
    local_bbox: np.ndarray,
    bbox_color,
    axis_colors,
    bbox_thickness: int,
    axis_thickness: int,
    axis_ratio: float,
):
    bbox_cam = (pose_cam[:3, :3] @ local_bbox.T).T + pose_cam[:3, 3]
    uv, valid = project_points(k, bbox_cam)
    draw_bbox3d(img, uv, valid, color=bbox_color, thickness=bbox_thickness)

    diag = float(np.linalg.norm(local_bbox.max(axis=0) - local_bbox.min(axis=0)))
    axis_len = float(max(1e-4, diag * axis_ratio))
    draw_pose_axis(img, k, pose_cam, axis_len=axis_len, colors=axis_colors, thickness=axis_thickness)


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


def perturb_pose(pose: np.ndarray, rng: np.random.Generator, max_rot_deg: float, max_trans_m: float) -> np.ndarray:
    out = pose.copy().astype(np.float64)
    axis = rng.normal(size=3)
    angle = np.deg2rad(float(max_rot_deg)) * rng.uniform(-1.0, 1.0)
    r_delta = axis_angle_to_matrix(axis, angle)
    out[:3, :3] = r_delta @ out[:3, :3]

    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-12 and max_trans_m > 0.0:
        direction = direction / norm
        out[:3, 3] += direction * rng.uniform(0.0, float(max_trans_m))
    return out


def annotate(img: np.ndarray, lines: List[str]):
    y = 26
    for line in lines:
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def visualize_frame(
    split_root: str,
    frame_id: str,
    args,
    rng: np.random.Generator,
    target_links: Optional[set] = None,
) -> Tuple[bool, dict]:
    rgb_path = find_image_path(split_root, frame_id)
    if not rgb_path:
        return False, {"frame_id": frame_id, "status": "missing_rgb"}
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        return False, {"frame_id": frame_id, "status": "read_rgb_failed", "rgb_path": rgb_path}

    k = load_intrinsic(split_root, frame_id)
    if k is None:
        return False, {"frame_id": frame_id, "status": "missing_intrinsic", "rgb_path": rgb_path}

    ann = load_frame_annotation(split_root, frame_id)
    if not ann:
        return False, {"frame_id": frame_id, "status": "missing_annotation_bbox", "rgb_path": rgb_path}

    drawn = 0
    links = sorted(ann.keys(), key=natural_sort_key)
    if target_links is not None:
        links = [link for link in links if link in target_links]
    if args.links:
        keep_links = set(parse_csv(args.links))
        links = [link for link in links if link in keep_links]
    if args.max_parts > 0:
        links = links[: args.max_parts]

    records = []
    for link_name in links:
        pose_path = os.path.join(split_root, args.gt_pose_dir, f"{frame_id}__{link_name}.txt")
        gt_pose = load_pose_txt(pose_path)
        if gt_pose is None:
            continue
        local_bbox = local_bbox_from_annotation(ann[link_name])
        if local_bbox is None:
            continue
        perturbed_pose = perturb_pose(gt_pose, rng, args.max_rot_deg, args.max_trans_m)

        draw_pose_with_annotation_bbox(
            img,
            k,
            gt_pose,
            local_bbox,
            bbox_color=(0, 255, 0),
            axis_colors=((0, 0, 255), (0, 255, 0), (255, 0, 0)),
            bbox_thickness=args.gt_bbox_thickness,
            axis_thickness=args.axis_thickness,
            axis_ratio=args.axis_ratio,
        )
        draw_pose_with_annotation_bbox(
            img,
            k,
            perturbed_pose,
            local_bbox,
            bbox_color=(0, 0, 255),
            axis_colors=((0, 165, 255), (255, 255, 0), (255, 0, 255)),
            bbox_thickness=args.perturbed_bbox_thickness,
            axis_thickness=args.axis_thickness,
            axis_ratio=args.axis_ratio,
        )
        drawn += 1
        records.append(
            {
                "link_name": link_name,
                "gt_pose_path": pose_path,
                "max_rot_deg": args.max_rot_deg,
                "max_trans_m": args.max_trans_m,
            }
        )

    if drawn <= 0:
        return False, {"frame_id": frame_id, "status": "nothing_drawn", "rgb_path": rgb_path}

    if args.draw_labels:
        annotate(
            img,
            [
                "GT annotation bbox green, perturbation red",
                f"{os.path.basename(split_root)}/{frame_id} parts={drawn}",
                f"perturb <= {args.max_rot_deg:g} deg, {args.max_trans_m:g} m",
            ],
        )

    split_name = os.path.basename(split_root.rstrip(os.sep))
    out_path = os.path.join(args.output_dir, split_name, f"{frame_id}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return True, {
        "frame_id": frame_id,
        "status": "ok",
        "rgb_path": rgb_path,
        "output_path": out_path,
        "parts_drawn": drawn,
        "records": records,
    }


def run(args):
    rng = np.random.default_rng(int(args.seed))
    explicit_objects = parse_csv(args.objects)
    selected_bad_objects = []
    summary_split = ""
    targets = select_bad_frame_link_targets(args)
    if targets:
        target_splits = sorted({split for split, _ in targets.keys()}, key=natural_sort_key)
        if args.splits == "auto":
            args.splits = ",".join(target_splits)
        print(
            "[INFO] selected bad_rt frame/link targets from detail: "
            f"frames={len(targets)} links={sum(len(v or []) for v in targets.values())}"
        )
    elif args.selection_mode == "bad" and not explicit_objects:
        selected_bad_objects, summary_split = select_bad_objects_from_summary(args)
        if selected_bad_objects:
            print(f"[INFO] selected bad objects from summary fallback: {','.join(selected_bad_objects)}")
            explicit_objects = selected_bad_objects
            if summary_split and args.splits == "auto":
                args.splits = summary_split

    if args.splits == "auto":
        args.splits = "test_intra,test_inter"
    split_roots = resolve_split_roots(args.dataset_root, args.splits)
    if not split_roots:
        raise FileNotFoundError(f"No valid split root found under {args.dataset_root}")

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []
    ok = 0
    for split_root in split_roots:
        split_name = os.path.basename(split_root.rstrip(os.sep))
        target_frames = None
        if targets:
            target_frames = {frame for split, frame in targets.keys() if split == split_name}
        frames = collect_frame_ids(split_root, explicit_objects, args.max_frames, target_frames=target_frames)
        for frame_id in frames:
            target_links = targets.get((split_name, frame_id), None) if targets else None
            success, info = visualize_frame(split_root, frame_id, args, rng, target_links=target_links)
            info["split_root"] = split_root
            outputs.append(info)
            ok += int(success)
            if success:
                print(f"[OK] {info['output_path']}")
            else:
                print(f"[SKIP] {os.path.basename(split_root)}/{frame_id}: {info['status']}")

    summary = {
        "dataset_root": args.dataset_root,
        "splits": args.splits,
        "objects": explicit_objects,
        "summary_json": args.summary_json,
        "detail_json": args.detail_json,
        "selection_source": "all_frames" if args.selection_mode == "all" else (
            "detail_json_bad_rt" if targets else "summary_json_objects"
        ),
        "bad_objects_from_summary": selected_bad_objects,
        "num_bad_rt_target_frames": len(targets),
        "num_bad_rt_target_links": sum(len(v or []) for v in targets.values()),
        "output_dir": args.output_dir,
        "num_frames": len(outputs),
        "num_visualized": ok,
        "max_rot_deg": args.max_rot_deg,
        "max_trans_m": args.max_trans_m,
        "records": outputs,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] visualized={ok}/{len(outputs)} summary={summary_path}")


def get_args():
    parser = argparse.ArgumentParser(
        "Draw GT poses like annotate.py and overlay a small pose perturbation."
    )
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--splits", type=str, default="auto")
    parser.add_argument(
        "--objects",
        type=str,
        default="",
        help="Comma-separated object names. Used only when no bad_rt detail targets are available.",
    )
    parser.add_argument("--links", type=str, default="", help="Comma-separated links, e.g. link_0,link_1. Empty means all.")
    parser.add_argument("--output-dir", type=str, default="gt_pose_tiny_perturb_vis")
    parser.add_argument("--gt-pose-dir", type=str, default="gt_pose_from_ann")
    parser.add_argument(
        "--selection-mode",
        choices=["all", "bad"],
        default="all",
        help="all draws every selected frame/object; bad keeps the old one2any bad-case selection.",
    )
    parser.add_argument("--summary-json", type=str, default="one2any_eval_rt_summary.json")
    parser.add_argument(
        "--detail-json",
        type=str,
        default="one2any_eval_rt_all_splits_detail.json",
        help="Detail json used to select exactly the same high-Re/Te frames as visualize_one2any_bad_rt.py.",
    )
    parser.add_argument("--bad-object-top-k", type=int, default=10)
    parser.add_argument("--bad-object-sort-by", choices=["max", "sum", "mean_re", "mean_te_scaled", "low_acc"], default="max")
    parser.add_argument("--bad-object-acc-key", type=str, default="acc_10deg_10cm")
    parser.add_argument("--require-ok-object", action="store_true", default=True)
    parser.add_argument("--max-frames", type=int, default=0, help="Per invocation cap after sorting frames. 0 means all.")
    parser.add_argument("--max-parts", type=int, default=0, help="Per-frame cap. 0 means all.")
    parser.add_argument("--top-k", type=int, default=50, help="Used when --detail-json is provided.")
    parser.add_argument("--max-per-object", type=int, default=3, help="Used when --detail-json is provided. 0 means no per-object limit.")
    parser.add_argument("--sort-by", choices=["max", "sum", "re", "te", "te_scaled"], default="max")
    parser.add_argument("--re-ref", type=float, default=90.0)
    parser.add_argument("--te-scaled-ref", type=float, default=10.0)
    parser.add_argument("--te-unit-scale", type=float, default=100.0)
    parser.add_argument("--min-re", type=float, default=None)
    parser.add_argument("--min-te", type=float, default=None)
    parser.add_argument("--min-te-scaled", type=float, default=None)
    parser.add_argument("--threshold-mode", choices=["or", "and"], default="or")
    parser.add_argument("--require-ok", action="store_true", default=True)
    parser.add_argument("--include-non-ok", dest="require_ok", action="store_false")
    parser.add_argument("--max-rot-deg", type=float, default=2.0)
    parser.add_argument("--max-trans-m", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--axis-ratio", type=float, default=0.35)
    parser.add_argument("--gt-bbox-thickness", type=int, default=3)
    parser.add_argument("--perturbed-bbox-thickness", type=int, default=2)
    parser.add_argument("--axis-thickness", type=int, default=2)
    parser.add_argument("--draw-labels", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
