import argparse
from collections import defaultdict
import json
import os
import re
import shutil

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from .match import DINOv2CADMatcher
    from .render_score import compute_multiview_render_score
except Exception:
    try:
        from eccv.dino_match.match import DINOv2CADMatcher
        from eccv.dino_match.render_score import compute_multiview_render_score
    except Exception:
        try:
            from match import DINOv2CADMatcher
            from render_score import compute_multiview_render_score
        except Exception:
            DINOv2CADMatcher = None
            compute_multiview_render_score = None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", str(s))]


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for adaptive rerank.")
    return cv2


def _require_np():
    if np is None:
        raise RuntimeError("NumPy is required for adaptive rerank.")
    return np


def _find_intrinsic_path(obj_dir):
    for name in ("intrinsic.txt", "cam_K.txt", "camera_intrinsic.txt", "K.txt"):
        path = os.path.join(obj_dir, name)
        if os.path.exists(path):
            return path
    return None


def _load_intrinsic(path):
    np_mod = _require_np()
    k = np_mod.loadtxt(path, dtype=np_mod.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"invalid intrinsic shape {k.shape} from {path}")
    return k.astype(np_mod.float32)


def _find_rgb_path(obj_dir, frame_id):
    rgb_dir = os.path.join(obj_dir, "rgb")
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(rgb_dir, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def _find_depth_path(obj_dir, frame_id):
    depth_dir = os.path.join(obj_dir, "depth")
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy", ".exr"):
        path = os.path.join(depth_dir, f"{frame_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def _load_depth(path):
    cv2_mod = _require_cv2()
    np_mod = _require_np()
    if path is None:
        return None
    if path.lower().endswith(".npy"):
        depth = np_mod.load(path).astype(np_mod.float32)
    else:
        depth = cv2_mod.imread(path, cv2_mod.IMREAD_UNCHANGED)
        if depth is None:
            return None
        depth = depth.astype(np_mod.float32)
    depth[(~np_mod.isfinite(depth)) | (depth < 0.0)] = 0.0
    return depth


def _load_mask(path):
    cv2_mod = _require_cv2()
    np_mod = _require_np()
    mask = cv2_mod.imread(path, cv2_mod.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np_mod.uint8)


def _bbox_from_mask(mask_np):
    np_mod = _require_np()
    ys, xs = np_mod.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _encode_query_descriptors(image_rgb, mask, semantic_matcher, appearance_matcher):
    if image_rgb is None or mask is None:
        return None, None
    box = _bbox_from_mask(mask)
    if box is None:
        return None, None
    try:
        import torch
    except Exception:
        return None, None
    mask_t = torch.as_tensor(mask[None, ...], dtype=torch.float32)
    box_t = torch.as_tensor([box], dtype=torch.float32)
    sem_desc = semantic_matcher.encode_proposals(image_rgb, mask_t, box_t)
    appe_desc = appearance_matcher.encode_proposals(image_rgb, mask_t, box_t)
    return sem_desc[0:1], appe_desc[0:1]


def _weighted_score(orig_score, render_score, sam6d_weight, render_weight):
    total = float(sam6d_weight)
    score = float(sam6d_weight) * float(orig_score)
    if render_score is not None:
        total += float(render_weight)
        score += float(render_weight) * float(render_score)
    return float(score / max(total, 1e-8))


def _build_sorted_candidates_per_cad(frame_candidates):
    by_cad = defaultdict(list)
    for idx, cand in enumerate(frame_candidates):
        cad_part_id = int(cand.get("cad_part_id", -1))
        if cad_part_id < 0:
            continue
        item = dict(cand)
        item["_frame_candidate_index"] = int(idx)
        item["_proposal_index"] = int(cand.get("proposal_index", -1))
        by_cad[cad_part_id].append(item)

    out = {}
    for cad_part_id, items in by_cad.items():
        out[int(cad_part_id)] = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out


def _evaluate_candidates(
    frame_candidates,
    image_rgb,
    depth,
    intrinsic,
    semantic_matcher=None,
    appearance_matcher=None,
    sam6d_weight=0.5,
    render_weight=0.15,
):
    evals = []
    for i, cand in enumerate(frame_candidates):
        model_dir = cand.get("cad_model_dir", "")
        mask_path = cand.get("saved_mask_path", "") or cand.get("mask_path", "")
        item = {
            "candidate_index": int(i),
            "orig_score": float(cand.get("score", 0.0)),
            "cad_part_id": int(cand.get("cad_part_id", -1)),
            "mask_path": mask_path,
            "render_score": None,
            "render_semantic_score": None,
            "render_appearance_score": None,
            "adaptive_weighted_score": None,
            "valid": False,
            "reason": "",
        }

        if (not mask_path) or (not os.path.exists(mask_path)):
            item["reason"] = "mask_missing"
            evals.append(item)
            continue
        mask = _load_mask(mask_path)
        if mask is None:
            item["reason"] = "mask_read_failed"
            evals.append(item)
            continue

        if (
            render_weight > 0.0
            and image_rgb is not None
            and depth is not None
            and intrinsic is not None
            and semantic_matcher is not None
            and appearance_matcher is not None
            and compute_multiview_render_score is not None
            and model_dir
        ):
            sem_desc, appe_desc = _encode_query_descriptors(
                image_rgb=image_rgb,
                mask=mask,
                semantic_matcher=semantic_matcher,
                appearance_matcher=appearance_matcher,
            )
            if sem_desc is not None and appe_desc is not None:
                render_res = compute_multiview_render_score(
                    image_rgb=image_rgb,
                    depth=depth,
                    k=intrinsic,
                    query_mask=mask,
                    query_sem_desc=sem_desc,
                    query_appe_desc=appe_desc,
                    semantic_matcher=semantic_matcher,
                    appearance_matcher=appearance_matcher,
                    cad_model_dir=model_dir,
                    mesh_path=os.path.join(model_dir, "model.obj"),
                )
                if render_res is not None:
                    item["render_score"] = float(render_res["score"])
                    item["render_semantic_score"] = float(render_res["semantic_score"])
                    item["render_appearance_score"] = float(render_res["appearance_score"])
                    item["render_view_index"] = int(render_res["view_index"])

        item["adaptive_weighted_score"] = _weighted_score(
            orig_score=item["orig_score"],
            render_score=item["render_score"],
            sam6d_weight=sam6d_weight,
            render_weight=render_weight,
        )
        item["valid"] = True
        item["reason"] = "render_rerank" if item["render_score"] is not None else "sam6d_score_only"
        evals.append(item)
    return evals


def _pick_best_for_cad_with_used(
    cad_candidates_sorted,
    topk_per_cad,
    used_masks,
    image_rgb=None,
    depth=None,
    intrinsic=None,
    semantic_matcher=None,
    appearance_matcher=None,
    sam6d_weight=0.5,
    render_weight=0.15,
):
    available = []
    for cand in cad_candidates_sorted:
        pidx = int(cand.get("_proposal_index", -1))
        if pidx >= 0 and pidx in used_masks:
            continue
        available.append(cand)
        if len(available) >= max(1, int(topk_per_cad)):
            break
    if not available:
        return None, [], "no_available_mask_after_used_filter"

    evals = _evaluate_candidates(
        frame_candidates=available,
        image_rgb=image_rgb,
        depth=depth,
        intrinsic=intrinsic,
        semantic_matcher=semantic_matcher,
        appearance_matcher=appearance_matcher,
        sam6d_weight=sam6d_weight,
        render_weight=render_weight,
    )
    valid = [x for x in evals if x.get("valid", False)]
    if valid:
        valid.sort(key=lambda x: float(x.get("adaptive_weighted_score", 0.0)), reverse=True)
        best_local_idx = int(valid[0]["candidate_index"])
    else:
        np_mod = _require_np()
        best_local_idx = int(np_mod.argmax([float(x.get("score", 0.0)) for x in available]))

    selected = dict(available[best_local_idx])
    selected_eval = None
    for ev in evals:
        if int(ev.get("candidate_index", -1)) == int(best_local_idx):
            selected_eval = ev
            break
    if selected_eval is not None:
        for key in (
            "render_score",
            "render_semantic_score",
            "render_appearance_score",
            "render_view_index",
            "adaptive_weighted_score",
        ):
            selected[key] = selected_eval.get(key)
        reason = selected_eval.get("reason", "adaptive_rerank")
    else:
        reason = "fallback_to_original_score"
    return selected, evals, reason


def run_adaptive_rerank_for_object(
    obj_dir,
    match_out_dir,
    reranked_mask_subdir="matched_pred_mask_direct_match_adaptive",
    match_result_json_name="match_results_sam6d_style.json",
    reranked_json_name="match_results_adaptive_weight.json",
    topk_per_cad=3,
    sam6d_weight=0.5,
    render_weight=0.15,
    render_model_name="dinov2_vitl14",
):
    match_json_path = os.path.join(match_out_dir, match_result_json_name)
    if not os.path.exists(match_json_path):
        raise FileNotFoundError(f"match result json not found: {match_json_path}")

    intrinsic = None
    intrinsic_path = _find_intrinsic_path(obj_dir)
    if intrinsic_path is not None:
        intrinsic = _load_intrinsic(intrinsic_path)

    with open(match_json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    out_mask_root = os.path.join(obj_dir, reranked_mask_subdir)
    os.makedirs(out_mask_root, exist_ok=True)

    semantic_matcher = None
    appearance_matcher = None
    if render_weight > 0.0 and DINOv2CADMatcher is not None and compute_multiview_render_score is not None:
        semantic_matcher = DINOv2CADMatcher(
            model_name=render_model_name,
            proposal_size=224,
            chunk_size=16,
            background_mean_fill=True,
            use_multi_layer_fusion=True,
            fusion_layers=8,
        )
        appearance_matcher = DINOv2CADMatcher(
            model_name=render_model_name,
            proposal_size=224,
            chunk_size=16,
            background_mean_fill=False,
            use_multi_layer_fusion=True,
            fusion_layers=8,
        )

    reranked_results = {}
    for frame_id in sorted(all_results.keys(), key=natural_sort_key):
        frame_candidates = all_results.get(frame_id, []) or []
        frame_out_dir = os.path.join(out_mask_root, frame_id)
        os.makedirs(frame_out_dir, exist_ok=True)
        for name in os.listdir(frame_out_dir):
            if name.startswith("mask_") and name.lower().endswith(".png"):
                path = os.path.join(frame_out_dir, name)
                if os.path.isfile(path):
                    os.remove(path)

        if not frame_candidates:
            reranked_results[frame_id] = []
            continue

        image_rgb = None
        depth = None
        if render_weight > 0.0:
            cv2_mod = _require_cv2()
            rgb_path = _find_rgb_path(obj_dir, frame_id)
            if rgb_path is not None:
                image_bgr = cv2_mod.imread(rgb_path, cv2_mod.IMREAD_COLOR)
                if image_bgr is not None:
                    image_rgb = cv2_mod.cvtColor(image_bgr, cv2_mod.COLOR_BGR2RGB)
            depth = _load_depth(_find_depth_path(obj_dir, frame_id))

        by_cad = _build_sorted_candidates_per_cad(frame_candidates)
        selected_items = []
        used_cads = set()
        used_masks = set()
        for cad_part_id in sorted(by_cad.keys()):
            if cad_part_id in used_cads:
                continue
            selected, evals, reason = _pick_best_for_cad_with_used(
                cad_candidates_sorted=by_cad[cad_part_id],
                topk_per_cad=topk_per_cad,
                used_masks=used_masks,
                image_rgb=image_rgb,
                depth=depth,
                intrinsic=intrinsic,
                semantic_matcher=semantic_matcher,
                appearance_matcher=appearance_matcher,
                sam6d_weight=sam6d_weight,
                render_weight=render_weight,
            )

            selected_out = None
            if selected is not None:
                src_mask = selected.get("saved_mask_path", "") or selected.get("mask_path", "")
                if src_mask and os.path.exists(src_mask):
                    selected_out = os.path.join(frame_out_dir, f"mask_{int(cad_part_id):04d}.png")
                    shutil.copyfile(src_mask, selected_out)
                pidx = int(selected.get("_proposal_index", -1))
                if pidx >= 0:
                    used_masks.add(pidx)
                used_cads.add(int(cad_part_id))

                out_item = dict(selected)
                out_item["selected_mask_saved_path"] = selected_out
                if selected_out is not None:
                    out_item["saved_mask_path"] = selected_out
                out_item["adaptive_reason"] = reason
                out_item["adaptive_candidates_eval"] = evals
                selected_items.append(out_item)

        selected_items = sorted(selected_items, key=lambda x: int(x.get("cad_part_id", -1)))
        reranked_results[frame_id] = selected_items
        print(
            f"[ADAPTIVE] {frame_id}: candidates={len(frame_candidates)}, "
            f"cad_models={len(by_cad)}, selected={len(selected_items)}, topk_per_cad={int(topk_per_cad)}, "
            f"weights=sam6d:{float(sam6d_weight):.3f},render:{float(render_weight):.3f}"
        )

    out_json = os.path.join(match_out_dir, reranked_json_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] adaptive rerank result saved: {out_json}")
    return out_json


def main():
    parser = argparse.ArgumentParser(description="Adaptive render rerank for direct_match top-k results.")
    parser.add_argument("--obj-dir", type=str, required=True)
    parser.add_argument("--match-out-dir", type=str, required=True)
    parser.add_argument("--reranked-mask-subdir", type=str, default="matched_pred_mask_direct_match_adaptive")
    parser.add_argument("--match-result-json-name", type=str, default="match_results_sam6d_style.json")
    parser.add_argument("--reranked-json-name", type=str, default="match_results_adaptive_weight.json")
    parser.add_argument("--topk-per-cad", type=int, default=3)
    parser.add_argument("--sam6d-weight", type=float, default=0.5)
    parser.add_argument("--render-weight", type=float, default=0.15)
    parser.add_argument("--render-model-name", type=str, default="dinov2_vitl14")
    args = parser.parse_args()

    run_adaptive_rerank_for_object(
        obj_dir=args.obj_dir,
        match_out_dir=args.match_out_dir,
        reranked_mask_subdir=args.reranked_mask_subdir,
        match_result_json_name=args.match_result_json_name,
        reranked_json_name=args.reranked_json_name,
        topk_per_cad=args.topk_per_cad,
        sam6d_weight=args.sam6d_weight,
        render_weight=args.render_weight,
        render_model_name=args.render_model_name,
    )


if __name__ == "__main__":
    main()
