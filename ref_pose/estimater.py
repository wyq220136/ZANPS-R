import os, sys
import json
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from foundationpose import *
from dataloader import QueryLoader
from pose_consistency import score_pose
from reference_evidence import optimize_pose_with_reference_correspondence


def natural_sort_key(s):
    """实现自然排序，处理文件名中的数字"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def pose_estimation(mesh_dir, query_scene_dir, debug_dir, use_nvdiffrast=True):
    """
        mesh_dir: CAD model path
        query_scene_dir: query image directory
        debug_dir: output directory
    """
    mesh = trimesh.load(mesh_dir, force='mesh')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        glctx=None,
        load_render_models=use_nvdiffrast,
    )
    
    reader = QueryLoader(files_dir=query_scene_dir)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
    print(len(reader.color_files))
    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)    
        # 第一帧估计初始化位姿
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            with torch.inference_mode():
                pose = est.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=7,
                    use_nvdiffrast=use_nvdiffrast,
                )
        else:
            with torch.inference_mode():
                pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    iteration=5,
                    use_nvdiffrast=use_nvdiffrast,
                )
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, K=reader.K, thickness=3, transparency=0,
                            is_input_rgb=True, bbox=bbox, auto_scale=True)
        np.savetxt(f'{debug_dir}/ob_in_cam1/{reader.id_strs[i]}.txt', pose.reshape(4,4))
        imageio.imwrite(f'{debug_dir}/track_vis1/{reader.id_strs[i]}.png', vis)
        



def pose_single_estimation(
    loader: SingleLoader,
    parts_num: int,
    debug_dir: str,
    use_nvdiffrast: bool = True,
    max_parts_per_frame: int | None = None,
    force_recompute: bool = False,
    init_pose_overrides: dict[int, np.ndarray] | None = None,
    init_pose_candidate_overrides: dict[int, list] | None = None,
    pose_rerank: bool = True,
    max_pose_candidates: int = 4,
    use_fp_local_candidates: bool = True,
    use_reference_correspondence: bool = True,
    output_tag: str = "",
):
    suffix = f"_{output_tag}" if output_tag else ""
    track_vis_dir = os.path.join(debug_dir, f"track_vis2{suffix}")
    res_dir = os.path.join(debug_dir, f"ob_in_cam2{suffix}")
    
    os.makedirs(track_vis_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    color, save_name = loader.get_rgb()
    depth = loader.get_depth()
    vis = color.copy()
    
    est = None
    
    if force_recompute or (not os.path.exists(os.path.join(track_vis_dir, save_name))):
        part_count = min(parts_num, len(loader.models))
        if max_parts_per_frame is not None and max_parts_per_frame > 0:
            part_count = min(part_count, max_parts_per_frame)
        for i in range(part_count):
            mesh = loader.get_mesh(i)
            part_id = loader.get_part_id(i)
            if mesh is None or len(mesh.vertices) == 0:
                print(f"警告: 发现空 Mesh，跳过当前估计。")
                continue
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            if est is None:
                est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, 
                    scorer=loader.scorer, refiner=loader.refiner, debug_dir=debug_dir, glctx=loader.glctx,
                    load_render_models=use_nvdiffrast)
            else:
                est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)
            
            mask = loader.get_mask(i).astype(bool)
            init_pose = loader.get_init_pose(i)
            if init_pose_overrides is not None and part_id in init_pose_overrides:
                init_pose = init_pose_overrides[part_id]
            candidates = []
            if init_pose_candidate_overrides is not None and part_id in init_pose_candidate_overrides:
                raw_candidates = init_pose_candidate_overrides[part_id]
                if not isinstance(raw_candidates, (list, tuple)):
                    raw_candidates = [raw_candidates]
                for cand in raw_candidates:
                    cand_pose = cand.get("pose", cand) if isinstance(cand, dict) else cand
                    try:
                        cand_pose = np.asarray(cand_pose, dtype=np.float32).reshape(4, 4)
                    except Exception:
                        continue
                    candidates.append(cand_pose)
            if not candidates:
                base_init_pose = np.asarray(init_pose, dtype=np.float32).reshape(4, 4)
                if use_fp_local_candidates:
                    candidates = list(est.sample_local_pose_candidates(
                        K=loader.K,
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        init_pose=base_init_pose,
                        max_candidates=max_pose_candidates,
                    ))
                else:
                    candidates = [base_init_pose]
            if max_pose_candidates is not None and max_pose_candidates > 0:
                candidates = candidates[:max_pose_candidates]
            
            candidate_results = []
            try:
                for cand_idx, cand_pose in enumerate(candidates):
                    corr_info = {"available": False, "updated": False, "reason": "disabled"}
                    pose_for_refine = cand_pose
                    if use_reference_correspondence:
                        pose_for_refine, corr_info = optimize_pose_with_reference_correspondence(
                            evidence=loader.get_reference_evidence(i),
                            pose_obj_to_cam=cand_pose,
                            depth=depth,
                            mask=mask,
                            K=loader.K,
                        )
                    with torch.inference_mode():
                        pose_cur = est.register(
                            K=loader.K,
                            rgb=color,
                            depth=depth,
                            ob_mask=mask,
                            init_pose=pose_for_refine,
                            iteration=7,
                            use_nvdiffrast=use_nvdiffrast,
                        )
                    validity_mean = None
                    refine_trace = getattr(loader.refiner, "last_refine_trace", None) if loader.refiner is not None else None
                    if isinstance(refine_trace, dict):
                        validity_mean = refine_trace.get("final_validity_mean", None)
                    score = score_pose(
                        mesh=mesh,
                        pose=pose_cur,
                        depth=depth,
                        mask=mask,
                        K=loader.K,
                        init_pose=cand_pose,
                        validity_mean=validity_mean,
                        reference_evidence=loader.get_reference_evidence(i),
                    )
                    candidate_results.append({
                        "candidate_index": int(cand_idx),
                        "pose": pose_cur,
                        "init_pose": cand_pose,
                        "correspondence_pose": pose_for_refine,
                        "correspondence": corr_info,
                        "score": score,
                        "refine_trace": refine_trace,
                    })
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM on part {part_id}, skipped.")
                    torch.cuda.empty_cache()
                    continue
                raise
            if not candidate_results:
                continue
            if pose_rerank:
                candidate_results.sort(key=lambda item: float(item["score"].get("score", -1e9)), reverse=True)
            pose = candidate_results[0]["pose"]
            
            center_pose = pose@np.linalg.inv(to_origin)
        
            vis = draw_posed_3d_box(loader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, K=loader.K, thickness=3, transparency=0,
                                is_input_rgb=True, bbox=bbox, auto_scale=True)
            
            pose_res_dir = os.path.join(res_dir, os.path.splitext(save_name)[0])
            os.makedirs(pose_res_dir, exist_ok=True)
            np.savetxt(os.path.join(pose_res_dir, f'pose_{part_id:04d}.txt'), pose.reshape(4,4))
            score_payload = []
            for item in candidate_results:
                score_payload.append({
                    "candidate_index": item["candidate_index"],
                    "selected": bool(item is candidate_results[0]),
                    "score": item["score"],
                    "refine_trace": item["refine_trace"],
                    "init_pose": np.asarray(item["init_pose"], dtype=float).reshape(4, 4).tolist(),
                    "correspondence_pose": np.asarray(item["correspondence_pose"], dtype=float).reshape(4, 4).tolist(),
                    "correspondence": item["correspondence"],
                    "pose": np.asarray(item["pose"], dtype=float).reshape(4, 4).tolist(),
                })
            with open(os.path.join(pose_res_dir, f'pose_{part_id:04d}_score.json'), "w", encoding="utf-8") as f:
                json.dump(score_payload, f, ensure_ascii=False, indent=2)
            torch.cuda.empty_cache()
            
        imageio.imwrite(os.path.join(track_vis_dir, save_name), vis)
        
    
            

        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="pose estimation")
    parser.add_argument("--mesh_path", type=int, default=0, help="Start index of object list")
    parser.add_argument("--query_dir", type=int, default=None, help="End index of object list")
    parser.add_argument("--debug_dir", type=str, default="/data1/yuquan/CAPNet/data/test_inter/objs", help="Root directory")
    args = parser.parse_args()
    pose_estimation(args.mesh_path, args.query_dir, args.debug_dir, use_nvdiffrast=True)
