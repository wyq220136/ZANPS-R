import os, sys
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from foundationpose import *
from dataloader import QueryLoader


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
            
            try:
                with torch.inference_mode():
                    pose = est.register(
                        K=loader.K,
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        init_pose=init_pose,
                        iteration=7,
                        use_nvdiffrast=use_nvdiffrast,
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM on part {part_id}, skipped.")
                    torch.cuda.empty_cache()
                    continue
                raise
            
            center_pose = pose@np.linalg.inv(to_origin)
        
            vis = draw_posed_3d_box(loader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, K=loader.K, thickness=3, transparency=0,
                                is_input_rgb=True, bbox=bbox, auto_scale=True)
            
            pose_res_dir = os.path.join(res_dir, os.path.splitext(save_name)[0])
            os.makedirs(pose_res_dir, exist_ok=True)
            np.savetxt(os.path.join(pose_res_dir, f'pose_{part_id:04d}.txt'), pose.reshape(4,4))
            torch.cuda.empty_cache()
            
        imageio.imwrite(os.path.join(track_vis_dir, save_name), vis)
        
    
            

        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="pose estimation")
    parser.add_argument("--mesh_path", type=int, default=0, help="Start index of object list")
    parser.add_argument("--query_dir", type=int, default=None, help="End index of object list")
    parser.add_argument("--debug_dir", type=str, default="/data1/yuquan/CAPNet/data/test_inter/objs", help="Root directory")
    args = parser.parse_args()
    pose_estimation(args.mesh_path, args.query_dir, args.debug_dir, use_nvdiffrast=True)
