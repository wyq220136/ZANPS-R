from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from recon_utils import (
    DatasetObject,
    backproject,
    ensure_dir,
    find_image,
    frames_for_part,
    list_parts,
    load_depth_m,
    load_k,
    load_mask,
    mask_path_for_part_frame,
    method_models_dir,
    method_object_dir,
    method_pose_ready_dir,
    model_obj_path,
    part_model_name,
    write_json,
)


@dataclass
class Observation:
    frame: str
    points_cam: np.ndarray
    mask: np.ndarray
    depth_m: np.ndarray
    rgb_hw: Tuple[int, int]


@dataclass
class Keyframe:
    observation: Observation
    cam_in_ob: np.ndarray
    ob_in_cam: np.ndarray
    pose_info: Dict[str, object]


def _require_base(args: argparse.Namespace, obj: DatasetObject, base_method: str) -> None:
    parts = list_parts(obj)
    base_root = method_pose_ready_dir(Path(args.work_root).resolve(), base_method, args.split, obj.name)
    missing = [
        part_model_name(p, i)
        for i, p in enumerate(parts)
        if not model_obj_path(base_root, part_model_name(p, i)).exists()
    ]
    if not missing:
        return
    if not getattr(args, "build_base_if_missing", False):
        raise FileNotFoundError(
            f"base method '{base_method}' missing models for {obj.name}: {missing[:5]}. "
            "Run the base reconstruction first or pass --build-base-if-missing."
        )
    if base_method == "sam3d":
        from run.recon_sam3d import reconstruct_object
    elif base_method == "hunyuan3d":
        from run.recon_hunyuan3d import reconstruct_object
    elif base_method == "sam3d_tsdf":
        from run.recon_sam3d_tsdf import reconstruct_object
    elif base_method == "hunyuan3d_tsdf":
        from run.recon_hunyuan3d_tsdf import reconstruct_object
    else:
        raise ValueError(f"unknown base method: {base_method}")
    reconstruct_object(obj, args)


def _require_pytorch3d():
    global torch
    try:
        import torch as torch_mod
        from pytorch3d.ops import iterative_closest_point, knn_points, sample_points_from_meshes
        from pytorch3d.structures import Meshes
        from pytorch3d.transforms import axis_angle_to_matrix
    except Exception as exc:
        raise RuntimeError(
            "PyTorch3D is required for the DLMesh replacement backend. "
            "This path intentionally has no slower scipy/numpy fallback."
        ) from exc
    torch = torch_mod
    return Meshes, sample_points_from_meshes, iterative_closest_point, knn_points, axis_angle_to_matrix


def _trimesh():
    import trimesh

    return trimesh


def _as_trimesh(mesh_obj):
    tm = _trimesh()
    if isinstance(mesh_obj, tm.Scene):
        geoms = [g for g in mesh_obj.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            raise ValueError("mesh scene is empty")
        mesh_obj = tm.util.concatenate(geoms)
    if not isinstance(mesh_obj, tm.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh_obj)!r}")
    if len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("mesh has no vertices/faces")
    return tm.Trimesh(
        vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
        faces=np.asarray(mesh_obj.faces, dtype=np.int64),
        process=False,
    )


def _copy_converted_tree(src_root: Path, dst_root: Path, overwrite: bool) -> int:
    count = 0
    if not src_root.exists():
        return count
    for src_obj in src_root.rglob("model.obj"):
        rel_dir = src_obj.parent.relative_to(src_root)
        dst_dir = dst_root / rel_dir
        dst_obj = dst_dir / "model.obj"
        if dst_obj.exists() and not overwrite:
            count += 1
            continue
        if dst_dir.exists() and overwrite:
            shutil.rmtree(dst_dir)
        shutil.copytree(src_obj.parent, dst_dir, dirs_exist_ok=True)
        count += 1
    return count


def _largest_components(mesh, min_faces: int):
    tm = _trimesh()
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    kept = [p for p in parts if len(p.faces) >= int(min_faces)]
    if not kept:
        kept = [max(parts, key=lambda p: len(p.faces))]
    return _as_trimesh(tm.util.concatenate(kept))


def _simplify_mesh(mesh, target_faces: int):
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh, "not_needed"
    try:
        out = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        return _as_trimesh(out), "trimesh_quadric"
    except TypeError:
        out = mesh.simplify_quadric_decimation(int(target_faces))
        return _as_trimesh(out), "trimesh_quadric"


def _preprocess_mesh(base_obj: Path, out_path: Path, args: argparse.Namespace):
    tm = _trimesh()
    mesh = _as_trimesh(tm.load(str(base_obj), force="mesh", process=False))
    before = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    if bool(args.dlmesh_remove_small_components):
        mesh = _largest_components(mesh, int(args.dlmesh_min_component_faces))
    simplify_status = "disabled"
    if int(args.dlmesh_target_faces) > 0:
        mesh, simplify_status = _simplify_mesh(mesh, int(args.dlmesh_target_faces))
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    after = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    return mesh, {"before": before, "after": after, "simplify": simplify_status, "output": str(out_path)}


def _load_observation(
    obj: DatasetObject,
    part_name: str,
    frame: str,
    k: np.ndarray,
    args: argparse.Namespace,
) -> Optional[Observation]:
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    depth_path = find_image(obj.depth_dir, frame)
    rgb_path = find_image(obj.rgb_dir, frame)
    if mask_path is None or depth_path is None or rgb_path is None:
        return None
    depth_m = load_depth_m(depth_path, args.depth_scale)
    mask = load_mask(mask_path, depth_m.shape[:2])
    if int(np.count_nonzero(mask)) < int(args.dlmesh_min_mask_pixels):
        return None
    points_cam = backproject(depth_m, mask, k)
    if len(points_cam) < int(args.dlmesh_min_points):
        return None
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    return Observation(
        frame=frame,
        points_cam=points_cam.astype(np.float32),
        mask=mask.astype(bool),
        depth_m=depth_m.astype(np.float32),
        rgb_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
    )


def _sample_np(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), int(max_points), replace=False)]


def _mesh_to_pytorch3d(mesh, device: torch.device):
    Meshes, _, _, _, _ = _require_pytorch3d()
    verts = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int64), dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])


def _sample_mesh_points(mesh, n_points: int, device: torch.device) -> torch.Tensor:
    _, sample_points_from_meshes, _, _, _ = _require_pytorch3d()
    p3d_mesh = _mesh_to_pytorch3d(mesh, device)
    pts = sample_points_from_meshes(p3d_mesh, num_samples=int(n_points))[0]
    return pts.contiguous()


def _as_homo(tf: torch.Tensor) -> torch.Tensor:
    out = torch.eye(4, dtype=tf.dtype, device=tf.device)
    out[:3, :4] = tf[:3, :4]
    return out


def _transform_points_torch(points: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    return points @ tf[:3, :3].T + tf[:3, 3]


def _icp_cam_points_to_mesh(
    points_cam: np.ndarray,
    mesh,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    _, _, iterative_closest_point, knn_points, _ = _require_pytorch3d()
    obs = torch.as_tensor(
        _sample_np(points_cam, int(args.dlmesh_icp_points), seed),
        dtype=torch.float32,
        device=device,
    )[None]
    dst = _sample_mesh_points(mesh, int(args.dlmesh_mesh_samples), device)[None]
    sol = iterative_closest_point(
        obs,
        dst,
        max_iterations=int(args.dlmesh_icp_iters),
        estimate_scale=False,
        allow_reflection=False,
    )
    src_aligned = sol.RTs.s * torch.bmm(obs, sol.RTs.R) + sol.RTs.T[:, None, :]
    # PyTorch3D ICP applies x @ R + T. Store as camera -> object column-vector transform.
    cam_in_ob = torch.eye(4, dtype=torch.float32, device=device)
    cam_in_ob[:3, :3] = sol.RTs.R[0].T
    cam_in_ob[:3, 3] = sol.RTs.T[0]
    d2 = knn_points(src_aligned, dst, K=1).dists[0, :, 0]
    return cam_in_ob.detach().cpu().numpy().astype(np.float32), {
        "icp_converged": bool(getattr(sol, "converged", False)),
        "icp_rmse": float(torch.sqrt(torch.mean(d2)).detach().cpu()),
        "icp_mean_dist": float(torch.mean(torch.sqrt(torch.clamp(d2, min=1e-12))).detach().cpu()),
        "icp_src": "local_observation_points_camera",
        "icp_dst": "mesh_points_object",
        "icp_output": "cam_in_ob",
    }


def _optimize_cam_in_ob_pose(
    init_cam_in_ob: np.ndarray,
    points_cam: np.ndarray,
    mesh,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    _, _, _, knn_points, axis_angle_to_matrix = _require_pytorch3d()
    src = torch.as_tensor(
        _sample_np(points_cam, int(args.dlmesh_pose_points), seed),
        dtype=torch.float32,
        device=device,
    )[None]
    dst = _sample_mesh_points(mesh, int(args.dlmesh_mesh_samples), device)[None].detach()
    init = torch.as_tensor(init_cam_in_ob, dtype=torch.float32, device=device)
    rot_delta = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    trans_delta = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([rot_delta, trans_delta], lr=float(args.dlmesh_pose_lr))
    last_loss = 0.0
    for _ in range(max(1, int(args.dlmesh_pose_steps))):
        opt.zero_grad(set_to_none=True)
        r_delta = axis_angle_to_matrix(rot_delta[None])[0]
        cur = torch.eye(4, dtype=torch.float32, device=device)
        cur[:3, :3] = r_delta @ init[:3, :3]
        cur[:3, 3] = r_delta @ init[:3, 3] + trans_delta
        moved = _transform_points_torch(src[0], cur)[None]
        d2 = knn_points(moved, dst, K=1).dists[0, :, 0]
        q = torch.quantile(d2.detach(), float(args.dlmesh_pose_trim_quantile))
        keep = d2 <= q
        if int(keep.sum().item()) < 16:
            keep = torch.ones_like(d2, dtype=torch.bool)
        chamfer = torch.mean(d2[keep])
        prior = torch.sum(rot_delta * rot_delta) * float(args.dlmesh_pose_rot_prior)
        prior = prior + torch.sum(trans_delta * trans_delta) * float(args.dlmesh_pose_trans_prior)
        loss = chamfer + prior
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    with torch.no_grad():
        r_delta = axis_angle_to_matrix(rot_delta[None])[0]
        cur = torch.eye(4, dtype=torch.float32, device=device)
        cur[:3, :3] = r_delta @ init[:3, :3]
        cur[:3, 3] = r_delta @ init[:3, 3] + trans_delta
        moved = _transform_points_torch(src[0], cur)[None]
        d2 = knn_points(moved, dst, K=1).dists[0, :, 0]
        mean_dist = torch.mean(torch.sqrt(torch.clamp(d2, min=1e-12)))
    return cur.detach().cpu().numpy().astype(np.float32), {
        "pose_optimizer": "pytorch3d_knn_chamfer",
        "pose_steps": int(args.dlmesh_pose_steps),
        "pose_final_loss": last_loss,
        "pose_mean_dist": float(mean_dist.detach().cpu()),
        "pose_input": "cam_in_ob_from_icp",
        "pose_output": "cam_in_ob",
    }


def _pose_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3] @ b[:3, :3].T
    cos = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _pose_translation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _pose_pool_accept(ob_in_cam: np.ndarray, accepted: Sequence[Keyframe], args: argparse.Namespace) -> Tuple[bool, Dict[str, float]]:
    if not accepted:
        return True, {"nearest_angle_deg": 0.0, "nearest_translation": 0.0}
    accepted_poses = [k.ob_in_cam for k in accepted]
    angles = np.asarray([_pose_angle_deg(ob_in_cam, p) for p in accepted_poses], dtype=np.float32)
    translations = np.asarray([_pose_translation(ob_in_cam, p) for p in accepted_poses], dtype=np.float32)
    nearest_angle = float(np.min(angles))
    nearest_translation = float(np.min(translations))
    too_close = nearest_angle < float(args.dlmesh_min_angle) and nearest_translation < float(args.dlmesh_min_translation)
    too_far = nearest_angle > float(args.dlmesh_max_angle) or nearest_translation > float(args.dlmesh_max_translation)
    return (not too_close and not too_far), {
        "nearest_angle_deg": nearest_angle,
        "nearest_translation": nearest_translation,
        "too_close": float(too_close),
        "too_far": float(too_far),
    }


def _collect_keyframes(
    obj: DatasetObject,
    part_name: str,
    mesh,
    args: argparse.Namespace,
    device: torch.device,
    round_idx: int,
) -> Tuple[List[Keyframe], List[Dict[str, object]]]:
    k = load_k(obj)
    accepted: List[Keyframe] = []
    reports: List[Dict[str, object]] = []
    frames = frames_for_part(obj, part_name, args.max_frames, args.frame_stride)
    for frame_idx, frame in enumerate(frames):
        if len(accepted) >= int(args.dlmesh_max_keyframes):
            break
        obs = _load_observation(obj, part_name, frame, k, args)
        if obs is None:
            reports.append({"frame": frame, "status": "skipped", "reason": "invalid_observation"})
            continue
        try:
            icp_pose, icp_info = _icp_cam_points_to_mesh(
                obs.points_cam, mesh, args, device, seed=round_idx * 100000 + frame_idx
            )
            cam_in_ob, pose_info = _optimize_cam_in_ob_pose(
                icp_pose, obs.points_cam, mesh, args, device, seed=round_idx * 100000 + frame_idx + 17
            )
        except Exception as exc:
            reports.append({"frame": frame, "status": "failed", "reason": "pose_init_failed", "error": str(exc)})
            continue
        ob_in_cam = np.linalg.inv(cam_in_ob).astype(np.float32)
        if float(pose_info["pose_mean_dist"]) > float(args.dlmesh_pose_max_mean_dist):
            reports.append(
                {
                    "frame": frame,
                    "status": "skipped",
                    "reason": "pose_mean_dist_too_large",
                    "pose_info": pose_info,
                    "icp_info": icp_info,
                }
            )
            continue
        ok, pool_info = _pose_pool_accept(ob_in_cam, accepted, args)
        if not ok:
            reports.append(
                {
                    "frame": frame,
                    "status": "skipped",
                    "reason": "pose_pool_filter",
                    "pool_info": pool_info,
                    "pose_info": pose_info,
                    "icp_info": icp_info,
                }
            )
            continue
        merged_info = {**icp_info, **pose_info, **pool_info}
        accepted.append(Keyframe(obs, cam_in_ob, ob_in_cam, merged_info))
        reports.append({"frame": frame, "status": "accepted", "pose_info": merged_info})
    return accepted, reports


def _vertex_laplacian_loss(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ],
        dim=0,
    )
    src, dst = edges[:, 0], edges[:, 1]
    accum = torch.zeros_like(verts)
    deg = torch.zeros((verts.shape[0], 1), dtype=verts.dtype, device=verts.device)
    accum.index_add_(0, src, verts[dst])
    deg.index_add_(0, src, torch.ones((len(src), 1), dtype=verts.dtype, device=verts.device))
    mean = accum / torch.clamp(deg, min=1.0)
    return torch.mean((verts - mean) ** 2)


def _project_points(points_cam: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = torch.clamp(points_cam[:, 2], min=1e-6)
    u = k[0, 0] * points_cam[:, 0] / z + k[0, 2]
    v = k[1, 1] * points_cam[:, 1] / z + k[1, 2]
    return torch.stack([u, v], dim=-1), z


def _sample_observed_for_stage(keyframes: Sequence[Keyframe], max_points: int, device: torch.device, seed: int):
    out = []
    for i, kf in enumerate(keyframes):
        pts = _sample_np(kf.observation.points_cam, max_points, seed + i)
        cam_in_ob = np.asarray(kf.cam_in_ob, dtype=np.float32)
        pts_obj = (cam_in_ob[:3, :3] @ pts.T).T + cam_in_ob[:3, 3]
        out.append(torch.as_tensor(pts_obj, dtype=torch.float32, device=device))
    return out


def _render_depth_mask_loss(
    verts: torch.Tensor,
    faces: torch.Tensor,
    keyframes: Sequence[Keyframe],
    k_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    # Lightweight differentiable projection supervision: project vertices with ob_in_cam
    # and compare sampled visible projected vertices against the observed mask/depth.
    k = torch.as_tensor(k_np, dtype=torch.float32, device=device)
    loss = torch.zeros((), dtype=torch.float32, device=device)
    stride = max(1, int(math.ceil(verts.shape[0] / max(1, int(args.dlmesh_project_vertices)))))
    verts_sub = verts[::stride]
    for kf in keyframes:
        ob_in_cam = torch.as_tensor(kf.ob_in_cam, dtype=torch.float32, device=device)
        pts_cam = _transform_points_torch(verts_sub, ob_in_cam)
        uv, z = _project_points(pts_cam, k)
        h, w = kf.observation.depth_m.shape[:2]
        inside = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if int(inside.sum().item()) < 8:
            continue
        uv_i = torch.round(uv[inside]).long()
        depth = torch.as_tensor(kf.observation.depth_m, dtype=torch.float32, device=device)
        mask = torch.as_tensor(kf.observation.mask, dtype=torch.bool, device=device)
        obs_depth = depth[uv_i[:, 1].clamp(0, h - 1), uv_i[:, 0].clamp(0, w - 1)]
        obs_mask = mask[uv_i[:, 1].clamp(0, h - 1), uv_i[:, 0].clamp(0, w - 1)]
        z_sel = z[inside]
        valid = obs_mask & (obs_depth > 1e-6)
        if int(valid.sum().item()) < 8:
            loss = loss + torch.mean(torch.relu(float(args.dlmesh_mask_margin) - z_sel) * (~obs_mask).float())
            continue
        depth_res = torch.abs(z_sel[valid] - obs_depth[valid])
        loss = loss + torch.mean(torch.clamp(depth_res, max=float(args.dlmesh_depth_trunc)))
        loss = loss + torch.mean((~obs_mask).float()) * float(args.dlmesh_silhouette_weight)
    return loss / max(1, len(keyframes))


def _optimize_mesh_vertices(
    mesh,
    keyframes: Sequence[Keyframe],
    k_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    round_idx: int,
    stage_idx: int,
) -> Tuple[object, Dict[str, object]]:
    _, _, _, knn_points, _ = _require_pytorch3d()
    tm = _trimesh()
    verts0 = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int64), dtype=torch.int64, device=device)
    verts = verts0.clone().detach().requires_grad_(True)
    observed_obj_sets = _sample_observed_for_stage(
        keyframes,
        int(args.dlmesh_stage_points_per_frame),
        device,
        seed=round_idx * 100000 + stage_idx * 1000,
    )
    opt = torch.optim.Adam([verts], lr=float(args.dlmesh_lr))
    last = {}
    for step in range(max(1, int(args.dlmesh_steps_per_stage))):
        opt.zero_grad(set_to_none=True)
        point_loss = torch.zeros((), dtype=torch.float32, device=device)
        for obs_obj in observed_obj_sets:
            if obs_obj.numel() == 0:
                continue
            d2 = knn_points(obs_obj[None], verts[None], K=1).dists[0, :, 0]
            q = torch.quantile(d2.detach(), float(args.dlmesh_point_trim_quantile))
            keep = d2 <= q
            if int(keep.sum().item()) < 16:
                keep = torch.ones_like(d2, dtype=torch.bool)
            point_loss = point_loss + torch.mean(d2[keep])
        point_loss = point_loss / max(1, len(observed_obj_sets))
        prior_loss = torch.mean((verts - verts0) ** 2)
        lap_loss = _vertex_laplacian_loss(verts, faces)
        render_loss = _render_depth_mask_loss(verts, faces, keyframes, k_np, args, device)
        total = (
            point_loss * float(args.dlmesh_point_weight)
            + prior_loss * float(args.dlmesh_prior_weight)
            + lap_loss * float(args.dlmesh_laplace_weight)
            + render_loss * float(args.dlmesh_depth_weight)
        )
        total.backward()
        opt.step()
        if int(args.dlmesh_max_vertex_step) > 0:
            with torch.no_grad():
                delta = verts - verts0
                norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)
                scale = torch.clamp(float(args.dlmesh_max_vertex_step) / norm, max=1.0)
                verts.copy_(verts0 + delta * scale)
        if step == int(args.dlmesh_steps_per_stage) - 1:
            last = {
                "total_loss": float(total.detach().cpu()),
                "point_loss": float(point_loss.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
                "laplace_loss": float(lap_loss.detach().cpu()),
                "depth_mask_loss": float(render_loss.detach().cpu()),
            }
    out = tm.Trimesh(
        vertices=verts.detach().cpu().numpy().astype(np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    out.remove_degenerate_faces()
    out.remove_unreferenced_vertices()
    return _as_trimesh(out), {"stage": int(stage_idx), "keyframes": len(keyframes), **last}


def _run_dlmesh_refinement(base_obj: Path, out_dir: Path, obj: DatasetObject, part_name: str, args: argparse.Namespace) -> Dict[str, object]:
    _require_pytorch3d()
    device = torch.device(args.dlmesh_device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DLMesh replacement requires CUDA because PyTorch3D/nvdiffrast refinement is GPU-only.")
    out_dir.mkdir(parents=True, exist_ok=True)
    pre_obj = out_dir / "remesh_preprocessed.obj"
    mesh, pre_info = _preprocess_mesh(base_obj, pre_obj, args)
    k = load_k(obj)
    summary: Dict[str, object] = {
        "backend": "remesh_pytorch3d_pose_dlmesh",
        "replaces": "dmesh",
        "source_model": str(base_obj),
        "remesh": pre_info,
        "rounds": [],
        "pytorch3d_required": True,
        "icp_src": "local_observation_points_camera",
        "icp_dst": "mesh_points_object",
        "icp_output": "cam_in_ob",
        "dlmesh_pose": "ob_in_cam = inverse(cam_in_ob)",
        "optimize_light": False,
    }
    for round_idx in range(max(1, int(args.dlmesh_outer_iters))):
        round_dir = ensure_dir(out_dir / f"round_{round_idx:02d}")
        keyframes, reports = _collect_keyframes(obj, part_name, mesh, args, device, round_idx)
        round_summary: Dict[str, object] = {
            "round": int(round_idx),
            "accepted_keyframes": [k.observation.frame for k in keyframes],
            "reports": reports,
            "stages": [],
        }
        pose_dir = ensure_dir(round_dir / "poses")
        for kf in keyframes:
            np.savetxt(pose_dir / f"{kf.observation.frame}_cam_in_ob.txt", kf.cam_in_ob, fmt="%.8f")
            np.savetxt(pose_dir / f"{kf.observation.frame}_ob_in_cam.txt", kf.ob_in_cam, fmt="%.8f")
        if not keyframes:
            round_summary["status"] = "skipped_no_keyframes"
            summary["rounds"].append(round_summary)
            continue
        stage_size = max(1, int(args.dlmesh_stage_size))
        for stage_end in range(stage_size, len(keyframes) + stage_size, stage_size):
            stage_kfs = keyframes[: min(len(keyframes), stage_end)]
            stage_idx = len(round_summary["stages"])
            mesh, stage_info = _optimize_mesh_vertices(mesh, stage_kfs, k, args, device, round_idx, stage_idx)
            stage_path = round_dir / f"stage_{stage_idx:02d}.obj"
            mesh.export(str(stage_path))
            stage_info["model"] = str(stage_path)
            round_summary["stages"].append(stage_info)
            if stage_end >= len(keyframes):
                break
        round_summary["status"] = "success"
        summary["rounds"].append(round_summary)
    final_obj = out_dir / "model.obj"
    mesh.export(str(final_obj))
    summary["status"] = "success"
    summary["output_model"] = str(final_obj)
    return summary


def run_dmesh_object(obj: DatasetObject, args: argparse.Namespace, base_method: str, method: str) -> Dict[str, object]:
    work_root = Path(args.work_root).resolve()
    _require_base(args, obj, base_method)

    base_pose_root = method_pose_ready_dir(work_root, base_method, args.split, obj.name)
    out_pose_root = ensure_dir(method_pose_ready_dir(work_root, method, args.split, obj.name))
    out_model_root = ensure_dir(method_models_dir(work_root, method, args.split, obj.name))

    parts = list_parts(obj)
    summary = {
        "method": method,
        "base_method": base_method,
        "object": obj.name,
        "backend": "remesh_pytorch3d_pose_dlmesh",
        "placeholder": bool(args.copy_base_as_placeholder),
        "parts": [],
    }
    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        base_obj = model_obj_path(base_pose_root, part_model)
        out_obj = model_obj_path(out_pose_root, part_model)
        if not base_obj.exists():
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "base_model_missing"})
            continue
        if out_obj.exists() and not args.overwrite:
            status = "cached"
            result = {"status": "cached", "output_model": str(out_obj), "backend": "remesh_pytorch3d_pose_dlmesh"}
        elif args.copy_base_as_placeholder:
            out_obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_obj, out_obj)
            status = "placeholder_copied"
            result = {"status": status, "source_model": str(base_obj), "output_model": str(out_obj)}
        else:
            result = _run_dlmesh_refinement(base_obj, out_obj.parent, obj, part_name, args)
            status = str(result.get("status", "converted"))
        summary["parts"].append(
            {
                "part": part_name,
                "part_model": part_model,
                "status": status,
                "frames_available": len(frames_for_part(obj, part_name, args.max_frames, args.frame_stride)),
                "model": str(out_obj),
                "source_model": str(base_obj),
                "dlmesh_result": result,
            }
        )
    _copy_converted_tree(out_pose_root, out_model_root, overwrite=True)
    write_json(method_object_dir(work_root, method, args.split, obj.name) / "summary.json", summary)
    return summary


def add_dmesh_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build-base-if-missing", action="store_true", help="Run the base method if shared cache is missing.")
    parser.add_argument("--dmesh-device", type=str, default="cuda:0", help="Deprecated alias for --dlmesh-device.")
    parser.add_argument("--dmesh-steps", type=int, default=0, help="Deprecated alias for --dlmesh-steps-per-stage if set > 0.")
    parser.add_argument("--dmesh-save-step", type=int, default=0, help="Deprecated compatibility option; unused.")
    parser.add_argument("--dmesh-refresh-points-step", type=int, default=0, help="Deprecated compatibility option; unused.")
    parser.add_argument("--dmesh-gt-max-perturb", type=float, default=0.0, help="Deprecated compatibility option; unused.")
    parser.add_argument("--dmesh-seed", type=int, default=1, help="Deprecated compatibility option; unused.")
    parser.add_argument("--dmesh-root", type=str, default="", help="Deprecated compatibility option; DMesh is no longer called.")
    parser.add_argument(
        "--dmesh-output-variant",
        type=str,
        default="auto",
        choices=["auto", "perfect", "best_recovery_ratio", "best_false_positive_ratio", "last"],
        help="Deprecated compatibility option; unused.",
    )
    parser.add_argument(
        "--copy-base-as-placeholder",
        action="store_true",
        help="Only for IO/eval dry-runs: copy base mesh into output without DLMesh optimization.",
    )
    parser.add_argument("--dlmesh-device", type=str, default="")
    parser.add_argument("--dlmesh-outer-iters", type=int, default=3)
    parser.add_argument("--dlmesh-steps-per-stage", type=int, default=300)
    parser.add_argument("--dlmesh-lr", type=float, default=1e-3)
    parser.add_argument("--dlmesh-target-faces", type=int, default=8000)
    parser.add_argument("--dlmesh-remove-small-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dlmesh-min-component-faces", type=int, default=64)
    parser.add_argument("--dlmesh-max-keyframes", type=int, default=12)
    parser.add_argument("--dlmesh-stage-size", type=int, default=4)
    parser.add_argument("--dlmesh-min-mask-pixels", type=int, default=500)
    parser.add_argument("--dlmesh-min-points", type=int, default=200)
    parser.add_argument("--dlmesh-icp-iters", type=int, default=30)
    parser.add_argument("--dlmesh-icp-points", type=int, default=4096)
    parser.add_argument("--dlmesh-mesh-samples", type=int, default=8192)
    parser.add_argument("--dlmesh-pose-points", type=int, default=4096)
    parser.add_argument("--dlmesh-pose-steps", type=int, default=80)
    parser.add_argument("--dlmesh-pose-lr", type=float, default=1e-2)
    parser.add_argument("--dlmesh-pose-trim-quantile", type=float, default=0.8)
    parser.add_argument("--dlmesh-pose-rot-prior", type=float, default=1e-3)
    parser.add_argument("--dlmesh-pose-trans-prior", type=float, default=1e-2)
    parser.add_argument("--dlmesh-pose-max-mean-dist", type=float, default=0.08)
    parser.add_argument("--dlmesh-min-angle", type=float, default=8.0)
    parser.add_argument("--dlmesh-max-angle", type=float, default=70.0)
    parser.add_argument("--dlmesh-min-translation", type=float, default=0.01)
    parser.add_argument("--dlmesh-max-translation", type=float, default=0.25)
    parser.add_argument("--dlmesh-stage-points-per-frame", type=int, default=2048)
    parser.add_argument("--dlmesh-point-trim-quantile", type=float, default=0.8)
    parser.add_argument("--dlmesh-point-weight", type=float, default=1.0)
    parser.add_argument("--dlmesh-prior-weight", type=float, default=10.0)
    parser.add_argument("--dlmesh-laplace-weight", type=float, default=1.0)
    parser.add_argument("--dlmesh-depth-weight", type=float, default=0.2)
    parser.add_argument("--dlmesh-silhouette-weight", type=float, default=1.0)
    parser.add_argument("--dlmesh-depth-trunc", type=float, default=0.05)
    parser.add_argument("--dlmesh-mask-margin", type=float, default=0.02)
    parser.add_argument("--dlmesh-project-vertices", type=int, default=4096)
    parser.add_argument("--dlmesh-max-vertex-step", type=float, default=0.05)

    old_parse = parser.parse_args

    def parse_args_with_alias(*args, **kwargs):
        ns = old_parse(*args, **kwargs)
        if not ns.dlmesh_device:
            ns.dlmesh_device = ns.dmesh_device
        if int(ns.dmesh_steps) > 0:
            ns.dlmesh_steps_per_stage = int(ns.dmesh_steps)
        return ns

    parser.parse_args = parse_args_with_alias
