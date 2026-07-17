"""
迭代式 TSDF 重建的公共实现。

这个模块不负责生成初始单目网格，而是在已有基础网格（SAM3D、Hunyuan3D
或 InstantMesh）的基础上，利用多帧 RGB-D 观测完成以下工作：

1. 读取每一帧的 RGB、深度、部件掩码和初始位姿。
2. 检查基础网格与真实深度点云是否存在明显的尺度不一致。
3. 选择一个可靠的 seed 帧，通过多组全局位姿假设初始化物体坐标系。
4. 逐帧执行网格到深度点云的 ICP 位姿细化。
5. 使用双向深度一致性过滤错误视角。
6. 将通过验收的 RGB-D 帧融合进 Open3D TSDF。
7. 可选地联合优化已接收帧的位姿，并检查新网格是否真正提高整体效用。
8. 保存最终网格、逐帧位姿和完整诊断日志，便于定位失败原因。

坐标系约定（维护本文件时务必保持一致）：

- ``points_cam``：相机坐标系中的深度点，单位为米。
- 网格顶点和 TSDF world：物体坐标系。
- ``ob_in_cam`` / ``pose``：4x4 的 object-to-camera 变换。
- Open3D TSDF 的 extrinsic 参数使用 world-to-camera；这里 world 就是物体坐标系，
  因而可以直接传入 object-to-camera 位姿。

主入口为 ``run_tsdf_object``，单个部件的核心状态机位于
``_iterative_tsdf_part``。
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    load_pose,
    mask_path_for_part_frame,
    method_object_dir,
    method_pose_ready_dir,
    model_obj_path,
    part_model_name,
    pose_path_for_part_frame,
    write_json,
)
from pose_delta_optimization import PoseDeltaOptimizationConfig, optimize_pose_deltas
from pose_utility import PoseUtilityConfig, evaluate_pose_utility, pose_delta_egocentric, utility_improved


def _trimesh():
    """延迟导入 trimesh，避免仅解析命令行参数时就加载较重的几何依赖。"""
    import trimesh

    return trimesh


def _as_trimesh(mesh_obj):
    """
    将 trimesh.Scene 或 Trimesh 统一转换成干净的 Trimesh。

    Scene 中可能包含多个子网格，这里会把所有非空子网格拼接起来。返回的新网格
    禁用 ``process``，避免 trimesh 自动合并顶点或修改拓扑，确保后续配准使用的
    几何与输入模型一致。
    """
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


def _o3d_mesh_to_trimesh(mesh):
    """把 Open3D TSDF 输出网格转换成后续配准使用的 trimesh 网格。"""
    tm = _trimesh()
    return _as_trimesh(
        tm.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.triangles, dtype=np.int64),
            process=False,
        )
    )


def _pose_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个 object-to-camera 位姿之间的旋转夹角，单位为度。"""
    rel = a[:3, :3] @ b[:3, :3].T
    cos = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _pose_translation(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个位姿平移向量之间的欧氏距离，单位与深度一致（通常为米）。"""
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _score_pose_distance(
    pose: np.ndarray,
    accepted_poses: List[np.ndarray],
    min_angle: float,
    max_angle: float,
    min_translation: float,
    max_translation: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    判断候选视角与已接收视角之间的距离是否适合参与下一轮融合。

    候选视角过近时不会带来足够的新表面信息；过远时初始位姿和 ICP 又更容易失败。
    因此旋转和平移都设置了最小/最大门限。返回值中的统计量会写入迭代日志。
    """
    if not accepted_poses:
        return True, {"nearest_angle_deg": 0.0, "nearest_translation": 0.0}
    angles = np.asarray([_pose_angle_deg(pose, p) for p in accepted_poses], dtype=np.float32)
    translations = np.asarray([_pose_translation(pose, p) for p in accepted_poses], dtype=np.float32)
    nearest_idx = int(np.argmin(angles + translations))
    nearest_angle = float(angles[nearest_idx])
    nearest_translation = float(translations[nearest_idx])
    too_dense = nearest_angle < float(min_angle) and nearest_translation < float(min_translation)
    too_far = float(np.min(angles)) > float(max_angle) or float(np.min(translations)) > float(max_translation)
    return (not too_dense and not too_far), {
        "nearest_angle_deg": nearest_angle,
        "nearest_translation": nearest_translation,
        "min_angle_deg": float(np.min(angles)),
        "min_translation": float(np.min(translations)),
    }


def _load_frame_observation(
    obj: DatasetObject,
    part_name: str,
    frame: str,
    k: np.ndarray,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    """
    加载某个部件在单帧中的完整观测。

    返回字典包含 RGB、米制深度、布尔掩码、初始 object-to-camera 位姿，以及由
    掩码深度反投影得到的相机坐标点云。任何必要文件缺失、掩码过小或有效点过少
    都返回 ``None``，让上层统一跳过该帧。
    """
    mask_path = mask_path_for_part_frame(obj, part_name, frame)
    depth_path = find_image(obj.depth_dir, frame)
    rgb_path = find_image(obj.rgb_dir, frame)
    pose_path = pose_path_for_part_frame(obj, part_name, frame)
    if mask_path is None or depth_path is None or rgb_path is None or pose_path is None:
        return None
    depth_m = load_depth_m(depth_path, args.depth_scale)
    mask = load_mask(mask_path, depth_m.shape[:2])
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels < int(args.min_mask_pixels):
        return None
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    pose = load_pose(pose_path, args.pose_convention)
    points_cam = backproject(depth_m, mask, k)
    if len(points_cam) < int(args.iter_tsdf_min_points):
        return None
    return {
        "frame": frame,
        "rgb": rgb,
        "depth_m": depth_m,
        "mask": mask,
        "mask_pixels": mask_pixels,
        "raw_pose": pose.astype(np.float32),
        "points_cam": points_cam.astype(np.float32),
    }


def _make_rgbd(obs: Dict[str, object], args: argparse.Namespace, o3d):
    """
    构造供 Open3D TSDF 融合使用的 RGBDImage。

    掩码外深度清零，防止相邻部件或背景被融合进当前部件。内部深度以米保存，
    Open3D 输入则转换成毫米 uint16，并通过 ``depth_scale=1000`` 还原为米。
    """
    depth_m = np.asarray(obs["depth_m"], dtype=np.float32)
    mask = np.asarray(obs["mask"], dtype=bool)
    rgb = np.asarray(obs["rgb"])
    depth_masked = depth_m.copy()
    depth_masked[~mask] = 0.0
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(np.clip(depth_masked * 1000.0, 0, 65535).astype(np.uint16)),
        depth_scale=1000.0,
        depth_trunc=float(args.depth_trunc),
        convert_rgb_to_intensity=False,
    )


def _camera_intrinsic(obs: Dict[str, object], k: np.ndarray, o3d):
    """根据图像尺寸和 3x3 内参矩阵创建 Open3D 针孔相机模型。"""
    depth_m = np.asarray(obs["depth_m"], dtype=np.float32)
    h, w = depth_m.shape[:2]
    return o3d.camera.PinholeCameraIntrinsic(
        int(w), int(h), float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    )


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """确定性随机下采样点云，控制 ICP 和一致性检查的计算量。"""
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(max_points), replace=False)]


def _points_extent_stats(points: np.ndarray) -> Dict[str, float]:
    """为 seed/view 排序统计可见点云的几何约束强度。"""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 8:
        return {
            "diag": 0.0,
            "volume": 0.0,
            "nonplanarity": 0.0,
            "linearity": 0.0,
            "valid": 0.0,
        }
    centered = points - np.mean(points, axis=0, keepdims=True)
    extent = np.ptp(points, axis=0)
    diag = float(np.linalg.norm(extent))
    volume = float(np.prod(np.maximum(extent, 1e-6)))
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    eig = np.sort(np.linalg.eigvalsh(cov).astype(np.float64))[::-1]
    denom = max(float(eig[0]), 1e-9)
    return {
        "diag": diag,
        "volume": volume,
        "nonplanarity": float(eig[-1] / denom),
        "linearity": float(1.0 - eig[1] / denom),
        "valid": 1.0,
    }


def _observation_seed_score(obs: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
    """
    用可见 3D extent / 非平面性 / mask 支撑度给 seed 排序。

    这样可以避免选到“mask 很大但几何退化”的初始化帧。
    """
    points = np.asarray(obs.get("points_cam", np.zeros((0, 3))), dtype=np.float32)
    stats = _points_extent_stats(points)
    mask_pixels = float(obs.get("mask_pixels", 0))
    mask_term = np.log1p(max(mask_pixels, 0.0))
    diag_term = np.log1p(max(stats["diag"], 0.0) * 100.0)
    volume_term = np.log1p(max(stats["volume"], 0.0) * 1e6)
    score = float(
        mask_term
        + 1.5 * diag_term
        + 0.5 * volume_term
        + 8.0 * float(stats["nonplanarity"])
        - max(0.0, float(stats["linearity"]) - 0.8)
    )
    return score, {**stats, "mask_pixels": mask_pixels, "score": score}


def _points_cam_to_obj(points_cam: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    """使用 object-to-camera 位姿的逆变换，将相机点云转换到物体坐标系。"""
    cam_to_ob = np.linalg.inv(ob_in_cam).astype(np.float32)
    return (cam_to_ob[:3, :3] @ points_cam.T).T + cam_to_ob[:3, 3]


def _points_obj_to_cam(points_obj: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    """使用 object-to-camera 位姿，将物体坐标点变换到相机坐标系。"""
    return (ob_in_cam[:3, :3] @ points_obj.T).T + ob_in_cam[:3, 3]


def _estimate_pose_from_obj_cam_points(points_obj: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """
    用成对的物体点/相机点估计刚体 object-to-camera 变换。

    实现为无尺度的 SVD/Kabsch 对齐；若 SVD 产生反射矩阵，会翻转最后一个奇异向量
    保证旋转矩阵行列式为正。
    """
    n = min(int(len(points_obj)), int(len(points_cam)))
    if n < 3:
        raise ValueError(f"not enough points for pose estimation: {n}")
    src = points_obj[:n].astype(np.float64)
    dst = points_cam[:n].astype(np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xs.T @ xd) / float(n)
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = r.astype(np.float32)
    tf[:3, 3] = t.astype(np.float32)
    return tf


def _robust_geometry_stats(points: np.ndarray) -> Dict[str, object]:
    """计算对少量离群点不敏感的中心、包围盒尺度和 PCA 主轴。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError(f"not enough points for geometry statistics: {len(points)}")
    lo = np.percentile(points, 2.0, axis=0)
    hi = np.percentile(points, 98.0, axis=0)
    center = np.median(points, axis=0)
    extents = np.maximum(hi - lo, 1e-6)
    centered = points - center[None]
    cov = centered.T @ centered / float(max(1, len(centered)))
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    axes = eigvecs[:, order]
    # PCA 轴的符号本来就是不确定的；这里先固定成右手系，后续再显式枚举符号和排列。
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return {
        "center": center.astype(np.float32),
        "bbox_min": lo.astype(np.float32),
        "bbox_max": hi.astype(np.float32),
        "extents": extents.astype(np.float32),
        "diagonal": float(np.linalg.norm(extents)),
        "axes": axes.astype(np.float32),
        "eigenvalues": eigvals.astype(np.float32),
    }


def _stats_to_json(stats: Dict[str, object]) -> Dict[str, object]:
    """把包含 ndarray 的几何统计转换成可写入 JSON 的基础 Python 类型。"""
    return {
        "center": np.asarray(stats["center"], dtype=float).tolist(),
        "bbox_min": np.asarray(stats["bbox_min"], dtype=float).tolist(),
        "bbox_max": np.asarray(stats["bbox_max"], dtype=float).tolist(),
        "extents": np.asarray(stats["extents"], dtype=float).tolist(),
        "diagonal": float(stats["diagonal"]),
        "eigenvalues": np.asarray(stats["eigenvalues"], dtype=float).tolist(),
    }


def _prepare_registration_mesh(
    base_mesh,
    observations: List[Dict[str, object]],
    part_name: str,
    args: argparse.Namespace,
) -> Tuple[object, Dict[str, object]]:
    """
    在 TSDF 内部检查单目重建 mesh 的尺度和坐标状态。

    深度掩码点先用数据集原始位姿变换到物体坐标系，再与 mesh 的稳健包围盒比较。
    只有尺度比超过明确的数量级阈值时才缩放注册副本；原始 SAM3D/Hunyuan3D/
    InstantMesh 文件以及后续模块看到的缓存都不会被修改。
    """
    mesh = base_mesh.copy()
    mesh_stats_before = _robust_geometry_stats(np.asarray(mesh.vertices, dtype=np.float32))
    observation_stats = []
    for obs in observations:
        points_obj = _points_cam_to_obj(
            np.asarray(obs["points_cam"], dtype=np.float32),
            np.asarray(obs["raw_pose"], dtype=np.float32),
        )
        if len(points_obj) < 3:
            continue
        stats = _robust_geometry_stats(points_obj)
        observation_stats.append((str(obs["frame"]), stats))

    report: Dict[str, object] = {
        "part": str(part_name),
        "mesh_before": _stats_to_json(mesh_stats_before),
        "observation_count": int(len(observation_stats)),
        "auto_scale_enabled": bool(args.iter_tsdf_auto_scale_mesh),
        "scale_applied": 1.0,
        "scale_status": "not_checked",
    }
    if not observation_stats:
        report["scale_status"] = "no_observation_geometry"
        return mesh, report

    obs_diagonals = np.asarray([float(s["diagonal"]) for _, s in observation_stats], dtype=np.float32)
    obs_centers = np.stack([np.asarray(s["center"], dtype=np.float32) for _, s in observation_stats], axis=0)
    observed_diagonal = float(np.median(obs_diagonals))
    observed_center = np.median(obs_centers, axis=0).astype(np.float32)
    mesh_diagonal = float(mesh_stats_before["diagonal"])
    ratio = mesh_diagonal / max(observed_diagonal, 1e-8)
    center_offset = observed_center - np.asarray(mesh_stats_before["center"], dtype=np.float32)
    report.update(
        {
            "observation_diagonal_median": observed_diagonal,
            "mesh_to_observation_scale_ratio": float(ratio),
            "observation_center_median_in_object": observed_center.astype(float).tolist(),
            "mesh_center_offset_to_observation": center_offset.astype(float).tolist(),
            "normalized_center_offset": float(
                np.linalg.norm(center_offset) / max(mesh_diagonal, observed_diagonal, 1e-8)
            ),
            "observation_frames": [frame for frame, _ in observation_stats],
        }
    )

    min_ratio = float(args.iter_tsdf_mesh_scale_ratio_min)
    max_ratio = float(args.iter_tsdf_mesh_scale_ratio_max)
    scale_is_abnormal = ratio < min_ratio or ratio > max_ratio
    if not scale_is_abnormal:
        report["scale_status"] = "metric_scale_plausible"
        report["mesh_after"] = report["mesh_before"]
        return mesh, report
    if not bool(args.iter_tsdf_auto_scale_mesh):
        report["scale_status"] = "abnormal_detected_not_corrected"
        report["mesh_after"] = report["mesh_before"]
        return mesh, report

    # 观测通常只覆盖可见表面，因此只修复明显的数量级错误，并限制最大缩放幅度。
    scale = float(np.clip(observed_diagonal / max(mesh_diagonal, 1e-8), 0.05, 20.0))
    center = np.asarray(mesh_stats_before["center"], dtype=np.float32)
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float32) - center[None]) * scale + center[None]
    mesh_stats_after = _robust_geometry_stats(np.asarray(mesh.vertices, dtype=np.float32))
    report["scale_applied"] = scale
    report["scale_status"] = "abnormal_corrected_for_tsdf_registration"
    report["mesh_after"] = _stats_to_json(mesh_stats_after)
    return mesh, report


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """根据旋转轴和角度构造 3x3 旋转矩阵，使用 Rodrigues 公式。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _center_pose_translation(rotation: np.ndarray, mesh_center: np.ndarray, cloud_center: np.ndarray) -> np.ndarray:
    """
    在给定旋转的前提下，把 mesh 中心移动到观测点云中心。

    这用于构造 seed 位姿：先猜一个旋转，再用中心对齐得到合理平移，降低 ICP
    一开始就完全对不上的概率。
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32)
    pose[:3, 3] = np.asarray(cloud_center, dtype=np.float32) - pose[:3, :3] @ np.asarray(
        mesh_center, dtype=np.float32
    )
    return pose


def _right_handed_axis_maps() -> List[np.ndarray]:
    """枚举 PCA/包围盒三条轴之间全部 24 个右手旋转对应关系。"""
    maps = []
    eye = np.eye(3, dtype=np.float32)
    for perm in itertools.permutations(range(3)):
        permuted = eye[:, list(perm)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = permuted @ np.diag(np.asarray(signs, dtype=np.float32))
            if np.linalg.det(mapping) > 0.0:
                maps.append(mapping.astype(np.float32))
    return maps


def _axisymmetric_mesh_axis(
    mesh_stats: Dict[str, object],
    part_name: str,
    symmetry_extent_tolerance: float,
) -> Optional[np.ndarray]:
    """
    根据部件几何判断是否存在轴对称，并返回对称轴。

    lid 等薄片类部件优先使用最短包围盒轴；其他部件只有在另外两个尺度足够接近时，
    才认为绕当前轴旋转会产生近似等价的外观。
    """
    extents = np.asarray(mesh_stats["extents"], dtype=np.float32)
    axes = np.asarray(mesh_stats["axes"], dtype=np.float32)
    order = np.argsort(extents)
    # lid 通常是扁平旋转体，其对称轴是最短包围盒轴。
    if "lid" in str(part_name).lower():
        return axes[:, int(order[0])]
    # 对一般零件，仅当另外两个尺度足够接近时才认为存在轴对称性。
    for axis_idx in range(3):
        other = [i for i in range(3) if i != axis_idx]
        similarity = abs(float(extents[other[0]] - extents[other[1]])) / max(
            float(max(extents[other[0]], extents[other[1]])), 1e-8
        )
        if similarity <= float(symmetry_extent_tolerance):
            return axes[:, axis_idx]
    return None


def _deduplicate_pose_hypotheses(
    hypotheses: List[Tuple[str, np.ndarray]], max_count: int
) -> List[Tuple[str, np.ndarray]]:
    """去掉旋转和平移几乎相同的 seed 位姿假设，避免重复 ICP 浪费时间。"""
    unique: List[Tuple[str, np.ndarray]] = []
    for name, pose in hypotheses:
        duplicate = any(
            _pose_angle_deg(pose, existing) < 0.5 and _pose_translation(pose, existing) < 1e-4
            for _, existing in unique
        )
        if not duplicate:
            unique.append((name, pose.astype(np.float32)))
        if int(max_count) > 0 and len(unique) >= int(max_count):
            break
    return unique


def _build_seed_pose_hypotheses(
    mesh,
    obs: Dict[str, object],
    init_pose: np.ndarray,
    part_name: str,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, np.ndarray]], Dict[str, object]]:
    """
    构造 seed 帧的全局初始位姿集合。

    依次加入：
    1. 原始 cam_params 位姿。
    2. 保留原始旋转、但把 mesh 中心平移到 mask 深度点云中心的位姿。
    3. 对 lid/轴对称零件，绕局部对称轴枚举旋转。
    4. 可选的 PCA/包围盒 24 种右手轴映射，用于纠正较大的初始方向错误。
    """
    mesh_stats = _robust_geometry_stats(np.asarray(mesh.vertices, dtype=np.float32))
    cloud_points = np.asarray(obs["points_cam"], dtype=np.float32)
    cloud_stats = _robust_geometry_stats(cloud_points)
    mesh_center = np.asarray(mesh_stats["center"], dtype=np.float32)
    cloud_center = np.asarray(cloud_stats["center"], dtype=np.float32)
    init_pose = np.asarray(init_pose, dtype=np.float32)

    hypotheses: List[Tuple[str, np.ndarray]] = [("raw_pose", init_pose.copy())]
    centered_raw = _center_pose_translation(init_pose[:3, :3], mesh_center, cloud_center)
    hypotheses.append(("raw_rotation_mask_center_translation", centered_raw))

    symmetry_axis = _axisymmetric_mesh_axis(
        mesh_stats,
        part_name,
        symmetry_extent_tolerance=float(args.iter_tsdf_axis_symmetry_extent_tolerance),
    )
    symmetry_steps = max(1, int(args.iter_tsdf_axis_symmetry_steps))
    if symmetry_axis is not None and symmetry_steps > 1:
        for step in range(1, symmetry_steps):
            local_rotation = _axis_angle_rotation(symmetry_axis, 2.0 * np.pi * step / symmetry_steps)
            rotation = centered_raw[:3, :3] @ local_rotation
            hypotheses.append(
                (
                    f"axis_symmetry_{step:02d}_of_{symmetry_steps:02d}",
                    _center_pose_translation(rotation, mesh_center, cloud_center),
                )
            )

    if bool(args.iter_tsdf_seed_global_init):
        mesh_axes = np.asarray(mesh_stats["axes"], dtype=np.float32)
        cloud_axes = np.asarray(cloud_stats["axes"], dtype=np.float32)
        for idx, axis_map in enumerate(_right_handed_axis_maps()):
            rotation = cloud_axes @ axis_map @ mesh_axes.T
            hypotheses.append(
                (
                    f"pca_bbox_axis_map_{idx:02d}",
                    _center_pose_translation(rotation, mesh_center, cloud_center),
                )
            )

    hypotheses = _deduplicate_pose_hypotheses(
        hypotheses,
        max_count=int(args.iter_tsdf_seed_max_hypotheses),
    )
    return hypotheses, {
        "mesh": _stats_to_json(mesh_stats),
        "mask_depth_cloud": _stats_to_json(cloud_stats),
        "axis_symmetric": bool(symmetry_axis is not None),
        "symmetry_axis_object": None if symmetry_axis is None else symmetry_axis.astype(float).tolist(),
        "hypothesis_count": int(len(hypotheses)),
        "hypothesis_names": [name for name, _ in hypotheses],
    }


def _pose_mesh_fit_metrics(
    points_cam: np.ndarray,
    pose: np.ndarray,
    mesh_pts: np.ndarray,
    mesh_normals: np.ndarray,
    tree,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    评估某个位姿下深度点云与 mesh 表面的贴合程度。

    流程是把观测点从相机系变到物体系，查找最近的 mesh 采样点，再计算
    point-to-plane 残差、最近邻距离和内点比例。这些指标用于 seed 排序和 ICP
    验收。
    """
    points_obj = _points_cam_to_obj(points_cam, pose)
    dist, idx = tree.query(points_obj, k=1, workers=-1)
    keep = dist <= float(args.iter_tsdf_refine_max_dist)
    if int(np.count_nonzero(keep)) < 6 and len(dist) > 0:
        cutoff = np.quantile(dist, min(0.8, max(0.1, float(args.iter_tsdf_refine_trim_quantile))))
        keep = dist <= cutoff
    if int(np.count_nonzero(keep)) < 6:
        return {
            "mean_abs_point_plane": float("inf"),
            "inlier_ratio": 0.0,
            "mean_nn_dist": float(np.mean(dist)) if len(dist) else float("inf"),
            "kept_points": int(np.count_nonzero(keep)),
        }
    residual = np.sum(
        (points_obj[keep] - mesh_pts[idx[keep]]) * mesh_normals[idx[keep]],
        axis=1,
    )
    return {
        "mean_abs_point_plane": float(np.mean(np.abs(residual))),
        "inlier_ratio": float(np.count_nonzero(keep) / max(1, len(points_cam))),
        "mean_nn_dist": float(np.mean(dist[keep])),
        "kept_points": int(np.count_nonzero(keep)),
    }


def _estimate_pose_point_to_plane_step(
    src_obj: np.ndarray,
    dst_cam: np.ndarray,
    normals_obj: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    """
    在当前位姿附近求一次 point-to-plane SE(3) 增量。

    残差在相机坐标系中构造，因此旋转和平移增量都直接左乘到 ``ob_in_cam``，
    与 Open3D/TSDF 使用的 object-to-camera 位姿约定保持一致。
    """
    """
    在当前位姿附近求一次 point-to-plane SE(3) 增量。

    残差在相机坐标系中构造，因此旋转和平移增量都直接左乘到 ob_in_cam，
    与 Open3D/TSDF 使用的 object-to-camera 位姿约定保持一致。
    """
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    src_cam = (rotation @ np.asarray(src_obj, dtype=np.float64).T).T + np.asarray(
        pose[:3, 3], dtype=np.float64
    )
    normals_cam = (rotation @ np.asarray(normals_obj, dtype=np.float64).T).T
    dst_cam = np.asarray(dst_cam, dtype=np.float64)
    a_rot = np.cross(src_cam, normals_cam)
    a = np.concatenate([a_rot, normals_cam], axis=1)
    b = np.sum(normals_cam * (dst_cam - src_cam), axis=1)
    if len(a) < 6:
        return np.asarray(pose, dtype=np.float32)
    delta, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    omega = delta[:3]
    translation = delta[3:]
    angle = float(np.linalg.norm(omega))
    if angle > 1e-12:
        delta_rotation = _axis_angle_rotation(omega / angle, angle).astype(np.float64)
    else:
        delta_rotation = np.eye(3, dtype=np.float64)
    out = np.asarray(pose, dtype=np.float64).copy()
    out[:3, :3] = delta_rotation @ out[:3, :3]
    out[:3, 3] = delta_rotation @ out[:3, 3] + translation
    return out.astype(np.float32)


def _run_icp_stage(
    points_cam: np.ndarray,
    mesh_pts: np.ndarray,
    mesh_normals: np.ndarray,
    tree,
    init_pose: np.ndarray,
    max_dist: float,
    iterations: int,
    trim_quantile: float,
    mode: str,
    angle_eps: float,
    trans_eps: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    执行一段 ICP，并记录每轮迭代的内点数、距离和位姿变化。

    ``mode`` 可以是 point-to-point 或 point-to-plane。若固定距离门限下对应点过少，
    会退回到分位数裁剪，尽量保留最可靠的一批近邻。
    """
    pose = np.asarray(init_pose, dtype=np.float32).copy()
    logs = []
    for iteration in range(max(1, int(iterations))):
        points_obj = _points_cam_to_obj(points_cam, pose)
        dist, idx = tree.query(points_obj, k=1, workers=-1)
        keep = dist <= float(max_dist)
        if int(np.count_nonzero(keep)) < 6 and len(dist) > 0:
            cutoff = np.quantile(dist, min(0.95, max(0.1, float(trim_quantile))))
            keep = dist <= cutoff
        kept = int(np.count_nonzero(keep))
        if kept < 6:
            logs.append({"iteration": iteration, "status": "too_few_correspondences", "kept_points": kept})
            break

        src_obj = mesh_pts[idx[keep]]
        dst_cam = points_cam[keep]
        if mode == "point_to_point":
            new_pose = _estimate_pose_from_obj_cam_points(src_obj, dst_cam)
        elif mode == "point_to_plane":
            new_pose = _estimate_pose_point_to_plane_step(
                src_obj,
                dst_cam,
                mesh_normals[idx[keep]],
                pose,
            )
        else:
            raise ValueError(f"unsupported ICP mode: {mode}")
        delta_angle = _pose_angle_deg(new_pose, pose)
        delta_trans = _pose_translation(new_pose, pose)
        logs.append(
            {
                "iteration": int(iteration),
                "status": "updated",
                "kept_points": kept,
                "mean_nn_dist": float(np.mean(dist[keep])),
                "delta_angle_deg": float(delta_angle),
                "delta_translation": float(delta_trans),
            }
        )
        pose = new_pose
        if delta_angle < float(angle_eps) and delta_trans < float(trans_eps):
            break
    return pose.astype(np.float32), {"mode": mode, "max_dist": float(max_dist), "iterations": logs}


def _coarse_to_fine_refine_pose(
    mesh,
    obs: Dict[str, object],
    init_pose: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    seed 专用的粗到细 ICP。

    粗阶段使用较大距离门限的 point-to-point，把较大的初始旋转/平移误差拉回；
    细阶段使用较小门限的 point-to-plane，提高表面贴合精度。
    """
    from scipy.spatial import cKDTree

    points_cam = _sample_points(
        np.asarray(obs["points_cam"], dtype=np.float32),
        int(args.iter_tsdf_refine_points),
        int(seed),
    )
    tm = _trimesh()
    sample_count = max(int(args.iter_tsdf_mesh_samples), 1024)
    mesh_pts, face_idx = tm.sample.sample_surface(mesh, sample_count)
    mesh_pts = np.asarray(mesh_pts, dtype=np.float32)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    mesh_normals = face_normals[np.asarray(face_idx, dtype=np.int64)]
    tree = cKDTree(mesh_pts)

    pose, coarse_log = _run_icp_stage(
        points_cam,
        mesh_pts,
        mesh_normals,
        tree,
        init_pose,
        max_dist=float(args.iter_tsdf_seed_coarse_max_dist),
        iterations=int(args.iter_tsdf_seed_coarse_iters),
        trim_quantile=float(args.iter_tsdf_seed_coarse_trim_quantile),
        mode="point_to_point",
        angle_eps=float(args.iter_tsdf_refine_angle_eps),
        trans_eps=float(args.iter_tsdf_refine_trans_eps),
    )
    pose, fine_log = _run_icp_stage(
        points_cam,
        mesh_pts,
        mesh_normals,
        tree,
        pose,
        max_dist=float(args.iter_tsdf_refine_max_dist),
        iterations=int(args.iter_tsdf_refine_iters),
        trim_quantile=float(args.iter_tsdf_refine_trim_quantile),
        mode="point_to_plane",
        angle_eps=float(args.iter_tsdf_refine_angle_eps),
        trans_eps=float(args.iter_tsdf_refine_trans_eps),
    )
    metrics = _pose_mesh_fit_metrics(points_cam, pose, mesh_pts, mesh_normals, tree, args)
    return pose, {"coarse_point_to_point": coarse_log, "fine_point_to_plane": fine_log, "fit": metrics}


def _seed_hypothesis_score(metrics: Dict[str, object]) -> float:
    """把 seed 粗配准指标压成一个越小越好的分数，用于选取待严格验证的 top-k。"""
    mean_nn = float(metrics.get("mean_nn_dist", float("inf")))
    point_plane = float(metrics.get("mean_abs_point_plane", float("inf")))
    inlier = float(metrics.get("inlier_ratio", 0.0))
    if not np.isfinite(mean_nn) or not np.isfinite(point_plane):
        return float("inf")
    return mean_nn + point_plane - 0.02 * inlier


def _refine_seed_pose_to_mesh(
    mesh,
    obs: Dict[str, object],
    init_pose: np.ndarray,
    part_name: str,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    对 seed 帧枚举多个全局位姿假设，粗筛后对最优候选执行严格 refinement/验收。

    seed 的位姿决定初始 TSDF 物体坐标系，因此这里允许比普通帧更大的旋转和平移
    修正范围，并把所有尝试写入日志，方便事后查看 seed 为什么成功或失败。
    """
    hypotheses, hypothesis_meta = _build_seed_pose_hypotheses(mesh, obs, init_pose, part_name, args)
    attempts = []
    ranked = []
    for idx, (name, hypothesis) in enumerate(hypotheses):
        coarse_pose, coarse_info = _coarse_to_fine_refine_pose(
            mesh,
            obs,
            hypothesis,
            args,
            seed=int(seed) * 100 + idx,
        )
        metrics = dict(coarse_info["fit"])
        score = _seed_hypothesis_score(metrics)
        ranked.append((score, name, coarse_pose, hypothesis, coarse_info))
        attempts.append(
            {
                "name": name,
                "score": float(score),
                "initial_pose": np.asarray(hypothesis, dtype=float).tolist(),
                "coarse_to_fine": coarse_info,
            }
        )

    ranked.sort(key=lambda x: x[0])
    verify_count = min(len(ranked), max(1, int(args.iter_tsdf_seed_verify_topk)))
    verified = []
    best_pose = np.asarray(init_pose, dtype=np.float32)
    best_info: Optional[Dict[str, object]] = None
    for rank, (score, name, coarse_pose, hypothesis, coarse_info) in enumerate(ranked[:verify_count]):
        refined_pose, refine_info = _refine_pose_to_mesh(
            mesh,
            obs,
            coarse_pose,
            args,
            seed=int(seed) * 1000 + rank,
            prior_pose=np.asarray(init_pose, dtype=np.float32),
            max_prior_angle=float(args.iter_tsdf_seed_max_prior_angle),
            max_prior_translation=float(args.iter_tsdf_seed_max_prior_translation),
        )
        verified.append(
            {
                "rank": int(rank),
                "name": name,
                "coarse_score": float(score),
                "coarse_initial_pose": np.asarray(hypothesis, dtype=float).tolist(),
                "coarse_to_fine": coarse_info,
                "strict_refine": refine_info,
            }
        )
        if bool(refine_info.get("ok", False)):
            best_pose = refined_pose
            best_info = refine_info
            break
        if best_info is None:
            best_pose = refined_pose
            best_info = refine_info

    if best_info is None:
        best_info = {
            "ok": False,
            "fail_reasons": ["no_seed_hypothesis"],
            "mean_abs_point_plane": float("inf"),
            "inlier_ratio": 0.0,
        }
    return best_pose.astype(np.float32), {
        **best_info,
        "seed_registration": {
            "hypothesis_meta": hypothesis_meta,
            "all_hypotheses": attempts,
            "verified_topk": verified,
        },
    }


def _pose_fit_ok(metrics: Dict[str, float], args: argparse.Namespace) -> bool:
    """判断一组点云-网格贴合指标是否达到可接受阈值。"""
    mean_abs = float(metrics.get("mean_abs_point_plane", float("inf")))
    inlier_ratio = float(metrics.get("inlier_ratio", 0.0))
    return (
        np.isfinite(mean_abs)
        and mean_abs <= float(args.iter_tsdf_refine_residual_thresh)
        and inlier_ratio >= float(args.iter_tsdf_min_initial_inlier_ratio)
    )


def _refine_pose_to_mesh(
    mesh,
    obs: Dict[str, object],
    init_pose: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    prior_pose: Optional[np.ndarray] = None,
    max_prior_angle: Optional[float] = None,
    max_prior_translation: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    将单帧初始位姿细化到当前 mesh 上，并执行安全验收。

    普通帧的 refinement 不能无限漂移：最终位姿必须满足残差阈值，并且相对 prior
    的旋转/平移变化不能超过配置上限。若 ICP 漂移但原始位姿本身已经足够贴合，
    可按配置回退并接受初始位姿。
    """
    from scipy.spatial import cKDTree

    points_cam = _sample_points(
        np.asarray(obs["points_cam"], dtype=np.float32),
        int(args.iter_tsdf_refine_points),
        int(seed),
    )
    tm = _trimesh()
    mesh_pts, face_idx = tm.sample.sample_surface(mesh, max(int(args.iter_tsdf_mesh_samples), 1024))
    mesh_pts = np.asarray(mesh_pts, dtype=np.float32)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    mesh_normals = face_normals[np.asarray(face_idx, dtype=np.int64)]
    tree = cKDTree(mesh_pts)

    init_pose = init_pose.astype(np.float32).copy()
    prior_pose = init_pose.copy() if prior_pose is None else np.asarray(prior_pose, dtype=np.float32).copy()
    init_metrics = _pose_mesh_fit_metrics(points_cam, init_pose, mesh_pts, mesh_normals, tree, args)
    pose = init_pose.copy()
    for _ in range(max(1, int(args.iter_tsdf_refine_iters))):
        points_obj = _points_cam_to_obj(points_cam, pose)
        dist, idx = tree.query(points_obj, k=1, workers=-1)
        keep = dist <= float(args.iter_tsdf_refine_max_dist)
        if int(np.count_nonzero(keep)) < 6:
            cutoff = np.quantile(dist, min(0.8, max(0.1, float(args.iter_tsdf_refine_trim_quantile))))
            keep = dist <= cutoff
        if int(np.count_nonzero(keep)) < 6:
            break
        src_obj = mesh_pts[idx[keep]]
        dst_cam = points_cam[keep]
        new_pose = _estimate_pose_from_obj_cam_points(src_obj, dst_cam)
        delta_angle = _pose_angle_deg(new_pose, pose)
        delta_trans = _pose_translation(new_pose, pose)
        pose = new_pose
        if delta_angle < float(args.iter_tsdf_refine_angle_eps) and delta_trans < float(args.iter_tsdf_refine_trans_eps):
            break

    final_metrics = _pose_mesh_fit_metrics(points_cam, pose, mesh_pts, mesh_normals, tree, args)
    prior_angle = _pose_angle_deg(pose, prior_pose)
    prior_trans = _pose_translation(pose, prior_pose)
    prior_delta = pose_delta_egocentric(prior_pose, pose)
    max_angle = float(args.iter_tsdf_max_refine_angle) if max_prior_angle is None else float(max_prior_angle)
    max_translation = (
        float(args.iter_tsdf_max_refine_translation)
        if max_prior_translation is None
        else float(max_prior_translation)
    )
    fail_reasons = []
    mean_abs = float(final_metrics["mean_abs_point_plane"])
    if not np.isfinite(mean_abs):
        fail_reasons.append("nonfinite_residual")
    if mean_abs > float(args.iter_tsdf_refine_residual_thresh):
        fail_reasons.append("residual_too_large")
    if prior_angle > max_angle:
        fail_reasons.append("rotation_delta_too_large")
    if prior_trans > max_translation:
        fail_reasons.append("translation_delta_too_large")
    ok = len(fail_reasons) == 0
    pose_source = "refined"
    can_fallback_to_init = (
        _pose_angle_deg(init_pose, prior_pose) <= max_angle
        and _pose_translation(init_pose, prior_pose) <= max_translation
    )
    if (
        bool(args.iter_tsdf_accept_initial_pose)
        and (not ok)
        and can_fallback_to_init
        and _pose_fit_ok(init_metrics, args)
    ):
        pose = init_pose.copy()
        pose_source = "initial_pose"
        prior_angle = _pose_angle_deg(pose, prior_pose)
        prior_trans = _pose_translation(pose, prior_pose)
        prior_delta = pose_delta_egocentric(prior_pose, pose)
        fail_reasons = []
        final_metrics = init_metrics
        ok = True
    return pose.astype(np.float32), {
        "ok": bool(ok),
        "pose_source": pose_source,
        "fail_reasons": fail_reasons,
        "mean_abs_point_plane": float(final_metrics["mean_abs_point_plane"]),
        "inlier_ratio": float(final_metrics["inlier_ratio"]),
        "mean_nn_dist": float(final_metrics["mean_nn_dist"]),
        "kept_points": int(final_metrics["kept_points"]),
        "initial_pose_fit": init_metrics,
        "max_prior_angle_deg": float(max_angle),
        "max_prior_translation": float(max_translation),
        "prior_angle_deg": float(prior_angle),
        "prior_translation": float(prior_trans),
        "pose_delta": prior_delta,
    }


def _pose_utility_config_from_args(args: argparse.Namespace) -> PoseUtilityConfig:
    """从命令行参数构造 pose utility 检查所需的配置对象。"""
    return PoseUtilityConfig(
        max_points=int(args.pose_utility_max_points),
        mesh_samples=int(args.pose_utility_mesh_samples),
        depth_inlier_thresh=float(args.pose_utility_depth_inlier_thresh),
        projection_depth_thresh=float(args.pose_utility_projection_depth_thresh),
        min_projected_points=int(args.pose_utility_min_projected_points),
        depth_weight=float(args.pose_utility_depth_weight),
        inlier_weight=float(args.pose_utility_inlier_weight),
        mask_weight=float(args.pose_utility_mask_weight),
        dt_weight=float(args.pose_utility_dt_weight),
        dr_weight=float(args.pose_utility_dr_weight),
    )


def _pose_delta_config_from_args(args: argparse.Namespace, anchor_frame: str) -> PoseDeltaOptimizationConfig:
    """
    从命令行参数构造联合位姿增量优化配置。

    当 pose-delta 参数未显式设置时，沿用 TSDF 单帧 refinement 的最大旋转/平移
    门限，避免两个验收标准互相矛盾。
    """
    max_dt = float(args.pose_delta_max_dt)
    if max_dt <= 0:
        max_dt = float(args.iter_tsdf_max_refine_translation)
    max_dr = float(args.pose_delta_max_dr_deg)
    if max_dr <= 0:
        max_dr = float(args.iter_tsdf_max_refine_angle)
    return PoseDeltaOptimizationConfig(
        rounds=int(args.pose_delta_rounds),
        max_dt=max_dt,
        max_dr_deg=max_dr,
        anchor_frame=str(anchor_frame),
        refine_anchor=bool(args.pose_delta_refine_anchor),
    )


def _pose_map_from_accepted(accepted: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """把 accepted 列表转换成 {frame: pose}，供联合优化和效用评估使用。"""
    return {
        str(item["obs"]["frame"]): np.asarray(item["pose"], dtype=np.float32).reshape(4, 4)
        for item in accepted
    }


def _obs_list_from_accepted(accepted: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """从 accepted 列表中抽出观测对象，保持与 pose_map 相同的数据来源。"""
    return [item["obs"] for item in accepted]


def _replace_accepted_poses(accepted: List[Dict[str, object]], pose_map: Dict[str, np.ndarray]) -> None:
    """用联合优化后的 pose_map 原地更新 accepted 中的位姿。"""
    for item in accepted:
        frame = str(item["obs"]["frame"])
        if frame in pose_map:
            item["pose"] = np.asarray(pose_map[frame], dtype=np.float32).reshape(4, 4)


def _copy_accepted(accepted: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """复制 accepted 状态；obs 只读共享，pose 做数值拷贝。"""
    return [
        {
            "obs": item["obs"],
            "pose": np.asarray(item["pose"], dtype=np.float32).reshape(4, 4).copy(),
        }
        for item in accepted
    ]


def _geometry_proxy_eval(
    mesh,
    accepted: List[Dict[str, object]],
    raw_pose_map: Dict[str, np.ndarray],
    k: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, object]:
    """
    用接近最终目标的可观测几何 proxy 给临时 TSDF 结果打分。

    这里仍只使用估计 pose 和观测数据，不需要 GT pose。score 来自 pose utility，
    额外加一点 accepted frame coverage，避免过度偏好只含少数帧的局部结果。
    """
    pose_map = _pose_map_from_accepted(accepted)
    init_pose_map = {
        frame: np.asarray(raw_pose_map.get(frame, pose), dtype=np.float32).reshape(4, 4)
        for frame, pose in pose_map.items()
    }
    eval_info = evaluate_pose_utility(
        mesh=mesh,
        observations=_obs_list_from_accepted(accepted),
        pose_map=pose_map,
        init_pose_map=init_pose_map,
        k=k,
        cfg=_pose_utility_config_from_args(args),
        seed=int(seed),
    )
    eval_score = float(eval_info.get("score", float("-inf")))
    coverage_bonus = float(args.iter_tsdf_geometry_coverage_weight) * np.log1p(len(accepted))
    score = eval_score + coverage_bonus
    return {
        "score": float(score),
        "utility_score": eval_score,
        "coverage_bonus": float(coverage_bonus),
        "frames": int(len(accepted)),
        "utility": eval_info,
    }


def _project_points(points_cam: np.ndarray, k: np.ndarray, shape: Tuple[int, int]):
    """
    将相机坐标点投影到像素平面，并返回像素坐标与有效投影掩码。

    有效点需要满足正深度且落在图像范围内；后续深度一致性检查会进一步验证 mask
    和深度残差。
    """
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = np.zeros(len(points_cam), dtype=np.int64)
    v = np.zeros(len(points_cam), dtype=np.int64)
    u[valid] = np.rint(points_cam[valid, 0] * float(k[0, 0]) / z[valid] + float(k[0, 2])).astype(np.int64)
    v[valid] = np.rint(points_cam[valid, 1] * float(k[1, 1]) / z[valid] + float(k[1, 2])).astype(np.int64)
    h, w = shape
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, valid


def _depth_consistency_one_way(
    src_obs: Dict[str, object],
    src_pose: np.ndarray,
    dst_obs: Dict[str, object],
    dst_pose: np.ndarray,
    k: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    """
    单向检查：把源帧的深度点经物体系投影到目标帧，比较目标深度。

    如果同一物体表面在两个视角下位姿一致，投影点应落在目标 mask 内，且预测深度
    与目标深度图接近。该函数只做 src -> dst，双向检查由上层组合完成。
    """
    src_pts_cam = _sample_points(
        np.asarray(src_obs["points_cam"], dtype=np.float32),
        int(args.iter_tsdf_consistency_points),
        int(seed),
    )
    points_obj = _points_cam_to_obj(src_pts_cam, src_pose)
    dst_pts_cam = _points_obj_to_cam(points_obj, dst_pose)
    dst_depth = np.asarray(dst_obs["depth_m"], dtype=np.float32)
    dst_mask = np.asarray(dst_obs["mask"], dtype=bool)
    u, v, valid = _project_points(dst_pts_cam, k, dst_depth.shape[:2])
    if int(np.count_nonzero(valid)) == 0:
        return {"valid_ratio": 0.0, "mean_abs_depth": float("inf"), "inlier_ratio": 0.0}
    valid_idx = np.where(valid)[0]
    in_mask = dst_mask[v[valid_idx], u[valid_idx]]
    depth_vals = dst_depth[v[valid_idx], u[valid_idx]]
    has_depth = depth_vals > 1e-6
    keep_local = in_mask & has_depth
    if int(np.count_nonzero(keep_local)) == 0:
        return {"valid_ratio": 0.0, "mean_abs_depth": float("inf"), "inlier_ratio": 0.0}
    kept = valid_idx[keep_local]
    residual = np.abs(dst_pts_cam[kept, 2] - depth_vals[keep_local])
    return {
        "valid_ratio": float(len(kept) / max(1, len(src_pts_cam))),
        "mean_abs_depth": float(np.mean(residual)),
        "inlier_ratio": float(np.mean(residual <= float(args.iter_tsdf_depth_residual_thresh))),
    }


def _candidate_information_gain(
    candidate_obs: Dict[str, object],
    candidate_pose: np.ndarray,
    accepted: List[Dict[str, object]],
    k: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, object]:
    """
    估计候选帧相对已融合帧的 overlap consistency 和 new-surface utility。

    overlap 部分要求候选点投影到已接收帧时与已有表面一致；new-surface 部分
    统计候选可见、但已有帧尚未覆盖的比例，用于鼓励真正补几何的视角进入 TSDF。
    """
    points_cam = _sample_points(
        np.asarray(candidate_obs["points_cam"], dtype=np.float32),
        int(args.iter_tsdf_consistency_points),
        int(seed),
    )
    if len(points_cam) == 0:
        return {
            "score": float("-inf"),
            "overlap_ratio": 0.0,
            "overlap_inlier_ratio": 0.0,
            "overlap_mean_abs_depth": float("inf"),
            "new_surface_ratio": 0.0,
            "covered_ratio": 0.0,
            "pairs": [],
        }
    if not accepted:
        return {
            "score": 1.0,
            "overlap_ratio": 0.0,
            "overlap_inlier_ratio": 1.0,
            "overlap_mean_abs_depth": 0.0,
            "new_surface_ratio": 1.0,
            "covered_ratio": 0.0,
            "pairs": [],
        }

    points_obj = _points_cam_to_obj(points_cam, candidate_pose)
    covered_any = np.zeros(len(points_cam), dtype=bool)
    inlier_any = np.zeros(len(points_cam), dtype=bool)
    residuals = []
    pair_logs = []
    for idx, item in enumerate(accepted):
        acc_obs = item["obs"]
        acc_pose = np.asarray(item["pose"], dtype=np.float32)
        dst_pts_cam = _points_obj_to_cam(points_obj, acc_pose)
        dst_depth = np.asarray(acc_obs["depth_m"], dtype=np.float32)
        dst_mask = np.asarray(acc_obs["mask"], dtype=bool)
        u, v, valid = _project_points(dst_pts_cam, k, dst_depth.shape[:2])
        if int(np.count_nonzero(valid)) == 0:
            pair_logs.append({"frame": acc_obs["frame"], "covered_ratio": 0.0, "inlier_ratio": 0.0})
            continue
        valid_idx = np.where(valid)[0]
        in_mask = dst_mask[v[valid_idx], u[valid_idx]]
        depth_vals = dst_depth[v[valid_idx], u[valid_idx]]
        has_depth = depth_vals > 1e-6
        overlap_local = in_mask & has_depth
        if int(np.count_nonzero(overlap_local)) == 0:
            pair_logs.append({"frame": acc_obs["frame"], "covered_ratio": 0.0, "inlier_ratio": 0.0})
            continue
        overlap_idx = valid_idx[overlap_local]
        residual = np.abs(dst_pts_cam[overlap_idx, 2] - depth_vals[overlap_local])
        inlier_local = residual <= float(args.iter_tsdf_depth_residual_thresh)
        covered_any[overlap_idx] = True
        inlier_any[overlap_idx[inlier_local]] = True
        residuals.append(residual.astype(np.float32))
        pair_logs.append(
            {
                "frame": acc_obs["frame"],
                "covered_ratio": float(len(overlap_idx) / max(1, len(points_cam))),
                "inlier_ratio": float(np.mean(inlier_local)),
                "mean_abs_depth": float(np.mean(residual)),
            }
        )

    overlap_ratio = float(np.mean(covered_any))
    new_surface_ratio = float(np.mean(~covered_any))
    if residuals:
        residual_all = np.concatenate(residuals, axis=0)
        mean_abs = float(np.mean(residual_all))
    else:
        mean_abs = float("inf")
    overlap_inlier = float(np.count_nonzero(inlier_any) / max(1, np.count_nonzero(covered_any)))
    depth_penalty = 0.0 if not np.isfinite(mean_abs) else mean_abs / max(float(args.iter_tsdf_depth_residual_thresh), 1e-6)
    score = (
        float(args.iter_tsdf_info_overlap_weight) * overlap_inlier
        + float(args.iter_tsdf_info_new_surface_weight) * new_surface_ratio
        - float(args.iter_tsdf_info_depth_penalty) * depth_penalty
    )
    return {
        "score": float(score),
        "overlap_ratio": overlap_ratio,
        "overlap_inlier_ratio": overlap_inlier,
        "overlap_mean_abs_depth": mean_abs,
        "new_surface_ratio": new_surface_ratio,
        "covered_ratio": overlap_ratio,
        "pairs": pair_logs,
    }


def _check_depth_consistency(
    candidate_obs: Dict[str, object],
    candidate_pose: np.ndarray,
    accepted: List[Dict[str, object]],
    k: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, object]]:
    """
    检查候选帧与已接收帧之间的双向深度一致性。

    对每个已接收帧同时计算 candidate -> accepted 和 accepted -> candidate，
    再选择“有效投影多且深度残差小”的最佳配对作为验收依据。这样可以降低遮挡、
    mask 不完整或单向投影失败对决策的影响。
    """
    if not accepted:
        return True, {"checked_pairs": 0}
    info_gain = _candidate_information_gain(
        candidate_obs,
        candidate_pose,
        accepted,
        k,
        args,
        seed=911,
    )
    pair_logs = []
    for idx, item in enumerate(accepted):
        acc_obs = item["obs"]
        acc_pose = item["pose"]
        forward = _depth_consistency_one_way(
            candidate_obs,
            candidate_pose,
            acc_obs,
            acc_pose,
            k,
            args,
            seed=idx + 13,
        )
        backward = _depth_consistency_one_way(
            acc_obs,
            acc_pose,
            candidate_obs,
            candidate_pose,
            k,
            args,
            seed=idx + 113,
        )
        pair_logs.append({"frame": acc_obs["frame"], "candidate_to_accepted": forward, "accepted_to_candidate": backward})
    valid = [min(p["candidate_to_accepted"]["valid_ratio"], p["accepted_to_candidate"]["valid_ratio"]) for p in pair_logs]
    depth = [max(p["candidate_to_accepted"]["mean_abs_depth"], p["accepted_to_candidate"]["mean_abs_depth"]) for p in pair_logs]
    inlier = [min(p["candidate_to_accepted"]["inlier_ratio"], p["accepted_to_candidate"]["inlier_ratio"]) for p in pair_logs]
    best_idx = int(np.argmax(np.asarray(valid, dtype=np.float32) - np.asarray(depth, dtype=np.float32)))
    strict_overlap_ok = (
        valid[best_idx] >= float(args.iter_tsdf_min_valid_proj_ratio)
        and depth[best_idx] <= float(args.iter_tsdf_depth_residual_thresh)
        and inlier[best_idx] >= float(args.iter_tsdf_min_depth_inlier_ratio)
    )
    info_overlap_ok = (
        float(info_gain["overlap_ratio"]) >= float(args.iter_tsdf_min_overlap_ratio)
        and float(info_gain["overlap_mean_abs_depth"]) <= float(args.iter_tsdf_depth_residual_thresh)
        and float(info_gain["overlap_inlier_ratio"]) >= float(args.iter_tsdf_min_depth_inlier_ratio)
    )
    new_surface_ok = (
        bool(args.iter_tsdf_accept_new_surface)
        and float(info_gain["new_surface_ratio"]) >= float(args.iter_tsdf_min_new_surface_ratio)
        and (
            float(info_gain["overlap_ratio"]) < float(args.iter_tsdf_min_overlap_ratio)
            or float(info_gain["overlap_inlier_ratio"]) >= float(args.iter_tsdf_min_new_surface_overlap_inlier)
        )
    )
    ok = strict_overlap_ok or info_overlap_ok or new_surface_ok
    return bool(ok), {
        "checked_pairs": len(pair_logs),
        "decision": "strict_overlap" if strict_overlap_ok else ("info_overlap" if info_overlap_ok else ("new_surface" if new_surface_ok else "rejected")),
        "best_pair_frame": pair_logs[best_idx]["frame"],
        "best_valid_ratio": float(valid[best_idx]),
        "best_mean_abs_depth": float(depth[best_idx]),
        "best_inlier_ratio": float(inlier[best_idx]),
        "information_gain": info_gain,
        "pairs": pair_logs,
    }


def _rebuild_tsdf_mesh(accepted: List[Dict[str, object]], out_path: Optional[Path], k: np.ndarray, args: argparse.Namespace, o3d):
    """
    用当前已接收帧重新融合 TSDF 并提取三角网格。

    每次接收新帧后都会调用一次：如果新融合网格为空，说明该帧破坏了融合结果，
    上层会撤销这次接收。最终调用时传入 ``out_path`` 保存正式模型。
    """
    schedules = [(float(args.voxel_length), float(args.sdf_trunc), "fine")]
    if bool(args.iter_tsdf_coarse_to_fine):
        coarse = (
            float(args.voxel_length) * float(args.iter_tsdf_coarse_voxel_multiplier),
            float(args.sdf_trunc) * float(args.iter_tsdf_coarse_trunc_multiplier),
            "coarse_fallback",
        )
        schedules.append(coarse)
    for voxel_length, sdf_trunc, _stage in schedules:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_length),
            sdf_trunc=float(sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for item in accepted:
            obs = item["obs"]
            rgbd = _make_rgbd(obs, args, o3d)
            intrinsic = _camera_intrinsic(obs, k, o3d)
            # Open3D integrate uses camera extrinsic (world-to-camera). Here the TSDF world is object frame.
            volume.integrate(rgbd, intrinsic, np.asarray(item["pose"], dtype=np.float64))
        mesh = volume.extract_triangle_mesh()
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=True)
            return mesh
    return None


def _select_seed(observations: List[Dict[str, object]]) -> int:
    """选择掩码像素最多的观测作为默认 seed；保留给旧逻辑或调试使用。"""
    if not observations:
        return -1
    return int(np.argmax([int(o["mask_pixels"]) for o in observations]))


def _seed_candidate_indices(observations: List[Dict[str, object]], max_attempts: int = 0) -> List[int]:
    """按可见几何约束强度生成 seed 候选序列；max_attempts<=0 表示尝试全部。"""
    order = sorted(
        range(len(observations)),
        key=lambda i: _observation_seed_score(observations[i])[0],
        reverse=True,
    )
    if int(max_attempts) > 0:
        order = order[: int(max_attempts)]
    return order


def _select_next_candidate(
    observations: List[Dict[str, object]],
    pending: List[int],
    accepted: List[Dict[str, object]],
    raw_pose_map: Dict[str, np.ndarray],
    k: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Optional[int], Dict[str, object]]:
    """
    从尚未处理的帧中选择下一帧。

    首先用视角距离门控过滤明显重复/过远帧；在剩余候选中优先选择
    overlap consistency 和 new-surface utility 更高的帧。
    """
    accepted_raw = [raw_pose_map[str(item["obs"]["frame"])] for item in accepted]
    best = None
    rejected = []
    for idx in list(pending):
        obs = observations[idx]
        raw_pose = raw_pose_map[str(obs["frame"])]
        ok, info = _score_pose_distance(
            raw_pose,
            accepted_raw,
            float(args.iter_tsdf_min_view_angle),
            float(args.iter_tsdf_max_view_angle),
            float(args.iter_tsdf_min_translation),
            float(args.iter_tsdf_max_translation),
        )
        gain = _candidate_information_gain(
            obs,
            raw_pose,
            accepted,
            k,
            args,
            seed=7000 + int(idx),
        )
        if not ok and not (
            bool(args.iter_tsdf_info_gain_overrides_view_gate)
            and float(gain["new_surface_ratio"]) >= float(args.iter_tsdf_min_new_surface_ratio)
        ):
            rejected.append({"frame": obs["frame"], "reason": "view_gate", **info, "information_gain": gain})
            continue
        score = (
            float(gain["score"])
            + 0.05 * _observation_seed_score(obs)[0]
            - float(args.iter_tsdf_redundancy_penalty) * max(0.0, 1.0 - float(gain["new_surface_ratio"]))
        )
        if not ok:
            score -= float(args.iter_tsdf_view_gate_override_penalty)
        if best is None or score > best[0]:
            best = (score, idx, info, gain)
    if best is None:
        return None, {"view_gate_rejections": rejected}
    return int(best[1]), {
        "selected_info": best[2],
        "information_gain": best[3],
        "selection_score": float(best[0]),
        "view_gate_rejections": rejected,
    }


def _refresh_raw_poses_after_shape_update(
    current_mesh,
    observations: List[Dict[str, object]],
    raw_pose_map: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed_offset: int,
) -> Dict[str, object]:
    """
    TSDF 网格更新后，用新形状重新细化所有候选帧的 raw pose。

    刷新后的 raw_pose_map 只影响后续候选选择和初始化；已经选中的候选在进入融合前
    仍会单独执行一次 refinement。
    """
    logs = {}
    for idx, obs in enumerate(observations):
        frame = str(obs["frame"])
        new_pose, info = _refine_pose_to_mesh(
            current_mesh,
            obs,
            raw_pose_map[frame],
            args,
            seed=seed_offset + idx,
        )
        if bool(info.get("ok", False)):
            raw_pose_map[frame] = new_pose
        logs[frame] = info
    return logs


def _iterative_tsdf_part(
    obj: DatasetObject,
    part_name: str,
    part_model: str,
    base_obj: Path,
    out_obj: Path,
    work_root: Path,
    method: str,
    k: np.ndarray,
    args: argparse.Namespace,
    o3d,
) -> Dict[str, object]:
    """
    单个部件的迭代式 TSDF 主流程。

    高层状态机：
    1. 收集所有可用 RGB-D 观测。
    2. 加载基础单目网格，并生成只在 TSDF 内部使用的配准副本。
    3. 从大 mask 帧开始尝试 seed，建立初始物体坐标系。
    4. 循环选择新候选帧，执行位姿 refinement 和深度一致性验收。
    5. 接收候选后重建 TSDF，必要时做联合位姿优化和 utility gate。
    6. 保存最终 mesh、accepted/rejected 帧列表、逐帧位姿和 summary.json。
    """
    observations = []
    for frame in frames_for_part(obj, part_name, args.max_frames, args.frame_stride):
        obs = _load_frame_observation(obj, part_name, frame, k, args)
        if obs is not None:
            observations.append(obs)
    if not observations:
        part_summary = {
            "part": part_name,
            "part_model": part_model,
            "status": "failed",
            "reason": "no_valid_observations",
            "frames": 0,
            "model": "",
        }
        return part_summary

    tm = _trimesh()
    base_mesh = _as_trimesh(tm.load(str(base_obj), force="mesh", process=False))
    # current_mesh 始终表示“当前用于配准和评估的形状”。第一次来自基础网格副本，
    # 每接收一个新视角后会被 TSDF 融合结果替换。
    current_mesh, mesh_diagnostics = _prepare_registration_mesh(base_mesh, observations, part_name, args)
    raw_pose_map = {str(o["frame"]): np.asarray(o["raw_pose"], dtype=np.float32) for o in observations}
    seed_attempts = []
    seed_idx = -1
    seed_obs = None
    seed_pose = None
    seed_refine = None
    for attempt_idx, cand_seed_idx in enumerate(
        _seed_candidate_indices(observations, int(args.iter_tsdf_seed_attempts))
    ):
        cand_seed_obs = observations[cand_seed_idx]
        cand_seed_pose, cand_seed_refine = _refine_seed_pose_to_mesh(
            current_mesh,
            cand_seed_obs,
            np.asarray(cand_seed_obs["raw_pose"], dtype=np.float32),
            part_name,
            args,
            seed=17 + attempt_idx,
        )
        seed_attempts.append(
            {
                "attempt": int(attempt_idx),
                "frame": str(cand_seed_obs["frame"]),
                "mask_pixels": int(cand_seed_obs["mask_pixels"]),
                "refine": cand_seed_refine,
            }
        )
        if bool(cand_seed_refine.get("ok", False)):
            seed_idx = int(cand_seed_idx)
            seed_obs = cand_seed_obs
            seed_pose = cand_seed_pose
            seed_refine = cand_seed_refine
            break
    if seed_obs is None or seed_pose is None or seed_refine is None:
        part_summary = {
            "part": part_name,
            "part_model": part_model,
            "status": "failed",
            "reason": "all_seed_pose_refine_failed",
            "frames": 0,
            "candidate_frames": len(observations),
            "mesh_diagnostics": mesh_diagnostics,
            "seed_attempts": seed_attempts,
            "model": "",
        }
        return part_summary
    # accepted 是 TSDF 融合的真实状态：只有通过 seed/refine/一致性/utility 的帧才进入。
    accepted = [{"obs": seed_obs, "pose": seed_pose}]
    accepted_frames = {str(seed_obs["frame"])}
    pending = [i for i in range(len(observations)) if i != seed_idx]
    rejected = []
    iter_logs = [
        {
            "iter": 0,
            "frame": seed_obs["frame"],
            "status": "accepted_seed",
            "mask_pixels": int(seed_obs["mask_pixels"]),
            "mesh_diagnostics": mesh_diagnostics,
            "refine": seed_refine,
            "seed_attempts": seed_attempts,
        }
    ]
    best_accepted = _copy_accepted(accepted)
    best_proxy: Dict[str, object] = {"score": float("-inf"), "reason": "not_evaluated"}
    rollback_logs = []
    if bool(args.iter_tsdf_geometry_rollback):
        seed_mesh_o3d = _rebuild_tsdf_mesh(accepted, None, k, args, o3d)
        if seed_mesh_o3d is not None:
            seed_mesh = _o3d_mesh_to_trimesh(seed_mesh_o3d)
            best_proxy = _geometry_proxy_eval(seed_mesh, accepted, raw_pose_map, k, args, seed=2500)

    max_accept = int(args.iter_tsdf_max_frames)
    if max_accept <= 0:
        max_accept = len(observations)
    max_rounds = max(len(observations) * 2, max_accept)
    for it in range(1, max_rounds + 1):
        if len(accepted) >= max_accept or not pending:
            break
        # 先从视角分布上挑一个有价值的候选，再做较昂贵的 ICP 与 TSDF 融合。
        cand_idx, select_info = _select_next_candidate(observations, pending, accepted, raw_pose_map, k, args)
        if cand_idx is None:
            rejected.extend(select_info.get("view_gate_rejections", []))
            break
        pending.remove(cand_idx)
        cand_obs = observations[cand_idx]
        cand_pose_init = raw_pose_map[str(cand_obs["frame"])]
        refined_pose, refine_info = _refine_pose_to_mesh(current_mesh, cand_obs, cand_pose_init, args, seed=it + 31)
        if not bool(refine_info.get("ok", False)):
            item = {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "rejected",
                "reason": "pose_refine_failed",
                "select": select_info,
                "refine": refine_info,
            }
            rejected.append(item)
            iter_logs.append(item)
            continue
        consistency_ok, consistency_info = _check_depth_consistency(cand_obs, refined_pose, accepted, k, args)
        if not consistency_ok:
            item = {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "rejected",
                "reason": "depth_consistency_failed",
                "select": select_info,
                "refine": refine_info,
                "consistency": consistency_info,
            }
            rejected.append(item)
            iter_logs.append(item)
            continue

        old_mesh = current_mesh
        accepted.append({"obs": cand_obs, "pose": refined_pose})
        accepted_frames.add(str(cand_obs["frame"]))
        # 先临时融合。如果提取出的 TSDF mesh 为空，说明这帧不能保留，需要回滚 accepted。
        mesh_o3d = _rebuild_tsdf_mesh(
            accepted,
            None,
            k,
            args,
            o3d,
        )
        if mesh_o3d is None:
            item = {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "rejected",
                "reason": "tsdf_mesh_empty_after_accept",
                "select": select_info,
                "refine": refine_info,
                "consistency": consistency_info,
            }
            rejected.append(item)
            accepted.pop()
            accepted_frames.discard(str(cand_obs["frame"]))
            iter_logs.append(item)
            continue
        proposed_mesh = _o3d_mesh_to_trimesh(mesh_o3d)
        pose_map_before_utility = _pose_map_from_accepted(accepted)
        init_pose_map_for_utility = {
            frame: np.asarray(raw_pose_map.get(frame, pose), dtype=np.float32).reshape(4, 4)
            for frame, pose in pose_map_before_utility.items()
        }
        accepted_obs = _obs_list_from_accepted(accepted)
        pose_delta_info: Dict[str, object] = {
            "status": "disabled",
            "reason": "pose_delta_opt disabled",
        }
        pose_map_after_delta = pose_map_before_utility
        if bool(args.pose_delta_opt):
            def _refine_for_pose_delta(mesh_obj, obs_obj, init_pose_obj, seed_obj):
                return _refine_pose_to_mesh(mesh_obj, dict(obs_obj), np.asarray(init_pose_obj, dtype=np.float32), args, int(seed_obj))

            # 联合优化只调整已接收帧之间的相对位姿，不改变候选集合。
            pose_map_after_delta, pose_delta_info = optimize_pose_deltas(
                mesh=proposed_mesh,
                observations=accepted_obs,
                init_pose_map=pose_map_before_utility,
                refine_pose_fn=_refine_for_pose_delta,
                cfg=_pose_delta_config_from_args(args, anchor_frame=str(seed_obs["frame"])),
                seed=2000 + it * 100,
            )

        pose_utility_info: Dict[str, object] = {
            "status": "disabled",
            "reason": "pose_utility_check disabled",
        }
        if bool(args.pose_utility_check) and len(accepted) > 1:
            # utility gate 比较“接收前旧形状”和“接收后新形状”，防止虽然 TSDF 非空、
            # 但整体投影/深度匹配质量变差的帧进入结果。
            utility_cfg = _pose_utility_config_from_args(args)
            old_eval = evaluate_pose_utility(
                mesh=old_mesh,
                observations=accepted_obs,
                pose_map=pose_map_before_utility,
                init_pose_map=init_pose_map_for_utility,
                k=k,
                cfg=utility_cfg,
                seed=3000 + it * 100,
            )
            new_eval = evaluate_pose_utility(
                mesh=proposed_mesh,
                observations=accepted_obs,
                pose_map=pose_map_after_delta,
                init_pose_map=init_pose_map_for_utility,
                k=k,
                cfg=utility_cfg,
                seed=4000 + it * 100,
            )
            utility_ok, utility_gate = utility_improved(
                old_eval,
                new_eval,
                min_delta=float(args.pose_utility_min_score_delta),
                max_frame_drop=float(args.pose_utility_max_frame_drop),
            )
            pose_utility_info = {
                "status": "accepted" if utility_ok else "rejected",
                "gate": utility_gate,
                "old_eval": old_eval,
                "new_eval": new_eval,
            }
            if not utility_ok:
                item = {
                    "iter": it,
                    "frame": cand_obs["frame"],
                    "status": "rejected",
                    "reason": "negative_pose_utility",
                    "select": select_info,
                    "refine": refine_info,
                    "consistency": consistency_info,
                    "pose_delta_optimization": pose_delta_info,
                    "pose_utility": pose_utility_info,
                }
                rejected.append(item)
                accepted.pop()
                accepted_frames.discard(str(cand_obs["frame"]))
                iter_logs.append(item)
                continue

        _replace_accepted_poses(accepted, pose_map_after_delta)
        current_mesh = proposed_mesh
        geometry_proxy_info: Dict[str, object] = {
            "status": "disabled",
            "reason": "geometry rollback disabled",
        }
        if bool(args.iter_tsdf_geometry_rollback):
            geometry_proxy_info = _geometry_proxy_eval(
                current_mesh,
                accepted,
                raw_pose_map,
                k,
                args,
                seed=5000 + it * 100,
            )
            if float(geometry_proxy_info["score"]) >= float(best_proxy.get("score", float("-inf"))):
                best_proxy = geometry_proxy_info
                best_accepted = _copy_accepted(accepted)
        # Raw poses are refreshed only after shape update, then used by the next
        # candidate-selection round. The selected candidate still only receives
        # pose refinement before fusion.
        raw_update_logs = _refresh_raw_poses_after_shape_update(
            current_mesh,
            observations,
            raw_pose_map,
            args,
            seed_offset=1000 + it * 100,
        )
        iter_logs.append(
            {
                "iter": it,
                "frame": cand_obs["frame"],
                "status": "accepted",
                "accepted_count": len(accepted),
                "select": select_info,
                "refine": refine_info,
                "consistency": consistency_info,
                "pose_delta_optimization": pose_delta_info,
                "pose_utility": pose_utility_info,
                "geometry_proxy": geometry_proxy_info,
                "raw_pose_refresh": raw_update_logs,
            }
        )

    final_accepted = accepted
    if bool(args.iter_tsdf_geometry_rollback) and best_accepted:
        current_proxy = _geometry_proxy_eval(
            current_mesh,
            accepted,
            raw_pose_map,
            k,
            args,
            seed=9000,
        ) if len(accepted) > 0 else {"score": float("-inf")}
        if float(best_proxy.get("score", float("-inf"))) > float(current_proxy.get("score", float("-inf"))) + float(args.iter_tsdf_final_rollback_margin):
            rollback_logs.append(
                {
                    "status": "rolled_back_to_best_proxy",
                    "current_proxy": current_proxy,
                    "best_proxy": best_proxy,
                    "current_frames": [str(item["obs"]["frame"]) for item in accepted],
                    "best_frames": [str(item["obs"]["frame"]) for item in best_accepted],
                }
            )
            final_accepted = best_accepted
        else:
            rollback_logs.append(
                {
                    "status": "kept_current",
                    "current_proxy": current_proxy,
                    "best_proxy": best_proxy,
                }
            )

    final_mesh_o3d = _rebuild_tsdf_mesh(final_accepted, out_obj, k, args, o3d)
    status = "success" if final_mesh_o3d is not None and len(accepted) > 0 else "failed"

    part_summary = {
        "part": part_name,
        "part_model": part_model,
        "status": status,
        "frames": len(final_accepted) if status == "success" else 0,
        "reason": "" if status == "success" else "tsdf_mesh_empty",
        "candidate_frames": len(observations),
        "mesh_diagnostics": mesh_diagnostics,
        "accepted_frames": [str(item["obs"]["frame"]) for item in final_accepted],
        "rejected": rejected,
        "seed_frame": str(seed_obs["frame"]),
        "model": str(out_obj),
        "geometry_rollback": rollback_logs,
        "iterations": iter_logs,
    }
    return part_summary


def _require_base(args: argparse.Namespace, obj: DatasetObject, base_method: str) -> None:
    """
    确认基础方法的 pose-ready 模型已经存在。

    TSDF 只负责多帧融合，不直接生成单目基础网格。若缓存缺失且用户开启
    ``--build-base-if-missing``，这里会调用对应基础重建方法补齐模型。
    """
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
    elif base_method == "instantmesh":
        from run.recon_instantmesh import reconstruct_object
    else:
        raise ValueError(f"unknown base method: {base_method}")
    reconstruct_object(obj, args)


def run_tsdf_object(obj: DatasetObject, args: argparse.Namespace, base_method: str, method: str) -> Dict[str, object]:
    """
    对一个对象运行指定 base_method 上的 TSDF 后处理。

    对象级职责是准备目录、遍历部件、汇总状态并复制最终模型树；真正的单部件迭代
    融合逻辑在 ``_iterative_tsdf_part`` 中完成。
    """
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("Open3D is required for TSDF reconstruction.") from e

    work_root = Path(args.work_root).resolve()
    _require_base(args, obj, base_method)
    base_pose_root = method_pose_ready_dir(work_root, base_method, args.split, obj.name)
    out_pose_root = ensure_dir(method_pose_ready_dir(work_root, method, args.split, obj.name))

    k = load_k(obj)
    parts = list_parts(obj)
    summary = {"method": method, "base_method": base_method, "object": obj.name, "parts": []}
    status_counts: Dict[str, int] = {}

    for part_idx, part_name in enumerate(parts):
        part_model = part_model_name(part_name, part_idx)
        base_obj = model_obj_path(base_pose_root, part_model)
        out_obj = model_obj_path(out_pose_root, part_model)
        # 已有结果且未要求 overwrite 时直接复用，避免长时间重复融合。
        if out_obj.exists() and not args.overwrite:
            summary["parts"].append({"part": part_name, "status": "cached", "model": str(out_obj)})
            continue
        if not base_obj.exists():
            summary["parts"].append({"part": part_name, "status": "skipped", "reason": "base_model_missing"})
            continue

        try:
            part_summary = _iterative_tsdf_part(
                obj=obj,
                part_name=part_name,
                part_model=part_model,
                base_obj=base_obj,
                out_obj=out_obj,
                work_root=work_root,
                method=method,
                k=k,
                args=args,
                o3d=o3d,
            )
        except Exception as e:
            err = traceback.format_exc()
            print(
                f"[TSDF-FAIL] method={method} object={obj.name} part={part_name} "
                f"part_model={part_model}: {e}\n{err}",
                flush=True,
            )
            part_summary = {
                "part": part_name,
                "part_model": part_model,
                "status": "failed",
                "reason": "iterative_tsdf_failed",
                "error": str(e),
                "traceback": err,
                "frames": 0,
                "model": "",
            }
        status = str(part_summary.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        summary["parts"].append(
            {
                **part_summary,
                "source_model": str(base_obj),
            }
        )

    summary["status_counts"] = status_counts
    total_parts = len(summary["parts"])
    usable_parts = status_counts.get("success", 0) + status_counts.get("cached", 0)
    failed_parts = status_counts.get("failed", 0)
    skipped_parts = status_counts.get("skipped", 0)
    print(
        f"[TSDF-SUMMARY] method={method} object={obj.name} parts={total_parts} "
        f"success={status_counts.get('success', 0)} cached={status_counts.get('cached', 0)} "
        f"failed={status_counts.get('failed', 0)} "
        f"skipped={skipped_parts}",
        flush=True,
    )
    write_json(method_object_dir(work_root, method, args.split, obj.name) / "summary.json", summary)
    if total_parts > 0 and usable_parts == 0:
        first_errors = [
            str(p.get("error", p.get("reason", "")))
            for p in summary["parts"]
            if p.get("status") == "failed"
        ]
        summary["status"] = "skipped_no_usable_parts"
        summary["skip_reason"] = "tsdf_mesh_empty"
        summary["first_errors"] = first_errors[:3]
        print(
            f"[TSDF-SKIP] method={method} object={obj.name} no usable parts; "
            f"status_counts={status_counts}; first_errors={first_errors[:3]}",
            flush=True,
        )
    return summary


def add_tsdf_args(parser: argparse.ArgumentParser) -> None:
    """
    注册 TSDF 后处理相关命令行参数。

    参数大致分为：TSDF 体素设置、候选帧/seed 选择、ICP refinement、尺度诊断、
    深度一致性、pose utility gate、pose delta 联合优化，以及失败处理策略。
    """
    parser.add_argument("--build-base-if-missing", action="store_true", help="Run the base method if shared cache is missing.")
    parser.add_argument("--voxel-length", type=float, default=0.005)
    parser.add_argument("--sdf-trunc", type=float, default=0.02)
    parser.add_argument("--depth-trunc", type=float, default=10.0)
    parser.add_argument(
        "--iter-tsdf-coarse-to-fine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try the requested fine TSDF first; if extraction is empty, retry with a coarser voxel/truncation schedule.",
    )
    parser.add_argument("--iter-tsdf-coarse-voxel-multiplier", type=float, default=2.0)
    parser.add_argument("--iter-tsdf-coarse-trunc-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--fail-on-empty-tsdf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the object when TSDF has no success/cached parts instead of silently skipping it.",
    )
    parser.add_argument("--iter-tsdf-max-frames", type=int, default=16, help="Maximum accepted frames per part; <=0 allows all candidates.")
    parser.add_argument(
        "--iter-tsdf-seed-attempts",
        type=int,
        default=0,
        help="Number of candidate seed frames to try before failing; <=0 tries all valid observations.",
    )
    parser.add_argument("--iter-tsdf-min-points", type=int, default=64, help="Minimum valid masked depth points for a candidate frame.")
    parser.add_argument("--iter-tsdf-min-view-angle", type=float, default=3.0, help="Reject frames that are too close to accepted views.")
    parser.add_argument("--iter-tsdf-max-view-angle", type=float, default=35.0, help="Reject frames whose nearest accepted view is too far in rotation.")
    parser.add_argument("--iter-tsdf-min-translation", type=float, default=0.002, help="Reject frames that are too close to accepted views in translation.")
    parser.add_argument("--iter-tsdf-max-translation", type=float, default=0.08, help="Reject frames whose nearest accepted view is too far in translation.")
    parser.add_argument("--iter-tsdf-refine-iters", type=int, default=15)
    parser.add_argument("--iter-tsdf-refine-points", type=int, default=2000)
    parser.add_argument("--iter-tsdf-mesh-samples", type=int, default=20000)
    parser.add_argument("--iter-tsdf-refine-max-dist", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-refine-trim-quantile", type=float, default=0.8)
    parser.add_argument("--iter-tsdf-refine-residual-thresh", type=float, default=0.02)
    parser.add_argument(
        "--iter-tsdf-auto-scale-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only inside TSDF, detect order-of-magnitude mesh/depth scale mismatch and scale the registration copy.",
    )
    parser.add_argument(
        "--iter-tsdf-mesh-scale-ratio-min",
        type=float,
        default=0.2,
        help="If mesh bbox diagonal / observed depth bbox diagonal is below this value, the mesh scale is considered abnormal.",
    )
    parser.add_argument(
        "--iter-tsdf-mesh-scale-ratio-max",
        type=float,
        default=5.0,
        help="If mesh bbox diagonal / observed depth bbox diagonal is above this value, the mesh scale is considered abnormal.",
    )
    parser.add_argument(
        "--iter-tsdf-seed-global-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build PCA/bounding-box pose hypotheses for the seed frame before strict refinement.",
    )
    parser.add_argument("--iter-tsdf-seed-max-hypotheses", type=int, default=32)
    parser.add_argument("--iter-tsdf-seed-verify-topk", type=int, default=5)
    parser.add_argument("--iter-tsdf-seed-coarse-iters", type=int, default=8)
    parser.add_argument("--iter-tsdf-seed-coarse-max-dist", type=float, default=0.15)
    parser.add_argument("--iter-tsdf-seed-coarse-trim-quantile", type=float, default=0.9)
    parser.add_argument(
        "--iter-tsdf-seed-max-prior-angle",
        type=float,
        default=180.0,
        help="Seed hypotheses may intentionally correct large raw-pose rotation errors; later frames still use --iter-tsdf-max-refine-angle.",
    )
    parser.add_argument(
        "--iter-tsdf-seed-max-prior-translation",
        type=float,
        default=0.3,
        help="Seed hypotheses may intentionally correct large raw-pose translation errors; later frames still use --iter-tsdf-max-refine-translation.",
    )
    parser.add_argument("--iter-tsdf-axis-symmetry-steps", type=int, default=8)
    parser.add_argument("--iter-tsdf-axis-symmetry-extent-tolerance", type=float, default=0.15)
    parser.add_argument(
        "--iter-tsdf-accept-initial-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accept the original cam_params pose when it already fits the mesh/depth, even if ICP refinement drifts.",
    )
    parser.add_argument(
        "--iter-tsdf-min-initial-inlier-ratio",
        type=float,
        default=0.1,
        help="Minimum inlier ratio required when accepting the original pose without ICP refinement.",
    )
    parser.add_argument("--iter-tsdf-refine-angle-eps", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-refine-trans-eps", type=float, default=1e-4)
    parser.add_argument("--iter-tsdf-max-refine-angle", type=float, default=20.0)
    parser.add_argument("--iter-tsdf-max-refine-translation", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-consistency-points", type=int, default=1500)
    parser.add_argument("--iter-tsdf-depth-residual-thresh", type=float, default=0.02)
    parser.add_argument("--iter-tsdf-min-valid-proj-ratio", type=float, default=0.05)
    parser.add_argument("--iter-tsdf-min-depth-inlier-ratio", type=float, default=0.5)
    parser.add_argument(
        "--iter-tsdf-min-overlap-ratio",
        type=float,
        default=0.03,
        help="Minimum existing-surface overlap ratio for information-gain consistency.",
    )
    parser.add_argument(
        "--iter-tsdf-accept-new-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accept a well-refined frame when it contributes enough previously unseen surface.",
    )
    parser.add_argument(
        "--iter-tsdf-min-new-surface-ratio",
        type=float,
        default=0.35,
        help="Minimum candidate visible points not covered by accepted frames for new-surface acceptance.",
    )
    parser.add_argument(
        "--iter-tsdf-min-new-surface-overlap-inlier",
        type=float,
        default=0.35,
        help="When a new-surface frame still overlaps old frames, require this overlap inlier ratio.",
    )
    parser.add_argument("--iter-tsdf-info-overlap-weight", type=float, default=1.0)
    parser.add_argument("--iter-tsdf-info-new-surface-weight", type=float, default=0.75)
    parser.add_argument("--iter-tsdf-info-depth-penalty", type=float, default=0.25)
    parser.add_argument("--iter-tsdf-redundancy-penalty", type=float, default=0.2)
    parser.add_argument(
        "--iter-tsdf-info-gain-overrides-view-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow high new-surface candidates to pass the old angle/translation gate.",
    )
    parser.add_argument("--iter-tsdf-view-gate-override-penalty", type=float, default=0.25)
    parser.add_argument(
        "--iter-tsdf-geometry-rollback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the best temporary TSDF state by observable geometry proxy and roll back the final output if needed.",
    )
    parser.add_argument("--iter-tsdf-geometry-coverage-weight", type=float, default=0.03)
    parser.add_argument("--iter-tsdf-final-rollback-margin", type=float, default=0.0)
    parser.add_argument(
        "--pose-utility-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each accepted TSDF candidate, validate whether the new mesh improves pose-oriented metrics.",
    )
    parser.add_argument(
        "--pose-utility-min-score-delta",
        type=float,
        default=-0.02,
        help="Minimum allowed new_score-old_score for accepting a fused mesh; negative allows small temporary drops.",
    )
    parser.add_argument(
        "--pose-utility-max-frame-drop",
        type=float,
        default=0.15,
        help="Maximum allowed drop in the worst per-frame utility score after fusion.",
    )
    parser.add_argument("--pose-utility-max-points", type=int, default=2048)
    parser.add_argument("--pose-utility-mesh-samples", type=int, default=8192)
    parser.add_argument("--pose-utility-depth-inlier-thresh", type=float, default=0.02)
    parser.add_argument("--pose-utility-projection-depth-thresh", type=float, default=0.03)
    parser.add_argument("--pose-utility-min-projected-points", type=int, default=64)
    parser.add_argument("--pose-utility-depth-weight", type=float, default=4.0)
    parser.add_argument("--pose-utility-inlier-weight", type=float, default=1.0)
    parser.add_argument("--pose-utility-mask-weight", type=float, default=1.0)
    parser.add_argument("--pose-utility-dt-weight", type=float, default=0.5)
    parser.add_argument("--pose-utility-dr-weight", type=float, default=0.01)
    parser.add_argument(
        "--pose-delta-opt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh accepted-frame ICP pose deltas against the new mesh and anchor them to the seed frame.",
    )
    parser.add_argument("--pose-delta-rounds", type=int, default=1)
    parser.add_argument(
        "--pose-delta-max-dt",
        type=float,
        default=0.0,
        help="Maximum accepted per-refresh translation delta in meters; <=0 reuses --iter-tsdf-max-refine-translation.",
    )
    parser.add_argument(
        "--pose-delta-max-dr-deg",
        type=float,
        default=0.0,
        help="Maximum accepted per-refresh rotation delta in degrees; <=0 reuses --iter-tsdf-max-refine-angle.",
    )
    parser.add_argument(
        "--pose-delta-refine-anchor",
        action="store_true",
        help="Also refine the seed/reference pose before applying the object-frame anchor offset.",
    )
