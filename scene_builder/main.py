import argparse
import concurrent.futures
import gc
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import stat
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from .config import parse_args
from .environment import (
    _add_ground_plane,
    _composite_object_mask,
    _composite_rgb_background,
    _configure_scene_lighting,
    _instance_background_seed,
    _make_render_material,
    _setup_scene_background,
    configure_ray_tracing,
)
from .geometry_views import (
    _dir_key,
    _link_name,
    _link_pose_matrix,
    _look_at,
    _safe_articulation_qlimits,
    _safe_articulation_qpos,
    _set_articulation_qpos,
    _small_joint_motion_qpos,
    angular_distance,
    diversity_bonus,
    generate_diverse_view_dirs,
    min_az_elev_distance,
    parse_bounding_box,
    perturb_view_dirs,
)
from .instance_parts import (
    InstanceInfo,
    PartUrdfInfo,
    SapienCameraBufferError,
    SkipInstanceError,
    build_link_layout,
    build_part_models,
    discover_instances,
    filter_part_layout,
    filter_part_layout_by_movable_joints,
    find_missing_mesh_files,
    parse_exclude_keywords,
    parse_mobility_links,
    parse_urdf_link_infos,
    part_folder_name,
    select_instances,
)
from .visibility_framing import (
    _camera_picture,
    choose_entity_segmentation_channel,
    mask_bbox_center_offset_ratio,
    mask_center_offset_ratio,
    mask_edge_margin_ratio,
    mask_framing_ok,
    mask_is_complete_and_sized,
    part_visibility_from_pixel_counts,
    view_quality_score,
)


def is_missing_asset_error(message: str) -> bool:
    msg = message.lower()
    return (
        "file not found" in msg
        or "no such file" in msg
        or ("textured_objs" in msg and ".obj" in msg)
    )


def is_render_view_selection_error(message: str) -> bool:
    msg = message.lower()
    return (
        "no valid render views found" in msg
        or "failed final framing check" in msg
        or "became invalid after joint motion" in msg
        or "part mask pixels below" in msg
    )


def is_renderer_resource_error(message: str) -> bool:
    msg = message.lower()
    return (
        "cannot create image" in msg
        or "user allocator returned null" in msg
        or "out of memory" in msg
        or "vk_error_out_of_device_memory" in msg
        or "vk_error_out_of_host_memory" in msg
    )


def write_skip_reason(out_dir: Path, reason: str, details: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "skipped", "reason": reason, "details": details}
    with (out_dir / "skip_reason.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_failure_reason(out_dir: Path, reason: str, details: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "failed", "reason": reason, "details": details}
    with (out_dir / "failure_reason.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _rmtree_onerror(func: object, path: str, exc_info: object) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def robust_rmtree(path: Path, retries: int = 8, delay: float = 0.25) -> None:
    if not path.exists():
        return
    last_error: Optional[Exception] = None
    for attempt in range(max(1, int(retries))):
        gc.collect()
        try:
            shutil.rmtree(path, onerror=_rmtree_onerror)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last_error = e
            time.sleep(float(delay) * (attempt + 1))
    if path.exists():
        raise last_error or OSError(f"failed to remove directory: {path}")


def clear_render_outputs(out_dir: Path) -> None:
    for name in ("rgb", "depth", "masks", "object_mask", "cam_params"):
        path = out_dir / name
        if path.exists():
            robust_rmtree(path)


def _safe_call_method(obj: object, method_name: str, *args: object) -> None:
    if obj is None:
        return
    method = getattr(obj, method_name, None)
    if not callable(method):
        return
    try:
        method(*args)
    except Exception:
        pass


def cleanup_sapien_scene(scene: object, camera: object = None, articulation: object = None) -> None:
    """Release per-instance SAPIEN resources before rendering the next object."""
    if scene is None:
        gc.collect()
        return

    if articulation is not None:
        _safe_call_method(scene, "remove_articulation", articulation)

    if camera is not None:
        _safe_call_method(scene, "remove_camera", camera)

    get_actors = getattr(scene, "get_all_actors", None)
    if not callable(get_actors):
        get_actors = getattr(scene, "get_actors", None)
    if callable(get_actors):
        try:
            for actor in list(get_actors()):
                _safe_call_method(scene, "remove_actor", actor)
        except Exception:
            pass

    _safe_call_method(scene, "update_render")
    gc.collect()


def prepare_camera_probe(
    resource_tracker: Optional[Dict[str, object]],
    phase: str,
    probe_counter: List[int],
    collect_every: int = 16,
) -> None:
    if resource_tracker is not None:
        resource_tracker["phase"] = phase
    probe_counter[0] += 1
    if probe_counter[0] % max(1, int(collect_every)) == 0:
        gc.collect()


def part_mask_stats(mask: np.ndarray, min_part_pixels: int) -> Dict[str, object]:
    mask_bool = mask.astype(bool)
    pixels = int(mask_bool.sum())
    h, w = mask_bool.shape
    stats: Dict[str, object] = {
        "visible": pixels >= int(min_part_pixels),
        "pixels": pixels,
        "coverage": float(pixels) / float(max(1, h * w)),
        "bbox_xyxy": None,
        "center_offset": 1.0,
        "bbox_center_offset": 1.0,
        "edge_margin": 0.0,
    }
    if pixels <= 0:
        return stats
    ys, xs = np.where(mask_bool)
    stats["bbox_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    stats["center_offset"] = float(mask_center_offset_ratio(mask_bool))
    stats["bbox_center_offset"] = float(mask_bbox_center_offset_ratio(mask_bool))
    stats["edge_margin"] = float(mask_edge_margin_ratio(mask_bool))
    return stats


def _rpy_xyz_to_matrix(xyz: Optional[List[float]], rpy: Optional[List[float]]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    if xyz is not None and len(xyz) >= 3:
        mat[:3, 3] = np.asarray(xyz[:3], dtype=np.float32)
    if rpy is None or len(rpy) < 3:
        return mat
    roll, pitch, yaw = [float(v) for v in rpy[:3]]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    mat[:3, :3] = rz @ ry @ rx
    return mat


def _add_mesh_visual_compatible(
    builder: object,
    sapien_module: object,
    mesh_path: Path,
    scale: Optional[List[float]],
) -> None:
    add_visual = getattr(builder, "add_visual_from_file", None)
    if not callable(add_visual):
        add_visual = getattr(builder, "add_mesh_visual", None)
    if not callable(add_visual):
        raise RuntimeError("SAPIEN actor builder does not expose add_visual_from_file/add_mesh_visual.")

    pose = sapien_module.Pose()
    scale_arg = [float(v) for v in scale[:3]] if scale is not None and len(scale) >= 3 else [1.0, 1.0, 1.0]
    attempts = (
        lambda: add_visual(str(mesh_path), pose=pose, scale=scale_arg),
        lambda: add_visual(filename=str(mesh_path), pose=pose, scale=scale_arg),
        lambda: add_visual(str(mesh_path), pose, scale_arg),
        lambda: add_visual(str(mesh_path)),
    )
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            attempt()
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to add part-only mesh visual {mesh_path}: {last_error}") from last_error


def _build_part_only_probe(
    sapien_module: object,
    instance: InstanceInfo,
    part_layout: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
    width: int,
    height: int,
    fov_deg: float,
) -> Optional[Dict[str, object]]:
    scene = sapien_module.Scene()
    scene.set_timestep(1 / 100.0)
    scene.set_ambient_light([1.0, 1.0, 1.0])
    camera = scene.add_camera(
        "part_only_camera",
        width=width,
        height=height,
        fovy=np.deg2rad(fov_deg),
        near=0.1,
        far=100.0,
    )
    actors_by_part: Dict[int, List[object]] = {}
    visual_local_by_part: Dict[int, np.ndarray] = {}
    for part_id in part_layout.keys():
        info = urdf_part_infos.get(part_id)
        if info is None or not info.mesh_relpaths:
            continue
        actors: List[object] = []
        for rel in info.mesh_relpaths:
            mesh_path = (instance.instance_dir / rel).resolve()
            if not mesh_path.exists():
                continue
            builder = scene.create_actor_builder()
            _add_mesh_visual_compatible(builder, sapien_module, mesh_path, info.mesh_scale)
            try:
                actor = builder.build_static(name=f"part_only_{part_id}")
            except TypeError:
                actor = builder.build_static(f"part_only_{part_id}")
            actors.append(actor)
        if actors:
            actors_by_part[part_id] = actors
            visual_local_by_part[part_id] = _rpy_xyz_to_matrix(info.visual_origin_xyz, info.visual_origin_rpy)
    if not actors_by_part:
        cleanup_sapien_scene(scene, camera=camera)
        return None
    return {
        "scene": scene,
        "camera": camera,
        "actors_by_part": actors_by_part,
        "visual_local_by_part": visual_local_by_part,
    }


def _set_actor_pose(actor: object, sapien_module: object, pose44: np.ndarray) -> None:
    try:
        pose = sapien_module.Pose.from_transformation_matrix(pose44)
    except Exception:
        try:
            pose = sapien_module.Pose(pose44)
        except Exception:
            pose = sapien_module.Pose(pose44[:3, 3])
    set_pose = getattr(actor, "set_pose", None)
    if callable(set_pose):
        set_pose(pose)
        return
    setattr(actor, "pose", pose)


def _part_only_projected_pixels(
    probe: Dict[str, object],
    sapien_module: object,
    part_id: int,
    link_pose44: np.ndarray,
    camera_pose44: np.ndarray,
) -> int:
    scene = probe["scene"]
    camera = probe["camera"]
    actors_by_part = probe["actors_by_part"]
    visual_local_by_part = probe["visual_local_by_part"]
    actors = actors_by_part.get(part_id, [])
    if not actors:
        return 0

    hidden_pose = np.eye(4, dtype=np.float32)
    hidden_pose[:3, 3] = np.asarray([1000.0, 1000.0, 1000.0], dtype=np.float32)
    for pid, pid_actors in actors_by_part.items():
        pose44 = link_pose44 @ visual_local_by_part.get(pid, np.eye(4, dtype=np.float32)) if pid == part_id else hidden_pose
        for actor in pid_actors:
            _set_actor_pose(actor, sapien_module, pose44)
    camera.entity.set_pose(sapien_module.Pose(camera_pose44))
    scene.step()
    scene.update_render()
    camera.take_picture()
    seg = _camera_picture(camera, "Segmentation")
    if seg.ndim < 3:
        return 0
    return int(np.count_nonzero(seg[..., 1] if seg.shape[2] > 1 else seg[..., 0]))


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        pass
    shutil.copy2(src, dst)
def _render_instance_impl(
    instance: InstanceInfo,
    out_dir: Path,
    part_layout: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
    n_views: int,
    width: int,
    height: int,
    fov_deg: float,
    radius_scale: float,
    target_max_extent: float,
    min_object_scale: float,
    max_object_scale: float,
    min_object_coverage: float,
    max_object_coverage: float,
    min_part_mask_pixels: int,
    min_part_mask_coverage: float,
    require_all_part_visible: bool,
    view_candidate_multiplier: int,
    part_occlusion_check: bool,
    min_part_visible_ratio: float,
    use_rt: bool,
    render_ground: bool,
    rt_camera_shader: bool,
    rt_spp: int,
    rt_path_depth: int,
    rt_denoiser: str,
    background_mode: str,
    background_variants: int,
    background_seed: int,
    joint_motion: bool,
    joint_motion_fraction: float,
    joint_motion_max_delta: float,
    _resource_tracker: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    try:
        import sapien  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "sapien is required for rendering. Install sapien or run with --skip-render."
        ) from e

    background_mode = str(background_mode).lower().strip()
    background_variants = max(1, int(background_variants))
    if background_mode in {"plain", "scene"}:
        background_variants = 1
    per_instance_bg_seed = _instance_background_seed(instance.instance_name, background_seed)
    use_scene_background = background_mode == "scene"
    rt_disabled_for_scene_background = False
    if use_scene_background and use_rt:
        # Some SAPIEN builds cannot export Position/Segmentation buffers under
        # ray-tracing scene backgrounds. Those buffers are required for depth,
        # masks, and per-part cam_params, so prefer a valid dataset export.
        use_rt = False
        rt_disabled_for_scene_background = True
    has_rendered_background_geometry = bool(render_ground or use_scene_background)
    scene_background_info: Dict[str, object] = {}

    if use_rt:
        if _resource_tracker is not None:
            _resource_tracker["phase"] = "configure_ray_tracing"
        configure_ray_tracing(
            sapien_module=sapien,
            rt_camera_shader=rt_camera_shader,
            rt_spp=rt_spp,
            rt_path_depth=rt_path_depth,
            rt_denoiser=rt_denoiser,
        )

    if _resource_tracker is not None:
        _resource_tracker["phase"] = "create_scene"
    scene = sapien.Scene()
    if _resource_tracker is not None:
        _resource_tracker["scene"] = scene
    scene.set_timestep(1 / 100.0)
    lighting_info = _configure_scene_lighting(
        scene,
        seed=per_instance_bg_seed,
        rich_scene=use_scene_background,
    )

    if _resource_tracker is not None:
        _resource_tracker["phase"] = "add_camera"
    camera = scene.add_camera(
        "camera",
        width=width,
        height=height,
        fovy=np.deg2rad(fov_deg),
        near=0.1,
        far=100.0,
    )
    if _resource_tracker is not None:
        _resource_tracker["camera"] = camera

    if _resource_tracker is not None:
        _resource_tracker["phase"] = "load_urdf"
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    bbox = parse_bounding_box(instance.instance_dir)
    object_scale = 1.0
    if bbox is not None:
        bmin, bmax = bbox
        extent = bmax - bmin
        max_extent = float(np.max(extent))
        if max_extent > 1e-6:
            scale_lo = max(1e-6, float(min_object_scale))
            scale_hi = max(scale_lo, float(max_object_scale))
            object_scale = float(np.clip(target_max_extent / max_extent, scale_lo, scale_hi))
    loader.scale = object_scale
    art = loader.load(str(instance.urdf_path))
    if art is None:
        raise RuntimeError(f"failed to load urdf: {instance.urdf_path}")
    if _resource_tracker is not None:
        _resource_tracker["articulation"] = art
    base_qpos = _safe_articulation_qpos(art)
    qlimits = _safe_articulation_qlimits(art, len(base_qpos) if base_qpos is not None else 0)
    if bbox is not None:
        bmin0, bmax0 = bbox
        extent0 = bmax0 - bmin0
        extent_z_raw = float(extent0[2])
    else:
        bmin0 = np.array([-0.25, -0.25, 0.0], dtype=np.float32)
        bmax0 = np.array([0.25, 0.25, 0.5], dtype=np.float32)
        extent_z_raw = 0.5
    # Lift more aggressively to avoid sinking/intersection with optional scene geometry.
    object_lift = max(0.05 * object_scale, 0.20 * extent_z_raw * object_scale)
    art.set_pose(sapien.Pose([0.0, 0.0, object_lift]))
    for link in art.get_links():
        link.set_disable_gravity(True)

    scaled_bmin = np.asarray(bmin0, dtype=np.float32) * float(object_scale)
    scaled_bmax = np.asarray(bmax0, dtype=np.float32) * float(object_scale)
    object_bottom_z = float(scaled_bmin[2] + object_lift)
    ground_clearance = max(
        0.025,
        min(0.06, 0.04 * max(1e-6, float(extent_z_raw) * float(object_scale))),
    )
    safe_ground_altitude = object_bottom_z - ground_clearance

    if use_scene_background:
        extent_for_scene = (scaled_bmax - scaled_bmin).astype(np.float32)
        radius_for_scene = float(np.linalg.norm(extent_for_scene) * 0.5)
        scene_background_info = _setup_scene_background(
            scene=scene,
            sapien_module=sapien,
            seed=per_instance_bg_seed,
            extent=extent_for_scene,
            radius_obj=radius_for_scene,
            ground_altitude=safe_ground_altitude,
            ground_clearance=ground_clearance,
        )
    elif render_ground:
        render_ground_altitude = safe_ground_altitude
        if use_rt:
            ground_material = _make_render_material(
                sapien,
                (np.array([202, 164, 114, 256], dtype=np.float32) / 256.0).tolist(),
                roughness=0.75,
                specular=0.5,
            )
            _add_ground_plane(scene, sapien, altitude=render_ground_altitude, material=ground_material)
        else:
            _add_ground_plane(scene, sapien, altitude=render_ground_altitude)
        scene_background_info = {
            "name": "ground",
            "ground_altitude": float(render_ground_altitude),
            "ground_clearance": float(ground_clearance),
        }

    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    mask_root = out_dir / "masks"
    object_mask_dir = out_dir / "object_mask"
    model_root = out_dir / "models"
    cam_dir = out_dir / "cam_params"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    object_mask_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    for part_id, part_name in part_layout.items():
        sub = part_folder_name(part_id, part_name)
        (mask_root / sub).mkdir(parents=True, exist_ok=True)
        (model_root / sub).mkdir(parents=True, exist_ok=True)
        (cam_dir / sub).mkdir(parents=True, exist_ok=True)

    id_to_entity = {int(e.per_scene_id): str(e.name) for e in scene.get_entities()}
    linkname_to_entity_ids: Dict[str, List[int]] = {}
    for eid, name in id_to_entity.items():
        if not name:
            continue
        linkname_to_entity_ids.setdefault(name, []).append(eid)

    part_to_segmentation_ids: Dict[int, List[int]] = {}
    part_to_entity_ids: Dict[int, List[int]] = {}
    for part_id, info in urdf_part_infos.items():
        if part_id not in part_layout:
            continue
        part_to_entity_ids[part_id] = linkname_to_entity_ids.get(info.link_name, [])
        part_to_segmentation_ids[part_id] = list(info.visual_part_ids)
    missing = [
        pid
        for pid in part_layout.keys()
        if pid not in part_to_entity_ids
        or not part_to_entity_ids[pid]
        or pid not in part_to_segmentation_ids
        or not part_to_segmentation_ids[pid]
    ]
    if missing:
        raise RuntimeError(
            f"Missing URDF/link or visual segmentation mapping for part ids: {missing}. "
            "Cannot build mask correspondence."
        )
    linkname_to_link = {_link_name(link): link for link in art.get_links()}
    object_entity_ids = {
        eid
        for link_name in linkname_to_link.keys()
        for eid in linkname_to_entity_ids.get(link_name, [])
    }
    if not object_entity_ids:
        object_entity_ids = {eid for entity_ids in part_to_entity_ids.values() for eid in entity_ids}
    object_entity_ids_np = np.asarray(sorted(object_entity_ids), dtype=np.int32)
    union_segmentation_ids = np.asarray(
        sorted({seg_id for seg_ids in part_to_segmentation_ids.values() for seg_id in seg_ids}),
        dtype=np.int32,
    )
    part_segmentation_ids_np: Dict[int, np.ndarray] = {
        part_id: np.asarray(sorted(set(seg_ids)), dtype=np.int32)
        for part_id, seg_ids in part_to_segmentation_ids.items()
    }
    part_entity_ids_np: Dict[int, np.ndarray] = {
        part_id: np.asarray(sorted(set(entity_ids)), dtype=np.int32)
        for part_id, entity_ids in part_to_entity_ids.items()
    }

    def object_mask_from_segmentation(seg: np.ndarray, entity_channel: int) -> np.ndarray:
        if (
            seg.ndim == 3
            and seg.shape[2] > int(entity_channel)
            and object_entity_ids_np.size > 0
        ):
            return np.isin(seg[..., int(entity_channel)].astype(np.int32), object_entity_ids_np)
        return np.zeros(seg.shape[:2], dtype=bool)

    def _segmentation_channel(seg: np.ndarray, channel: Optional[int]) -> np.ndarray:
        if seg.ndim == 3 and channel is not None and seg.shape[2] > int(channel):
            return seg[..., int(channel)].astype(np.int32)
        return np.zeros(seg.shape[:2], dtype=np.int32)

    def _masks_from_part_ids(
        seg_channel: np.ndarray,
        ids_by_part: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        masks: Dict[int, np.ndarray] = {}
        pixels: Dict[int, int] = {}
        for part_id in part_layout.keys():
            ids = ids_by_part.get(part_id)
            if ids is None or ids.size == 0:
                mask = np.zeros_like(seg_channel, dtype=np.uint8)
            else:
                mask = np.isin(seg_channel, ids).astype(np.uint8) * 255
            masks[part_id] = mask
            pixels[part_id] = int(np.count_nonzero(mask))
        return masks, pixels

    def _part_pixels_ok(part_pixels: Dict[int, int], require_all_parts: bool) -> bool:
        if not part_pixels:
            return False
        threshold = int(min_effective_part_pixels)
        if bool(require_all_parts):
            return all(int(pix) >= threshold for pix in part_pixels.values())
        return max(int(pix) for pix in part_pixels.values()) >= threshold

    def _part_mask_source_score(part_pixels: Dict[int, int]) -> Tuple[int, int]:
        threshold = int(min_effective_part_pixels)
        visible_count = sum(1 for pix in part_pixels.values() if int(pix) >= threshold)
        total_pixels = sum(int(pix) for pix in part_pixels.values())
        return visible_count, total_pixels

    def _choose_visual_segmentation_channel(seg: np.ndarray, entity_channel: Optional[int]) -> int:
        if seg.ndim != 3 or seg.shape[2] < 2:
            return 0
        valid_ids = union_segmentation_ids.astype(np.int64)
        if valid_ids.size == 0:
            return 0
        scored: List[Tuple[int, int, int]] = []
        for ch in range(min(2, seg.shape[2])):
            channel = seg[..., ch].astype(np.int64)
            pixels = int(np.isin(channel, valid_ids).sum())
            overlap = sum(1 for v in np.unique(channel).tolist() if int(v) in set(valid_ids.tolist()))
            scored.append((pixels, overlap, ch))
        non_entity = [
            item for item in scored
            if item[2] != int(entity_channel) and item[0] > 0
        ]
        if non_entity:
            non_entity.sort(reverse=True)
            return non_entity[0][2]
        scored.sort(reverse=True)
        return scored[0][2]

    def part_masks_from_segmentation(
        seg: np.ndarray,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int], str, int]:
        nonlocal entity_seg_channel, visual_seg_channel
        if entity_seg_channel is None:
            entity_seg_channel = choose_entity_segmentation_channel(
                seg,
                set(object_entity_ids_np.tolist()),
            )
        if visual_seg_channel is None:
            visual_seg_channel = _choose_visual_segmentation_channel(seg, entity_seg_channel)

        entity_channel = _segmentation_channel(seg, entity_seg_channel)
        entity_masks, entity_pixels = _masks_from_part_ids(entity_channel, part_entity_ids_np)

        entity_score = _part_mask_source_score(entity_pixels)
        if entity_score[1] > 0:
            return entity_masks, entity_pixels, "entity", int(entity_seg_channel)

        visual_channel = _segmentation_channel(seg, visual_seg_channel)
        visual_masks, visual_pixels = _masks_from_part_ids(visual_channel, part_segmentation_ids_np)
        visual_score = _part_mask_source_score(visual_pixels)
        if visual_score[1] > 0:
            return visual_masks, visual_pixels, "visual_fallback", int(visual_seg_channel)
        return entity_masks, entity_pixels, "entity", int(entity_seg_channel)

    if bbox is not None:
        bmin, bmax = bbox
        center = ((bmin + bmax) * 0.5 * object_scale).astype(np.float32)
        extent = ((bmax - bmin) * object_scale).astype(np.float32)
        radius_obj = float(np.linalg.norm(extent) * 0.5)
    else:
        center = np.array([0.0, 0.0, 0.5 * object_scale], dtype=np.float32)
        extent = np.array([0.6, 0.6, 0.6], dtype=np.float32) * object_scale
        radius_obj = float(np.linalg.norm(extent) * 0.5)

    # Raise camera target slightly to keep full object above lower image boundary.
    center[2] += (0.12 * float(extent[2]) + object_lift)

    fovy = np.deg2rad(fov_deg)
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * (float(width) / float(height)))
    min_half_fov = min(fovx, fovy) * 0.5
    base_distance = (radius_obj / max(1e-6, math.sin(min_half_fov))) * radius_scale
    base_distance = max(base_distance, radius_obj * 2.0)

    target = center
    candidate_count = max(n_views * max(1, int(view_candidate_multiplier)), 160)
    candidate_dirs = generate_diverse_view_dirs(candidate_count)
    target_ratio = (min_object_coverage + max_object_coverage) * 0.5
    min_effective_part_pixels = max(
        int(min_part_mask_pixels),
        int(width * height * float(min_part_mask_coverage)),
    )

    # Calibrate distance with staged constraints to avoid all-invalid failure.
    test_dirs = candidate_dirs[: min(16, len(candidate_dirs))]
    current_distance = base_distance
    entity_seg_channel: Optional[int] = None
    visual_seg_channel: Optional[int] = None
    probe_counter = [0]
    staged_thresholds = [
        (min_object_coverage, max_object_coverage),
        (min_object_coverage * 0.75, min(0.95, max_object_coverage * 1.2)),
        (min_object_coverage * 0.55, min(0.98, max_object_coverage * 1.4)),
    ]
    active_min_cov = min_object_coverage
    active_max_cov = max_object_coverage
    for min_cov, max_cov in staged_thresholds:
        active_min_cov, active_max_cov = min_cov, max_cov
        found_stage = False
        for _ in range(10):
            valid_count = 0
            ratios: List[float] = []
            edge_touches = 0
            for d in test_dirs:
                prepare_camera_probe(
                    _resource_tracker,
                    "distance_calibration_take_picture",
                    probe_counter,
                )
                cam_pos = target + d * current_distance
                pose44 = _look_at(cam_pos, target)
                camera.entity.set_pose(sapien.Pose(pose44))
                scene.step()
                scene.update_render()
                camera.take_picture()
                seg = _camera_picture(camera, "Segmentation")
                if entity_seg_channel is None:
                    entity_seg_channel = choose_entity_segmentation_channel(
                        seg,
                        set(object_entity_ids_np.tolist()),
                    )
                if visual_seg_channel is None:
                    visual_seg_channel = _choose_visual_segmentation_channel(seg, entity_seg_channel)
                part_masks, _, _, _ = part_masks_from_segmentation(seg)
                part_union_mask = np.zeros(seg.shape[:2], dtype=bool)
                for part_mask in part_masks.values():
                    part_union_mask |= part_mask.astype(bool)
                object_mask_for_framing = object_mask_from_segmentation(seg, entity_seg_channel)
                if int(object_mask_for_framing.sum()) == 0 and int(part_union_mask.sum()) > 0:
                    object_mask_for_framing = part_union_mask
                if int(object_mask_for_framing.sum()) == 0:
                    if has_rendered_background_geometry:
                        ratios.append(0.0)
                        continue
                    # Fallback when segmentation ids are unreliable in this build/view.
                    position = _camera_picture(camera, "Position")
                    depth_mask = (-position[..., 2]) > 0
                    if int(depth_mask.sum()) == 0:
                        ratios.append(0.0)
                        continue
                    object_mask_for_framing = depth_mask
                h, w = object_mask_for_framing.shape
                edge_touch = (
                    object_mask_for_framing[0, :].any()
                    or object_mask_for_framing[h - 1, :].any()
                    or object_mask_for_framing[:, 0].any()
                    or object_mask_for_framing[:, w - 1].any()
                )
                if edge_touch:
                    edge_touches += 1
                ratio = float(object_mask_for_framing.sum()) / float(h * w)
                ratios.append(ratio)
                if (not edge_touch) and (min_cov <= ratio <= max_cov):
                    valid_count += 1

            if valid_count >= max(2, min(6, len(test_dirs) // 3)):
                found_stage = True
                break

            mean_ratio = float(np.mean(ratios)) if ratios else 0.0
            if edge_touches > len(test_dirs) * 0.3 or mean_ratio > max_cov:
                current_distance *= 1.18
            elif mean_ratio < min_cov and mean_ratio > 1e-6:
                current_distance *= 0.92
            else:
                current_distance *= 1.08
        if found_stage:
            break

    camera_poses: Dict[str, List[float]] = {}
    accepted_dirs: List[np.ndarray] = []
    fallback_scored_dirs: List[Tuple[float, np.ndarray, float]] = []
    relaxed_scored_dirs: List[Tuple[float, np.ndarray, float]] = []
    visible_seed_dirs: List[np.ndarray] = []
    accepted_dir_keys: Set[Tuple[int, int, int]] = set()
    accepted_distances: Dict[Tuple[int, int, int], float] = {}
    min_separation = math.radians(12.0)
    min_az_separation = math.radians(16.0)
    min_elev_separation = math.radians(10.0)
    max_center_offset = 0.30
    min_edge_margin = 0.025
    distance_factors = (0.82, 0.92, 1.0, 1.10, 1.22, 1.38, 1.55)
    for d in candidate_dirs:
        if len(accepted_dirs) >= n_views:
            break
        best: Optional[Tuple[np.ndarray, float, float, bool, bool, float]] = None
        for fac in distance_factors:
            prepare_camera_probe(
                _resource_tracker,
                "view_candidate_take_picture",
                probe_counter,
            )
            dist = current_distance * fac
            cam_pos = target + d * dist
            pose44 = _look_at(cam_pos, target)
            camera.entity.set_pose(sapien.Pose(pose44))
            scene.step()
            scene.update_render()
            camera.take_picture()
            seg = _camera_picture(camera, "Segmentation")
            if entity_seg_channel is None:
                entity_seg_channel = choose_entity_segmentation_channel(
                    seg,
                    set(object_entity_ids_np.tolist()),
                )
            if visual_seg_channel is None:
                visual_seg_channel = _choose_visual_segmentation_channel(seg, entity_seg_channel)
            part_masks, part_areas, _, _ = part_masks_from_segmentation(seg)
            part_union_mask = np.zeros(seg.shape[:2], dtype=bool)
            for part_mask in part_masks.values():
                part_union_mask |= part_mask.astype(bool)
            object_mask_for_framing = object_mask_from_segmentation(seg, entity_seg_channel)
            object_seg_reliable = int(object_mask_for_framing.sum()) > 0
            if int(object_mask_for_framing.sum()) <= 0 and int(part_union_mask.sum()) > 0:
                object_mask_for_framing = part_union_mask
            part_seg_reliable = sum(int(pix) for pix in part_areas.values()) > 0
            if int(object_mask_for_framing.sum()) <= 0:
                if has_rendered_background_geometry:
                    continue
                # Fallback when segmentation ids are unreliable in this build/view.
                position = _camera_picture(camera, "Position")
                depth_mask = (-position[..., 2]) > 0
                if int(depth_mask.sum()) <= 0:
                    continue
                object_mask_for_framing = depth_mask
            ratio = float(object_mask_for_framing.sum()) / float(
                object_mask_for_framing.shape[0] * object_mask_for_framing.shape[1]
            )
            center_offset = mask_center_offset_ratio(object_mask_for_framing)
            if object_seg_reliable:
                ok, _ = mask_is_complete_and_sized(
                    object_mask_for_framing,
                    min_ratio=active_min_cov,
                    max_ratio=active_max_cov,
                )
            else:
                ok, _ = mask_is_complete_and_sized(
                    object_mask_for_framing,
                    min_ratio=active_min_cov * 0.75,
                    max_ratio=min(0.98, active_max_cov * 1.35),
                )
            has_part = part_seg_reliable and _part_pixels_ok(
                part_areas,
                require_all_parts=bool(require_all_part_visible),
            )
            framing_ok, _ = mask_framing_ok(
                mask=object_mask_for_framing,
                ratio=ratio,
                min_ratio=active_min_cov,
                max_ratio=active_max_cov,
                max_center_offset=max_center_offset,
                min_edge_margin=min_edge_margin,
            )
            valid = bool(ok and has_part and framing_ok)
            candidate_score = view_quality_score(object_mask_for_framing, ratio, target_ratio)
            if valid:
                if best is None or not best[3] or candidate_score > view_quality_score(best[0], best[1], target_ratio):
                    best = (object_mask_for_framing, ratio, center_offset, True, bool(has_part), float(dist))
            elif best is None or (not best[3] and candidate_score > view_quality_score(best[0], best[1], target_ratio)):
                best = (object_mask_for_framing, ratio, center_offset, False, bool(has_part), float(dist))

        if best is None:
            continue
        union_mask, ratio, center_offset, is_valid, has_part, best_dist = best
        score = view_quality_score(union_mask, ratio, target_ratio)
        if has_part and is_valid:
            visible_seed_dirs.append(d)
            score += 0.25 * diversity_bonus(d, accepted_dirs)
            fallback_scored_dirs.append((score, d, best_dist))
        # Keep a softer candidate pool to avoid hard failure on difficult objects.
        if has_part and is_valid:
            relaxed_score = score - 0.15 * abs(ratio - target_ratio)
            relaxed_scored_dirs.append((relaxed_score, d, best_dist))
        if not is_valid:
            continue
        if accepted_dirs:
            min_ang = diversity_bonus(d, accepted_dirs)
            min_az, min_elev = min_az_elev_distance(d, accepted_dirs)
            if (
                min_ang < min_separation
                and min_az < min_az_separation
                and min_elev < min_elev_separation
            ):
                continue
        accepted_dirs.append(d)
        key = _dir_key(d)
        accepted_dir_keys.add(key)
        accepted_distances[key] = float(best_dist)

    if len(accepted_dirs) < n_views:
        fallback_scored_dirs.sort(key=lambda x: x[0], reverse=True)
        for _, d, dist in fallback_scored_dirs:
            if len(accepted_dirs) >= n_views:
                break
            if _dir_key(d) in accepted_dir_keys:
                continue
            if accepted_dirs:
                min_ang = diversity_bonus(d, accepted_dirs)
                min_az, min_elev = min_az_elev_distance(d, accepted_dirs)
                if (
                    min_ang < min_separation * 0.5
                    and min_az < min_az_separation * 0.7
                    and min_elev < min_elev_separation * 0.7
                ):
                    continue
            accepted_dirs.append(d)
            key = _dir_key(d)
            accepted_dir_keys.add(key)
            accepted_distances[key] = float(dist)

    if len(accepted_dirs) < n_views:
        relaxed_scored_dirs.sort(key=lambda x: x[0], reverse=True)
        for _, d, dist in relaxed_scored_dirs:
            if len(accepted_dirs) >= n_views:
                break
            if _dir_key(d) in accepted_dir_keys:
                continue
            if accepted_dirs:
                min_ang = diversity_bonus(d, accepted_dirs)
                min_az, min_elev = min_az_elev_distance(d, accepted_dirs)
                if (
                    min_ang < min_separation * 0.35
                    and min_az < min_az_separation * 0.5
                    and min_elev < min_elev_separation * 0.5
                ):
                    continue
            accepted_dirs.append(d)
            key = _dir_key(d)
            accepted_dir_keys.add(key)
            accepted_distances[key] = float(dist)

    if len(accepted_dirs) < n_views:
        seed_dirs = accepted_dirs or visible_seed_dirs or [d for _, d, _ in fallback_scored_dirs]
        for seed in list(seed_dirs):
            if len(accepted_dirs) >= n_views:
                break
            for d in perturb_view_dirs(seed):
                if len(accepted_dirs) >= n_views:
                    break
                prepare_camera_probe(
                    _resource_tracker,
                    "perturb_candidate_take_picture",
                    probe_counter,
                )
                if accepted_dirs:
                    min_ang = diversity_bonus(d, accepted_dirs)
                    min_az, min_elev = min_az_elev_distance(d, accepted_dirs)
                    if (
                        min_ang < min_separation * 0.5
                        and min_az < min_az_separation * 0.7
                        and min_elev < min_elev_separation * 0.7
                    ):
                        continue
                cam_pos = target + d * current_distance
                pose44 = _look_at(cam_pos, target)
                camera.entity.set_pose(sapien.Pose(pose44))
                scene.step()
                scene.update_render()
                camera.take_picture()
                seg = _camera_picture(camera, "Segmentation")
                if entity_seg_channel is None:
                    entity_seg_channel = choose_entity_segmentation_channel(
                        seg,
                        set(object_entity_ids_np.tolist()),
                    )
                if visual_seg_channel is None:
                    visual_seg_channel = _choose_visual_segmentation_channel(seg, entity_seg_channel)
                part_masks, part_areas, _, _ = part_masks_from_segmentation(seg)
                part_union_mask = np.zeros(seg.shape[:2], dtype=bool)
                for part_mask in part_masks.values():
                    part_union_mask |= part_mask.astype(bool)
                object_mask_for_framing = object_mask_from_segmentation(seg, entity_seg_channel)
                object_seg_reliable = int(object_mask_for_framing.sum()) > 0
                if int(object_mask_for_framing.sum()) <= 0 and int(part_union_mask.sum()) > 0:
                    object_mask_for_framing = part_union_mask
                part_seg_reliable = sum(int(pix) for pix in part_areas.values()) > 0
                if int(object_mask_for_framing.sum()) <= 0:
                    if has_rendered_background_geometry:
                        continue
                    # Fallback when segmentation ids are unreliable in this build/view.
                    position = _camera_picture(camera, "Position")
                    depth_mask = (-position[..., 2]) > 0
                    if int(depth_mask.sum()) <= 0:
                        continue
                    object_mask_for_framing = depth_mask
                ratio = float(object_mask_for_framing.sum()) / float(
                    object_mask_for_framing.shape[0] * object_mask_for_framing.shape[1]
                )
                framing_ok, _ = mask_framing_ok(
                    mask=object_mask_for_framing,
                    ratio=ratio,
                    min_ratio=active_min_cov,
                    max_ratio=active_max_cov,
                    max_center_offset=max_center_offset,
                    min_edge_margin=min_edge_margin,
                )
                if not framing_ok:
                    continue
                if object_seg_reliable:
                    ok, _ = mask_is_complete_and_sized(
                        object_mask_for_framing,
                        min_ratio=active_min_cov,
                        max_ratio=active_max_cov,
                    )
                else:
                    ok, _ = mask_is_complete_and_sized(
                        object_mask_for_framing,
                        min_ratio=active_min_cov * 0.75,
                        max_ratio=min(0.98, active_max_cov * 1.35),
                    )
                has_part = part_seg_reliable and _part_pixels_ok(
                    part_areas,
                    require_all_parts=bool(require_all_part_visible),
                )
                if ok and has_part:
                    accepted_dirs.append(d)
                    key = _dir_key(d)
                    accepted_dir_keys.add(key)
                    accepted_distances[key] = float(current_distance)

    if not accepted_dirs:
        # Last-resort fallback: use top candidate dirs by score even if they fail strict rules.
        any_scored = fallback_scored_dirs or relaxed_scored_dirs
        if any_scored:
            any_scored.sort(key=lambda x: x[0], reverse=True)
            for _, d, dist in any_scored:
                if len(accepted_dirs) >= n_views:
                    break
                if _dir_key(d) in accepted_dir_keys:
                    continue
                accepted_dirs.append(d)
                key = _dir_key(d)
                accepted_dir_keys.add(key)
                accepted_distances[key] = float(dist)
        if not accepted_dirs:
            raise RuntimeError(
                "No valid render views found with the required movable-part visibility. "
                f"Try lowering --min-part-mask-pixels/--min-part-mask-coverage, increasing "
                f"--view-candidate-multiplier, or disabling --require-all-part-visible. "
                f"threshold_pixels={min_effective_part_pixels}; "
                f"part_to_segmentation_ids={part_to_segmentation_ids}"
            )
    while len(accepted_dirs) < n_views:
        src = accepted_dirs[len(accepted_dirs) % len(accepted_dirs)]
        accepted_dirs.append(src)

    background_records: Dict[str, Dict[str, object]] = {}
    qpos_records: Dict[str, List[float]] = {}
    view_distance_records: Dict[str, float] = {}
    part_visibility_records: Dict[str, Dict[str, int]] = {}
    visible_part_records: Dict[str, List[str]] = {}
    invisible_part_records: Dict[str, List[str]] = {}
    part_visibility_flag_records: Dict[str, Dict[str, bool]] = {}
    part_visibility_stats_records: Dict[str, Dict[str, Dict[str, object]]] = {}
    part_segmentation_source_records: Dict[str, Dict[str, object]] = {}
    part_only_probe: Optional[Dict[str, object]] = None
    if bool(part_occlusion_check):
        part_only_probe = _build_part_only_probe(
            sapien_module=sapien,
            instance=instance,
            part_layout=part_layout,
            urdf_part_infos=urdf_part_infos,
            width=width,
            height=height,
            fov_deg=fov_deg,
        )
        if part_only_probe is None:
            raise RuntimeError("part occlusion check requested, but no part-only visual meshes could be built.")

    saved_view_idx = 0
    rejected_occlusion_views: List[Dict[str, object]] = []
    for view_idx, d in enumerate(accepted_dirs):
        if _resource_tracker is not None:
            _resource_tracker["phase"] = "final_render_take_picture"
        view_qpos = (
            _small_joint_motion_qpos(
                base_qpos=base_qpos,
                qlimits=qlimits,
                view_idx=view_idx,
                n_views=len(accepted_dirs),
                fraction=joint_motion_fraction,
                max_delta=joint_motion_max_delta,
            )
            if joint_motion
            else base_qpos
        )
        _set_articulation_qpos(art, view_qpos)
        view_distance = float(accepted_distances.get(_dir_key(d), current_distance))
        cam_pos = target + d * view_distance
        pose44 = _look_at(cam_pos, target)
        camera.entity.set_pose(sapien.Pose(pose44))
        scene.step()
        scene.update_render()
        camera.take_picture()

        rgba = _camera_picture(camera, "Color")
        base_rgb = (np.clip(rgba[..., :3], 0.0, 1.0) * 255).astype(np.uint8)

        # Depth (same convention as test_obj.py): -Z in camera space, saved as uint16 millimeters.
        position = _camera_picture(camera, "Position")
        depth = -position[..., 2]
        depth = np.where(depth > 0, depth, 0.0)
        depth_image = (depth * 1000.0).astype(np.uint16)

        seg = _camera_picture(camera, "Segmentation")
        if entity_seg_channel is None:
            entity_seg_channel = choose_entity_segmentation_channel(
                seg,
                set(object_entity_ids_np.tolist()),
            )
        if visual_seg_channel is None:
            visual_seg_channel = _choose_visual_segmentation_channel(seg, entity_seg_channel)
        per_part_masks, per_part_visible_pixels, part_segmentation_source, part_mask_channel = (
            part_masks_from_segmentation(seg)
        )
        part_union_mask = np.zeros(seg.shape[:2], dtype=bool)
        for part_mask in per_part_masks.values():
            part_union_mask |= part_mask.astype(bool)
        object_mask_for_framing = object_mask_from_segmentation(seg, entity_seg_channel)
        if int(object_mask_for_framing.sum()) <= 0 and int(part_union_mask.sum()) > 0:
            object_mask_for_framing = part_union_mask
        if int(object_mask_for_framing.sum()) <= 0 and not has_rendered_background_geometry:
            object_mask_for_framing = depth > 0
        final_ratio = float(object_mask_for_framing.sum()) / float(
            object_mask_for_framing.shape[0] * object_mask_for_framing.shape[1]
        )
        final_framing_ok, final_framing_stats = mask_framing_ok(
            mask=object_mask_for_framing,
            ratio=final_ratio,
            min_ratio=min_object_coverage,
            max_ratio=max_object_coverage,
            max_center_offset=max_center_offset,
            min_edge_margin=min_edge_margin,
        )
        if not final_framing_ok:
            raise RuntimeError(
                f"Accepted view {view_idx} failed final framing check after joint motion: "
                f"{final_framing_stats}; thresholds="
                f"min_ratio={min_object_coverage:.4f}, max_ratio={max_object_coverage:.4f}, "
                f"max_center_offset={max_center_offset:.4f}, min_edge_margin={min_edge_margin:.4f}"
            )
        composite_mask = _composite_object_mask(
            union_mask=object_mask_for_framing,
            seg=seg,
            rgba=rgba,
            depth=depth,
            has_rendered_background_geometry=has_rendered_background_geometry,
        )

        has_visible_part, part_visibility_flags = part_visibility_from_pixel_counts(
            per_part_visible_pixels,
            int(min_effective_part_pixels),
        )
        if not has_visible_part:
            raise RuntimeError(
                f"Accepted view {view_idx} has no visible movable part after joint motion: "
                f"threshold_pixels={min_effective_part_pixels}, "
                f"part_pixels={{"
                + ", ".join(
                    f"{part_folder_name(pid, part_layout[pid])}: {pix}"
                    for pid, pix in per_part_visible_pixels.items()
                )
                + "}"
            )

        if bool(require_all_part_visible):
            too_small = {
                part_folder_name(pid, part_layout[pid]): pix
                for pid, pix in per_part_visible_pixels.items()
                if pix < int(min_effective_part_pixels)
            }
            if too_small:
                raise RuntimeError(
                    f"Accepted view {view_idx} became invalid after joint motion: "
                    f"part mask pixels below {min_effective_part_pixels}: {too_small}"
                )

        world_in_cam = np.linalg.inv(pose44).astype(np.float32)
        per_part_pose: Dict[int, np.ndarray] = {}
        part_only_projected_pixels: Dict[int, int] = {}
        part_visible_ratios: Dict[int, float] = {}
        for part_id, part_name in part_layout.items():
            link_name = urdf_part_infos[part_id].link_name
            link = linkname_to_link.get(link_name)
            if link is None:
                raise RuntimeError(f"Link '{link_name}' for part {part_id} not found in articulation.")
            ob_in_world = _link_pose_matrix(link)
            per_part_pose[part_id] = world_in_cam @ ob_in_world
            if part_only_probe is not None:
                projected = _part_only_projected_pixels(
                    probe=part_only_probe,
                    sapien_module=sapien,
                    part_id=part_id,
                    link_pose44=ob_in_world,
                    camera_pose44=pose44,
                )
                visible = int(per_part_visible_pixels.get(part_id, 0))
                ratio = float(visible) / float(max(1, projected))
                part_only_projected_pixels[part_id] = int(projected)
                part_visible_ratios[part_id] = float(ratio)

        if part_only_probe is not None:
            min_ratio = float(min_part_visible_ratio)
            valid_ratio_flags = {
                part_id: (
                    int(per_part_visible_pixels.get(part_id, 0)) >= int(min_effective_part_pixels)
                    and float(part_visible_ratios.get(part_id, 0.0)) >= min_ratio
                )
                for part_id in part_layout.keys()
            }
            ratio_ok = all(valid_ratio_flags.values()) if bool(require_all_part_visible) else any(valid_ratio_flags.values())
            if not ratio_ok:
                details = {
                    part_folder_name(pid, part_layout[pid]): {
                        "visible_pixels": int(per_part_visible_pixels.get(pid, 0)),
                        "part_only_projected_pixels": int(part_only_projected_pixels.get(pid, 0)),
                        "visible_ratio": float(part_visible_ratios.get(pid, 0.0)),
                    }
                    for pid in part_layout.keys()
                }
                rejected_occlusion_views.append(
                    {
                        "view_idx": int(view_idx),
                        "min_part_visible_ratio": float(min_ratio),
                        "details": details,
                    }
                )
                continue

        # For background variants of the same view, depth/object-mask/part-masks/cam-params
        # are identical. Save once, then hardlink/copy to the remaining frame indices.
        output_view_idx = saved_view_idx
        first_frame_name = f"{output_view_idx * background_variants:06d}"
        first_depth_path = depth_dir / f"{first_frame_name}.png"
        first_objmask_path = object_mask_dir / f"{first_frame_name}.png"
        first_part_mask_paths: Dict[int, Path] = {}
        first_part_cam_paths: Dict[int, Path] = {}

        for bg_idx in range(background_variants):
            frame_idx = output_view_idx * background_variants + bg_idx
            frame_name = f"{frame_idx:06d}"

            if background_mode == "composite":
                rgb, bg_name = _composite_rgb_background(
                    rgb=base_rgb,
                    object_mask=composite_mask,
                    variant_idx=bg_idx,
                    seed=per_instance_bg_seed,
                )
            else:
                rgb = base_rgb
                bg_name = str(scene_background_info.get("name", "scene" if use_scene_background else "plain"))

            Image.fromarray(rgb).save(rgb_dir / f"{frame_name}.png")
            if bg_idx == 0:
                Image.fromarray(depth_image).save(first_depth_path)
                Image.fromarray(composite_mask.astype(np.uint8) * 255, mode="L").save(first_objmask_path)

                for part_id, part_name in part_layout.items():
                    part_mask_path = mask_root / part_folder_name(part_id, part_name) / f"{first_frame_name}.png"
                    Image.fromarray(per_part_masks[part_id], mode="L").save(part_mask_path)
                    first_part_mask_paths[part_id] = part_mask_path

                for part_id, part_name in part_layout.items():
                    part_cam_file = cam_dir / part_folder_name(part_id, part_name) / f"{first_frame_name}.txt"
                    np.savetxt(part_cam_file, per_part_pose[part_id], fmt="%.8f")
                    first_part_cam_paths[part_id] = part_cam_file
            else:
                _link_or_copy_file(first_depth_path, depth_dir / f"{frame_name}.png")
                _link_or_copy_file(first_objmask_path, object_mask_dir / f"{frame_name}.png")
                for part_id, part_name in part_layout.items():
                    dst_part_mask = mask_root / part_folder_name(part_id, part_name) / f"{frame_name}.png"
                    _link_or_copy_file(first_part_mask_paths[part_id], dst_part_mask)
                for part_id, part_name in part_layout.items():
                    dst_part_cam = cam_dir / part_folder_name(part_id, part_name) / f"{frame_name}.txt"
                    _link_or_copy_file(first_part_cam_paths[part_id], dst_part_cam)

            camera_poses[frame_name] = pose44.reshape(-1).tolist()
            if view_qpos is not None:
                qpos_records[frame_name] = np.asarray(view_qpos, dtype=np.float32).reshape(-1).tolist()
            background_records[frame_name] = {
                "mode": background_mode,
                "variant_idx": bg_idx,
                "name": bg_name,
                "source_view_idx": view_idx,
            }
            view_distance_records[frame_name] = view_distance
            part_visibility_records[frame_name] = {
                part_folder_name(pid, part_layout[pid]): int(pix)
                for pid, pix in per_part_visible_pixels.items()
            }
            part_visibility_flag_records[frame_name] = {
                part_folder_name(pid, part_layout[pid]): bool(part_visibility_flags.get(pid, False))
                for pid in part_layout.keys()
            }
            part_visibility_stats_records[frame_name] = {
                part_folder_name(pid, part_layout[pid]): part_mask_stats(
                    per_part_masks[pid],
                    int(min_effective_part_pixels),
                )
                for pid in part_layout.keys()
            }
            part_segmentation_source_records[frame_name] = {
                "source": part_segmentation_source,
                "channel": int(part_mask_channel),
            }
            visible_part_records[frame_name] = [
                part_folder_name(pid, part_layout[pid])
                for pid in part_layout.keys()
                if bool(part_visibility_flags.get(pid, False))
            ]
            invisible_part_records[frame_name] = [
                part_folder_name(pid, part_layout[pid])
                for pid in part_layout.keys()
                if not bool(part_visibility_flags.get(pid, False))
            ]
        saved_view_idx += 1

    if saved_view_idx <= 0:
        raise RuntimeError(
            "No accepted views survived part-only occlusion check. "
            f"min_part_visible_ratio={float(min_part_visible_ratio):.4f}; "
            f"rejected_views={rejected_occlusion_views[:5]}"
        )
    if part_only_probe is not None:
        cleanup_sapien_scene(
            scene=part_only_probe.get("scene"),
            camera=part_only_probe.get("camera"),
        )

    intrinsic = camera.get_intrinsic_matrix()
    np.savetxt(out_dir / "K.txt", intrinsic, fmt="%.8f")

    return {
        "intrinsic": intrinsic.reshape(-1).tolist(),
        "views": saved_view_idx * background_variants,
        "requested_views": int(n_views),
        "views_per_background": saved_view_idx,
        "background_variants": background_variants,
        "background_mode": background_mode,
        "background_records": background_records,
        "scene_background": scene_background_info,
        "lighting": lighting_info,
        "rendered_background_geometry": has_rendered_background_geometry,
        "joint_motion": bool(joint_motion),
        "joint_motion_fraction": float(joint_motion_fraction),
        "joint_motion_max_delta": float(joint_motion_max_delta),
        "qpos_records": qpos_records,
        "base_qpos": [] if base_qpos is None else np.asarray(base_qpos, dtype=np.float32).reshape(-1).tolist(),
        "rt_enabled": bool(use_rt),
        "rt_disabled_for_scene_background": bool(rt_disabled_for_scene_background),
        "rt_camera_shader": bool(rt_camera_shader),
        "rt_spp": int(rt_spp),
        "rt_path_depth": int(rt_path_depth),
        "rt_denoiser": str(rt_denoiser),
        "width": width,
        "height": height,
        "cam_params_format": "cam_params/<part_id_name>/<frame>.txt -> 4x4 ob_in_cam",
        "object_scale": object_scale,
        "min_object_scale": float(min_object_scale),
        "max_object_scale": float(max_object_scale),
        "target": target.tolist(),
        "camera_distance": current_distance,
        "view_distances": view_distance_records,
        "part_visible_pixels": part_visibility_records,
        "part_visibility": part_visibility_flag_records,
        "part_visibility_stats": part_visibility_stats_records,
        "part_segmentation_source": part_segmentation_source_records,
        "visible_parts": visible_part_records,
        "invisible_parts": invisible_part_records,
        "min_part_mask_pixels": int(min_effective_part_pixels),
        "require_all_part_visible": bool(require_all_part_visible),
        "view_candidate_count": int(candidate_count),
        "part_occlusion_check": bool(part_occlusion_check),
        "min_part_visible_ratio": float(min_part_visible_ratio),
        "saved_view_count": int(saved_view_idx),
        "rejected_occlusion_view_count": int(len(rejected_occlusion_views)),
        "rejected_occlusion_views": rejected_occlusion_views[:20],
        "bbox_extent_scaled": extent.tolist(),
        "segmentation_channel": visual_seg_channel,
        "entity_segmentation_channel": entity_seg_channel,
        "visual_segmentation_channel": visual_seg_channel,
        "part_to_link": {str(pid): urdf_part_infos[pid].link_name for pid in part_layout.keys()},
        "part_to_segmentation_ids": {
            str(pid): part_to_segmentation_ids[pid] for pid in part_layout.keys()
        },
        "id_to_entity": {str(k): v for k, v in id_to_entity.items()},
        "linkname_to_entity_ids": {str(k): v for k, v in linkname_to_entity_ids.items()},
        "object_lift": object_lift,
        "camera_poses": camera_poses,
    }


def render_instance(
    instance: InstanceInfo,
    out_dir: Path,
    part_layout: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
    n_views: int,
    width: int,
    height: int,
    fov_deg: float,
    radius_scale: float,
    target_max_extent: float,
    min_object_scale: float,
    max_object_scale: float,
    min_object_coverage: float,
    max_object_coverage: float,
    min_part_mask_pixels: int,
    min_part_mask_coverage: float,
    require_all_part_visible: bool,
    view_candidate_multiplier: int,
    part_occlusion_check: bool,
    min_part_visible_ratio: float,
    use_rt: bool,
    render_ground: bool,
    rt_camera_shader: bool,
    rt_spp: int,
    rt_path_depth: int,
    rt_denoiser: str,
    background_mode: str,
    background_variants: int,
    background_seed: int,
    joint_motion: bool,
    joint_motion_fraction: float,
    joint_motion_max_delta: float,
) -> Dict[str, object]:
    resource_tracker: Dict[str, object] = {}
    try:
        return _render_instance_impl(
            instance=instance,
            out_dir=out_dir,
            part_layout=part_layout,
            urdf_part_infos=urdf_part_infos,
            n_views=n_views,
            width=width,
            height=height,
            fov_deg=fov_deg,
            radius_scale=radius_scale,
            target_max_extent=target_max_extent,
            min_object_scale=min_object_scale,
            max_object_scale=max_object_scale,
            min_object_coverage=min_object_coverage,
            max_object_coverage=max_object_coverage,
            min_part_mask_pixels=min_part_mask_pixels,
            min_part_mask_coverage=min_part_mask_coverage,
            require_all_part_visible=require_all_part_visible,
            view_candidate_multiplier=view_candidate_multiplier,
            part_occlusion_check=part_occlusion_check,
            min_part_visible_ratio=min_part_visible_ratio,
            use_rt=use_rt,
            render_ground=render_ground,
            rt_camera_shader=rt_camera_shader,
            rt_spp=rt_spp,
            rt_path_depth=rt_path_depth,
            rt_denoiser=rt_denoiser,
            background_mode=background_mode,
            background_variants=background_variants,
            background_seed=background_seed,
            joint_motion=joint_motion,
            joint_motion_fraction=joint_motion_fraction,
            joint_motion_max_delta=joint_motion_max_delta,
            _resource_tracker=resource_tracker,
        )
    except Exception as e:
        if is_renderer_resource_error(str(e)):
            phase = str(resource_tracker.get("phase", "unknown"))
            raise RuntimeError(
                f"{e}; render_phase={phase}; width={int(width)}, height={int(height)}, "
                f"rt={bool(use_rt)}, background_mode={background_mode}"
            ) from e
        raise
    finally:
        cleanup_sapien_scene(
            scene=resource_tracker.get("scene"),
            camera=resource_tracker.get("camera"),
            articulation=resource_tracker.get("articulation"),
        )
        resource_tracker.clear()


def build_one_instance(instance: InstanceInfo, args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.output_root) / instance.instance_name
    stale_incomplete_output = out_dir.exists() and not (out_dir / "meta.json").exists()
    cleanup_on_error = args.overwrite or not out_dir.exists() or stale_incomplete_output
    if out_dir.exists() and (args.overwrite or stale_incomplete_output):
        robust_rmtree(out_dir)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        mobility_link_map = parse_mobility_links(instance.mobility_path)
        urdf_part_infos = parse_urdf_link_infos(instance.urdf_path, mobility_link_map)
        part_layout = build_link_layout(mobility_link_map, urdf_part_infos)
        exclude_keywords = parse_exclude_keywords(args.exclude_part_keywords)
        part_layout, excluded_part_layout = filter_part_layout(part_layout, exclude_keywords)
        part_layout, excluded_by_movable = filter_part_layout_by_movable_joints(
            instance=instance,
            part_layout=part_layout,
            urdf_part_infos=urdf_part_infos,
        )
        excluded_part_layout = {**excluded_part_layout, **excluded_by_movable}
        if not part_layout:
            raise RuntimeError(
                f"No valid mobility link ids remain after filtering in mobility_v2.json: {instance.mobility_path}"
            )
        selected_urdf_part_infos = {
            part_id: urdf_part_infos[part_id]
            for part_id in part_layout.keys()
            if part_id in urdf_part_infos
        }
        missing_mesh_files = find_missing_mesh_files(instance.instance_dir, selected_urdf_part_infos)
        if missing_mesh_files:
            write_skip_reason(
                out_dir,
                "missing_mesh_files",
                {
                    "count": len(missing_mesh_files),
                    "sample": missing_mesh_files[:20],
                },
            )
            raise SkipInstanceError(
                "missing_mesh_files",
                details={
                    "count": len(missing_mesh_files),
                    "sample": missing_mesh_files[:20],
                },
            )
        mask_part_ids = sorted(part_layout.keys())

        part_model_result: Dict[int, Dict[str, object]] = {}
        if not args.skip_model:
            try:
                part_model_result = build_part_models(
                    instance=instance,
                    part_layout=part_layout,
                    urdf_part_infos=urdf_part_infos,
                    output_models_root_dir=out_dir / "models",
                )
            except Exception as e:
                if is_missing_asset_error(str(e)):
                    write_skip_reason(
                        out_dir,
                        "missing_assets_in_model_build",
                        {"error": str(e)},
                    )
                    raise SkipInstanceError(
                        "missing_assets_in_model_build",
                        details={"error": str(e)},
                    ) from e
                raise

        render_result: Dict[str, object] = {}
        if not args.skip_render:
            render_kwargs = dict(
                instance=instance,
                out_dir=out_dir,
                part_layout=part_layout,
                urdf_part_infos=urdf_part_infos,
                n_views=args.views,
                width=args.width,
                height=args.height,
                fov_deg=args.fov_deg,
                radius_scale=args.radius_scale,
                target_max_extent=args.target_max_extent,
                min_object_scale=args.min_object_scale,
                max_object_scale=args.max_object_scale,
                min_object_coverage=args.min_object_coverage,
                max_object_coverage=args.max_object_coverage,
                min_part_mask_pixels=args.min_part_mask_pixels,
                min_part_mask_coverage=args.min_part_mask_coverage,
                require_all_part_visible=args.require_all_part_visible,
                view_candidate_multiplier=args.view_candidate_multiplier,
                part_occlusion_check=args.part_occlusion_check,
                min_part_visible_ratio=args.min_part_visible_ratio,
                use_rt=args.rt,
                render_ground=args.render_ground,
                rt_camera_shader=args.rt_camera_shader,
                rt_spp=args.rt_spp,
                rt_path_depth=args.rt_path_depth,
                rt_denoiser=args.rt_denoiser,
                background_mode=args.background_mode,
                background_variants=args.background_variants,
                background_seed=args.background_seed,
                joint_motion=args.joint_motion,
                joint_motion_fraction=args.joint_motion_fraction,
                joint_motion_max_delta=args.joint_motion_max_delta,
            )
            try:
                render_result = render_instance(**render_kwargs)
            except SapienCameraBufferError as e:
                if bool(args.rt):
                    clear_render_outputs(out_dir)
                    fallback_kwargs = dict(render_kwargs)
                    fallback_kwargs["use_rt"] = False
                    try:
                        render_result = render_instance(**fallback_kwargs)
                    except SapienCameraBufferError as fallback_error:
                        write_failure_reason(
                            out_dir,
                            "sapien_camera_buffer_unavailable",
                            {
                                "buffer": fallback_error.buffer_name,
                                "error": str(fallback_error),
                                "first_error": str(e),
                                "background_mode": args.background_mode,
                                "rt": bool(args.rt),
                                "fallback_no_rt_attempted": True,
                            },
                        )
                        raise RuntimeError(
                            "Required SAPIEN camera buffer is unavailable even after retrying with --no-rt. "
                            f"buffer={fallback_error.buffer_name}, rt={bool(args.rt)}, "
                            f"background_mode={args.background_mode}. "
                            "This is a renderer/runtime configuration problem, not an instance to skip."
                        ) from fallback_error
                    render_result["rt_fallback"] = {
                        "enabled": True,
                        "from_rt": True,
                        "to_rt": False,
                        "reason": str(e),
                    }
                    render_result["rt_requested"] = True
                    render_result["rt_effective"] = False
                else:
                    write_failure_reason(
                        out_dir,
                        "sapien_camera_buffer_unavailable",
                        {
                            "buffer": e.buffer_name,
                            "error": str(e),
                            "background_mode": args.background_mode,
                            "rt": bool(args.rt),
                            "fallback_no_rt_attempted": False,
                        },
                    )
                    raise RuntimeError(
                        "Required SAPIEN camera buffer is unavailable. "
                        f"buffer={e.buffer_name}, rt={bool(args.rt)}, "
                        f"background_mode={args.background_mode}. "
                        "This is a renderer/runtime configuration problem, not an instance to skip."
                    ) from e
            except Exception as e:
                if is_missing_asset_error(str(e)):
                    write_skip_reason(
                        out_dir,
                        "missing_assets_in_render",
                        {"error": str(e)},
                    )
                    raise SkipInstanceError(
                        "missing_assets_in_render",
                        details={"error": str(e)},
                    ) from e
                if is_render_view_selection_error(str(e)):
                    write_skip_reason(
                        out_dir,
                        "render_view_selection_failed",
                        {
                            "error": str(e),
                            "min_part_mask_pixels": int(args.min_part_mask_pixels),
                            "min_part_mask_coverage": float(args.min_part_mask_coverage),
                            "require_all_part_visible": bool(args.require_all_part_visible),
                            "view_candidate_multiplier": int(args.view_candidate_multiplier),
                        },
                    )
                    raise SkipInstanceError(
                        "render_view_selection_failed",
                        details={
                            "error": str(e),
                            "min_part_mask_pixels": int(args.min_part_mask_pixels),
                            "min_part_mask_coverage": float(args.min_part_mask_coverage),
                            "require_all_part_visible": bool(args.require_all_part_visible),
                            "view_candidate_multiplier": int(args.view_candidate_multiplier),
                        },
                    ) from e
                if is_renderer_resource_error(str(e)):
                    write_failure_reason(
                        out_dir,
                        "renderer_resource_unavailable",
                        {
                            "error": str(e),
                            "width": int(args.width),
                            "height": int(args.height),
                            "rt": bool(args.rt),
                            "background_mode": args.background_mode,
                        },
                    )
                    raise RuntimeError(
                        "Renderer resource creation failed. "
                        f"error={e}; width={int(args.width)}, height={int(args.height)}, "
                        f"rt={bool(args.rt)}, background_mode={args.background_mode}. "
                        "This is a renderer/runtime configuration problem, not an instance to skip. "
                        "Use a lower resolution or fix the SAPIEN/Vulkan device resource limit."
                    ) from e
                raise
        meta = {
            "instance_name": instance.instance_name,
            "category": instance.category,
            "instance_id": instance.instance_id,
            "source_dir": str(instance.instance_dir),
            "urdf_path": str(instance.urdf_path),
            "mobility_path": str(instance.mobility_path),
            "result_path": str(instance.result_path),
            "part_count": len(part_layout),
            "part_mode": "link_based",
            "part_layout": {str(k): v for k, v in part_layout.items()},
            "excluded_part_keywords": exclude_keywords,
            "excluded_part_layout": {str(k): v for k, v in excluded_part_layout.items()},
            "movable_joint_filter_enabled": True,
            "mask_part_ids": mask_part_ids,
            "part_models": {str(k): v for k, v in part_model_result.items()},
            "render": render_result,
        }
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta
    except SkipInstanceError:
        raise
    except Exception as e:
        if is_renderer_resource_error(str(e)):
            write_failure_reason(
                out_dir,
                "renderer_resource_unavailable",
                {
                    "error": str(e),
                    "width": int(getattr(args, "width", -1)),
                    "height": int(getattr(args, "height", -1)),
                    "rt": bool(getattr(args, "rt", False)),
                    "background_mode": str(getattr(args, "background_mode", "unknown")),
                },
            )
            raise RuntimeError(
                "Renderer resource creation failed. "
                f"error={e}; width={int(getattr(args, 'width', -1))}, "
                f"height={int(getattr(args, 'height', -1))}, "
                f"rt={bool(getattr(args, 'rt', False))}, "
                f"background_mode={getattr(args, 'background_mode', 'unknown')}. "
                "This is a renderer/runtime configuration problem, not an instance to skip."
            ) from e
        if is_render_view_selection_error(str(e)):
            write_skip_reason(
                out_dir,
                "render_view_selection_failed",
                {"error": str(e)},
            )
            raise SkipInstanceError(
                "render_view_selection_failed",
                details={"error": str(e)},
            ) from e
        if cleanup_on_error and out_dir.exists():
            robust_rmtree(out_dir)
        raise


def _run_one_instance_worker(instance: InstanceInfo, args_dict: Dict[str, object]) -> Dict[str, object]:
    args = argparse.Namespace(**args_dict)
    try:
        build_one_instance(instance, args)
        return {
            "instance_name": instance.instance_name,
            "status": "success",
        }
    except SkipInstanceError as e:
        return {
            "instance_name": instance.instance_name,
            "status": "skipped",
            "error": e.reason,
            "details": e.details,
        }
    except Exception as e:
        return {
            "instance_name": instance.instance_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        gc.collect()


def _run_one_instance_worker_queue(
    instance: InstanceInfo,
    args_dict: Dict[str, object],
    result_queue: object,
) -> None:
    result_queue.put(_run_one_instance_worker(instance, args_dict))


def run_one_instance_isolated(
    instance: InstanceInfo,
    args_dict: Dict[str, object],
    timeout_seconds: int,
) -> Dict[str, object]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_one_instance_worker_queue,
        args=(instance, args_dict, result_queue),
    )
    proc.start()
    proc.join(timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        out_dir = Path(str(args_dict.get("output_root", "dataset_train"))) / instance.instance_name
        write_skip_reason(
            out_dir,
            "instance_timeout",
            {"timeout_seconds": int(timeout_seconds), "instance_name": instance.instance_name},
        )
        return {
            "instance_name": instance.instance_name,
            "status": "skipped",
            "error": "instance_timeout",
            "details": {"timeout_seconds": int(timeout_seconds)},
        }
    try:
        return result_queue.get_nowait()
    except queue_module.Empty:
        if proc.exitcode == 0:
            return {"instance_name": instance.instance_name, "status": "success"}
        return {
            "instance_name": instance.instance_name,
            "status": "failed",
            "error": f"worker exited without result, exitcode={proc.exitcode}",
        }


def resolve_num_processes(requested: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if requested <= 0:
        requested = max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(requested, n_tasks))


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root)
    if not models_root.exists():
        raise FileNotFoundError(f"models root not found: {models_root}")

    all_instances = discover_instances(models_root)
    if not all_instances:
        raise RuntimeError("No valid instances found under models root.")

    targets = select_instances(all_instances, args.instance, args.start_instance, args.end_instance)
    print(f"[Info] discovered {len(all_instances)} instances, selected {len(targets)}")
    num_workers = resolve_num_processes(args.processes, len(targets))
    print(f"[Info] worker_processes={num_workers}")

    ok = 0
    skipped = 0
    skipped_items: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []
    args_dict = vars(args).copy()

    use_instance_timeout = int(args.instance_timeout) > 0
    recycle_workers = int(args.max_tasks_per_worker) > 0 and len(targets) > 1

    if num_workers <= 1 and not recycle_workers and not use_instance_timeout:
        for idx, inst in enumerate(targets, start=1):
            print(f"[Build {idx}/{len(targets)}] {inst.instance_name}")
            result = _run_one_instance_worker(inst, args_dict)
            if result["status"] == "success":
                ok += 1
                continue
            if result["status"] == "skipped":
                skipped += 1
                err = str(result.get("error", "skipped"))
                skipped_items.append((inst.instance_name, err))
                print(f"[Skip] {inst.instance_name}: {err}")
                continue
            err = str(result.get("error", "unknown_error"))
            failed.append((inst.instance_name, err))
            print(f"[Error] {inst.instance_name}: {err}")
    elif num_workers <= 1:
        for idx, inst in enumerate(targets, start=1):
            print(f"[Build {idx}/{len(targets)}] {inst.instance_name}")
            result = run_one_instance_isolated(inst, args_dict, int(args.instance_timeout))
            if result["status"] == "success":
                ok += 1
                continue
            if result["status"] == "skipped":
                skipped += 1
                err = str(result.get("error", "skipped"))
                skipped_items.append((inst.instance_name, err))
                print(f"[Skip] {inst.instance_name}: {err}")
                continue
            err = str(result.get("error", "unknown_error"))
            failed.append((inst.instance_name, err))
            print(f"[Error] {inst.instance_name}: {err}")
    else:
        tasks_per_worker = int(args.max_tasks_per_worker) if int(args.max_tasks_per_worker) > 0 else len(targets)
        batch_size = max(1, tasks_per_worker)
        finished = 0
        total = len(targets)
        ctx = mp.get_context("spawn")
        print(f"[Info] worker_recycle_every={tasks_per_worker} tasks/worker, batch_size={batch_size}")
        for batch_start in range(0, len(targets), batch_size):
            batch = targets[batch_start : batch_start + batch_size]
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                future_to_instance = {
                    executor.submit(_run_one_instance_worker, inst, args_dict): inst.instance_name
                    for inst in batch
                }
            for future in concurrent.futures.as_completed(future_to_instance):
                finished += 1
                inst_name = future_to_instance[future]
                try:
                    result = future.result()
                except Exception as e:
                    failed.append((inst_name, str(e)))
                    print(f"[Error {finished}/{total}] {inst_name}: {e}")
                    continue
                status = str(result.get("status", "failed"))
                if status == "success":
                    ok += 1
                    print(f"[OK {finished}/{total}] {inst_name}")
                elif status == "skipped":
                    skipped += 1
                    err = str(result.get("error", "skipped"))
                    skipped_items.append((inst_name, err))
                    print(f"[Skip {finished}/{total}] {inst_name}: {err}")
                else:
                    err = str(result.get("error", "unknown_error"))
                    failed.append((inst_name, err))
                    print(f"[Error {finished}/{total}] {inst_name}: {err}")

    print(f"[Done] success={ok}, skipped={skipped}, failed={len(failed)}")
    if skipped_items:
        print("[Skipped Details]")
        for name, err in skipped_items:
            print(f"  - {name}: {err}")
    if failed:
        print("[Failed Details]")
        for name, err in failed:
            print(f"  - {name}: {err}")
