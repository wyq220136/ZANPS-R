import hashlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_set_render_attr(obj: object, name: str, value: object) -> None:
    """Set a SAPIEN render attribute when it exists in the installed version."""
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def configure_ray_tracing(
    sapien_module: object,
    rt_camera_shader: bool,
    rt_spp: int,
    rt_path_depth: int,
    rt_denoiser: str,
) -> None:
    """
    Configure SAPIEN ray tracing before creating the scene/camera.

    For dataset generation, masks and depth rely on Segmentation/Position buffers.
    Some SAPIEN builds do not expose those buffers under the rt camera shader, so
    keep that shader opt-in and default to the dataset-compatible camera shader.
    """
    render = sapien_module.render
    if hasattr(render, "set_viewer_shader_dir"):
        render.set_viewer_shader_dir("rt")
    if rt_camera_shader and hasattr(render, "set_camera_shader_dir"):
        render.set_camera_shader_dir("rt")
    if hasattr(render, "set_ray_tracing_samples_per_pixel"):
        render.set_ray_tracing_samples_per_pixel(max(1, int(rt_spp)))
    if hasattr(render, "set_ray_tracing_path_depth"):
        render.set_ray_tracing_path_depth(max(1, int(rt_path_depth)))
    denoiser = str(rt_denoiser).lower().strip()
    if denoiser not in {"", "none", "off"} and hasattr(render, "set_ray_tracing_denoiser"):
        render.set_ray_tracing_denoiser(denoiser)


def _make_render_material(
    sapien_module: object,
    base_color: List[float],
    roughness: float = 0.8,
    specular: float = 0.25,
    metallic: float = 0.0,
) -> object:
    material = sapien_module.render.RenderMaterial()
    _safe_set_render_attr(material, "base_color", np.asarray(base_color, dtype=np.float32))
    _safe_set_render_attr(material, "roughness", float(roughness))
    _safe_set_render_attr(material, "specular", float(specular))
    _safe_set_render_attr(material, "metallic", float(metallic))
    return material


def _add_ground_plane(
    scene: object,
    sapien_module: object,
    altitude: float,
    material: Optional[object] = None,
) -> None:
    if material is None:
        scene.add_ground(altitude=float(altitude))
        return
    try:
        scene.add_ground(altitude=float(altitude), render_material=material)
    except TypeError:
        scene.add_ground(altitude=float(altitude))


def _add_box_visual_actor(
    scene: object,
    sapien_module: object,
    name: str,
    pose_xyz: List[float],
    half_size: List[float],
    material: object,
) -> Optional[object]:
    builder = scene.create_actor_builder()
    pose = sapien_module.Pose(pose_xyz)
    try:
        builder.add_box_visual(pose=pose, half_size=half_size, material=material)
    except TypeError:
        try:
            builder.add_box_visual(half_size=half_size, material=material, pose=pose)
        except TypeError:
            builder.add_box_visual(pose, half_size, material)
    try:
        return builder.build_static(name=name)
    except TypeError:
        return builder.build_static(name)


def _configure_scene_lighting(scene: object, seed: int, rich_scene: bool) -> Dict[str, object]:
    if not rich_scene:
        scene.set_ambient_light([0.55, 0.55, 0.55])
        scene.add_directional_light([0, 1, -1], [0.6, 0.6, 0.6], shadow=True)
        scene.add_point_light([1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        scene.add_point_light([-1.0, -1.0, 2.0], [1.0, 1.0, 1.0])
        return {"mode": "default"}

    rng = np.random.default_rng(int(seed) + 7717)
    ambient = np.clip(rng.uniform(0.28, 0.52, size=3), 0.0, 1.0)
    key_dir = np.asarray(
        [
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.6, 1.0),
            rng.uniform(-1.2, -0.7),
        ],
        dtype=np.float32,
    )
    key_dir = key_dir / max(1e-6, float(np.linalg.norm(key_dir)))
    key_color = np.asarray([1.0, rng.uniform(0.88, 1.0), rng.uniform(0.78, 0.95)], dtype=np.float32)
    key_color *= rng.uniform(0.55, 0.9)
    fill_pos = [
        float(rng.uniform(-1.4, 1.4)),
        float(rng.uniform(-1.4, 1.4)),
        float(rng.uniform(1.2, 2.4)),
    ]
    fill_color = np.asarray([rng.uniform(0.55, 0.9), rng.uniform(0.65, 0.95), 1.0], dtype=np.float32)
    fill_color *= rng.uniform(0.35, 0.8)

    scene.set_ambient_light(ambient.tolist())
    scene.add_directional_light(key_dir.tolist(), key_color.tolist(), shadow=True)
    scene.add_point_light(fill_pos, fill_color.tolist())
    return {
        "mode": "randomized_scene",
        "ambient": ambient.tolist(),
        "key_dir": key_dir.tolist(),
        "key_color": key_color.tolist(),
        "fill_pos": fill_pos,
        "fill_color": fill_color.tolist(),
    }


def _setup_scene_background(
    scene: object,
    sapien_module: object,
    seed: int,
    extent: np.ndarray,
    radius_obj: float,
    ground_altitude: float,
    ground_clearance: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(seed) + 4703)
    palettes = [
        ("warm_table_blue_wall", [0.58, 0.42, 0.26, 1.0], [0.62, 0.74, 0.82, 1.0]),
        ("green_workbench_light_wall", [0.31, 0.48, 0.37, 1.0], [0.78, 0.78, 0.70, 1.0]),
        ("gray_lab_soft_wall", [0.48, 0.50, 0.52, 1.0], [0.68, 0.72, 0.76, 1.0]),
        ("muted_red_table_cool_wall", [0.55, 0.32, 0.29, 1.0], [0.61, 0.68, 0.72, 1.0]),
    ]
    name, floor_color, wall_color = palettes[int(rng.integers(0, len(palettes)))]
    floor_material = _make_render_material(
        sapien_module,
        floor_color,
        roughness=float(rng.uniform(0.65, 0.95)),
        specular=float(rng.uniform(0.12, 0.35)),
    )
    wall_material = _make_render_material(
        sapien_module,
        wall_color,
        roughness=float(rng.uniform(0.75, 1.0)),
        specular=float(rng.uniform(0.05, 0.18)),
    )

    _add_ground_plane(scene, sapien_module, altitude=ground_altitude, material=floor_material)

    scene_radius = max(1.8, float(radius_obj) * 5.5, float(np.max(extent[:2])) * 3.5)
    wall_width = scene_radius * 2.8
    wall_height = max(1.1, float(extent[2]) * 2.4, float(radius_obj) * 2.4)
    wall_y = scene_radius
    wall_z = float(ground_altitude) + wall_height * 0.5
    _add_box_visual_actor(
        scene=scene,
        sapien_module=sapien_module,
        name="scene_background_wall",
        pose_xyz=[0.0, float(wall_y), float(wall_z)],
        half_size=[float(wall_width * 0.5), 0.025, float(wall_height * 0.5)],
        material=wall_material,
    )

    return {
        "name": name,
        "ground_altitude": float(ground_altitude),
        "ground_clearance": float(ground_clearance),
        "wall_y": float(wall_y),
        "wall_width": float(wall_width),
        "wall_height": float(wall_height),
    }


def _instance_background_seed(instance_name: str, seed: int) -> int:
    digest = hashlib.sha1(instance_name.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) + int(seed)) % (2**31 - 1)


def _make_synthetic_background(
    height: int,
    width: int,
    variant_idx: int,
    seed: int,
) -> Tuple[np.ndarray, str]:
    """
    Generate a deterministic, non-gray, scene-like RGB background.

    The background is intentionally synthetic rather than photo-real: it provides
    color and texture diversity without introducing unknown object/background
    collisions, occlusions, or depth changes.
    """
    palettes = [
        ("warm_wood_table", np.array([0.72, 0.55, 0.36]), np.array([0.94, 0.83, 0.66])),
        ("blue_lab_wall", np.array([0.42, 0.56, 0.70]), np.array([0.78, 0.86, 0.92])),
        ("green_workbench", np.array([0.38, 0.55, 0.44]), np.array([0.76, 0.86, 0.72])),
        ("sand_studio", np.array([0.73, 0.64, 0.50]), np.array([0.93, 0.88, 0.76])),
        ("purple_indoor", np.array([0.52, 0.45, 0.65]), np.array([0.82, 0.78, 0.92])),
        ("teal_factory", np.array([0.32, 0.58, 0.61]), np.array([0.76, 0.90, 0.88])),
    ]
    name, c_floor, c_wall = palettes[variant_idx % len(palettes)]
    rng = np.random.default_rng(seed + variant_idx * 10007)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xx / max(1.0, float(width - 1))
    yn = yy / max(1.0, float(height - 1))

    horizon = 0.52 + 0.08 * math.sin(variant_idx * 1.37)
    wall_weight = np.clip((horizon - yn) / max(horizon, 1e-6), 0.0, 1.0)[..., None]
    floor_weight = 1.0 - wall_weight

    wall = c_wall[None, None, :] * (0.82 + 0.18 * (1.0 - yn[..., None]))
    floor = c_floor[None, None, :] * (0.75 + 0.25 * yn[..., None])
    bg = wall * wall_weight + floor * floor_weight

    # Mild perspective-table stripes below the horizon.
    stripe_freq = 10.0 + float(variant_idx % 4) * 3.0
    stripe = 0.035 * np.sin((xn * stripe_freq + yn * 4.0 + rng.uniform(0, 2 * math.pi)) * 2 * math.pi)
    stripe *= (yn > horizon).astype(np.float32)
    bg = bg + stripe[..., None]

    # Low-frequency illumination variation, avoiding a flat color field.
    vignette = ((xn - 0.5) ** 2 + (yn - 0.48) ** 2)
    bg = bg * (1.05 - 0.38 * vignette[..., None])

    # Very small texture noise; deterministic but enough to avoid color collision.
    noise = rng.normal(0.0, 0.012, size=(height, width, 1)).astype(np.float32)
    bg = np.clip(bg + noise, 0.0, 1.0)
    return (bg * 255.0).astype(np.uint8), name


def _composite_rgb_background(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    variant_idx: int,
    seed: int,
) -> Tuple[np.ndarray, str]:
    bg, bg_name = _make_synthetic_background(
        height=rgb.shape[0],
        width=rgb.shape[1],
        variant_idx=variant_idx,
        seed=seed,
    )
    keep = object_mask.astype(bool)[..., None]
    out = np.where(keep, rgb, bg).astype(np.uint8)
    return out, bg_name


def _composite_object_mask(
    union_mask: np.ndarray,
    seg: np.ndarray,
    rgba: np.ndarray,
    depth: np.ndarray,
    has_rendered_background_geometry: bool,
) -> np.ndarray:
    mask = union_mask.astype(bool).copy()
    if not has_rendered_background_geometry and seg.ndim == 3 and seg.shape[2] > 1:
        mask |= seg[..., 1].astype(np.int32) > 1
    if not has_rendered_background_geometry and rgba.ndim == 3 and rgba.shape[2] >= 4:
        alpha_mask = rgba[..., 3] > 0.05
        # Some renderer builds return alpha=1 everywhere. Only trust alpha if it
        # actually separates foreground from background.
        if alpha_mask.any() and not alpha_mask.all():
            mask |= alpha_mask
    if not has_rendered_background_geometry:
        # With the synthetic background path we normally do not render a real
        # ground plane, so positive depth is the most complete visible-geometry
        # mask and catches links missed by segmentation-name matching.
        mask |= depth > 0
    return mask
