import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def parse_bounding_box(instance_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = instance_dir / "bounding_box.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    bmin = data.get("min")
    bmax = data.get("max")
    if not (isinstance(bmin, list) and isinstance(bmax, list) and len(bmin) == 3 and len(bmax) == 3):
        return None
    return np.asarray(bmin, dtype=np.float32), np.asarray(bmax, dtype=np.float32)
def _look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-6:
        left = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        left = left / left_norm
    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = cam_pos
    return mat


def _pose_to_matrix(pose: object) -> np.ndarray:
    if hasattr(pose, "to_transformation_matrix"):
        mat = pose.to_transformation_matrix()
        return np.asarray(mat, dtype=np.float32)
    raise RuntimeError("Pose object does not support to_transformation_matrix().")


def _link_name(link: object) -> str:
    get_name = getattr(link, "get_name", None)
    if callable(get_name):
        return str(get_name())
    name = getattr(link, "name", None)
    if isinstance(name, str):
        return name
    return str(name)


def _link_pose_matrix(link: object) -> np.ndarray:
    get_entity_pose = getattr(link, "get_entity_pose", None)
    if callable(get_entity_pose):
        return _pose_to_matrix(get_entity_pose())
    get_pose = getattr(link, "get_pose", None)
    if callable(get_pose):
        return _pose_to_matrix(get_pose())
    pose = getattr(link, "pose", None)
    if pose is not None:
        return _pose_to_matrix(pose)
    raise RuntimeError("Link object does not expose pose/get_pose.")


FRONT_AZIMUTH_RAD = math.radians(0.0)
FRONT_AZIMUTH_HALF_WIDTH_RAD = math.radians(150.0)
FRONT_ELEV_MIN_RAD = math.radians(5.0)
FRONT_ELEV_MAX_RAD = math.radians(50.0)


def _wrap_angle_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp_front_azimuth(az: float) -> float:
    delta = _wrap_angle_pi(az - FRONT_AZIMUTH_RAD)
    delta = float(np.clip(delta, -FRONT_AZIMUTH_HALF_WIDTH_RAD, FRONT_AZIMUTH_HALF_WIDTH_RAD))
    return FRONT_AZIMUTH_RAD + delta


def _view_dir_from_angles(az: float, elev: float) -> np.ndarray:
    az = _clamp_front_azimuth(az)
    elev = float(np.clip(elev, FRONT_ELEV_MIN_RAD, FRONT_ELEV_MAX_RAD))
    x = math.cos(elev) * math.cos(az)
    y = math.cos(elev) * math.sin(az)
    z = math.sin(elev)
    v = np.asarray([x, y, z], dtype=np.float32)
    return v / np.linalg.norm(v)


def generate_diverse_view_dirs(n_views: int) -> List[np.ndarray]:
    if n_views <= 0:
        return []
    if n_views == 1:
        return [_view_dir_from_angles(FRONT_AZIMUTH_RAD, (FRONT_ELEV_MIN_RAD + FRONT_ELEV_MAX_RAD) * 0.5)]

    az_span = FRONT_AZIMUTH_HALF_WIDTH_RAD * 2.0
    elev_span = FRONT_ELEV_MAX_RAD - FRONT_ELEV_MIN_RAD
    golden_ratio = 0.61803398875
    dirs: List[np.ndarray] = []
    for i in range(n_views):
        az_t = (i + 0.5) / float(n_views)
        elev_t = ((i * golden_ratio) % 1.0)
        az = FRONT_AZIMUTH_RAD - FRONT_AZIMUTH_HALF_WIDTH_RAD + az_span * az_t
        elev = FRONT_ELEV_MIN_RAD + elev_span * elev_t
        dirs.append(_view_dir_from_angles(az, elev))
    return dirs
def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def diversity_bonus(d: np.ndarray, chosen: List[np.ndarray]) -> float:
    if not chosen:
        return math.pi
    return min(angular_distance(d, c) for c in chosen)


def _dir_to_az_elev(d: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(d, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return 0.0, 0.0
    v = v / n
    az = math.atan2(float(v[1]), float(v[0]))
    elev = math.asin(float(np.clip(v[2], -1.0, 1.0)))
    return az, elev


def min_az_elev_distance(d: np.ndarray, chosen: List[np.ndarray]) -> Tuple[float, float]:
    if not chosen:
        return math.pi, math.pi
    az, elev = _dir_to_az_elev(d)
    min_az = math.pi
    min_elev = math.pi
    for c in chosen:
        c_az, c_elev = _dir_to_az_elev(c)
        daz = abs(_wrap_angle_pi(az - c_az))
        de = abs(elev - c_elev)
        if daz < min_az:
            min_az = daz
        if de < min_elev:
            min_elev = de
    return min_az, min_elev


def perturb_view_dirs(base_dir: np.ndarray) -> List[np.ndarray]:
    elev = math.asin(float(np.clip(base_dir[2], 0.0, 1.0)))
    az = math.atan2(float(base_dir[1]), float(base_dir[0]))
    dirs: List[np.ndarray] = []
    for da_deg in (6.0, -6.0, 12.0, -12.0, 18.0, -18.0):
        for de_deg in (0.0, 4.0, -4.0, 8.0, -8.0):
            dirs.append(_view_dir_from_angles(az + math.radians(da_deg), elev + math.radians(de_deg)))
    return dirs


def _dir_key(d: np.ndarray) -> Tuple[int, int, int]:
    v = np.asarray(d, dtype=np.float64).reshape(3)
    return (int(round(v[0] * 1_000_000)), int(round(v[1] * 1_000_000)), int(round(v[2] * 1_000_000)))
def _safe_articulation_qpos(art: object) -> Optional[np.ndarray]:
    get_qpos = getattr(art, "get_qpos", None)
    if not callable(get_qpos):
        return None
    try:
        qpos = np.asarray(get_qpos(), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return qpos


def _safe_articulation_qlimits(art: object, ndof: int) -> Optional[np.ndarray]:
    get_qlimits = getattr(art, "get_qlimits", None)
    if not callable(get_qlimits):
        return None
    try:
        qlimits = np.asarray(get_qlimits(), dtype=np.float32)
    except Exception:
        return None
    if qlimits.shape != (ndof, 2):
        return None
    return qlimits


def _small_joint_motion_qpos(
    base_qpos: Optional[np.ndarray],
    qlimits: Optional[np.ndarray],
    view_idx: int,
    n_views: int,
    fraction: float,
    max_delta: float,
) -> Optional[np.ndarray]:
    if base_qpos is None or qlimits is None or len(base_qpos) == 0:
        return None
    fraction = float(np.clip(fraction, 0.0, 0.4))
    max_delta = max(0.0, float(max_delta))
    if fraction <= 0.0 or max_delta <= 0.0:
        return base_qpos.copy()

    qpos = base_qpos.copy()
    phase = 0.0 if n_views <= 1 else view_idx / float(max(1, n_views - 1))
    # Keep early frames close to the default pose, then gently increase and vary.
    amount = 0.35 + 0.65 * ((view_idx * 0.61803398875) % 1.0)
    amount *= 0.55 + 0.45 * phase

    for i, (lo, hi) in enumerate(qlimits):
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        base = float(np.clip(base_qpos[i], lo, hi))
        span = float(hi - lo)
        direction = 1.0 if (hi - base) >= (base - lo) else -1.0
        room = float(hi - base) if direction > 0 else float(base - lo)
        delta = min(room, span * fraction * amount, max_delta)
        if delta <= 1e-6:
            continue
        qpos[i] = base + direction * delta
    return qpos


def _set_articulation_qpos(art: object, qpos: Optional[np.ndarray]) -> None:
    if qpos is None:
        return
    set_qpos = getattr(art, "set_qpos", None)
    if callable(set_qpos):
        set_qpos(qpos)
