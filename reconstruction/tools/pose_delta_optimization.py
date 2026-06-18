from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from pose_utility import pose_angle_deg, pose_delta_egocentric


RefinePoseFn = Callable[[object, Mapping[str, object], np.ndarray, int], Tuple[np.ndarray, Mapping[str, object]]]


@dataclass
class PoseDeltaOptimizationConfig:
    """Controls the light-weight multi-frame pose-delta refresh.

    This is deliberately not a BundleSDF neural optimizer. It borrows the same
    idea of per-frame pose deltas plus a reference-frame offset, but it uses the
    repository's ICP/nearest-surface pose refinement as the only pose update.

    The first accepted frame acts as the reference anchor. Its pose defines the
    object coordinate system, so after refreshing other poses we right-multiply a
    single object-frame offset that keeps the anchor pose fixed. No non-anchor
    ground-truth pose is used here.
    """

    rounds: int = 1
    max_dt: float = 0.05
    max_dr_deg: float = 20.0
    anchor_frame: str = ""
    refine_anchor: bool = False


def _anchor_offset(anchor_init: np.ndarray, anchor_refined: np.ndarray) -> np.ndarray:
    """Return object-frame offset X such that anchor_refined @ X == anchor_init."""

    return np.linalg.inv(anchor_refined).astype(np.float32) @ anchor_init.astype(np.float32)


def _apply_object_offset(pose_map: Dict[str, np.ndarray], offset: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        frame: (np.asarray(pose, dtype=np.float32).reshape(4, 4) @ offset).astype(np.float32)
        for frame, pose in pose_map.items()
    }


def optimize_pose_deltas(
    mesh,
    observations: Sequence[Mapping[str, object]],
    init_pose_map: Mapping[str, np.ndarray],
    refine_pose_fn: RefinePoseFn,
    cfg: PoseDeltaOptimizationConfig,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Refresh a set of ICP poses against the same mesh and anchor coordinates.

    Parameters
    ----------
    mesh:
        Current reconstructed mesh used as the pose refinement target.
    observations:
        Frame observations. Each item must contain a string-like ``frame`` key.
    init_pose_map:
        Initial/current ICP poses for the same frames. These are not treated as
        ground truth except the anchor frame, whose pose fixes the coordinate
        system to avoid global drift.
    refine_pose_fn:
        Callback that performs one frame's pose refinement. It must use only
        observation data and the supplied init pose.
    cfg:
        Optimization limits and anchor policy.
    seed:
        Deterministic seed offset for sampling inside the callback.
    """

    if not observations:
        return {}, {"status": "empty", "frames": 0}

    frames = [str(obs["frame"]) for obs in observations]
    anchor = str(cfg.anchor_frame or frames[0])
    if anchor not in init_pose_map:
        anchor = frames[0]

    pose_map: Dict[str, np.ndarray] = {
        frame: np.asarray(init_pose_map[frame], dtype=np.float32).reshape(4, 4).copy()
        for frame in frames
        if frame in init_pose_map
    }
    reports: Dict[str, object] = {}
    rounds = max(1, int(cfg.rounds))

    for round_idx in range(rounds):
        round_reports: Dict[str, object] = {}
        for obs_idx, obs in enumerate(observations):
            frame = str(obs["frame"])
            if frame not in pose_map:
                continue
            if frame == anchor and not bool(cfg.refine_anchor):
                round_reports[frame] = {
                    "status": "anchor_kept",
                    "pose_delta": pose_delta_egocentric(pose_map[frame], pose_map[frame]),
                }
                continue

            init_pose = pose_map[frame]
            refined, info = refine_pose_fn(mesh, obs, init_pose, int(seed) + round_idx * 1000 + obs_idx)
            refined = np.asarray(refined, dtype=np.float32).reshape(4, 4)
            delta = pose_delta_egocentric(init_pose, refined)
            within_limit = (
                float(delta["dt_norm"]) <= float(cfg.max_dt)
                and float(delta["dR_angle_deg"]) <= float(cfg.max_dr_deg)
            )
            ok = bool(info.get("ok", True)) and bool(within_limit)
            if ok:
                pose_map[frame] = refined
            round_reports[frame] = {
                "status": "updated" if ok else "rejected_delta",
                "refine": dict(info),
                "pose_delta": delta,
                "max_dt": float(cfg.max_dt),
                "max_dr_deg": float(cfg.max_dr_deg),
            }
        reports[f"round_{round_idx:02d}"] = round_reports

    anchor_init = np.asarray(init_pose_map[anchor], dtype=np.float32).reshape(4, 4)
    anchor_refined = np.asarray(pose_map[anchor], dtype=np.float32).reshape(4, 4)
    offset = _anchor_offset(anchor_init, anchor_refined)
    aligned_pose_map = _apply_object_offset(pose_map, offset)

    drift_reports = {}
    for frame, pose in aligned_pose_map.items():
        init_pose = np.asarray(init_pose_map[frame], dtype=np.float32).reshape(4, 4)
        drift_reports[frame] = {
            "pose_delta_from_input": pose_delta_egocentric(init_pose, pose),
            "angle_from_input_deg": pose_angle_deg(pose, init_pose),
        }

    return aligned_pose_map, {
        "status": "success",
        "frames": int(len(aligned_pose_map)),
        "anchor_frame": anchor,
        "refine_anchor": bool(cfg.refine_anchor),
        "object_frame_offset": offset.astype(np.float32).reshape(-1).tolist(),
        "rounds": reports,
        "aligned_pose_deltas": drift_reports,
    }
