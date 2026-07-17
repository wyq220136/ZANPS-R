from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from recon_utils import (
    DatasetObject,
    list_parts,
    load_pose,
    natural_sort_key,
    parse_part_id,
    pose_path_for_part_frame,
    read_json,
    write_json,
)


def _part_semantic(part_name: str) -> str:
    tokens = [t for t in str(part_name).replace("-", "_").split("_") if t and not t.isdigit()]
    return "_".join(tokens) if tokens else str(part_name)


def _resolve_mobility_path(obj: DatasetObject, meta: Dict[str, object]) -> Optional[Path]:
    raw = str(meta.get("mobility_path", "") or "").strip()
    if raw:
        p = Path(raw)
        if p.exists():
            return p
        local = obj.root / p.name
        if local.exists():
            return local
    for name in ("mobility_v2.json", "mobility.json"):
        p = obj.root / name
        if p.exists():
            return p
    return None


def _meta_part_layout(meta: Dict[str, object]) -> Dict[int, str]:
    layout = meta.get("part_layout", {}) or {}
    out: Dict[int, str] = {}
    if isinstance(layout, dict):
        for k, v in layout.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    return out


def _mobility_by_id(mobility: object) -> Dict[int, Dict[str, object]]:
    out = {}
    if not isinstance(mobility, list):
        return out
    for item in mobility:
        if not isinstance(item, dict):
            continue
        try:
            out[int(item.get("id"))] = item
        except Exception:
            continue
    return out


def _reference_relative_pose(
    obj: DatasetObject,
    parent_name: str,
    child_name: str,
) -> tuple[Optional[str], Optional[List[float]]]:
    parent_dir = obj.cam_params_dir / parent_name
    child_dir = obj.cam_params_dir / child_name
    if not parent_dir.is_dir() or not child_dir.is_dir():
        return None, None
    parent_frames = {p.stem for p in parent_dir.glob("*.txt")}
    child_frames = {p.stem for p in child_dir.glob("*.txt")}
    for frame in sorted(parent_frames & child_frames, key=natural_sort_key):
        parent_path = pose_path_for_part_frame(obj, parent_name, frame)
        child_path = pose_path_for_part_frame(obj, child_name, frame)
        if parent_path is None or child_path is None:
            continue
        try:
            parent_pose = load_pose(parent_path, "cv")
            child_pose = load_pose(child_path, "cv")
            relative = np.linalg.inv(parent_pose) @ child_pose
            return str(frame), relative.astype(float).reshape(-1).tolist()
        except Exception:
            continue
    return None, None


def build_relation_graph(obj: DatasetObject) -> Dict[str, object]:
    """Build a training-free part relation graph for an object.

    The preferred source is ``meta.json -> mobility_path``. If mobility metadata
    is unavailable, the graph still contains nodes and can later be extended with
    RGB-D adjacency edges.
    """
    meta_path = obj.root / "meta.json"
    meta = read_json(meta_path, default={}) or {}
    mobility_path = _resolve_mobility_path(obj, meta)
    mobility = read_json(mobility_path, default=[]) if mobility_path is not None else []
    mobility_items = _mobility_by_id(mobility)
    layout = _meta_part_layout(meta)

    nodes: List[Dict[str, object]] = []
    for part_name in list_parts(obj):
        part_id = parse_part_id(part_name, fallback=len(nodes))
        mob = mobility_items.get(part_id, {})
        semantic = layout.get(part_id, str(mob.get("name", _part_semantic(part_name))))
        nodes.append(
            {
                "part_id": int(part_id),
                "node_id": str(part_name),
                "semantic": str(semantic),
                "mobility_name": str(mob.get("name", semantic)) if mob else None,
                "parent": None if not mob else int(mob.get("parent", -1)),
                "joint_type": None if not mob else str(mob.get("joint", "unknown")),
            }
        )

    known_part_ids = {int(n["part_id"]) for n in nodes}
    part_name_by_id = {int(n["part_id"]): str(n["node_id"]) for n in nodes}
    edges: List[Dict[str, object]] = []
    for node in nodes:
        part_id = int(node["part_id"])
        mob = mobility_items.get(part_id, {})
        if not mob:
            continue
        parent = int(mob.get("parent", -1))
        if parent < 0:
            continue
        joint_data = mob.get("jointData", {}) or {}
        axis = joint_data.get("axis", {}) if isinstance(joint_data, dict) else {}
        limit = joint_data.get("limit", {}) if isinstance(joint_data, dict) else {}
        reference_frame = None
        rest_relative_pose = None
        if parent in part_name_by_id:
            reference_frame, rest_relative_pose = _reference_relative_pose(
                obj,
                part_name_by_id[parent],
                part_name_by_id[part_id],
            )
        edges.append(
            {
                "parent": parent,
                "child": part_id,
                "parent_in_reconstruction": bool(parent in known_part_ids),
                "type": str(mob.get("joint", "unknown")),
                "axis_origin": axis.get("origin") if isinstance(axis, dict) else None,
                "axis_direction": axis.get("direction") if isinstance(axis, dict) else None,
                "limit": {
                    "a": limit.get("a") if isinstance(limit, dict) else None,
                    "b": limit.get("b") if isinstance(limit, dict) else None,
                    "noLimit": limit.get("noLimit") if isinstance(limit, dict) else None,
                },
                "reference_frame": reference_frame,
                "rest_relative_pose": rest_relative_pose,
            }
        )

    root_candidates = [int(n["part_id"]) for n in nodes if int(n.get("parent") if n.get("parent") is not None else -1) < 0]
    return {
        "object": obj.name,
        "split": obj.split,
        "source": "mobility_v2" if mobility_path is not None else "parts_only",
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "mobility_path": None if mobility_path is None else str(mobility_path),
        "root_candidates": root_candidates,
        "nodes": sorted(nodes, key=lambda x: natural_sort_key(x["node_id"])),
        "edges": edges,
    }


def write_relation_graph(obj: DatasetObject, out_path: Path) -> Dict[str, object]:
    graph = build_relation_graph(obj)
    write_json(out_path, graph)
    return graph


def find_graph_node(graph: Optional[Dict[str, object]], part_id: int, part_name: str = "") -> Optional[Dict[str, object]]:
    if not graph:
        return None
    for node in graph.get("nodes", []):
        if int(node.get("part_id", -999999)) == int(part_id):
            return node
        if part_name and str(node.get("node_id")) == str(part_name):
            return node
    return None
