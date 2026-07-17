import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional

from recon_utils import (
    DatasetObject,
    ensure_dir,
    list_parts,
    method_object_dir,
    method_pose_ready_dir,
    part_model_name,
    parse_part_id,
    read_json,
    write_json,
)
from relation_graph import find_graph_node, write_relation_graph


def _axis_alignment_for_part(model_dir: Path) -> Optional[Dict[str, object]]:
    path = model_dir / "axis_alignment.json"
    data = read_json(path, default=None)
    return data if isinstance(data, dict) else None


def run_relation_graph_export_object(
    obj: DatasetObject,
    args: argparse.Namespace,
    method: str,
    axis_align_method: str = "",
) -> Dict[str, object]:
    work_root = Path(args.work_root).resolve()
    obj_dir = ensure_dir(method_object_dir(work_root, method, args.split, obj.name))
    pose_root = method_pose_ready_dir(work_root, method, args.split, obj.name)
    graph = write_relation_graph(obj, obj_dir / "relation_graph.json")

    exported_nodes = []
    node_table = {}
    for part_idx, part_name in enumerate(list_parts(obj)):
        part_model = part_model_name(part_name, part_idx)
        part_id = parse_part_id(part_name, part_idx)
        model_dir = pose_root / part_model
        if axis_align_method:
            src_axis = method_pose_ready_dir(work_root, axis_align_method, args.split, obj.name) / part_model / "axis_alignment.json"
            dst_axis = model_dir / "axis_alignment.json"
            if src_axis.exists() and (not dst_axis.exists() or bool(getattr(args, "overwrite", False))):
                try:
                    model_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_axis, dst_axis)
                except Exception:
                    pass
        node = find_graph_node(graph, part_id, part_name) or {
            "part_id": int(part_id),
            "node_id": str(part_name),
            "semantic": str(part_name),
        }
        out = {
            **node,
            "part_name": str(part_name),
            "part_model": str(part_model),
            "mesh_path": str(model_dir / "model.obj"),
            "axis_alignment": _axis_alignment_for_part(model_dir),
            "coordinate_frame": "final_pose_ready_mesh",
        }
        write_json(model_dir / "graph_node.json", out)
        exported_nodes.append(out)
        node_table[str(part_model)] = out

    write_json(obj_dir / "graph_node.json", node_table)
    summary = {
        "method": method,
        "object": obj.name,
        "relation_graph": str(obj_dir / "relation_graph.json"),
        "graph_node_table": str(obj_dir / "graph_node.json"),
        "nodes": len(exported_nodes),
        "edges": len(graph.get("edges", [])) if isinstance(graph, dict) else 0,
        "status": "success",
    }
    write_json(obj_dir / "relation_graph_summary.json", summary)
    return summary
