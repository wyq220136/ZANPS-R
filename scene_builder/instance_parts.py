import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


OBJ_FACE_RE = re.compile(r"(\d+)(?:/(\d*))?(?:/(\d+))?")


@dataclass
class InstanceInfo:
    category: str
    instance_id: str
    instance_name: str
    instance_dir: Path
    urdf_path: Path
    mobility_path: Path
    result_path: Path


@dataclass
class PartEntry:
    part_id: int
    name: str
    obj_stems: List[str]


@dataclass
class PartUrdfInfo:
    part_id: int
    part_name: str
    link_name: str
    mesh_relpaths: List[str]
    visual_part_ids: List[int]
    visual_origin_xyz: Optional[List[float]] = None
    visual_origin_rpy: Optional[List[float]] = None
    mesh_scale: Optional[List[float]] = None


@dataclass
class MobilityLinkInfo:
    link_id: int
    link_name: str
    name: str
    part_ids: List[int]


class SkipInstanceError(RuntimeError):
    def __init__(self, reason: str, details: Optional[Dict[str, object]] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class SapienCameraBufferError(RuntimeError):
    def __init__(self, buffer_name: str, original_error: Exception):
        self.buffer_name = buffer_name
        self.original_error = original_error
        super().__init__(
            f"Failed to read SAPIEN camera buffer '{buffer_name}'. "
            "For dataset export this buffer is required. If this happened with "
            "--background-mode scene or ray tracing enabled, try --background-mode composite "
            f"or --no-rt for this SAPIEN build. SAPIEN error: {original_error}"
        )
def discover_instances(models_root: Path) -> List[InstanceInfo]:
    candidates: List[InstanceInfo] = []
    seen: Set[Tuple[str, str]] = set()

    def collect_from_category_root(category_root: Path) -> None:
        if not category_root.exists() or not category_root.is_dir():
            return
        for category_dir in sorted(category_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for instance_dir in sorted(category_dir.iterdir()):
                if not instance_dir.is_dir():
                    continue
                urdf_path = instance_dir / "mobility.urdf"
                mobility_path = instance_dir / "mobility_v2.json"
                result_path = instance_dir / "result.json"
                if not (urdf_path.exists() and mobility_path.exists() and result_path.exists()):
                    continue
                category = category_dir.name
                instance_id = instance_dir.name
                key = (category, instance_id)
                if key in seen:
                    continue
                seen.add(key)
                instance_name = f"{category}_{instance_id}"
                candidates.append(
                    InstanceInfo(
                        category=category,
                        instance_id=instance_id,
                        instance_name=instance_name,
                        instance_dir=instance_dir,
                        urdf_path=urdf_path,
                        mobility_path=mobility_path,
                        result_path=result_path,
                    )
                )

    # Legacy layout: Models/PartModels*/<category>/<instance_id>
    for part_root in sorted(models_root.glob("PartModels*")):
        collect_from_category_root(part_root)

    # Direct layout: Models/<category>/<instance_id>
    collect_from_category_root(models_root)

    return candidates


def select_instances(
    all_instances: List[InstanceInfo],
    single_instance_name: Optional[str],
    start_instance_name: Optional[str],
    end_instance_name: Optional[str],
) -> List[InstanceInfo]:
    if single_instance_name is None and start_instance_name is None and end_instance_name is None:
        return all_instances
    if single_instance_name is not None:
        if start_instance_name is not None and start_instance_name != single_instance_name:
            raise ValueError(
                "--start-instance cannot be different from --instance when building a single instance."
            )
        if end_instance_name is not None and end_instance_name != single_instance_name:
            raise ValueError(
                "--end-instance cannot be different from --instance when building a single instance."
            )
        hits = [x for x in all_instances if x.instance_name == single_instance_name]
        if not hits:
            raise ValueError(
                f"Instance '{single_instance_name}' not found. "
                "Check category/id or use --all."
            )
        return hits

    start_idx = 0
    end_idx = len(all_instances) - 1
    if start_instance_name is not None:
        for idx, info in enumerate(all_instances):
            if info.instance_name == start_instance_name:
                start_idx = idx
                break
        else:
            raise ValueError(
                f"Start instance '{start_instance_name}' not found. "
                "Check category/id or run without --start-instance."
            )
    if end_instance_name is not None:
        for idx, info in enumerate(all_instances):
            if info.instance_name == end_instance_name:
                end_idx = idx
                break
        else:
            raise ValueError(
                f"End instance '{end_instance_name}' not found. "
                "Check category/id or run without --end-instance."
            )
    if start_idx > end_idx:
        raise ValueError(
            f"Start instance '{start_instance_name}' comes after end instance '{end_instance_name}'."
        )
    return all_instances[start_idx : end_idx + 1]


def _walk_result_nodes(node: object, part_map: Dict[int, PartEntry]) -> None:
    if isinstance(node, dict):
        part_id = node.get("id")
        name = node.get("name")
        objs = node.get("objs", [])
        if isinstance(part_id, int) and isinstance(name, str):
            obj_list: List[str] = []
            if isinstance(objs, list):
                obj_list = [str(x) for x in objs if isinstance(x, str)]
            if part_id not in part_map:
                part_map[part_id] = PartEntry(part_id=part_id, name=name, obj_stems=obj_list)
            else:
                if obj_list:
                    merged = list(part_map[part_id].obj_stems)
                    for stem in obj_list:
                        if stem not in merged:
                            merged.append(stem)
                    part_map[part_id].obj_stems = merged
        children = node.get("children")
        if isinstance(children, list):
            for c in children:
                _walk_result_nodes(c, part_map)
    elif isinstance(node, list):
        for it in node:
            _walk_result_nodes(it, part_map)


def parse_result_parts(result_path: Path) -> Dict[int, PartEntry]:
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    part_map: Dict[int, PartEntry] = {}
    _walk_result_nodes(data, part_map)
    return part_map


def parse_mobility_ids(mobility_path: Path) -> List[int]:
    with mobility_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ids: Set[int] = set()
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            parts = item.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict):
                        p_id = p.get("id")
                        if isinstance(p_id, int):
                            ids.add(p_id)
    return sorted(ids)


def parse_mobility_parts(mobility_path: Path) -> Dict[int, str]:
    with mobility_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    part_map: Dict[int, str] = {}
    if not isinstance(data, list):
        return part_map
    for item in data:
        if not isinstance(item, dict):
            continue
        parts = item.get("parts")
        if not isinstance(parts, list):
            continue
        for p in parts:
            if not isinstance(p, dict):
                continue
            p_id = p.get("id")
            p_name = p.get("name")
            if isinstance(p_id, int):
                if isinstance(p_name, str) and p_name.strip():
                    part_map[p_id] = p_name.strip()
                else:
                    part_map.setdefault(p_id, f"part_{p_id}")
    return part_map


def parse_mobility_links(mobility_path: Path) -> Dict[int, MobilityLinkInfo]:
    with mobility_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, MobilityLinkInfo] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        link_id = item.get("id")
        if not isinstance(link_id, int) or link_id < 0:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"link_{link_id}"
        parts = item.get("parts")
        part_ids: List[int] = []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict):
                    p_id = p.get("id")
                    if isinstance(p_id, int):
                        part_ids.append(p_id)
        out[link_id] = MobilityLinkInfo(
            link_id=link_id,
            link_name=f"link_{link_id}",
            name=name.strip(),
            part_ids=part_ids,
        )
    return out


def parse_urdf_part_infos(urdf_path: Path, mobility_part_map: Dict[int, str]) -> Dict[int, PartUrdfInfo]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    infos: Dict[int, PartUrdfInfo] = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "").strip()
        if not link_name:
            continue
        per_part_meshes: Dict[int, List[str]] = {}
        for visual in link.findall("visual"):
            visual_name = visual.attrib.get("name", "")
            m = re.search(r"-(\d+)$", visual_name)
            if not m:
                continue
            part_id = int(m.group(1))
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_file = mesh.attrib.get("filename", "").strip()
            if not mesh_file:
                continue
            per_part_meshes.setdefault(part_id, []).append(mesh_file)
        for part_id, mesh_list in per_part_meshes.items():
            mesh_unique = []
            for p in mesh_list:
                if p not in mesh_unique:
                    mesh_unique.append(p)
            part_name = mobility_part_map.get(part_id, f"part_{part_id}")
            infos[part_id] = PartUrdfInfo(
                part_id=part_id,
                part_name=part_name,
                link_name=link_name,
                mesh_relpaths=mesh_unique,
                visual_part_ids=[part_id],
            )
    return infos


def parse_urdf_link_infos(urdf_path: Path, mobility_link_map: Dict[int, MobilityLinkInfo]) -> Dict[int, PartUrdfInfo]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    infos: Dict[int, PartUrdfInfo] = {}
    link_visual_map: Dict[str, List[Tuple[int, str]]] = {}
    link_visual_meta: Dict[str, Dict[int, Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]]] = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "").strip()
        if not link_name:
            continue
        visuals: List[Tuple[int, str]] = []
        visual_meta: Dict[int, Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]] = {}
        for visual in link.findall("visual"):
            visual_name = visual.attrib.get("name", "")
            part_match = re.search(r"-(\d+)$", visual_name)
            if not part_match:
                continue
            part_id = int(part_match.group(1))
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_file = mesh.attrib.get("filename", "").strip()
            if mesh_file:
                visuals.append((part_id, mesh_file))
                origin = visual.find("./origin")
                xyz: Optional[List[float]] = None
                rpy: Optional[List[float]] = None
                if origin is not None:
                    xyz_attr = origin.attrib.get("xyz", "").strip()
                    rpy_attr = origin.attrib.get("rpy", "").strip()
                    if xyz_attr:
                        try:
                            xyz = [float(v) for v in xyz_attr.split()]
                        except Exception:
                            xyz = None
                    if rpy_attr:
                        try:
                            rpy = [float(v) for v in rpy_attr.split()]
                        except Exception:
                            rpy = None
                scale_attr = mesh.attrib.get("scale", "").strip()
                scale: Optional[List[float]] = None
                if scale_attr:
                    try:
                        values = [float(v) for v in scale_attr.split()]
                        if len(values) == 1:
                            scale = values * 3
                        elif len(values) >= 3:
                            scale = values[:3]
                    except Exception:
                        scale = None
                visual_meta.setdefault(part_id, (xyz, rpy, scale))
        link_visual_map[link_name] = visuals
        link_visual_meta[link_name] = visual_meta

    for link_id, link_info in mobility_link_map.items():
        target_part_ids = set(link_info.part_ids)
        best_link_name = ""
        best_meshes: List[str] = []
        best_visual_part_ids: List[int] = []
        best_visual_meta: Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]] = (
            None,
            None,
            None,
        )
        best_match_count = 0

        for link_name, visuals in link_visual_map.items():
            matched_meshes: List[str] = []
            matched_part_ids: List[int] = []
            matched_visual_meta: Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]] = (
                None,
                None,
                None,
            )
            matched_count = 0
            for visual_part_id, mesh_file in visuals:
                if visual_part_id not in target_part_ids:
                    continue
                matched_count += 1
                if visual_part_id not in matched_part_ids:
                    matched_part_ids.append(visual_part_id)
                if mesh_file not in matched_meshes:
                    matched_meshes.append(mesh_file)
                if matched_visual_meta == (None, None, None):
                    matched_visual_meta = link_visual_meta.get(link_name, {}).get(
                        visual_part_id,
                        (None, None, None),
                    )
            if matched_count > best_match_count:
                best_match_count = matched_count
                best_link_name = link_name
                best_meshes = matched_meshes
                best_visual_part_ids = matched_part_ids
                best_visual_meta = matched_visual_meta

        # Do not blindly assume mobility id N maps to URDF link_N. Some PartNet
        # instances have sparse or shifted mobility ids; using link_N there
        # extracts the wrong mesh and wrong segmentation mask.
        if best_match_count <= 0:
            best_link_name = ""
            best_meshes = []
            best_visual_part_ids = []
            best_visual_meta = (None, None, None)

        best_xyz, best_rpy, best_scale = best_visual_meta

        infos[link_id] = PartUrdfInfo(
            part_id=link_id,
            part_name=link_info.name,
            link_name=best_link_name,
            mesh_relpaths=best_meshes,
            visual_part_ids=best_visual_part_ids,
            visual_origin_xyz=best_xyz,
            visual_origin_rpy=best_rpy,
            mesh_scale=best_scale,
        )
    return infos


def parse_urdf_movable_child_links(urdf_path: Path) -> Set[str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    movable_links: Set[str] = set()
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "").strip().lower()
        if not joint_type or joint_type == "fixed":
            continue
        child = joint.find("child")
        if child is None:
            continue
        child_link = child.attrib.get("link", "").strip()
        if child_link:
            movable_links.add(child_link)
    return movable_links


def build_part_layout(
    mobility_part_map: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
) -> Dict[int, str]:
    ids = sorted(set(mobility_part_map.keys()) | set(urdf_part_infos.keys()))
    out: Dict[int, str] = {}
    for pid in ids:
        out[pid] = mobility_part_map.get(pid, urdf_part_infos[pid].part_name)
    return out


def build_link_layout(
    mobility_link_map: Dict[int, MobilityLinkInfo],
    urdf_link_infos: Dict[int, PartUrdfInfo],
) -> Dict[int, str]:
    ids = sorted(set(mobility_link_map.keys()) | set(urdf_link_infos.keys()))
    out: Dict[int, str] = {}
    for pid in ids:
        if pid in mobility_link_map:
            out[pid] = mobility_link_map[pid].name
        else:
            out[pid] = urdf_link_infos[pid].part_name
    return out


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "part"


def part_folder_name(part_id: int, name: str) -> str:
    return f"{part_id:04d}_{sanitize_name(name)}"


def parse_exclude_keywords(value: str) -> List[str]:
    return [x.strip().lower() for x in str(value).split(",") if x.strip()]


def filter_part_layout(
    part_layout: Dict[int, str],
    exclude_keywords: List[str],
) -> Tuple[Dict[int, str], Dict[int, str]]:
    if not exclude_keywords:
        return dict(part_layout), {}
    kept: Dict[int, str] = {}
    excluded: Dict[int, str] = {}
    for part_id, part_name in part_layout.items():
        name_l = part_name.lower()
        if any(keyword in name_l for keyword in exclude_keywords):
            excluded[part_id] = part_name
        else:
            kept[part_id] = part_name
    return kept, excluded


def filter_part_layout_by_movable_joints(
    instance: InstanceInfo,
    part_layout: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
) -> Tuple[Dict[int, str], Dict[int, str]]:
    movable_links = parse_urdf_movable_child_links(instance.urdf_path)

    kept: Dict[int, str] = {}
    excluded: Dict[int, str] = {}
    for part_id, part_name in part_layout.items():
        info = urdf_part_infos.get(part_id)
        link_name = info.link_name if info is not None else ""
        if link_name and link_name in movable_links:
            kept[part_id] = part_name
        else:
            excluded[part_id] = part_name

    return kept, excluded


def _copy_texture(
    src_dir: Path,
    map_path: str,
    out_dir: Path,
    tex_prefix: str,
) -> str:
    src_tex = src_dir / map_path
    if not src_tex.exists():
        src_tex = src_dir / Path(map_path).name
    if not src_tex.exists():
        return map_path
    dst_name = f"{tex_prefix}_{src_tex.name}"
    dst_tex = out_dir / dst_name
    if not dst_tex.exists():
        shutil.copy2(src_tex, dst_tex)
    return dst_name


def resolve_obj_paths_from_urdf_meshes(instance_dir: Path, mesh_relpaths: List[str]) -> List[Path]:
    obj_paths: List[Path] = []
    for rel in mesh_relpaths:
        rel_path = Path(rel)
        candidate = instance_dir / rel_path
        if candidate.suffix.lower() != ".obj":
            continue
        if candidate.exists() and candidate not in obj_paths:
            obj_paths.append(candidate)
    return obj_paths


def _resolve_mesh_path(instance_dir: Path, mesh_relpath: str) -> Optional[Path]:
    rel = mesh_relpath.strip()
    if not rel:
        return None
    # Ignore external URI-like resources that are not local filesystem paths.
    if "://" in rel and not rel.lower().startswith("file://"):
        return None
    if rel.lower().startswith("file://"):
        rel = rel[7:]
    mesh_path = Path(rel)
    if not mesh_path.is_absolute():
        mesh_path = instance_dir / mesh_path
    return mesh_path


def find_missing_mesh_files(instance_dir: Path, urdf_part_infos: Dict[int, PartUrdfInfo]) -> List[str]:
    missing: List[str] = []
    seen: Set[str] = set()
    for info in urdf_part_infos.values():
        for rel in info.mesh_relpaths:
            if rel in seen:
                continue
            seen.add(rel)
            mesh_path = _resolve_mesh_path(instance_dir, rel)
            if mesh_path is None:
                continue
            if not mesh_path.exists():
                missing.append(rel)
    return missing
def merge_part_meshes(
    obj_paths: List[Path],
    output_obj: Path,
    output_mtl: Path,
) -> Dict[str, object]:
    obj_paths = [p for p in obj_paths if p.exists()]
    if not obj_paths:
        return {
            "ok": False,
            "reason": "no_obj_files",
            "requested_paths": [],
        }

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_obj.parent

    material_rename: Dict[Tuple[str, str], str] = {}
    merged_mtl_lines: List[str] = []
    mat_count = 0

    for obj_idx, obj_path in enumerate(obj_paths):
        src_dir = obj_path.parent
        local_mtl: Optional[Path] = None
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("mtllib "):
                    mtl_name = line.strip().split(" ", 1)[1]
                    mtl_path = src_dir / mtl_name
                    if mtl_path.exists():
                        local_mtl = mtl_path
                    break
        if local_mtl is None:
            guess = obj_path.with_suffix(".mtl")
            if guess.exists():
                local_mtl = guess
        if local_mtl is None:
            continue

        current_old_material: Optional[str] = None
        with local_mtl.open("r", encoding="utf-8", errors="ignore") as mf:
            for raw_line in mf:
                line = raw_line.rstrip("\n")
                if line.startswith("newmtl "):
                    old_name = line.split(" ", 1)[1].strip()
                    new_name = f"mat_{mat_count}"
                    mat_count += 1
                    material_rename[(obj_path.name, old_name)] = new_name
                    current_old_material = old_name
                    merged_mtl_lines.append(f"newmtl {new_name}\n")
                    continue
                if line.startswith("map_"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and current_old_material is not None:
                        copied_name = _copy_texture(
                            src_dir=local_mtl.parent,
                            map_path=parts[1].strip(),
                            out_dir=out_dir,
                            tex_prefix=f"tex{obj_idx}",
                        )
                        merged_mtl_lines.append(f"{parts[0]} {copied_name}\n")
                        continue
                merged_mtl_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")

    with output_mtl.open("w", encoding="utf-8") as out_mtl_f:
        out_mtl_f.writelines(merged_mtl_lines)

    v_offset = 0
    vt_offset = 0
    vn_offset = 0
    total_faces = 0

    with output_obj.open("w", encoding="utf-8") as out_obj_f:
        out_obj_f.write(f"mtllib {output_mtl.name}\n\n")
        for obj_path in obj_paths:
            local_v = 0
            local_vt = 0
            local_vn = 0
            out_obj_f.write(f"# merged from {obj_path.name}\n")
            with obj_path.open("r", encoding="utf-8", errors="ignore") as in_obj_f:
                for raw_line in in_obj_f:
                    line = raw_line.strip()
                    if not line:
                        out_obj_f.write("\n")
                        continue
                    if line.startswith("mtllib "):
                        continue
                    if line.startswith("v "):
                        out_obj_f.write(raw_line)
                        local_v += 1
                        continue
                    if line.startswith("vt "):
                        out_obj_f.write(raw_line)
                        local_vt += 1
                        continue
                    if line.startswith("vn "):
                        out_obj_f.write(raw_line)
                        local_vn += 1
                        continue
                    if line.startswith("usemtl "):
                        old_mtl = line.split(" ", 1)[1]
                        new_mtl = material_rename.get((obj_path.name, old_mtl), old_mtl)
                        out_obj_f.write(f"usemtl {new_mtl}\n")
                        continue
                    if line.startswith("f "):
                        def _shift_index(match: re.Match) -> str:
                            vi = int(match.group(1)) + v_offset
                            vti = match.group(2)
                            vni = match.group(3)
                            if vti is not None and vti != "":
                                vti_new = int(vti) + vt_offset
                            else:
                                vti_new = None
                            if vni is not None and vni != "":
                                vni_new = int(vni) + vn_offset
                            else:
                                vni_new = None
                            if vti_new is not None and vni_new is not None:
                                return f"{vi}/{vti_new}/{vni_new}"
                            if vti_new is not None:
                                return f"{vi}/{vti_new}"
                            if vni_new is not None:
                                return f"{vi}//{vni_new}"
                            return f"{vi}"

                        face = line[2:]
                        new_face = OBJ_FACE_RE.sub(_shift_index, face)
                        out_obj_f.write(f"f {new_face}\n")
                        total_faces += 1
                        continue
                    out_obj_f.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")

            v_offset += local_v
            vt_offset += local_vt
            vn_offset += local_vn
            out_obj_f.write("\n")

    return {
        "ok": True,
        "obj_count": len(obj_paths),
        "face_count": total_faces,
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
    }


def build_part_models(
    instance: InstanceInfo,
    part_layout: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
    output_models_root_dir: Path,
) -> Dict[int, Dict[str, object]]:
    output_models_root_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[int, Dict[str, object]] = {}
    for part_id, part_name in sorted(part_layout.items(), key=lambda x: x[0]):
        folder = output_models_root_dir / part_folder_name(part_id, part_name)
        folder.mkdir(parents=True, exist_ok=True)
        part_info = urdf_part_infos.get(part_id)
        if part_info is None:
            outputs[part_id] = {
                "ok": False,
                "reason": "part_not_found_in_urdf_visual",
                "name": part_name,
            }
            continue
        obj_paths = resolve_obj_paths_from_urdf_meshes(instance.instance_dir, part_info.mesh_relpaths)
        merged_obj = folder / "model.obj"
        merged_mtl = folder / "model.mtl"
        merge_info = merge_part_meshes(
            obj_paths=obj_paths,
            output_obj=merged_obj,
            output_mtl=merged_mtl,
        )
        merge_info["name"] = part_name
        merge_info["link_name"] = part_info.link_name
        merge_info["mesh_relpaths"] = part_info.mesh_relpaths
        outputs[part_id] = merge_info
    return outputs
