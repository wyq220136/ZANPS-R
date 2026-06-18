import argparse
import concurrent.futures
import json
import math
import hashlib
import os
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build articulated-part dataset from SAPIEN PartModels with SAPIEN ray tracing."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--instance",
        type=str,
        help="Single instance name, e.g. bottle_3398",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Build all discovered instances.",
    )
    parser.add_argument(
        "--start-instance",
        type=str,
        default=None,
        help="When used with --all, start building from this instance name (inclusive), e.g. bottle_3398.",
    )
    parser.add_argument(
        "--end-instance",
        type=str,
        default=None,
        help="When used with --all, stop building at this instance name (inclusive), e.g. bottle_3398.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/Models",
        help="Root containing PartModels folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="dataset_train",
        help="Dataset output root.",
    )
    parser.add_argument("--views", type=int, default=50, help="Views per instance.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-deg", type=float, default=35.0)
    parser.add_argument("--radius-scale", type=float, default=1.15)
    parser.add_argument("--target-max-extent", type=float, default=0.6)
    parser.add_argument("--min-object-coverage", type=float, default=0.08)
    parser.add_argument("--max-object-coverage", type=float, default=0.65)
    parser.add_argument(
        "--min-part-mask-pixels",
        type=int,
        default=30,
        help="Minimum visible mask pixels required for each movable joint/link part in an accepted view.",
    )
    parser.add_argument(
        "--min-part-mask-coverage",
        type=float,
        default=0.00015,
        help="Additional per-image minimum coverage for each movable joint/link part.",
    )
    parser.add_argument(
        "--require-all-part-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every exported movable joint/link part to be visible in accepted frames.",
    )
    parser.add_argument(
        "--view-candidate-multiplier",
        type=int,
        default=16,
        help="How many candidate camera directions to test per requested source view.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--exclude-part-keywords",
        type=str,
        default="base_body,frame,body",
        help=(
            "Comma-separated case-insensitive keywords for part/link names to exclude "
            "from exported models, masks, and cam_params. Empty string disables this filter."
        ),
    )
    parser.add_argument(
        "--render-ground",
        action="store_true",
        help="Render ground plane. Default is off to avoid object-ground intersection/occlusion.",
    )
    rt_group = parser.add_mutually_exclusive_group()
    rt_group.add_argument(
        "--rt",
        dest="rt",
        action="store_true",
        default=True,
        help="Use ray-tracing renderer settings. Enabled by default.",
    )
    rt_group.add_argument(
        "--no-rt",
        dest="rt",
        action="store_false",
        help="Disable ray-tracing renderer settings.",
    )
    rt_camera_group = parser.add_mutually_exclusive_group()
    rt_camera_group.add_argument(
        "--rt-camera-shader",
        dest="rt_camera_shader",
        action="store_true",
        default=False,
        help=(
            "Use the ray-tracing camera shader for saved RGB. Off by default because "
            "dataset export needs Position/Segmentation buffers for depth and masks."
        ),
    )
    rt_camera_group.add_argument(
        "--no-rt-camera-shader",
        dest="rt_camera_shader",
        action="store_false",
        help=(
            "Keep the default camera shader while retaining other rt settings. "
            "Use this if your local SAPIEN build cannot export Position/Segmentation under rt."
        ),
    )
    parser.add_argument(
        "--rt-spp",
        type=int,
        default=64,
        help="Ray-tracing samples per pixel. 32 is faster; 64/128 is cleaner.",
    )
    parser.add_argument(
        "--rt-path-depth",
        type=int,
        default=8,
        help="Ray-tracing path depth if supported by the installed SAPIEN version.",
    )
    parser.add_argument(
        "--rt-denoiser",
        type=str,
        default="optix",
        choices=["optix", "oidn", "none"],
        help="Ray-tracing denoiser. Use 'none' to disable denoising.",
    )
    parser.add_argument(
        "--background-mode",
        type=str,
        default="composite",
        choices=["plain", "composite", "scene"],
        help=(
            "plain keeps the original renderer background; composite replaces non-object "
            "pixels in RGB with deterministic non-gray studio/tabletop backgrounds while "
            "leaving depth, masks, segmentation, and poses unchanged; scene adds simple "
            "SAPIEN ground/wall geometry and randomized lighting."
        ),
    )
    parser.add_argument(
        "--background-variants",
        type=int,
        default=4,
        help=(
            "Number of RGB background variants per camera view. Total exported frames "
            "become views * background_variants when background_mode=composite. "
            "Plain and scene modes render one frame per view."
        ),
    )
    parser.add_argument(
        "--background-seed",
        type=int,
        default=2026,
        help="Seed for deterministic per-instance background generation.",
    )
    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument(
        "--joint-motion",
        dest="joint_motion",
        action="store_true",
        default=True,
        help="Apply small deterministic joint motion per source view. Enabled by default.",
    )
    joint_group.add_argument(
        "--no-joint-motion",
        dest="joint_motion",
        action="store_false",
        help="Keep all articulation joints at their loaded default positions.",
    )
    parser.add_argument(
        "--joint-motion-fraction",
        type=float,
        default=0.01,
        help="Maximum fraction of each limited joint range to use for motion.",
    )
    parser.add_argument(
        "--joint-motion-max-delta",
        type=float,
        default=3.0,
        help="Maximum absolute qpos delta for a moved joint; keeps poses modest.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes. Use 1 for single-process; <=0 means auto.",
    )
    return parser.parse_args()


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
            )
    return infos


def parse_urdf_link_infos(urdf_path: Path, mobility_link_map: Dict[int, MobilityLinkInfo]) -> Dict[int, PartUrdfInfo]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    infos: Dict[int, PartUrdfInfo] = {}
    link_mesh_map: Dict[str, List[str]] = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "").strip()
        if not link_name:
            continue
        meshes: List[str] = []
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_file = mesh.attrib.get("filename", "").strip()
            if mesh_file:
                meshes.append(mesh_file)
        uniq: List[str] = []
        for m in meshes:
            if m not in uniq:
                uniq.append(m)
        link_mesh_map[link_name] = uniq

    for link_id, link_info in mobility_link_map.items():
        mesh_list = link_mesh_map.get(link_info.link_name, [])
        infos[link_id] = PartUrdfInfo(
            part_id=link_id,
            part_name=link_info.name,
            link_name=link_info.link_name,
            mesh_relpaths=mesh_list,
        )
    return infos


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


def _joint_type_name(joint: object) -> str:
    get_type = getattr(joint, "get_type", None)
    if callable(get_type):
        try:
            jt = get_type()
            if isinstance(jt, str):
                return jt.lower().strip()
        except Exception:
            pass
    jt = getattr(joint, "type", None)
    if isinstance(jt, str):
        return jt.lower().strip()
    return ""


def _joint_dof_value(joint: object) -> Optional[int]:
    get_dof = getattr(joint, "get_dof", None)
    if callable(get_dof):
        try:
            v = get_dof()
            if isinstance(v, (int, np.integer)):
                return int(v)
        except Exception:
            pass
    dof = getattr(joint, "dof", None)
    if isinstance(dof, (int, np.integer)):
        return int(dof)
    return None


def _joint_child_link_name(joint: object) -> Optional[str]:
    for attr in ("get_child_link", "get_child_articulation_link"):
        fn = getattr(joint, attr, None)
        if callable(fn):
            try:
                link = fn()
            except Exception:
                link = None
            if link is not None:
                name = _link_name(link)
                if name:
                    return name
    link = getattr(joint, "child_link", None)
    if link is not None:
        name = _link_name(link)
        if name:
            return name
    return None


def filter_part_layout_by_movable_joints(
    instance: InstanceInfo,
    part_layout: Dict[int, str],
    urdf_part_infos: Dict[int, PartUrdfInfo],
) -> Tuple[Dict[int, str], Dict[int, str]]:
    try:
        import sapien  # type: ignore
    except Exception:
        return dict(part_layout), {}

    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load(str(instance.urdf_path))
    if art is None:
        return dict(part_layout), {}

    movable_links: Set[str] = set()
    for joint in art.get_joints():
        child_name = _joint_child_link_name(joint)
        if not child_name:
            continue
        joint_type = _joint_type_name(joint)
        dof = _joint_dof_value(joint)
        is_movable = False
        if dof is not None:
            is_movable = dof > 0
        elif joint_type:
            is_movable = joint_type not in {"fixed"}
        if is_movable:
            movable_links.add(child_name)

    kept: Dict[int, str] = {}
    excluded: Dict[int, str] = {}
    for part_id, part_name in part_layout.items():
        info = urdf_part_infos.get(part_id)
        link_name = info.link_name if info is not None else ""
        if link_name and link_name in movable_links:
            kept[part_id] = part_name
        else:
            excluded[part_id] = part_name

    if not kept:
        return dict(part_layout), {}
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


def is_missing_asset_error(message: str) -> bool:
    msg = message.lower()
    return (
        "file not found" in msg
        or "no such file" in msg
        or ("textured_objs" in msg and ".obj" in msg)
    )


def write_skip_reason(out_dir: Path, reason: str, details: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "skipped", "reason": reason, "details": details}
    with (out_dir / "skip_reason.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
FRONT_ELEV_MAX_RAD = math.radians(75.0)


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


def mask_is_complete_and_sized(
    mask: np.ndarray,
    min_ratio: float,
    max_ratio: float,
) -> Tuple[bool, float]:
    h, w = mask.shape
    area = float(mask.sum())
    ratio = area / float(h * w)
    if area <= 0:
        return False, 0.0
    edge_touch = (
        mask[0, :].any()
        or mask[h - 1, :].any()
        or mask[:, 0].any()
        or mask[:, w - 1].any()
    )
    ok = (not edge_touch) and (ratio >= min_ratio) and (ratio <= max_ratio)
    return ok, ratio


def choose_entity_segmentation_channel(seg: np.ndarray, valid_entity_ids: Set[int]) -> int:
    if seg.ndim != 3 or seg.shape[2] < 2:
        return 0
    scores: List[Tuple[int, int]] = []
    for ch in (0, 1):
        vals = np.unique(seg[..., ch].astype(np.int64))
        overlap = sum(1 for v in vals.tolist() if int(v) in valid_entity_ids)
        scores.append((overlap, ch))
    scores.sort(reverse=True)
    return scores[0][1]


def parse_link_index_from_entity_name(name: str) -> Optional[int]:
    m = re.search(r"(?:^|[^0-9])link_(\d+)(?:$|[^0-9])", name)
    if not m:
        return None
    return int(m.group(1))


def evaluate_all_parts_in_frame(
    seg_channel: np.ndarray,
    part_to_entity_ids: Dict[int, List[int]],
    min_object_coverage: float,
    max_object_coverage: float,
    union_entity_ids: Optional[np.ndarray] = None,
) -> Tuple[bool, float]:
    union_mask = compute_union_mask(
        seg_channel,
        part_to_entity_ids,
        union_entity_ids=union_entity_ids,
    )
    if int(union_mask.sum()) == 0:
        return False, 0.0
    union_ok, ratio = mask_is_complete_and_sized(
        union_mask,
        min_ratio=min_object_coverage,
        max_ratio=max_object_coverage,
    )
    return union_ok, ratio


def compute_union_mask(
    seg_channel: np.ndarray,
    part_to_entity_ids: Dict[int, List[int]],
    union_entity_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    if union_entity_ids is not None:
        if union_entity_ids.size == 0:
            return np.zeros_like(seg_channel, dtype=bool)
        return np.isin(seg_channel, union_entity_ids)
    union_mask = np.zeros_like(seg_channel, dtype=bool)
    for entity_ids in part_to_entity_ids.values():
        if not entity_ids:
            continue
        union_mask |= np.isin(seg_channel, entity_ids)
    return union_mask


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


def _camera_picture(camera: object, name: str) -> np.ndarray:
    try:
        return camera.get_picture(name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read SAPIEN camera buffer '{name}'. "
            "For dataset export this buffer is required. If this happened with "
            "--background-mode scene or ray tracing enabled, try --background-mode composite "
            "or --no-rt for this SAPIEN build."
        ) from exc


def max_visible_part_area(seg_channel: np.ndarray, part_to_entity_ids: Dict[int, List[int]]) -> int:
    max_area = 0
    for entity_ids in part_to_entity_ids.values():
        if not entity_ids:
            continue
        area = int(np.isin(seg_channel, entity_ids).sum())
        if area > max_area:
            max_area = area
    return max_area


def visible_part_areas(seg_channel: np.ndarray, part_to_entity_ids: Dict[int, List[int]]) -> Dict[int, int]:
    areas: Dict[int, int] = {}
    for part_id, entity_ids in part_to_entity_ids.items():
        if not entity_ids:
            areas[part_id] = 0
            continue
        areas[part_id] = int(np.isin(seg_channel, entity_ids).sum())
    return areas


def parts_visibility_ok(
    seg_channel: np.ndarray,
    part_to_entity_ids: Dict[int, List[int]],
    min_part_pixels: int,
    require_all_parts: bool,
) -> Tuple[bool, Dict[int, int]]:
    areas = visible_part_areas(seg_channel, part_to_entity_ids)
    if not areas:
        return False, areas
    if require_all_parts:
        ok = all(area >= int(min_part_pixels) for area in areas.values())
    else:
        ok = max(areas.values()) >= int(min_part_pixels)
    return bool(ok), areas


def mask_center_offset_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 1.0
    h, w = mask.shape
    cx = float(xs.mean())
    cy = float(ys.mean())
    ox = (cx - (w - 1) * 0.5) / max(1.0, w * 0.5)
    oy = (cy - (h - 1) * 0.5) / max(1.0, h * 0.5)
    return float(math.sqrt(ox * ox + oy * oy))


def view_quality_score(mask: np.ndarray, ratio: float, target_ratio: float) -> float:
    h, w = mask.shape
    if int(mask.sum()) == 0:
        return -1e9
    edge_touch = (
        mask[0, :].any()
        or mask[h - 1, :].any()
        or mask[:, 0].any()
        or mask[:, w - 1].any()
    )
    ratio_penalty = abs(math.log(max(ratio, 1e-6) / max(target_ratio, 1e-6)))
    center_penalty = mask_center_offset_ratio(mask) * 1.2
    score = 1.0 - ratio_penalty - center_penalty
    if edge_touch:
        score -= 2.0
    return score


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
        "ground_clearance": 0.015,
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
    min_object_coverage: float,
    max_object_coverage: float,
    min_part_mask_pixels: int,
    min_part_mask_coverage: float,
    require_all_part_visible: bool,
    view_candidate_multiplier: int,
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
        configure_ray_tracing(
            sapien_module=sapien,
            rt_camera_shader=rt_camera_shader,
            rt_spp=rt_spp,
            rt_path_depth=rt_path_depth,
            rt_denoiser=rt_denoiser,
        )

    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    lighting_info = _configure_scene_lighting(
        scene,
        seed=per_instance_bg_seed,
        rich_scene=use_scene_background,
    )

    camera = scene.add_camera(
        "camera",
        width=width,
        height=height,
        fovy=np.deg2rad(fov_deg),
        near=0.1,
        far=100.0,
    )

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    bbox = parse_bounding_box(instance.instance_dir)
    object_scale = 1.0
    if bbox is not None:
        bmin, bmax = bbox
        extent = bmax - bmin
        max_extent = float(np.max(extent))
        if max_extent > 1e-6:
            object_scale = float(np.clip(target_max_extent / max_extent, 0.25, 4.0))
    loader.scale = object_scale
    art = loader.load(str(instance.urdf_path))
    if art is None:
        raise RuntimeError(f"failed to load urdf: {instance.urdf_path}")
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
    ground_clearance = max(0.015, 0.025 * float(object_scale))
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

    part_to_entity_ids: Dict[int, List[int]] = {}
    for part_id, info in urdf_part_infos.items():
        if part_id not in part_layout:
            continue
        # Match by URDF link name (e.g. link_0, link_1) against id_to_entity values.
        part_to_entity_ids[part_id] = linkname_to_entity_ids.get(info.link_name, [])
    missing = [pid for pid in part_layout.keys() if pid not in part_to_entity_ids]
    if missing:
        raise RuntimeError(
            f"Missing URDF/link mapping for part ids: {missing}. "
            "Cannot build mask-link correspondence."
        )
    valid_entity_ids = {eid for entity_ids in part_to_entity_ids.values() for eid in entity_ids}
    if not valid_entity_ids:
        raise RuntimeError(
            "No SAPIEN entity ids matched exported part links. "
            "Cannot build mask-link correspondence."
        )
    # Cached flattened id arrays reduce repeated np.isin overhead in view search/render loops.
    union_entity_ids = np.asarray(sorted(valid_entity_ids), dtype=np.int32)
    part_entity_ids_np: Dict[int, np.ndarray] = {
        part_id: np.asarray(sorted(set(entity_ids)), dtype=np.int32)
        for part_id, entity_ids in part_to_entity_ids.items()
    }
    linkname_to_link = {_link_name(link): link for link in art.get_links()}

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
    chosen_seg_channel: Optional[int] = None
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
                cam_pos = target + d * current_distance
                pose44 = _look_at(cam_pos, target)
                camera.entity.set_pose(sapien.Pose(pose44))
                scene.step()
                scene.update_render()
                camera.take_picture()
                seg = _camera_picture(camera, "Segmentation")
                if chosen_seg_channel is None:
                    chosen_seg_channel = choose_entity_segmentation_channel(seg, valid_entity_ids)
                seg_channel = seg[..., chosen_seg_channel].astype(np.int32)
                union_mask = compute_union_mask(
                    seg_channel,
                    part_to_entity_ids,
                    union_entity_ids=union_entity_ids,
                )
                if int(union_mask.sum()) == 0:
                    if has_rendered_background_geometry:
                        ratios.append(0.0)
                        continue
                    # Fallback when segmentation ids are unreliable in this build/view.
                    position = _camera_picture(camera, "Position")
                    depth_mask = (-position[..., 2]) > 0
                    if int(depth_mask.sum()) == 0:
                        ratios.append(0.0)
                        continue
                    union_mask = depth_mask
                h, w = union_mask.shape
                edge_touch = (
                    union_mask[0, :].any()
                    or union_mask[h - 1, :].any()
                    or union_mask[:, 0].any()
                    or union_mask[:, w - 1].any()
                )
                if edge_touch:
                    edge_touches += 1
                ratio = float(union_mask.sum()) / float(h * w)
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
    max_center_offset = 0.38
    distance_factors = (0.95, 1.05, 1.18, 1.32, 1.50, 0.88)
    for d in candidate_dirs:
        if len(accepted_dirs) >= n_views:
            break
        best: Optional[Tuple[np.ndarray, float, float, bool, bool, float]] = None
        for fac in distance_factors:
            dist = current_distance * fac
            cam_pos = target + d * dist
            pose44 = _look_at(cam_pos, target)
            camera.entity.set_pose(sapien.Pose(pose44))
            scene.step()
            scene.update_render()
            camera.take_picture()
            seg = _camera_picture(camera, "Segmentation")
            if chosen_seg_channel is None:
                chosen_seg_channel = choose_entity_segmentation_channel(seg, valid_entity_ids)
            seg_channel = seg[..., chosen_seg_channel].astype(np.int32)
            union_mask = compute_union_mask(
                seg_channel,
                part_to_entity_ids,
                union_entity_ids=union_entity_ids,
            )
            seg_reliable = int(union_mask.sum()) > 0
            if int(union_mask.sum()) <= 0:
                if has_rendered_background_geometry:
                    continue
                # Fallback when segmentation ids are unreliable in this build/view.
                position = _camera_picture(camera, "Position")
                depth_mask = (-position[..., 2]) > 0
                if int(depth_mask.sum()) <= 0:
                    continue
                union_mask = depth_mask
            ratio = float(union_mask.sum()) / float(union_mask.shape[0] * union_mask.shape[1])
            center_offset = mask_center_offset_ratio(union_mask)
            if seg_reliable:
                ok, _ = evaluate_all_parts_in_frame(
                    seg_channel=seg_channel,
                    part_to_entity_ids=part_to_entity_ids,
                    min_object_coverage=active_min_cov,
                    max_object_coverage=active_max_cov,
                    union_entity_ids=union_entity_ids,
                )
                has_part, part_areas = parts_visibility_ok(
                    seg_channel,
                    part_to_entity_ids,
                    min_effective_part_pixels,
                    require_all_parts=bool(require_all_part_visible),
                )
            else:
                ok, _ = mask_is_complete_and_sized(
                    union_mask,
                    min_ratio=active_min_cov * 0.75,
                    max_ratio=min(0.98, active_max_cov * 1.35),
                )
                has_part = True
                part_areas = {}
            valid = bool(ok and has_part and center_offset <= max_center_offset)
            if valid:
                if best is None or abs(ratio - target_ratio) < abs(best[1] - target_ratio):
                    best = (union_mask, ratio, center_offset, True, bool(has_part), float(dist))
            elif best is None:
                best = (union_mask, ratio, center_offset, False, bool(has_part), float(dist))

        if best is None:
            continue
        union_mask, ratio, center_offset, is_valid, has_part, best_dist = best
        score = view_quality_score(union_mask, ratio, target_ratio)
        if has_part:
            visible_seed_dirs.append(d)
            score += 0.25 * diversity_bonus(d, accepted_dirs)
            fallback_scored_dirs.append((score, d, best_dist))
        # Keep a softer candidate pool to avoid hard failure on difficult objects.
        if has_part and center_offset <= 0.50:
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
                if chosen_seg_channel is None:
                    chosen_seg_channel = choose_entity_segmentation_channel(seg, valid_entity_ids)
                seg_channel = seg[..., chosen_seg_channel].astype(np.int32)
                union_mask = compute_union_mask(
                    seg_channel,
                    part_to_entity_ids,
                    union_entity_ids=union_entity_ids,
                )
                seg_reliable = int(union_mask.sum()) > 0
                if int(union_mask.sum()) <= 0:
                    if has_rendered_background_geometry:
                        continue
                    # Fallback when segmentation ids are unreliable in this build/view.
                    position = _camera_picture(camera, "Position")
                    depth_mask = (-position[..., 2]) > 0
                    if int(depth_mask.sum()) <= 0:
                        continue
                    union_mask = depth_mask
                if mask_center_offset_ratio(union_mask) > max_center_offset:
                    continue
                if seg_reliable:
                    ok, _ = evaluate_all_parts_in_frame(
                        seg_channel=seg_channel,
                        part_to_entity_ids=part_to_entity_ids,
                        min_object_coverage=active_min_cov,
                    max_object_coverage=active_max_cov,
                    union_entity_ids=union_entity_ids,
                )
                    has_part, part_areas = parts_visibility_ok(
                        seg_channel,
                        part_to_entity_ids,
                        min_effective_part_pixels,
                        require_all_parts=bool(require_all_part_visible),
                    )
                else:
                    ok, _ = mask_is_complete_and_sized(
                        union_mask,
                        min_ratio=active_min_cov * 0.75,
                        max_ratio=min(0.98, active_max_cov * 1.35),
                    )
                    has_part = True
                    part_areas = {}
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
        if not accepted_dirs and not bool(require_all_part_visible):
            # Final fallback: keep pipeline running even when segmentation buffers
            # are unavailable; rendering stage still writes RGB/depth and metadata.
            for d in candidate_dirs[: max(1, n_views)]:
                key = _dir_key(d)
                if key in accepted_dir_keys:
                    continue
                accepted_dirs.append(d)
                accepted_dir_keys.add(key)
                accepted_distances[key] = float(current_distance)
                if len(accepted_dirs) >= n_views:
                    break
        if not accepted_dirs:
            raise RuntimeError(
                "No valid render views found with all movable joint parts visible. "
                f"Try lowering --min-part-mask-pixels/--min-part-mask-coverage, increasing "
                f"--view-candidate-multiplier, or using --no-require-all-part-visible. "
                f"threshold_pixels={min_effective_part_pixels}"
            )
    while len(accepted_dirs) < n_views:
        src = accepted_dirs[len(accepted_dirs) % len(accepted_dirs)]
        accepted_dirs.append(src)

    background_records: Dict[str, Dict[str, object]] = {}
    qpos_records: Dict[str, List[float]] = {}
    view_distance_records: Dict[str, float] = {}
    part_visibility_records: Dict[str, Dict[str, int]] = {}

    for view_idx, d in enumerate(accepted_dirs):
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
        if chosen_seg_channel is None:
            chosen_seg_channel = choose_entity_segmentation_channel(seg, valid_entity_ids)
        seg_channel = seg[..., chosen_seg_channel].astype(np.int32)
        union_mask = compute_union_mask(
            seg_channel,
            part_to_entity_ids,
            union_entity_ids=union_entity_ids,
        )
        composite_mask = _composite_object_mask(
            union_mask=union_mask,
            seg=seg,
            rgba=rgba,
            depth=depth,
            has_rendered_background_geometry=has_rendered_background_geometry,
        )

        per_part_masks: Dict[int, np.ndarray] = {}
        per_part_visible_pixels: Dict[int, int] = {}
        for part_id, part_name in part_layout.items():
            entity_ids = part_entity_ids_np.get(part_id)
            if entity_ids is None or entity_ids.size == 0:
                per_part_masks[part_id] = np.zeros_like(seg_channel, dtype=np.uint8)
            else:
                per_part_masks[part_id] = np.isin(seg_channel, entity_ids).astype(np.uint8) * 255
            per_part_visible_pixels[part_id] = int(np.count_nonzero(per_part_masks[part_id]))

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
        for part_id, part_name in part_layout.items():
            link_name = urdf_part_infos[part_id].link_name
            link = linkname_to_link.get(link_name)
            if link is None:
                raise RuntimeError(f"Link '{link_name}' for part {part_id} not found in articulation.")
            ob_in_world = _link_pose_matrix(link)
            per_part_pose[part_id] = world_in_cam @ ob_in_world

        # For background variants of the same view, depth/object-mask/part-masks/cam-params
        # are identical. Save once, then hardlink/copy to the remaining frame indices.
        first_frame_name = f"{view_idx * background_variants:06d}"
        first_depth_path = depth_dir / f"{first_frame_name}.png"
        first_objmask_path = object_mask_dir / f"{first_frame_name}.png"
        first_part_mask_paths: Dict[int, Path] = {}
        first_part_cam_paths: Dict[int, Path] = {}

        for bg_idx in range(background_variants):
            frame_idx = view_idx * background_variants + bg_idx
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

    intrinsic = camera.get_intrinsic_matrix()
    np.savetxt(out_dir / "K.txt", intrinsic, fmt="%.8f")

    scene.remove_articulation(art)
    scene.update_render()

    return {
        "intrinsic": intrinsic.reshape(-1).tolist(),
        "views": n_views * background_variants,
        "views_per_background": n_views,
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
        "target": target.tolist(),
        "camera_distance": current_distance,
        "view_distances": view_distance_records,
        "part_visible_pixels": part_visibility_records,
        "min_part_mask_pixels": int(min_effective_part_pixels),
        "require_all_part_visible": bool(require_all_part_visible),
        "view_candidate_count": int(candidate_count),
        "bbox_extent_scaled": extent.tolist(),
        "segmentation_channel": chosen_seg_channel,
        "part_to_link": {str(pid): urdf_part_infos[pid].link_name for pid in part_layout.keys()},
        "part_to_entity_ids": {str(pid): part_to_entity_ids[pid] for pid in part_layout.keys()},
        "id_to_entity": {str(k): v for k, v in id_to_entity.items()},
        "linkname_to_entity_ids": {str(k): v for k, v in linkname_to_entity_ids.items()},
        "object_lift": object_lift,
        "camera_poses": camera_poses,
    }


def build_one_instance(instance: InstanceInfo, args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.output_root) / instance.instance_name
    cleanup_on_error = args.overwrite or not out_dir.exists()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

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
        missing_mesh_files = find_missing_mesh_files(instance.instance_dir, urdf_part_infos)
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
            try:
                render_result = render_instance(
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
                    min_object_coverage=args.min_object_coverage,
                    max_object_coverage=args.max_object_coverage,
                    min_part_mask_pixels=args.min_part_mask_pixels,
                    min_part_mask_coverage=args.min_part_mask_coverage,
                    require_all_part_visible=args.require_all_part_visible,
                    view_candidate_multiplier=args.view_candidate_multiplier,
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
    except Exception:
        if cleanup_on_error and out_dir.exists():
            shutil.rmtree(out_dir)
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

    if num_workers <= 1:
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
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_instance = {
                executor.submit(_run_one_instance_worker, inst, args_dict): inst.instance_name
                for inst in targets
            }
            finished = 0
            total = len(future_to_instance)
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


if __name__ == "__main__":
    main()



# python build_scene.py --instance bucket_100438 --processes=1 --overwrite --joint-motion
