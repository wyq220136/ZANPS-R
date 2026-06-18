import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import trimesh


def _log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(message).rstrip() + "\n")
        f.flush()


def _tail_file(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return f"{path}: <missing>"
    try:
        text = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])
    except Exception as e:
        text = f"<failed to read: {e}>"
    return f"{path}:\n{text}"


@dataclass
class DMeshConvertConfig:
    dmesh_root: str
    device: str = "cuda:0"
    num_step: int = 20000
    save_step: int = 100
    refresh_points_step: int = 5000
    gt_max_perturb: float = 0.0
    seed: int = 1
    output_variant: str = "auto"
    overwrite: bool = False

    def cache_key(self) -> Dict[str, Any]:
        out = asdict(self)
        out["dmesh_root"] = str(Path(self.dmesh_root).resolve())
        out.pop("overwrite", None)
        return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dmesh_root(dmesh_root: str | None = None) -> str:
    candidates = []
    if dmesh_root:
        candidates.append(Path(dmesh_root).expanduser())
    env_root = os.environ.get("DMESH_ROOT", "")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(_repo_root() / "dmesh")

    for p in candidates:
        p = p.resolve()
        if p.is_dir() and (p / "exp" / "1_mesh_to_dmesh.py").exists():
            return str(p)
    raise FileNotFoundError("DMesh root not found. Pass --dmesh-root or set DMESH_ROOT.")


def _load_official_mesh_to_dmesh_module(dmesh_root: str):
    dmesh_root = resolve_dmesh_root(dmesh_root)
    if dmesh_root not in sys.path:
        sys.path.insert(0, dmesh_root)
    module_path = Path(dmesh_root) / "exp" / "1_mesh_to_dmesh.py"
    spec = importlib.util.spec_from_file_location("_partnet_mesh_to_dmesh", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DMesh entry: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_mesh(mesh_obj) -> trimesh.Trimesh:
    if isinstance(mesh_obj, trimesh.Scene):
        geoms = [g for g in mesh_obj.geometry.values() if len(g.vertices) > 0 and len(g.faces) > 0]
        if not geoms:
            raise ValueError("mesh scene is empty")
        mesh_obj = trimesh.util.concatenate(geoms)
    if not isinstance(mesh_obj, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh_obj)!r}")
    if len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("mesh has no vertices/faces")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh_obj.vertices, dtype=np.float32),
        faces=np.asarray(mesh_obj.faces, dtype=np.int64),
        process=False,
    )


def _clean_mesh_for_dmesh(mesh: trimesh.Trimesh, merge_tol: float = 1e-6) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    stats: Dict[str, Any] = {
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
    }

    finite_v = np.all(np.isfinite(vertices), axis=1)
    if not np.all(finite_v):
        old_to_new = np.full(len(vertices), -1, dtype=np.int64)
        old_to_new[np.where(finite_v)[0]] = np.arange(int(np.count_nonzero(finite_v)), dtype=np.int64)
        keep_faces = np.all(finite_v[faces], axis=1)
        faces = old_to_new[faces[keep_faces]]
        vertices = vertices[finite_v]

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("mesh has no finite vertices/faces after cleaning")

    quant = np.round(vertices / float(merge_tol)).astype(np.int64)
    _, inverse = np.unique(quant, axis=0, return_inverse=True)
    if int(inverse.max()) + 1 < len(vertices):
        sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
        counts = np.bincount(inverse).astype(np.float64)
        np.add.at(sums, inverse, vertices)
        vertices = sums / counts[:, None]
        faces = inverse[faces]

    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[distinct]
    if len(faces) == 0:
        raise ValueError("mesh has no non-degenerate faces after duplicate vertex removal")

    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = faces[area2 > 1e-12]
    if len(faces) == 0:
        raise ValueError("mesh has no non-zero-area faces after cleaning")

    sorted_faces = np.sort(faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    faces = faces[np.sort(unique_idx)]

    used = np.unique(faces.reshape(-1))
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[used] = np.arange(len(used), dtype=np.int64)
    vertices = vertices[used]
    faces = old_to_new[faces]

    stats.update(
        {
            "output_vertices": int(len(vertices)),
            "output_faces": int(len(faces)),
            "removed_vertices": int(stats["input_vertices"] - len(vertices)),
            "removed_faces": int(stats["input_faces"] - len(faces)),
            "merge_tol": float(merge_tol),
        }
    )
    return trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces.astype(np.int64), process=False), stats


def _load_normalized_mesh(mesh_path: str, device: str, scale: float = 0.8):
    import torch as th

    mesh = _as_mesh(trimesh.load(mesh_path, force="mesh", process=False))
    mesh, clean_stats = _clean_mesh_for_dmesh(mesh)
    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    center = vertices_np.mean(axis=0, keepdims=True)
    centered = vertices_np - center
    max_norm = float(np.max(np.linalg.norm(centered, axis=1)) + 1e-6)
    normalized = (centered / max_norm) * float(scale)
    verts = th.tensor(normalized, dtype=th.float32, device=device)
    faces = th.tensor(np.asarray(mesh.faces, dtype=np.int64), dtype=th.long, device=device)
    return verts, faces, {"center": center.reshape(3), "max_norm": max_norm, "scale": float(scale), "clean_stats": clean_stats}


def _denormalize_mesh(mesh_path: Path, out_path: Path, norm: Dict[str, Any]) -> Dict[str, Any]:
    mesh = _as_mesh(trimesh.load(str(mesh_path), force="mesh", process=False))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices = (vertices / float(norm["scale"])) * float(norm["max_norm"]) + np.asarray(norm["center"], dtype=np.float32)
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(str(out_path))
    return {"vertices": int(len(out_mesh.vertices)), "faces": int(len(out_mesh.faces))}


def _select_dmesh_output(run_dir: Path, variant: str) -> Path:
    save_dir = run_dir / "save"
    ordered = []
    if variant == "auto":
        ordered = ["perfect", "best_recovery_ratio", "best_false_positive_ratio", "last"]
    else:
        ordered = [variant]
    for name in ordered:
        p = save_dir / name
        if (p / "mesh.obj").exists() and (p / "points.pth").exists():
            return p
    step_dirs = sorted(
        [p for p in save_dir.glob("step_*") if (p / "mesh.obj").exists() and (p / "points.pth").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if variant == "auto" and step_dirs:
        return step_dirs[-1]
    raise FileNotFoundError(f"No DMesh output found in {save_dir} for variant={variant}")


def _cache_is_valid(mesh_path: Path, dst_model_dir: Path, cfg: DMeshConvertConfig) -> bool:
    manifest_path = dst_model_dir / "dmesh_manifest.json"
    model_path = dst_model_dir / "model.obj"
    points_path = dst_model_dir / "dmesh" / "points.pth"
    if not (manifest_path.exists() and model_path.exists() and points_path.exists()):
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return (
            manifest.get("source_mesh") == str(mesh_path.resolve())
            and manifest.get("source_mtime") == mesh_path.stat().st_mtime
            and manifest.get("config") == cfg.cache_key()
        )
    except Exception:
        return False


def convert_model_to_dmesh(mesh_path: str, dst_model_dir: str, cfg: DMeshConvertConfig) -> Dict[str, Any]:
    mesh_path = Path(mesh_path).resolve()
    dst_model_dir = Path(dst_model_dir).resolve()
    model_path = dst_model_dir / "model.obj"
    if not cfg.overwrite and _cache_is_valid(mesh_path, dst_model_dir, cfg):
        return {"status": "cached", "source_mesh": str(mesh_path), "output_model": str(model_path)}

    if dst_model_dir.exists():
        shutil.rmtree(dst_model_dir)
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = dst_model_dir / "dmesh_run"
    progress_log = dst_model_dir / "dmesh_worker_progress.log"
    _log_line(
        progress_log,
        f"start mesh={mesh_path} dst={dst_model_dir} steps={cfg.num_step} "
        f"save_step={cfg.save_step} refresh={cfg.refresh_points_step} device={cfg.device}",
    )

    import torch as th

    dmesh_mod = _load_official_mesh_to_dmesh_module(cfg.dmesh_root)
    dmesh_mod.device = cfg.device
    th.random.manual_seed(int(cfg.seed))
    verts, faces, norm = _load_normalized_mesh(str(mesh_path), cfg.device, scale=0.8)
    _log_line(progress_log, f"loaded normalized mesh verts={int(verts.shape[0])} faces={int(faces.shape[0])}")
    _log_line(progress_log, f"mesh clean stats={norm.get('clean_stats', {})}")

    run_dir.mkdir(parents=True, exist_ok=True)
    _log_line(progress_log, "construct optimizer")
    optimizer = dmesh_mod.GtmeshOptimizer(verts, faces, str(run_dir))
    _log_line(progress_log, "optimize begin")
    optimizer.optimize(
        int(cfg.num_step),
        max(1, int(cfg.save_step)),
        max(1, int(cfg.refresh_points_step)),
        float(cfg.gt_max_perturb),
    )
    _log_line(progress_log, "optimize done")

    selected_dir = _select_dmesh_output(run_dir, cfg.output_variant)
    _log_line(progress_log, f"selected output {selected_dir}")
    dmesh_data_dir = dst_model_dir / "dmesh"
    dmesh_data_dir.mkdir(parents=True, exist_ok=True)
    official_mesh_path = dmesh_data_dir / "official_mesh.obj"
    shutil.copy2(selected_dir / "mesh.obj", official_mesh_path)
    shutil.copy2(selected_dir / "points.pth", dmesh_data_dir / "points.pth")
    if (selected_dir / "time_sec.txt").exists():
        shutil.copy2(selected_dir / "time_sec.txt", dmesh_data_dir / "time_sec.txt")

    mesh_stats = _denormalize_mesh(official_mesh_path, model_path, norm)
    manifest = {
        "status": "converted",
        "source_mesh": str(mesh_path),
        "source_mtime": mesh_path.stat().st_mtime,
        "output_model": str(model_path),
        "official_dmesh_mesh": str(official_mesh_path),
        "dmesh_points": str(dmesh_data_dir / "points.pth"),
        "selected_output": str(selected_dir),
        "config": cfg.cache_key(),
        "mesh_stats": mesh_stats,
    }
    with (dst_model_dir / "dmesh_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    _log_line(progress_log, f"done output_model={model_path}")
    return manifest


def convert_model_to_dmesh_subprocess(mesh_path: str, dst_model_dir: str, cfg: DMeshConvertConfig) -> Dict[str, Any]:
    mesh_path_p = Path(mesh_path).resolve()
    dst_model_dir_p = Path(dst_model_dir).resolve()
    model_path = dst_model_dir_p / "model.obj"
    if not cfg.overwrite and _cache_is_valid(mesh_path_p, dst_model_dir_p, cfg):
        return {"status": "cached", "source_mesh": str(mesh_path_p), "output_model": str(model_path)}

    log_dir = dst_model_dir_p.parent / f"{dst_model_dir_p.name}_dmesh_logs"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    request_path = log_dir / "dmesh_request.json"
    result_path = log_dir / "dmesh_result.json"
    log_path = log_dir / "dmesh_subprocess.log"
    request = {
        "mesh_path": str(mesh_path_p),
        "dst_model_dir": str(dst_model_dir_p),
        "cfg": asdict(cfg),
    }
    request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    dmesh_root = Path(cfg.dmesh_root).resolve()
    lib_paths = [
        str(dmesh_root / "external" / "oneTBB" / "install" / "lib"),
        str(Path(sys.prefix) / "lib"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths + [env.get("LD_LIBRARY_PATH", "")])
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"command={sys.executable} {Path(__file__).resolve()} --dmesh-worker {request_path} {result_path}\n")
        log_f.flush()
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dmesh-worker",
                str(request_path),
                str(result_path),
            ],
            cwd=str(_repo_root()),
            env=env,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        log_f.write(f"\nreturncode={proc.returncode}\n")
        log_f.flush()
    if proc.returncode != 0:
        detail = "\n\n".join(
            [
                _tail_file(log_path),
                _tail_file(log_dir / "dmesh_worker_error.log"),
                _tail_file(dst_model_dir_p / "dmesh_worker_progress.log"),
            ]
        )
        raise RuntimeError(
            f"DMesh subprocess failed with exit code {proc.returncode}. See log: {log_path}\n"
            f"log tail:\n{detail}"
        )
    if not result_path.exists():
        detail = "\n\n".join(
            [
                _tail_file(log_path),
                _tail_file(log_dir / "dmesh_worker_error.log"),
                _tail_file(dst_model_dir_p / "dmesh_worker_progress.log"),
            ]
        )
        raise RuntimeError(f"DMesh subprocess did not write result json: {result_path}\n\n{detail}")
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def _run_dmesh_worker(request_path: str, result_path: str) -> None:
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        cfg = DMeshConvertConfig(**request["cfg"])
        result = convert_model_to_dmesh(
            request["mesh_path"],
            request["dst_model_dir"],
            cfg,
        )
        Path(result_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        err = traceback.format_exc()
        error_path = Path(request_path).parent / "dmesh_worker_error.log"
        error_path.write_text(err, encoding="utf-8")
        print(err, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--dmesh-worker":
        _run_dmesh_worker(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("Usage: python reconstruction/dmesh_integration.py --dmesh-worker REQUEST_JSON RESULT_JSON")
