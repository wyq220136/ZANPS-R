import argparse
import os
import sys
import json
import hashlib
import pickle
import re
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


@dataclass
class PartRecord:
    obj_name: str
    frame_id: str
    part_name: str
    rgb_path: str
    mask_path: str
    depth_path: str
    K_path: str
    gt_mesh_path: str
    cam_param_path: str | None
    pose: np.ndarray | None = None


class Sam3DPartTrainDataset(Dataset):
    """
    Dataset loader for dataset_train structure:
      object/
        rgb/<frame>.png
        masks/<part_name>/<frame>.png
        models/<part_name>/model.obj
        K.txt
        cam_params/<part_name>/<frame>.txt

    It runs SAM3D reconstruction as preprocessing and caches only a small
    pose-covering set of reconstructed meshes per (object, part).
    """

    def __init__(
        self,
        dataset_root: str,
        cache_root: str,
        sam3d_project_root: str,
        sam3d_notebook_root: str,
        sam3d_config_path: str,
        cache_split: str | None = None,
        min_mask_pixels: int = 32,
        seed: int = 42,
        rebuild_recon: bool = False,
        allow_recon_write: bool = True,
        strict_mode: bool = False,
        fallback_to_gt_mesh_on_recon_fail: bool = True,
        depth_scale: float = 1000.0,
        recon_min_views_per_part: int = 3,
        recon_max_views_per_part: int = 5,
        recon_rot_threshold_deg: float = 15.0,
        recon_trans_threshold: float = 0.05,
        force_resample_recon: bool = False,
        use_real_depth_pointmap: bool = True,
        recon_model: str = "sam3d",
        recon_view_density_scale: float = 1.5,
        hunyuan_model_path: str | None = None,
        hunyuan_subfolder: str = "hunyuan3d-dit-v2-1",
        hunyuan_num_inference_steps: int = 50,
        hunyuan_octree_resolution: int = 384,
        hunyuan_guidance_scale: float = 5.5,
        instantmesh_root: str | None = None,
        instantmesh_config_path: str | None = None,
        instantmesh_diffusion_model: str = "sudo-ai/zero123plus-v1.2",
        instantmesh_dino_model: str = "",
        instantmesh_unet_path: str = "",
        instantmesh_model_path: str = "",
        instantmesh_diffusion_steps: int = 75,
        instantmesh_scale: float = 1.0,
        instantmesh_view: int = 6,
        instantmesh_foreground_ratio: float = 0.85,
        instantmesh_export_texmap: bool = False,
        defer_sample_io: bool = False,
        rebuild_records_index: bool = False,
    ):
        self.dataset_root = os.path.abspath(dataset_root)
        self.cache_root = os.path.abspath(cache_root)
        self.cache_split = str(cache_split or "").strip()
        self.sam3d_project_root = sam3d_project_root
        self.sam3d_notebook_root = sam3d_notebook_root
        self.sam3d_config_path = sam3d_config_path
        self.min_mask_pixels = int(min_mask_pixels)
        self.seed = int(seed)
        self.rebuild_recon = bool(rebuild_recon)
        self.allow_recon_write = bool(allow_recon_write)
        self.strict_mode = bool(strict_mode)
        self.fallback_to_gt_mesh_on_recon_fail = bool(fallback_to_gt_mesh_on_recon_fail)
        self.depth_scale = float(depth_scale)
        self.recon_min_views_per_part = int(recon_min_views_per_part)
        self.recon_max_views_per_part = int(recon_max_views_per_part)
        self.recon_rot_threshold_deg = float(recon_rot_threshold_deg)
        self.recon_trans_threshold = float(recon_trans_threshold)
        self.force_resample_recon = bool(force_resample_recon)
        self.use_real_depth_pointmap = bool(use_real_depth_pointmap)
        self.recon_model = str(recon_model).lower().strip()
        self.recon_view_density_scale = float(max(1.0, recon_view_density_scale))
        self.hunyuan_model_path = hunyuan_model_path
        self.hunyuan_subfolder = str(hunyuan_subfolder)
        self.hunyuan_num_inference_steps = int(hunyuan_num_inference_steps)
        self.hunyuan_octree_resolution = int(hunyuan_octree_resolution)
        self.hunyuan_guidance_scale = float(hunyuan_guidance_scale)
        self.instantmesh_root = instantmesh_root
        self.instantmesh_config_path = instantmesh_config_path
        self.instantmesh_diffusion_model = str(instantmesh_diffusion_model)
        self.instantmesh_dino_model = str(instantmesh_dino_model or "")
        self.instantmesh_unet_path = str(instantmesh_unet_path or "")
        self.instantmesh_model_path = str(instantmesh_model_path or "")
        self.instantmesh_diffusion_steps = int(instantmesh_diffusion_steps)
        self.instantmesh_scale = float(instantmesh_scale)
        self.instantmesh_view = int(instantmesh_view)
        self.instantmesh_foreground_ratio = float(instantmesh_foreground_ratio)
        self.instantmesh_export_texmap = bool(instantmesh_export_texmap)
        self.defer_sample_io = bool(defer_sample_io)
        self.rebuild_records_index = bool(rebuild_records_index)
        self._supported_recon_models = (
            "sam3d",
            "sam3d_tsdf",
            "sam3d_tsdf_dmesh",
            "hunyuan3d",
            "hunyuan3d_tsdf",
            "hunyuan3d_tsdf_dmesh",
            "instantmesh",
            "instantmesh_tsdf",
            "instantmesh_tsdf_dmesh",
        )
        if self.recon_model not in (*self._supported_recon_models, "all"):
            raise ValueError(
                f"unsupported recon_model={self.recon_model}, "
                f"expected one of {self._supported_recon_models + ('all',)}"
            )
        self.records: list[PartRecord] = []
        self.part_to_indices: dict[tuple[str, str], list[int]] = {}
        self.part_keys: list[tuple[str, str]] = []
        self._manifest_cache: dict[tuple[str, str], dict] = {}
        self._inference = None
        self._hunyuan_reconstructor = None
        self._instantmesh_reconstructor = None
        self._getitem_calls = 0
        self._recon_failures = 0
        self._bad_recon_records: set[tuple[str, str, str]] = set()
        self._object_recon_builds: set[tuple[str, str]] = set()
        self._bad_recon_skips = 0
        self._invisible_record_skips = 0
        self._fallback_uses = 0
        os.makedirs(self.cache_root, exist_ok=True)
        if self.rebuild_records_index:
            self._collect_records()
            self._write_records_index()
        elif not self._load_records_index():
            self._collect_records()
            self._write_records_index()
        if not self.allow_recon_write:
            self._filter_records_to_complete_recon_cache()
        print(
            f"[dataset] root={self.dataset_root} records={len(self.records)} "
            f"min_mask_pixels={self.min_mask_pixels} rebuild_recon={self.rebuild_recon} "
            f"allow_recon_write={self.allow_recon_write} strict_mode={self.strict_mode} "
            f"fallback_to_gt_mesh_on_recon_fail={self.fallback_to_gt_mesh_on_recon_fail} "
            f"depth_scale={self.depth_scale} "
            f"recon_views={self.recon_min_views_per_part}-{self.recon_max_views_per_part} "
            f"pose_thresholds(rot_deg={self.recon_rot_threshold_deg}, trans={self.recon_trans_threshold}) "
            f"use_real_depth_pointmap={self.use_real_depth_pointmap} "
            f"recon_model={self.recon_model} recon_view_density_scale={self.recon_view_density_scale:.2f} "
            f"defer_sample_io={self.defer_sample_io} "
            f"rebuild_records_index={self.rebuild_records_index}"
        )

    def _active_recon_models(self):
        if self.recon_model == "all":
            return list(self._supported_recon_models)
        return [self.recon_model]

    @staticmethod
    def _is_object_level_recon_model(recon_model: str):
        return str(recon_model).lower().strip().endswith(("_tsdf", "_tsdf_dmesh"))

    @staticmethod
    def _base_recon_model(recon_model: str):
        model = str(recon_model).lower().strip()
        if model.startswith("sam3d"):
            return "sam3d"
        if model.startswith("hunyuan3d"):
            return "hunyuan3d"
        if model.startswith("instantmesh"):
            return "instantmesh"
        return model

    @staticmethod
    def _part_model_name(part_name: str, fallback_idx: int = 0):
        match = re.search(r"(\d+)", str(part_name))
        idx = int(match.group(1)) if match else int(fallback_idx)
        return f"model_{idx:04d}"

    def _records_index_enabled(self):
        return os.environ.get("SAM3D_PART_DATASET_DISABLE_INDEX", "").strip().lower() not in ("1", "true", "yes")

    def _records_index_key(self):
        payload = {
            "version": 1,
            "dataset_root": os.path.abspath(self.dataset_root),
            "min_mask_pixels": int(self.min_mask_pixels),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _records_index_path(self):
        return os.path.join(self.cache_root, "_dataset_index", f"{self._records_index_key()}.json")

    def _records_index_pickle_path(self):
        return os.path.join(self.cache_root, "_dataset_index", f"{self._records_index_key()}.pkl")

    @staticmethod
    def _pose_to_json(pose):
        if pose is None:
            return None
        return np.asarray(pose, dtype=np.float32).astype(float).tolist()

    @staticmethod
    def _pose_from_json(pose):
        if pose is None:
            return None
        arr = np.asarray(pose, dtype=np.float32)
        if arr.shape == (16,):
            arr = arr.reshape(4, 4)
        if arr.shape != (4, 4):
            return None
        return arr

    def _record_to_json(self, rec: PartRecord):
        return {
            "obj_name": rec.obj_name,
            "frame_id": rec.frame_id,
            "part_name": rec.part_name,
            "rgb_path": rec.rgb_path,
            "mask_path": rec.mask_path,
            "depth_path": rec.depth_path,
            "K_path": rec.K_path,
            "gt_mesh_path": rec.gt_mesh_path,
            "cam_param_path": rec.cam_param_path,
            "pose": self._pose_to_json(rec.pose),
        }

    def _record_from_json(self, item: dict):
        return PartRecord(
            obj_name=str(item["obj_name"]),
            frame_id=str(item["frame_id"]),
            part_name=str(item["part_name"]),
            rgb_path=str(item["rgb_path"]),
            mask_path=str(item["mask_path"]),
            depth_path=str(item["depth_path"]),
            K_path=str(item["K_path"]),
            gt_mesh_path=str(item["gt_mesh_path"]),
            cam_param_path=item.get("cam_param_path"),
            pose=self._pose_from_json(item.get("pose")),
        )

    def _load_records_index(self):
        if not self._records_index_enabled():
            return False
        pkl_path = self._records_index_pickle_path()
        path = self._records_index_path()
        if os.path.exists(pkl_path):
            t0 = time.time()
            print(f"[dataset-index] loading pickle index: {pkl_path}", flush=True)
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                if data.get("version") != 1:
                    return False
                if os.path.abspath(data.get("dataset_root", "")) != os.path.abspath(self.dataset_root):
                    return False
                if int(data.get("min_mask_pixels", -1)) != int(self.min_mask_pixels):
                    return False
                records = [self._record_from_json(item) for item in data.get("records", [])]
                if not records:
                    return False
                self.records = records
                self._rebuild_part_index()
                print(
                    f"[dataset-index] loaded records={len(self.records)} "
                    f"path={pkl_path} seconds={time.time() - t0:.2f}",
                    flush=True,
                )
                return True
            except Exception as e:
                print(f"[dataset-index][warn] failed to load {pkl_path}: {repr(e)}", flush=True)

        if not os.path.exists(path):
            return False
        t0 = time.time()
        print(f"[dataset-index] loading json index: {path}", flush=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != 1:
                return False
            if os.path.abspath(data.get("dataset_root", "")) != os.path.abspath(self.dataset_root):
                return False
            if int(data.get("min_mask_pixels", -1)) != int(self.min_mask_pixels):
                return False
            records = [self._record_from_json(item) for item in data.get("records", [])]
            if not records:
                return False
            self.records = records
            self._rebuild_part_index()
            print(
                f"[dataset-index] loaded records={len(self.records)} "
                f"path={path} seconds={time.time() - t0:.2f}",
                flush=True,
            )
            self._write_records_index_pickle(data)
            return True
        except Exception as e:
            print(f"[dataset-index][warn] failed to load {path}: {repr(e)}", flush=True)
            return False

    def _write_records_index_pickle(self, data: dict):
        if not self._records_index_enabled():
            return
        path = self._records_index_pickle_path()
        tmp_path = f"{path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
            print(f"[dataset-index] wrote pickle path={path}", flush=True)
        except Exception as e:
            print(f"[dataset-index][warn] failed to write pickle {path}: {repr(e)}", flush=True)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _write_records_index(self):
        if not self._records_index_enabled():
            return
        path = self._records_index_path()
        tmp_path = f"{path}.{os.getpid()}.tmp"
        data = {
            "version": 1,
            "dataset_root": os.path.abspath(self.dataset_root),
            "min_mask_pixels": int(self.min_mask_pixels),
            "num_records": int(len(self.records)),
            "num_parts": int(len(self.part_keys)),
            "records": [self._record_to_json(rec) for rec in self.records],
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, path)
            self._write_records_index_pickle(data)
            print(f"[dataset-index] wrote records={len(self.records)} path={path}", flush=True)
        except Exception as e:
            print(f"[dataset-index][warn] failed to write {path}: {repr(e)}", flush=True)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _ensure_sam3d_imports(self):
        if self.sam3d_project_root not in sys.path:
            sys.path.append(self.sam3d_project_root)
        if self.sam3d_notebook_root not in sys.path:
            sys.path.append(self.sam3d_notebook_root)

    def _ensure_reconstruction_imports(self):
        repo_root = Path(__file__).resolve().parents[3]
        recon_root = repo_root / "reconstruction"
        tools_root = recon_root / "tools"
        for p in (str(repo_root), str(recon_root), str(tools_root)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def _get_inference(self):
        if self._inference is None:
            self._ensure_sam3d_imports()
            print(f"[dataset] loading SAM3D inference from config: {self.sam3d_config_path}")
            from inference import Inference  # noqa
            self._inference = Inference(
                self.sam3d_config_path,
                compile=False,
                use_depth_model=(not self.use_real_depth_pointmap),
            )
            print("[dataset] SAM3D inference is ready")
        return self._inference

    def _get_hunyuan_reconstructor(self):
        if self._hunyuan_reconstructor is None:
            self._ensure_reconstruction_imports()
            from reconstruct_hunyuan3d import HunyuanReconstructor, default_hunyuan_model_path  # noqa

            model_path = self.hunyuan_model_path
            if model_path is None or not str(model_path).strip():
                model_path = default_hunyuan_model_path()
            print(
                "[dataset] loading Hunyuan3D reconstructor "
                f"model_path={model_path} subfolder={self.hunyuan_subfolder} "
                f"steps={self.hunyuan_num_inference_steps} "
                f"octree={self.hunyuan_octree_resolution} "
                f"guidance={self.hunyuan_guidance_scale}",
                flush=True,
            )
            self._hunyuan_reconstructor = HunyuanReconstructor(
                model_path=str(model_path),
                subfolder=self.hunyuan_subfolder,
                num_inference_steps=self.hunyuan_num_inference_steps,
                octree_resolution=self.hunyuan_octree_resolution,
                guidance_scale=self.hunyuan_guidance_scale,
            )
            print("[dataset] Hunyuan3D reconstructor is ready", flush=True)
        return self._hunyuan_reconstructor

    def _get_instantmesh_reconstructor(self):
        if self._instantmesh_reconstructor is None:
            self._ensure_reconstruction_imports()
            from reconstruct_instantmesh import (  # noqa
                InstantMeshReconstructor,
                default_instantmesh_config_path,
                default_instantmesh_root,
            )

            instantmesh_root = self.instantmesh_root
            if instantmesh_root is None or not str(instantmesh_root).strip():
                instantmesh_root = default_instantmesh_root()
            config_path = self.instantmesh_config_path
            if config_path is None or not str(config_path).strip():
                config_path = default_instantmesh_config_path()
            print(
                "[dataset] loading InstantMesh reconstructor "
                f"root={instantmesh_root} config={config_path} "
                f"steps={self.instantmesh_diffusion_steps} "
                f"dino_model={self.instantmesh_dino_model} "
                f"view={self.instantmesh_view}",
                flush=True,
            )
            self._instantmesh_reconstructor = InstantMeshReconstructor(
                instantmesh_root=str(instantmesh_root),
                config_path=str(config_path),
                diffusion_model=self.instantmesh_diffusion_model,
                dino_model=self.instantmesh_dino_model,
                unet_path=self.instantmesh_unet_path,
                model_path=self.instantmesh_model_path,
                diffusion_steps=self.instantmesh_diffusion_steps,
                seed=self.seed,
                scale=self.instantmesh_scale,
                view=self.instantmesh_view,
                foreground_ratio=self.instantmesh_foreground_ratio,
                export_texmap=self.instantmesh_export_texmap,
            )
            print("[dataset] InstantMesh reconstructor is ready", flush=True)
        return self._instantmesh_reconstructor

    def _collect_records(self):
        t0 = time.time()
        obj_names = sorted(
            [d for d in os.listdir(self.dataset_root) if os.path.isdir(os.path.join(self.dataset_root, d))]
        )
        print(
            f"[dataset-index] collecting records root={self.dataset_root} "
            f"objects={len(obj_names)} min_mask_pixels={self.min_mask_pixels}",
            flush=True,
        )
        scanned_masks = 0
        kept_records = 0
        for obj_idx, obj_name in enumerate(obj_names, start=1):
            obj_dir = os.path.join(self.dataset_root, obj_name)
            rgb_dir = os.path.join(obj_dir, "rgb")
            masks_dir = os.path.join(obj_dir, "masks")
            models_dir = os.path.join(obj_dir, "models")
            K_path = os.path.join(obj_dir, "K.txt")
            cam_params_dir = os.path.join(obj_dir, "cam_params")
            if (not os.path.isdir(rgb_dir)) or (not os.path.isdir(masks_dir)) or (not os.path.isdir(models_dir)):
                continue
            if not os.path.exists(K_path):
                continue

            part_dirs = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))])
            if obj_idx == 1 or obj_idx % 10 == 0 or obj_idx == len(obj_names):
                print(
                    f"[dataset-index] scanning object {obj_idx}/{len(obj_names)} "
                    f"name={obj_name} parts={len(part_dirs)} "
                    f"scanned_masks={scanned_masks} kept_records={kept_records} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
            for part_name in part_dirs:
                part_mask_dir = os.path.join(masks_dir, part_name)
                gt_mesh_path = os.path.join(models_dir, part_name, "model.obj")
                if (not os.path.exists(gt_mesh_path)) or (not os.path.isdir(part_mask_dir)):
                    continue
                mask_files = sorted([f for f in os.listdir(part_mask_dir) if f.lower().endswith(".png")])
                for mf in mask_files:
                    scanned_masks += 1
                    if scanned_masks % 5000 == 0:
                        print(
                            f"[dataset-index] scanned_masks={scanned_masks} "
                            f"kept_records={kept_records} current={obj_name}/{part_name}/{mf} "
                            f"elapsed={time.time() - t0:.1f}s",
                            flush=True,
                        )
                    frame_id = os.path.splitext(mf)[0]
                    mask_path = os.path.join(part_mask_dir, mf)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    mask_bin = mask > 0
                    if int(np.count_nonzero(mask_bin)) < self.min_mask_pixels:
                        continue
                    rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")
                    if not os.path.exists(rgb_path):
                        rgb_path = os.path.join(rgb_dir, f"{frame_id}.jpg")
                    if not os.path.exists(rgb_path):
                        continue
                    depth_path = self._resolve_depth_path(obj_dir, frame_id)
                    if depth_path is None:
                        continue
                    cam_path = self._resolve_cam_param_path(cam_params_dir, part_name, frame_id)
                    if not os.path.exists(cam_path):
                        continue
                    try:
                        pose = self._load_pose(cam_path)
                    except Exception as e:
                        if self.strict_mode:
                            raise
                        print(f"[dataset][skip] invalid cam_params: {cam_path} err={repr(e)}")
                        continue
                    visible_ok, reason = self._visible_sample_ok(mask_bin)
                    if not visible_ok:
                        self._invisible_record_skips += 1
                        if self._invisible_record_skips <= 5 or (self._invisible_record_skips % 1000 == 0):
                            print(
                                f"[dataset][skip] invisible/weak sample ({self._invisible_record_skips}) "
                                f"obj={obj_name} part={part_name} frame={frame_id} reason={reason}"
                            )
                        continue
                    rec = PartRecord(
                        obj_name=obj_name,
                        frame_id=frame_id,
                        part_name=part_name,
                        rgb_path=rgb_path,
                        mask_path=mask_path,
                        depth_path=depth_path,
                        K_path=K_path,
                        gt_mesh_path=gt_mesh_path,
                        cam_param_path=cam_path,
                        pose=pose,
                    )
                    index = len(self.records)
                    self.records.append(rec)
                    self.part_to_indices.setdefault((obj_name, part_name), []).append(index)
                    kept_records += 1
        self.part_keys = sorted(self.part_to_indices.keys())
        print(
            f"[dataset-index] collected records={len(self.records)} parts={len(self.part_keys)} "
            f"scanned_masks={scanned_masks} seconds={time.time() - t0:.2f}",
            flush=True,
        )

    def _visible_sample_ok(self, mask_bin: np.ndarray):
        mask_bool = np.asarray(mask_bin) > 0
        mask_pixels = int(np.count_nonzero(mask_bool))
        if mask_pixels < self.min_mask_pixels:
            return False, f"mask_pixels={mask_pixels}<{self.min_mask_pixels}"
        return True, "ok"

    def _rebuild_part_index(self):
        self.part_to_indices = {}
        for index, rec in enumerate(self.records):
            self.part_to_indices.setdefault((rec.obj_name, rec.part_name), []).append(index)
        self.part_keys = sorted(self.part_to_indices.keys())

    def _filter_records_to_complete_recon_cache(self):
        keep_records = []
        skipped = 0
        for rec in self.records:
            model_manifests = self._load_manifests_for_part((rec.obj_name, rec.part_name))
            required_models = self._active_recon_models()
            complete_models = set()
            skipped_frame_ids = set()
            for recon_model, manifest in model_manifests.items():
                skipped_frame_ids.update(
                    str(item.get("frame_id"))
                    for item in (manifest or {}).get("skipped_frames", [])
                    if item.get("frame_id") is not None
                )
                if self._manifest_is_complete(manifest):
                    complete_models.add(recon_model)
            has_required_complete = all(m in complete_models for m in required_models)
            if has_required_complete and rec.frame_id not in skipped_frame_ids:
                keep_records.append(rec)
            else:
                skipped += 1
        if skipped > 0:
            print(
                f"[dataset][skip] filtered {skipped} records without complete reconstruction cache "
                f"(allow_recon_write=False)"
            )
        self.records = keep_records
        self._rebuild_part_index()

    @staticmethod
    def _resolve_cam_param_path(cam_params_dir: str, part_name: str, frame_id: str):
        part_path = os.path.join(cam_params_dir, part_name, f"{frame_id}.txt")
        return part_path

    @staticmethod
    def _load_pose(cam_param_path: str | None):
        if not cam_param_path or (not os.path.exists(cam_param_path)):
            return None
        cp = np.loadtxt(cam_param_path, dtype=np.float32)
        if cp.shape == (16,):
            cp = cp.reshape(4, 4)
        if cp.shape != (4, 4):
            raise RuntimeError(f"invalid cam_params shape at {cam_param_path}: {cp.shape}")
        return cp.astype(np.float32)

    @staticmethod
    def _rotation_angle_deg(R: np.ndarray):
        cos_theta = (float(np.trace(R)) - 1.0) * 0.5
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return float(np.degrees(np.arccos(cos_theta)))

    @classmethod
    def _pose_delta(cls, pose_a: np.ndarray, pose_b: np.ndarray):
        delta = np.linalg.inv(pose_a) @ pose_b
        return cls._rotation_angle_deg(delta[:3, :3]), float(np.linalg.norm(delta[:3, 3]))

    def _pose_score(self, pose_a: np.ndarray, pose_b: np.ndarray):
        rot, trans = self._pose_delta(pose_a, pose_b)
        rot_term = rot / max(self.recon_rot_threshold_deg, 1e-6)
        trans_term = trans / max(self.recon_trans_threshold, 1e-9)
        return max(rot_term, trans_term)

    def _select_recon_indices_for_part(self, indices: list[int]):
        valid = [i for i in indices if self.records[i].pose is not None]
        if not valid:
            raise RuntimeError("cannot sample reconstruction views without cam_params poses")
        max_views = max(1, int(round(self.recon_max_views_per_part * self.recon_view_density_scale)))
        min_views = min(max(1, self.recon_min_views_per_part), max_views, len(valid))

        selected = [valid[0]]
        last_pose = self.records[selected[-1]].pose
        dense_gate = 1.0 / max(self.recon_view_density_scale, 1e-6)
        for idx in valid[1:]:
            if len(selected) >= max_views:
                break
            if self._pose_score(last_pose, self.records[idx].pose) >= dense_gate:
                selected.append(idx)
                last_pose = self.records[idx].pose

        while len(selected) < min_views:
            remaining = [i for i in valid if i not in selected]
            if not remaining:
                break
            best_idx = max(
                remaining,
                key=lambda i: min(self._pose_score(self.records[i].pose, self.records[j].pose) for j in selected),
            )
            selected.append(best_idx)

        if len(selected) > max_views:
            selected = selected[:max_views]
        return sorted(selected, key=lambda i: self.records[i].frame_id)

    def get_part_cache_dir(self, obj_name: str, part_name: str, recon_model: str):
        if self.cache_split:
            return os.path.join(self.cache_root, recon_model, self.cache_split, obj_name, part_name)
        return os.path.join(self.cache_root, recon_model, obj_name, part_name)

    def get_part_manifest_path(self, obj_name: str, part_name: str, recon_model: str):
        return os.path.join(self.get_part_cache_dir(obj_name, part_name, recon_model), "manifest.json")

    def get_recon_path_for_record(self, rec: PartRecord, recon_model: str):
        recon_model = str(recon_model).lower().strip()
        if self._is_object_level_recon_model(recon_model):
            split_parts = [self.cache_root, recon_model]
            if self.cache_split:
                split_parts.append(self.cache_split)
            split_parts.extend(
                [
                    rec.obj_name,
                    "pose_ready_models",
                    "view_0",
                    self._part_model_name(rec.part_name),
                    "model.obj",
                ]
            )
            return os.path.join(*split_parts)
        return os.path.join(
            self.get_part_cache_dir(rec.obj_name, rec.part_name, recon_model),
            rec.frame_id,
            "model.obj",
        )

    def _write_manifest(
        self,
        part_key: tuple[str, str],
        recon_model: str,
        selected_indices: list[int],
        skipped_frames: list[dict] | None = None,
        complete: bool = True,
    ):
        obj_name, part_name = part_key
        selected = [self.records[idx] for idx in selected_indices]
        data = {
            "obj_name": obj_name,
            "part_name": part_name,
            "selected_frames": [
                {
                    "frame_id": rec.frame_id,
                    "pose": rec.pose.astype(float).tolist(),
                    "recon_mesh_path": self.get_recon_path_for_record(rec, recon_model),
                }
                for rec in selected
            ],
            "recon_model": recon_model,
            "sampling": {
                "min_views_per_part": self.recon_min_views_per_part,
                "max_views_per_part": self.recon_max_views_per_part,
                "rot_threshold_deg": self.recon_rot_threshold_deg,
                "trans_threshold": self.recon_trans_threshold,
            },
            "complete": bool(complete),
            "skipped_frames": skipped_frames or [],
        }
        os.makedirs(self.get_part_cache_dir(obj_name, part_name, recon_model), exist_ok=True)
        with open(self.get_part_manifest_path(obj_name, part_name, recon_model), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._manifest_cache[(part_key[0], part_key[1], recon_model)] = data
        return data

    def _load_manifest(self, part_key: tuple[str, str], recon_model: str):
        cache_key = (part_key[0], part_key[1], recon_model)
        if cache_key in self._manifest_cache:
            return self._manifest_cache[cache_key]
        obj_name, part_name = part_key
        path = self.get_part_manifest_path(obj_name, part_name, recon_model)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._manifest_cache[cache_key] = data
        return data

    def _load_manifests_for_part(self, part_key: tuple[str, str]):
        return {m: self._load_manifest(part_key, m) for m in self._active_recon_models()}

    def _manifest_is_complete(self, manifest: dict | None):
        if not manifest:
            return False
        if not bool(manifest.get("complete", True)):
            return False
        frames = manifest.get("selected_frames", [])
        part_key = (manifest.get("obj_name"), manifest.get("part_name"))
        valid_count = len(self.part_to_indices.get(part_key, []))
        if valid_count <= 0:
            return False
        expected_min = min(
            max(1, self.recon_min_views_per_part),
            max(1, self.recon_max_views_per_part),
            valid_count,
        )
        if len(frames) < expected_min:
            return False
        for item in frames:
            if not self._recon_output_is_complete(item.get("recon_mesh_path", "")):
                return False
        return True

    @staticmethod
    def _recon_output_is_complete(recon_mesh_path: str):
        if not recon_mesh_path or (not os.path.exists(recon_mesh_path)):
            return False
        meta_path = os.path.join(os.path.dirname(recon_mesh_path), "recon_meta.json")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return bool(meta.get("gt_pose_aligned", False))
        except Exception:
            return False

    def _write_object_recon_meta_if_needed(self, recon_mesh_path: str, recon_model: str, backend: str):
        if not recon_mesh_path or not os.path.exists(recon_mesh_path):
            return False
        meta_path = os.path.join(os.path.dirname(recon_mesh_path), "recon_meta.json")
        if os.path.exists(meta_path):
            return self._recon_output_is_complete(recon_mesh_path)
        data = {
            "gt_pose_aligned": True,
            "recon_model": str(recon_model),
            "backend": str(backend),
            "source": "reconstruction_object_level_pipeline",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True

    def _object_recon_runner(self, recon_model: str):
        self._ensure_reconstruction_imports()
        model = str(recon_model).lower().strip()
        if model == "sam3d_tsdf":
            from run.recon_sam3d_tsdf import reconstruct_object
        elif model == "sam3d_tsdf_dmesh":
            from run.recon_sam3d_tsdf_dmesh import reconstruct_object
        elif model == "hunyuan3d_tsdf":
            from run.recon_hunyuan3d_tsdf import reconstruct_object
        elif model == "hunyuan3d_tsdf_dmesh":
            from run.recon_hunyuan3d_tsdf_dmesh import reconstruct_object
        elif model == "instantmesh_tsdf":
            from run.recon_instantmesh_tsdf import reconstruct_object
        elif model == "instantmesh_tsdf_dmesh":
            from run.recon_instantmesh_tsdf_dmesh import reconstruct_object
        else:
            raise ValueError(f"not an object-level reconstruction model: {recon_model}")
        return reconstruct_object

    def _make_object_recon_args(self, recon_model: str, obj_name: str):
        self._ensure_reconstruction_imports()
        from recon_utils import add_common_args  # noqa
        from recon_tsdf_common import add_tsdf_args  # noqa

        parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
        add_common_args(parser, str(recon_model))
        add_tsdf_args(parser)
        if str(recon_model).lower().strip().endswith("_tsdf_dmesh"):
            from recon_dmesh_common import add_dmesh_args  # noqa

            add_dmesh_args(parser)
        if self._base_recon_model(recon_model) == "instantmesh":
            from run.recon_instantmesh import add_instantmesh_args  # noqa

            add_instantmesh_args(parser)
        args = parser.parse_args([])

        split_name = self.cache_split or Path(self.dataset_root).name
        data_root = str(Path(self.dataset_root).resolve().parent)
        args.data_root = data_root
        args.split = split_name
        args.work_root = self.cache_root
        args.objects = str(obj_name)
        args.object_source = "all"
        args.start = 0
        args.end = None
        args.mode = "single"
        args.num_workers = 1
        args.gpus = ""
        args.workers_per_gpu = ""
        args.coord_dir = ""
        args.reset_coord = False
        args.stale_lock_sec = 12 * 3600
        args.poll_interval_sec = 3.0
        args.overwrite = bool(self.rebuild_recon)
        args.build_base_if_missing = True
        args.min_mask_pixels = int(self.min_mask_pixels)
        args.depth_scale = float(self.depth_scale)
        args.pose_convention = "sapien"

        args.model_path = str(self.hunyuan_model_path or "")
        args.subfolder = str(self.hunyuan_subfolder)
        args.num_inference_steps = int(self.hunyuan_num_inference_steps)
        args.octree_resolution = int(self.hunyuan_octree_resolution)
        args.guidance_scale = float(self.hunyuan_guidance_scale)
        args.alignment_samples = getattr(args, "alignment_samples", 50000)
        args.alignment_seed = getattr(args, "alignment_seed", 2026)
        args.min_alignment_points = getattr(args, "min_alignment_points", 200)
        args.alignment_icp_iters = getattr(args, "alignment_icp_iters", 30)
        args.alignment_trim_quantile = getattr(args, "alignment_trim_quantile", 0.8)

        if self._base_recon_model(recon_model) == "instantmesh":
            args.instantmesh_root = str(self.instantmesh_root or "")
            args.instantmesh_config_path = str(self.instantmesh_config_path or "")
            args.instantmesh_diffusion_model = str(self.instantmesh_diffusion_model)
            args.instantmesh_dino_model = str(self.instantmesh_dino_model)
            args.instantmesh_unet_path = str(self.instantmesh_unet_path)
            args.instantmesh_model_path = str(self.instantmesh_model_path)
            args.instantmesh_diffusion_steps = int(self.instantmesh_diffusion_steps)
            args.instantmesh_seed = int(self.seed)
            args.instantmesh_scale = float(self.instantmesh_scale)
            args.instantmesh_view = int(self.instantmesh_view)
            args.instantmesh_foreground_ratio = float(self.instantmesh_foreground_ratio)
            args.instantmesh_export_texmap = bool(self.instantmesh_export_texmap)
        return args

    def _ensure_object_level_recon_for_part(
        self,
        part_key: tuple[str, str],
        rec: PartRecord,
        recon_model: str,
    ):
        target = self.get_recon_path_for_record(rec, recon_model)
        if (not self.rebuild_recon) and self._recon_output_is_complete(target):
            return True

        build_key = (rec.obj_name, str(recon_model))
        if build_key not in self._object_recon_builds or self.rebuild_recon:
            self._object_recon_builds.add(build_key)
            try:
                self._ensure_reconstruction_imports()
                from recon_utils import DatasetObject  # noqa

                args = self._make_object_recon_args(recon_model, rec.obj_name)
                obj = DatasetObject(
                    data_root=Path(args.data_root).resolve(),
                    split=str(args.split),
                    name=str(rec.obj_name),
                )
                print(
                    f"[dataset][object-recon] model={recon_model} obj={rec.obj_name} "
                    f"work_root={args.work_root} split={args.split}",
                    flush=True,
                )
                self._object_recon_runner(recon_model)(obj, args)
            except Exception as e:
                self._recon_failures += 1
                if self._recon_failures <= 5 or (self._recon_failures % 50 == 0):
                    print(
                        f"[dataset][warn] object-level recon failed ({self._recon_failures}) "
                        f"model={recon_model} obj={rec.obj_name} part={rec.part_name} err={repr(e)}",
                        flush=True,
                    )

        ok = os.path.exists(target)
        if ok:
            ok = self._write_object_recon_meta_if_needed(
                target,
                recon_model=recon_model,
                backend="tsdf_dmesh" if str(recon_model).endswith("_tsdf_dmesh") else "tsdf",
            )
        return bool(ok and self._recon_output_is_complete(target))

    def _retry_manifest_skipped_frames(
        self,
        part_key: tuple[str, str],
        recon_model: str,
        manifest: dict | None,
    ):
        """Retry frames that were recorded as skipped in the manifest.

        This is intentionally not an overwrite/rebuild path. It trusts the
        existing successful selected_frames, reads only skipped_frames from the
        JSON manifest, and retries those frame ids if their PartRecord still
        exists in the current dataset index. Successful retries are appended to
        selected_frames and removed from skipped_frames; failed retries stay in
        skipped_frames so another resume can try again later.
        """
        if not manifest or self.rebuild_recon or self.force_resample_recon:
            return manifest
        skipped_frames = list(manifest.get("skipped_frames", []) or [])
        if not skipped_frames:
            return manifest

        indices_by_frame = {
            str(self.records[idx].frame_id): idx
            for idx in self.part_to_indices.get(part_key, [])
        }
        selected_frames = list(manifest.get("selected_frames", []) or [])
        selected_frame_ids = {
            str(item.get("frame_id"))
            for item in selected_frames
            if item.get("frame_id") is not None
        }
        kept_skipped = []
        retried = 0
        recovered = 0

        for item in skipped_frames:
            frame_id = str(item.get("frame_id", ""))
            idx = indices_by_frame.get(frame_id)
            if idx is None or frame_id in selected_frame_ids:
                kept_skipped.append(item)
                continue
            rec = self.records[idx]
            recon_obj_path = self.get_recon_path_for_record(rec, recon_model)
            retried += 1
            if (not self.rebuild_recon) and self._recon_output_is_complete(recon_obj_path):
                ok = True
            else:
                ok = self._reconstruct_mesh_from_rgb_mask(
                    rec.rgb_path,
                    rec.mask_path,
                    rec.depth_path,
                    rec.K_path,
                    rec.pose,
                    recon_obj_path,
                    recon_model=recon_model,
                )
            if ok:
                recovered += 1
                selected_frames.append(
                    {
                        "frame_id": rec.frame_id,
                        "pose": rec.pose.astype(float).tolist(),
                        "recon_mesh_path": recon_obj_path,
                    }
                )
                selected_frame_ids.add(frame_id)
            else:
                kept_skipped.append(item)

        if retried <= 0:
            return manifest

        manifest = dict(manifest)
        manifest["selected_frames"] = selected_frames
        manifest["skipped_frames"] = kept_skipped
        if recovered > 0:
            manifest["complete"] = bool(manifest.get("complete", False) or self._manifest_is_complete(manifest))
        os.makedirs(self.get_part_cache_dir(part_key[0], part_key[1], recon_model), exist_ok=True)
        with open(self.get_part_manifest_path(part_key[0], part_key[1], recon_model), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        self._manifest_cache[(part_key[0], part_key[1], recon_model)] = manifest
        print(
            f"[dataset][resume] retried skipped frames model={recon_model} "
            f"obj={part_key[0]} part={part_key[1]} retried={retried} recovered={recovered} "
            f"remaining_skipped={len(kept_skipped)}",
            flush=True,
        )
        return manifest

    def ensure_recon_cache_for_part(self, part_key: tuple[str, str]):
        manifests = {}
        for recon_model in self._active_recon_models():
            manifest = None if self.force_resample_recon else self._load_manifest(part_key, recon_model)
            if (not self.rebuild_recon) and self._manifest_is_complete(manifest):
                manifest = self._retry_manifest_skipped_frames(part_key, recon_model, manifest)
                manifests[recon_model] = manifest
                continue

            indices = self.part_to_indices.get(part_key, [])
            if not indices:
                manifests[recon_model] = self._write_manifest(
                    part_key,
                    recon_model,
                    [],
                    skipped_frames=[{"reason": "no_visible_records"}],
                    complete=False,
                )
                continue

            selected_indices = self._select_recon_indices_for_part(indices)
            target_min = min(
                max(1, self.recon_min_views_per_part),
                max(1, int(round(self.recon_max_views_per_part * self.recon_view_density_scale))),
                len(indices),
            )
            target_max = min(max(1, int(round(self.recon_max_views_per_part * self.recon_view_density_scale))), len(indices))

            if self._is_object_level_recon_model(recon_model):
                rec0 = self.records[selected_indices[0]]
                ok = self._ensure_object_level_recon_for_part(part_key, rec0, recon_model)
                successful_indices = selected_indices[:target_max] if ok else []
                skipped_frames = [] if ok else [
                    {
                        "reason": "object_level_reconstruction_failed",
                        "recon_model": recon_model,
                        "expected_model": self.get_recon_path_for_record(rec0, recon_model),
                    }
                ]
                complete = len(successful_indices) >= target_min
                manifest = self._write_manifest(
                    part_key,
                    recon_model,
                    successful_indices,
                    skipped_frames=skipped_frames,
                    complete=complete,
                )
                manifests[recon_model] = manifest
                if not complete:
                    self._bad_recon_skips += 1
                    if self._bad_recon_skips <= 10 or (self._bad_recon_skips % 100 == 0):
                        print(
                            f"[dataset][skip] incomplete object-level reconstruction cache ({self._bad_recon_skips}) "
                            f"model={recon_model} obj={part_key[0]} part={part_key[1]} "
                            f"success={len(successful_indices)}/{target_min}"
                        )
                continue

            selected_set = set(selected_indices)
            remaining_indices = [idx for idx in indices if idx not in selected_set]
            candidate_indices = selected_indices + remaining_indices
            successful_indices = []
            skipped_frames = []

            for idx in candidate_indices:
                if len(successful_indices) >= target_max:
                    break
                rec = self.records[idx]
                recon_obj_path = self.get_recon_path_for_record(rec, recon_model)
                if (not self.rebuild_recon) and self._recon_output_is_complete(recon_obj_path):
                    successful_indices.append(idx)
                    continue
                ok = self._reconstruct_mesh_from_rgb_mask(
                    rec.rgb_path,
                    rec.mask_path,
                    rec.depth_path,
                    rec.K_path,
                    rec.pose,
                    recon_obj_path,
                    recon_model=recon_model,
                )
                if ok:
                    successful_indices.append(idx)
                else:
                    skipped_frames.append(
                        {
                            "frame_id": rec.frame_id,
                            "reason": "invisible_or_reconstruction_failed",
                            "rgb_path": rec.rgb_path,
                            "mask_path": rec.mask_path,
                        }
                    )

            complete = len(successful_indices) >= target_min
            manifest = self._write_manifest(
                part_key,
                recon_model,
                successful_indices,
                skipped_frames=skipped_frames,
                complete=complete,
            )
            manifests[recon_model] = manifest
            if not complete:
                self._bad_recon_skips += 1
                if self._bad_recon_skips <= 10 or (self._bad_recon_skips % 100 == 0):
                    print(
                        f"[dataset][skip] incomplete reconstruction cache ({self._bad_recon_skips}) "
                        f"model={recon_model} obj={part_key[0]} part={part_key[1]} "
                        f"success={len(successful_indices)}/{target_min} skipped={len(skipped_frames)}"
                    )
        return manifests if self.recon_model == "all" else manifests[self.recon_model]

    def _select_cached_recon_for_record(self, rec: PartRecord):
        part_key = (rec.obj_name, rec.part_name)
        out = []
        seen = set()
        for recon_model in self._active_recon_models():
            manifest = self._load_manifest(part_key, recon_model)
            if not self._manifest_is_complete(manifest):
                if not self.allow_recon_write:
                    raise FileNotFoundError(
                        f"reconstruction cache missing/incomplete for {rec.obj_name}/{rec.part_name}; "
                        f"manifest={self.get_part_manifest_path(rec.obj_name, rec.part_name, recon_model)}"
                    )
                ensured = self.ensure_recon_cache_for_part(part_key)
                manifest = ensured.get(recon_model) if isinstance(ensured, dict) else ensured

            frames = manifest.get("selected_frames", [])
            for item in frames:
                recon_mesh_path = item.get("recon_mesh_path")
                dedupe_key = (recon_model, recon_mesh_path)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                out.append(
                    {
                        "recon_model": recon_model,
                        "recon_frame_id": item.get("frame_id"),
                        "recon_mesh_path": recon_mesh_path,
                    }
                )
        return out

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _resolve_depth_path(obj_dir: str, frame_id: str):
        cands = [
            os.path.join(obj_dir, "depth", f"{frame_id}.png"),
            os.path.join(obj_dir, "depth", f"{frame_id}.npy"),
            os.path.join(obj_dir, "depth", f"{frame_id}.exr"),
        ]
        for p in cands:
            if os.path.exists(p):
                return p
        return None

    def _load_depth(self, depth_path: str):
        ext = os.path.splitext(depth_path)[1].lower()
        if ext == ".npy":
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"failed to read depth: {depth_path}")
            depth = depth.astype(np.float32)
        # Convert integer depth map (typically mm) to meter.
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if depth.size == 0:
            raise RuntimeError(f"empty depth map: {depth_path}")
        # Heuristic: if values are likely in millimeters, convert by scale.
        if np.nanmax(depth) > 100.0:
            depth = depth / max(self.depth_scale, 1.0)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return depth

    @staticmethod
    def _depth_to_camera_xyz(depth: np.ndarray, K: np.ndarray):
        H, W = depth.shape[:2]
        ys, xs = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij",
        )
        z = depth.astype(np.float32)
        x = (xs - float(K[0, 2])) * z / max(float(K[0, 0]), 1e-9)
        y = (ys - float(K[1, 2])) * z / max(float(K[1, 1]), 1e-9)
        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
        invalid = (~np.isfinite(z)) | (z <= 1e-6)
        xyz[invalid] = np.nan
        return xyz

    def _build_real_depth_pointmap(self, depth_path: str, K_path: str, image_shape, mask: np.ndarray | None = None):
        depth = self._load_depth(depth_path)
        K = np.loadtxt(K_path, dtype=np.float32).reshape(3, 3).copy()
        target_h, target_w = int(image_shape[0]), int(image_shape[1])
        if depth.shape[:2] != (target_h, target_w):
            src_h, src_w = depth.shape[:2]
            sx = float(target_w) / max(float(src_w), 1.0)
            sy = float(target_h) / max(float(src_h), 1.0)
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        if mask is not None and mask.shape[:2] != depth.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        xyz_camera = self._depth_to_camera_xyz(depth, K)
        # SAM3D's explicit pointmap branch expects PyTorch3D camera coordinates.
        xyz_camera[..., 0] *= -1.0
        xyz_camera[..., 1] *= -1.0
        return torch.from_numpy(xyz_camera)

    @staticmethod
    def _apply_transform_np(points: np.ndarray, transform: np.ndarray):
        if points.shape[0] == 0:
            return points.astype(np.float32)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([points.astype(np.float32), ones], axis=1)
        return (transform @ homo.T).T[:, :3].astype(np.float32)

    @staticmethod
    def _sample_points_np(points: np.ndarray, max_points: int, rng: np.random.Generator):
        points = np.asarray(points, dtype=np.float32)
        if points.shape[0] <= max_points:
            return points
        ids = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[ids].astype(np.float32)

    @staticmethod
    def _umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
        if src.shape != dst.shape or src.shape[0] < 3:
            raise ValueError(f"invalid paired points for similarity: {src.shape} vs {dst.shape}")
        src = src.astype(np.float64)
        dst = dst.astype(np.float64)
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean
        cov = (dst_c.T @ src_c) / max(src.shape[0], 1)
        u, s, vt = np.linalg.svd(cov)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1.0
            r = u @ vt
        if with_scale:
            var_src = float(np.sum(src_c * src_c) / max(src.shape[0], 1))
            scale = float(np.sum(s) / max(var_src, 1e-12))
            scale = float(np.clip(scale, 0.02, 50.0))
        else:
            scale = 1.0
        tf = np.eye(4, dtype=np.float32)
        tf[:3, :3] = (scale * r).astype(np.float32)
        tf[:3, 3] = (dst_mean - scale * (r @ src_mean)).astype(np.float32)
        return tf

    @staticmethod
    def _query_nearest(src: np.ndarray, dst: np.ndarray):
        if cKDTree is not None:
            dists, idx = cKDTree(dst).query(src, k=1, workers=-1)
            return dists.astype(np.float32), idx.astype(np.int64)
        idx = np.zeros((src.shape[0],), dtype=np.int64)
        dists = np.zeros((src.shape[0],), dtype=np.float32)
        chunk = 1024
        for s in range(0, src.shape[0], chunk):
            e = min(s + chunk, src.shape[0])
            diff = src[s:e, None, :] - dst[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            nn = np.argmin(d2, axis=1)
            idx[s:e] = nn
            dists[s:e] = np.sqrt(d2[np.arange(e - s), nn]).astype(np.float32)
        return dists, idx

    @staticmethod
    def _pca_axes(points: np.ndarray):
        centered = points - points.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(points.shape[0] - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        axes = vecs[:, order].astype(np.float32)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0
        return axes

    def _estimate_similarity_alignment(self, src_points: np.ndarray, dst_points: np.ndarray, rng: np.random.Generator):
        src = self._sample_points_np(src_points, 5000, rng)
        dst = self._sample_points_np(dst_points, 5000, rng)
        if src.shape[0] < 50 or dst.shape[0] < 50:
            raise ValueError(f"too few points for alignment: src={src.shape[0]} dst={dst.shape[0]}")

        src_ctr = src.mean(axis=0)
        dst_ctr = dst.mean(axis=0)
        src_span = np.percentile(src, 95, axis=0) - np.percentile(src, 5, axis=0)
        dst_span = np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0)
        scale0 = float(np.linalg.norm(dst_span) / max(np.linalg.norm(src_span), 1e-8))
        scale0 = float(np.clip(scale0, 0.02, 50.0))

        seeds = []
        base = np.eye(4, dtype=np.float32)
        base[:3, :3] *= scale0
        base[:3, 3] = (dst_ctr - scale0 * src_ctr).astype(np.float32)
        seeds.append(base)

        try:
            src_axes = self._pca_axes(src)
            dst_axes = self._pca_axes(dst)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        signs = np.diag([sx, sy, sz]).astype(np.float32)
                        if np.linalg.det(signs) < 0:
                            continue
                        r0 = (dst_axes @ signs @ src_axes.T).astype(np.float32)
                        tf = np.eye(4, dtype=np.float32)
                        tf[:3, :3] = scale0 * r0
                        tf[:3, 3] = (dst_ctr - scale0 * (r0 @ src_ctr)).astype(np.float32)
                        seeds.append(tf)
        except Exception:
            pass

        def score_tf(tf):
            pts = self._apply_transform_np(src, tf)
            dists, _ = self._query_nearest(pts, dst)
            if dists.shape[0] < 50:
                return np.inf
            keep = dists <= np.percentile(dists, 80)
            if int(np.sum(keep)) < 50:
                return np.inf
            return float(np.sqrt(np.mean(dists[keep] ** 2)))

        best_tf = None
        best_score = np.inf
        for seed in seeds:
            tf = seed.copy()
            for q in (70, 75, 80, 85, 90):
                pts_now = self._apply_transform_np(src, tf)
                dists, nn_idx = self._query_nearest(pts_now, dst)
                keep = dists <= np.percentile(dists, q)
                if int(np.sum(keep)) < 50:
                    continue
                delta = self._umeyama_similarity(pts_now[keep], dst[nn_idx[keep]], with_scale=True)
                tf = (delta @ tf).astype(np.float32)
            s = score_tf(tf)
            if s < best_score:
                best_score = s
                best_tf = tf.copy()
        if best_tf is None or not np.isfinite(best_score):
            raise ValueError("no valid similarity alignment")
        return best_tf, best_score

    def _backproject_mask_to_object_points(
        self,
        mask: np.ndarray,
        depth_path: str,
        K_path: str,
        pose: np.ndarray,
        image_shape,
        rng: np.random.Generator,
    ):
        depth = self._load_depth(depth_path)
        K = np.loadtxt(K_path, dtype=np.float32).reshape(3, 3).copy()
        target_h, target_w = int(image_shape[0]), int(image_shape[1])
        if depth.shape[:2] != (target_h, target_w):
            src_h, src_w = depth.shape[:2]
            sx = float(target_w) / max(float(src_w), 1.0)
            sy = float(target_h) / max(float(src_h), 1.0)
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        if mask.shape[:2] != depth.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        valid = (mask > 0) & np.isfinite(depth) & (depth > 1e-6)
        ys, xs = np.where(valid)
        if len(xs) < 50:
            raise ValueError(f"too few valid masked depth pixels: {len(xs)}")
        if len(xs) > 30000:
            pick = rng.choice(len(xs), size=30000, replace=False)
            ys = ys[pick]
            xs = xs[pick]
        z = depth[ys, xs].astype(np.float32)
        x = (xs.astype(np.float32) - float(K[0, 2])) * z / max(float(K[0, 0]), 1e-9)
        y = (ys.astype(np.float32) - float(K[1, 2])) * z / max(float(K[1, 1]), 1e-9)
        pts_cam = np.stack([x, y, z], axis=1).astype(np.float32)
        cam_to_obj = np.linalg.inv(np.asarray(pose, dtype=np.float32).reshape(4, 4))
        return self._apply_transform_np(pts_cam, cam_to_obj)

    def _align_mesh_to_gt_pose_frame(
        self,
        mesh: trimesh.Trimesh,
        mask: np.ndarray,
        depth_path: str,
        K_path: str,
        pose: np.ndarray | None,
        image_shape,
    ):
        if pose is None:
            raise ValueError("missing gt pose for reconstruction alignment")
        rng = np.random.default_rng(self.seed)
        target_obj = self._backproject_mask_to_object_points(mask, depth_path, K_path, pose, image_shape, rng)
        sample_count = max(3000, min(20000, int(len(mesh.vertices) * 4)))
        src_points = mesh.sample(sample_count).astype(np.float32)
        if src_points.shape[0] < 50:
            src_points = np.asarray(mesh.vertices, dtype=np.float32)
        tf, rmse = self._estimate_similarity_alignment(src_points, target_obj, rng)
        mesh.apply_transform(tf)
        return rmse

    def get_recon_path(self, rec: PartRecord):
        return self.get_recon_path_for_record(rec, self.recon_model if self.recon_model != "all" else "sam3d")

    @staticmethod
    def _mesh_from_any(mesh):
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            geoms = [g for g in mesh.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(tuple(geoms))
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        if (not hasattr(mesh, "vertices")) or (len(mesh.vertices) == 0):
            return None
        if (not hasattr(mesh, "faces")) or (len(mesh.faces) == 0):
            return None
        return mesh

    def _save_aligned_recon_mesh(
        self,
        mesh,
        mask: np.ndarray,
        depth_path: str,
        K_path: str,
        pose: np.ndarray | None,
        out_obj_path: str,
        image_shape,
        recon_model: str,
        backend: str,
        extra_meta: dict | None = None,
    ):
        mesh = self._mesh_from_any(mesh)
        if mesh is None:
            return False
        align_rmse = self._align_mesh_to_gt_pose_frame(
            mesh,
            mask,
            depth_path,
            K_path,
            pose,
            image_shape,
        )
        os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
        mesh.export(out_obj_path)
        meta = {
            "gt_pose_aligned": True,
            "alignment_rmse": float(align_rmse),
            "recon_model": recon_model,
            "backend": backend,
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path = os.path.join(os.path.dirname(out_obj_path), "recon_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return os.path.exists(out_obj_path)

    def build_recon_for_index(self, index: int):
        rec = self.records[index]
        manifest = self.ensure_recon_cache_for_part((rec.obj_name, rec.part_name))
        return manifest

    def _reconstruct_mesh_from_rgb_mask(
        self,
        rgb_path: str,
        mask_path: str,
        depth_path: str,
        K_path: str,
        pose: np.ndarray | None,
        out_obj_path: str,
        recon_model: str = "sam3d",
    ):
        recon_model = str(recon_model).lower().strip()
        if recon_model == "hunyuan3d":
            return self._reconstruct_mesh_hunyuan3d(
                rgb_path,
                mask_path,
                depth_path,
                K_path,
                pose,
                out_obj_path,
            )
        if recon_model == "instantmesh":
            return self._reconstruct_mesh_instantmesh(
                rgb_path,
                mask_path,
                depth_path,
                K_path,
                pose,
                out_obj_path,
            )
        if recon_model != "sam3d":
            raise ValueError(f"unsupported recon_model={recon_model}")
        return self._reconstruct_mesh_sam3d(
            rgb_path,
            mask_path,
            depth_path,
            K_path,
            pose,
            out_obj_path,
        )

    def _reconstruct_mesh_sam3d(
        self,
        rgb_path: str,
        mask_path: str,
        depth_path: str,
        K_path: str,
        pose: np.ndarray | None,
        out_obj_path: str,
    ):
        inf = self._get_inference()
        # Lazy import after path setup.
        from inference import load_image  # noqa

        image = load_image(rgb_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        # NOTE: SAM3D Inference.merge_mask_to_rgba() internally multiplies mask by 255.
        # Keep mask as binary {0,1} here to avoid uint8 overflow (255*255 -> 1).
        mask = (mask > 0).astype(np.uint8)
        visible_ok, reason = self._visible_sample_ok(mask)
        if not visible_ok:
            self._invisible_record_skips += 1
            if self._invisible_record_skips <= 5 or (self._invisible_record_skips % 1000 == 0):
                print(
                    f"[dataset][skip] invisible/weak recon input ({self._invisible_record_skips}) "
                    f"rgb={rgb_path} mask={mask_path} reason={reason}"
                )
            return False
        # Keep mask and image shape aligned to avoid downstream empty tensors.
        if hasattr(image, "shape") and len(image.shape) >= 2:
            ih, iw = int(image.shape[0]), int(image.shape[1])
            if (ih, iw) != mask.shape[:2]:
                mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
        try:
            pointmap = None
            if self.use_real_depth_pointmap:
                image_shape = image.shape[:2] if hasattr(image, "shape") and len(image.shape) >= 2 else mask.shape[:2]
                pointmap = self._build_real_depth_pointmap(depth_path, K_path, image_shape, mask=mask)
            output = inf(image, mask, seed=self.seed, pointmap=pointmap)
            output_mesh = inf._pipeline.postprocess_slat_output(
                output,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                use_vertex_color=False,
            )
            mesh = output_mesh.get("glb", None)
            return self._save_aligned_recon_mesh(
                mesh,
                mask,
                depth_path,
                K_path,
                pose,
                out_obj_path,
                image.shape[:2] if hasattr(image, "shape") and len(image.shape) >= 2 else mask.shape[:2],
                recon_model="sam3d",
                backend="sam3d",
                extra_meta={"used_real_depth_pointmap": bool(pointmap is not None)},
            )
        except Exception as e:
            self._recon_failures += 1
            if self._recon_failures <= 5 or (self._recon_failures % 50 == 0):
                print(
                    f"[dataset][warn] recon failed ({self._recon_failures}) "
                    f"rgb={rgb_path} mask={mask_path} err={repr(e)}"
                )
            return False

    def _reconstruct_mesh_hunyuan3d(
        self,
        rgb_path: str,
        mask_path: str,
        depth_path: str,
        K_path: str,
        pose: np.ndarray | None,
        out_obj_path: str,
    ):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        mask = (mask > 0).astype(np.uint8)
        visible_ok, reason = self._visible_sample_ok(mask)
        if not visible_ok:
            self._invisible_record_skips += 1
            if self._invisible_record_skips <= 5 or (self._invisible_record_skips % 1000 == 0):
                print(
                    f"[dataset][skip] invisible/weak Hunyuan3D input ({self._invisible_record_skips}) "
                    f"rgb={rgb_path} mask={mask_path} reason={reason}"
                )
            return False
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            return False
        if rgb_bgr.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            self._ensure_reconstruction_imports()
            from reconstruct_hunyuan3d import _prepare_rgba_from_mask  # noqa

            rgba = _prepare_rgba_from_mask(rgb_bgr, (mask * 255).astype(np.uint8))
            if rgba is None:
                return False
            reconstructor = self._get_hunyuan_reconstructor()
            raw_obj_path = os.path.join(os.path.dirname(out_obj_path), "raw_hunyuan3d.obj")
            reconstructor.reconstruct_part(rgba, raw_obj_path)
            mesh = trimesh.load(raw_obj_path, force="mesh", process=False)
            return self._save_aligned_recon_mesh(
                mesh,
                mask,
                depth_path,
                K_path,
                pose,
                out_obj_path,
                rgb_bgr.shape[:2],
                recon_model="hunyuan3d",
                backend="hunyuan3d",
                extra_meta={
                    "raw_mesh_path": raw_obj_path,
                    "hunyuan_model_path": str(self.hunyuan_model_path or ""),
                    "hunyuan_subfolder": self.hunyuan_subfolder,
                    "hunyuan_num_inference_steps": int(self.hunyuan_num_inference_steps),
                    "hunyuan_octree_resolution": int(self.hunyuan_octree_resolution),
                    "hunyuan_guidance_scale": float(self.hunyuan_guidance_scale),
                },
            )
        except Exception as e:
            self._recon_failures += 1
            if self._recon_failures <= 5 or (self._recon_failures % 50 == 0):
                print(
                    f"[dataset][warn] Hunyuan3D recon failed ({self._recon_failures}) "
                    f"rgb={rgb_path} mask={mask_path} err={repr(e)}"
            )
            return False

    def _reconstruct_mesh_instantmesh(
        self,
        rgb_path: str,
        mask_path: str,
        depth_path: str,
        K_path: str,
        pose: np.ndarray | None,
        out_obj_path: str,
    ):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        mask = (mask > 0).astype(np.uint8)
        visible_ok, reason = self._visible_sample_ok(mask)
        if not visible_ok:
            self._invisible_record_skips += 1
            if self._invisible_record_skips <= 5 or (self._invisible_record_skips % 1000 == 0):
                print(
                    f"[dataset][skip] invisible/weak InstantMesh input ({self._invisible_record_skips}) "
                    f"rgb={rgb_path} mask={mask_path} reason={reason}"
                )
            return False
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            return False
        if rgb_bgr.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            self._ensure_reconstruction_imports()
            from reconstruct_hunyuan3d import _prepare_rgba_from_mask  # noqa

            rgba = _prepare_rgba_from_mask(rgb_bgr, (mask * 255).astype(np.uint8))
            if rgba is None:
                return False
            reconstructor = self._get_instantmesh_reconstructor()
            raw_obj_path = os.path.join(os.path.dirname(out_obj_path), "raw_instantmesh.obj")
            reconstructor.reconstruct_part(rgba, raw_obj_path)
            mesh = trimesh.load(raw_obj_path, force="mesh", process=False)
            return self._save_aligned_recon_mesh(
                mesh,
                mask,
                depth_path,
                K_path,
                pose,
                out_obj_path,
                rgb_bgr.shape[:2],
                recon_model="instantmesh",
                backend="instantmesh",
                extra_meta={
                    "raw_mesh_path": raw_obj_path,
                    "instantmesh_root": str(self.instantmesh_root or ""),
                    "instantmesh_config_path": str(self.instantmesh_config_path or ""),
                    "instantmesh_diffusion_model": self.instantmesh_diffusion_model,
                    "instantmesh_dino_model": self.instantmesh_dino_model,
                    "instantmesh_unet_path": self.instantmesh_unet_path,
                    "instantmesh_model_path": self.instantmesh_model_path,
                    "instantmesh_diffusion_steps": int(self.instantmesh_diffusion_steps),
                    "instantmesh_scale": float(self.instantmesh_scale),
                    "instantmesh_view": int(self.instantmesh_view),
                    "instantmesh_foreground_ratio": float(self.instantmesh_foreground_ratio),
                    "instantmesh_export_texmap": bool(self.instantmesh_export_texmap),
                },
            )
        except Exception as e:
            self._recon_failures += 1
            if self._recon_failures <= 5 or (self._recon_failures % 50 == 0):
                print(
                    f"[dataset][warn] InstantMesh recon failed ({self._recon_failures}) "
                    f"rgb={rgb_path} mask={mask_path} err={repr(e)}"
                )
            return False

    def __getitem__(self, index):
        self._getitem_calls += 1
        if self._getitem_calls <= 3:
            print(f"[dataset] __getitem__ call={self._getitem_calls} index={index}")
        rec = self.records[index]
        try:
            recon_candidates = self._select_cached_recon_for_record(rec)
        except Exception:
            if not self.fallback_to_gt_mesh_on_recon_fail:
                raise
            recon_candidates = [
                {
                    "recon_model": "fallback_gt",
                    "recon_frame_id": rec.frame_id,
                    "recon_mesh_path": rec.gt_mesh_path,
                }
            ]
            self._fallback_uses += 1
            if self._fallback_uses <= 5 or (self._fallback_uses % 100 == 0):
                print(
                    f"[dataset][fallback] use gt mesh as recon ({self._fallback_uses}) "
                    f"{rec.obj_name}/{rec.part_name}/{rec.frame_id}"
                )

        init_pose = rec.pose
        if init_pose is None:
            raise RuntimeError(f"missing cam_params pose for {rec.obj_name}/{rec.part_name}/{rec.frame_id}")

        if self.defer_sample_io:
            return {
                "obj_name": rec.obj_name,
                "frame_id": rec.frame_id,
                "part_name": rec.part_name,
                "recon_candidates": recon_candidates,
                "rgb_path": rec.rgb_path,
                "mask_path": rec.mask_path,
                "depth_path": rec.depth_path,
                "K_path": rec.K_path,
                "depth_scale": self.depth_scale,
                "min_mask_pixels": self.min_mask_pixels,
                "recon_mesh_paths": [c["recon_mesh_path"] for c in recon_candidates if c.get("recon_mesh_path")],
                "gt_mesh_path": rec.gt_mesh_path,
                "init_pose": init_pose,
            }

        mask = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask: {rec.mask_path}")
        if int(np.count_nonzero(mask > 0)) < self.min_mask_pixels:
            raise RuntimeError(
                f"mask has too few valid pixels ({int(np.count_nonzero(mask > 0))}) in {rec.mask_path}"
            )

        rgb = cv2.imread(rec.rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read rgb: {rec.rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask_bin = (mask > 0).astype(np.uint8)
        depth = self._load_depth(rec.depth_path)
        if depth.shape[:2] != mask_bin.shape[:2]:
            raise RuntimeError(
                f"depth/mask shape mismatch: depth={depth.shape} mask={mask_bin.shape} file={rec.depth_path}"
            )
        K = np.loadtxt(rec.K_path, dtype=np.float32).reshape(3, 3)

        sample = {
            "obj_name": rec.obj_name,
            "frame_id": rec.frame_id,
            "part_name": rec.part_name,
            "recon_candidates": recon_candidates,
            "rgb": rgb,
            "mask": mask_bin,
            "depth": depth.astype(np.float32),
            "K": K,
            "recon_mesh_paths": [c["recon_mesh_path"] for c in recon_candidates if c.get("recon_mesh_path")],
            "gt_mesh_path": rec.gt_mesh_path,
            "init_pose": init_pose,
        }
        return sample
