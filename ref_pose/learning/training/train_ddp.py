import argparse
import inspect
import json
import math
import os
import random
import time
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import kornia
import nvdiffrast.torch as dr
import trimesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from omegaconf import OmegaConf

sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan")
sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/ref_pose")

from Utils import *  # noqa
from learning.models.refine_network import RefineNet
from learning.datasets.h5_dataset import PoseRefinePairH5Dataset
from learning.datasets.pose_dataset import BatchPoseData
from learning.datasets.sam3d_part_dataset import Sam3DPartTrainDataset
from ref_pose.learning.training.train_refine_validity_mask import (
    DEFAULT_REF_POSE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SAM3D_PROJECT_ROOT,
    DEFAULT_SAM3D_NOTEBOOK_ROOT,
    DEFAULT_RECON_CACHE_ROOT,
    _resolve_sam3d_config_path,
    _resolve_train_val_roots,
    _resolve_refiner_config_path,
    _resolve_refiner_base_ckpt,
    _load_refiner_backbone_only,
)

# mp.set_sharing_strategy("file_system")

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def collate_list(batch):
    return batch


TRAIN_DDP_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "configs",
    "train_ddp.yaml",
)
DEFAULT_PARTNET_ROOT = str(Path(__file__).resolve().parents[3])
DEFAULT_HUNYUAN_MODEL_PATH = os.path.join(DEFAULT_PARTNET_ROOT, "Hunyuan3D-2.1", "ckpts")
DEFAULT_INSTANTMESH_ROOT = os.path.join(DEFAULT_PARTNET_ROOT, "InstantMesh")
DEFAULT_INSTANTMESH_CONFIG_PATH = os.path.join(DEFAULT_INSTANTMESH_ROOT, "configs", "instant-mesh-large.yaml")


def _safe_vm_weight(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _validity_mask_gate(mask: torch.Tensor, vm_weight: float) -> torch.Tensor:
    w = _safe_vm_weight(vm_weight)
    return (1.0 - w) + w * mask


def _format_vm_weight_tag(vm_weight: float) -> str:
    return f"vmw{_safe_vm_weight(vm_weight):.3f}"


def _load_train_config_defaults(config_path: str) -> Dict[str, Any]:
    path = str(config_path).strip()
    if not path:
        raise ValueError("train config path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"train config not found: {path}")
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"train config must be a mapping/dict: {path}")
    return data


def _is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _ddp_info():
    if _is_dist():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _is_main_process():
    rank, _ = _ddp_info()
    return rank == 0


def _seed_everything(seed: int, rank: int):
    final_seed = int(seed) + int(rank)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def _setup_distributed(args):
    using_torchrun = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)
    if using_torchrun:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = int(args.rank)
        world_size = int(args.world_size)
        local_rank = int(args.local_rank)

    if world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = str(args.master_addr)
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(args.master_port)
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=args.backend,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    else:
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def _cleanup_distributed():
    if _is_dist():
        torch.distributed.destroy_process_group()


def _sync_barrier():
    if _is_dist():
        if torch.cuda.is_available() and torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        else:
            torch.distributed.barrier()


def _reduce_sum_scalar(value: float, device: torch.device):
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if _is_dist():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return float(t.item())


def _reduce_mean_scalar(value: float, device: torch.device):
    s = _reduce_sum_scalar(value, device)
    _, ws = _ddp_info()
    return s / max(1, ws)


def _parse_int_list(raw: str):
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]


def _parse_per_rank_value(raw: str, default_v: int, world_size: int):
    vals = _parse_int_list(raw)
    if not vals:
        return [int(default_v)] * world_size
    if len(vals) == 1:
        return [int(vals[0])] * world_size
    if len(vals) != world_size:
        raise ValueError(f"per-rank list length mismatch: expected {world_size}, got {len(vals)} in '{raw}'")
    return [int(v) for v in vals]


def _recon_uses_gpu_model(recon_model: str) -> bool:
    model = str(recon_model).lower().strip()
    return model in (
        "sam3d",
        "sam3d_tsdf",
        "sam3d_tsdf_dmesh",
        "hunyuan3d",
        "hunyuan3d_tsdf",
        "hunyuan3d_tsdf_dmesh",
        "instantmesh",
        "instantmesh_tsdf",
        "instantmesh_tsdf_dmesh",
        "all",
    )


def _cap_gpu_recon_workers_per_rank(recon_workers_per_rank: List[int], recon_model: str) -> List[int]:
    normalized = [max(1, int(v)) for v in recon_workers_per_rank]
    if _recon_uses_gpu_model(recon_model) and _is_main_process():
        print(
            "[prebuild] GPU reconstruction worker layout: "
            f"recon_model={recon_model} recon_workers_per_rank={normalized}",
            flush=True,
        )
    return normalized


class DistributedAlignedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset_size: int,
        batch_sizes: List[int],
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if len(batch_sizes) != world_size:
            raise ValueError("len(batch_sizes) must equal world_size")
        if min(batch_sizes) <= 0:
            raise ValueError(f"all per-rank batch sizes must be > 0, got {batch_sizes}")
        self.dataset_size = int(dataset_size)
        self.batch_sizes = [int(v) for v in batch_sizes]
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.global_batch = int(sum(self.batch_sizes))
        self.num_steps = int(math.ceil(self.dataset_size / float(self.global_batch)))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        indices = list(range(self.dataset_size))
        if self.shuffle:
            g.shuffle(indices)
        needed = self.num_steps * self.global_batch
        if needed > len(indices):
            pad = [indices[i % len(indices)] for i in range(needed - len(indices))]
            indices.extend(pad)
        local_bs = self.batch_sizes[self.rank]
        local_off = int(sum(self.batch_sizes[: self.rank]))
        for step in range(self.num_steps):
            base = step * self.global_batch
            chunk = indices[base + local_off : base + local_off + local_bs]
            if len(chunk) != local_bs:
                raise RuntimeError("internal sampler shape mismatch")
            yield chunk


def _has_trainable_params(module):
    return any(p.requires_grad for p in module.parameters())


def _set_model_mode(model: DDP, is_train: bool):
    module = model.module if isinstance(model, DDP) else model
    if is_train:
        model.train()
        # Keep frozen pretrained pose heads deterministic. Their gradients still
        # flow to trainable shared features, but dropout/BN state does not drift.
        module.encodeA.eval()
        module.trans_head.eval()
        module.rot_head.eval()
        module.mask_decoder.train()
        module.geom_head.train()
        module.pose_head.train()
        if _has_trainable_params(module.encodeAB):
            module.encodeAB.train()
        else:
            module.encodeAB.eval()
        if _has_trainable_params(module.encodeAB_adapters):
            module.encodeAB_adapters.train()
        else:
            module.encodeAB_adapters.eval()
    else:
        model.eval()


def _find_latest_train_ckpt(out_dir: str):
    if not os.path.isdir(out_dir):
        return None
    cands = []
    for name in os.listdir(out_dir):
        if name.startswith("train_epoch_") and name.endswith(".pth"):
            cands.append(os.path.join(out_dir, name))
    if not cands:
        latest = os.path.join(out_dir, "train_latest.pth")
        if os.path.exists(latest):
            return latest
        return None
    cands.sort(key=os.path.getmtime)
    return cands[-1]


def _find_latest_train_ckpt_for_vm_weight(out_dir: str, vm_weight_tag: str):
    if not os.path.isdir(out_dir):
        return None
    cands = []
    for name in os.listdir(out_dir):
        if name.startswith("train_epoch_") and name.endswith(".pth") and (f"_{vm_weight_tag}" in name):
            cands.append(os.path.join(out_dir, name))
    if not cands:
        latest = os.path.join(out_dir, f"train_latest_{vm_weight_tag}.pth")
        if os.path.exists(latest):
            return latest
        # Backward compatibility: allow plain latest if vm tag does not exist yet.
        legacy_latest = os.path.join(out_dir, "train_latest.pth")
        if os.path.exists(legacy_latest):
            return legacy_latest
        return None
    cands.sort(key=os.path.getmtime)
    return cands[-1]


def _load_training_ckpt_if_available(model, ckpt_path: str):
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        return 0
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid checkpoint format: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if _is_main_process():
        print(f"[resume] loaded ckpt: {ckpt_path}")
        print(f"[resume] missing keys: {len(missing)}")
        print(f"[resume] unexpected keys: {len(unexpected)}")
    return int(ckpt.get("epoch", 0))


def _enable_trainable_params(model: RefineNet, train_pose_backbone: bool = False, train_adapters: bool = False):
    for _, p in model.named_parameters():
        p.requires_grad = False

    for _, p in model.mask_decoder.named_parameters():
        p.requires_grad = True
    for _, p in model.geom_head.named_parameters():
        p.requires_grad = True
    for _, p in model.pose_head.named_parameters():
        p.requires_grad = True

    if train_pose_backbone:
        for _, p in model.encodeAB[3].named_parameters():
            p.requires_grad = True
        for _, p in model.encodeAB[4].named_parameters():
            p.requires_grad = True
    if train_adapters:
        for _, p in model.encodeAB_adapters.named_parameters():
            p.requires_grad = True

    if getattr(model, "enable_ref_coord", False):
        for _, p in model.coord_head.named_parameters():
            p.requires_grad = True
        for _, p in model.coord_conf_head.named_parameters():
            p.requires_grad = True


def _build_optimizer(model: RefineNet, base_lr: float, weight_decay: float):
    decoder_params = []
    adapter_params = []
    block_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encodeAB_adapters."):
            adapter_params.append(p)
        elif name.startswith("encodeAB.3.") or name.startswith("encodeAB.4."):
            block_params.append(p)
        elif name.startswith("coord_head.") or name.startswith("coord_conf_head."):
            decoder_params.append(p)
        else:
            decoder_params.append(p)
    groups = [
        {"params": decoder_params, "lr": float(base_lr), "name": "decoder_heads"},
        {"params": adapter_params, "lr": float(base_lr) * 0.5, "name": "adapters"},
        {"params": block_params, "lr": float(base_lr) * 0.1, "name": "encodeAB_last2"},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    optimizer = torch.optim.AdamW(groups, lr=float(base_lr), weight_decay=float(weight_decay))
    return optimizer, groups


def estimate_mesh_diameter_from_tensors(mesh_tensors, fallback: float = 1.0) -> float:
    pos = mesh_tensors.get("pos", None)
    if pos is None:
        return float(fallback)
    if (not torch.is_tensor(pos)) or pos.numel() < 3:
        return float(fallback)
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    diameter = torch.linalg.norm(extent).item()
    if (not np.isfinite(diameter)) or diameter <= 1e-8:
        return float(fallback)
    return float(diameter)


def render_crop_pair(
    mesh_tensors,
    pose,
    rgb,
    obs_mask,
    depth,
    K,
    cfg,
    dataset,
    glctx,
    tf_to_crops_override=None,
    mesh_diameter_override=None,
):
    H, W = obs_mask.shape[:2]
    pose_t = torch.as_tensor(pose[None], dtype=torch.float, device="cuda")
    mesh_diameter = (
        float(mesh_diameter_override)
        if mesh_diameter_override is not None
        else estimate_mesh_diameter_from_tensors(mesh_tensors, fallback=1.0)
    )
    if tf_to_crops_override is None:
        tf_to_crops = compute_crop_window_tf_batch(
            pts=mesh_tensors["pos"].data.cpu().numpy().reshape(-1, 3),
            H=H,
            W=W,
            poses=pose_t,
            K=K,
            crop_ratio=float(cfg["crop_ratio"]),
            out_size=(cfg["input_resize"][1], cfg["input_resize"][0]),
            method="box_3d",
            mesh_diameter=mesh_diameter,
        )
        tf_to_crops = torch.as_tensor(tf_to_crops, device="cuda", dtype=torch.float)
    else:
        tf_to_crops = tf_to_crops_override
    bbox2d_crop = torch.as_tensor(
        np.array([0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]).reshape(2, 2),
        device="cuda",
        dtype=torch.float,
    )
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1, 4)
    extra = {}
    rgb_r, depth_r, _ = nvdiffrast_render(
        K=K,
        H=H,
        W=W,
        ob_in_cams=pose_t,
        context="cuda",
        get_normal=False,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        output_size=cfg["input_resize"],
        bbox2d=bbox2d_ori,
        use_light=True,
        extra=extra,
    )
    rgb_r = rgb_r.permute(0, 3, 1, 2) * 255.0
    xyz_r = extra["xyz_map"].permute(0, 3, 1, 2)
    rgb_obs = torch.as_tensor(rgb, dtype=torch.float, device="cuda").permute(2, 0, 1)[None]
    depth_obs = torch.as_tensor(depth, dtype=torch.float, device="cuda")[None]
    xyz_obs = depth2xyzmap_batch(
        depth_obs,
        torch.as_tensor(K, dtype=torch.float, device="cuda")[None],
        zfar=np.inf,
    ).permute(0, 3, 1, 2)
    mask_obs = torch.as_tensor(obs_mask.astype(np.float32), dtype=torch.float, device="cuda")[None, None]
    rgb_obs = kornia.geometry.transform.warp_perspective(
        rgb_obs, tf_to_crops, dsize=cfg["input_resize"], mode="bilinear", align_corners=False
    )
    depth_obs = kornia.geometry.transform.warp_perspective(
        depth_obs[:, None], tf_to_crops, dsize=cfg["input_resize"], mode="nearest", align_corners=False
    )
    xyz_obs = kornia.geometry.transform.warp_perspective(
        xyz_obs, tf_to_crops, dsize=cfg["input_resize"], mode="nearest", align_corners=False
    )
    mask_obs = kornia.geometry.transform.warp_perspective(
        mask_obs, tf_to_crops, dsize=cfg["input_resize"], mode="nearest", align_corners=False
    )
    mask_obs = (mask_obs > 0.5).float()
    rgb_obs = rgb_obs * mask_obs
    xyz_obs = xyz_obs * mask_obs
    depth_obs = depth_obs * mask_obs
    pose_data = BatchPoseData(
        rgbAs=rgb_r,
        rgbBs=rgb_obs,
        depthAs=depth_r[..., None].permute(0, 3, 1, 2),
        depthBs=depth_obs,
        normalAs=None,
        normalBs=None,
        poseA=pose_t,
        poseB=None,
        xyz_mapAs=xyz_r,
        xyz_mapBs=xyz_obs,
        tf_to_crops=tf_to_crops,
        Ks=torch.as_tensor(K, device="cuda", dtype=torch.float).reshape(1, 3, 3),
        mesh_diameters=torch.full((1,), float(mesh_diameter), dtype=torch.float, device="cuda"),
    )
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)
    return pose_data


def _distance_transform_err(mask_a, mask_b):
    ma = (mask_a[0, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    mb = (mask_b[0, 0].detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    inv_a = (1 - ma).astype(np.uint8)
    inv_b = (1 - mb).astype(np.uint8)
    da = cv2.distanceTransform(inv_a, distanceType=cv2.DIST_L2, maskSize=3)
    db = cv2.distanceTransform(inv_b, distanceType=cv2.DIST_L2, maskSize=3)
    norm = float(max(ma.shape[0], ma.shape[1], 1))
    e = np.abs(da - db).astype(np.float32) / norm
    return torch.as_tensor(e, dtype=torch.float, device=mask_a.device)[None, None]


def _build_soft_validity_target(pd_recon, pd_gt, alpha: float, gamma: float, delta: float, xyz_normalized: bool):
    e_xyz = torch.linalg.norm(pd_recon.xyz_mapAs - pd_gt.xyz_mapAs, dim=1, keepdim=True)
    if not xyz_normalized:
        mesh_d = pd_recon.mesh_diameters.reshape(-1, 1, 1, 1).clamp(min=1e-4)
        e_xyz = e_xyz / mesh_d

    rgb_recon = pd_recon.rgbAs.clamp(0.0, 1.0)
    rgb_gt = pd_gt.rgbAs.clamp(0.0, 1.0)
    e_rgb = torch.mean(torch.abs(rgb_recon - rgb_gt), dim=1, keepdim=True)

    sil_recon = (pd_recon.depthAs > 1e-6).float()
    sil_gt = (pd_gt.depthAs > 1e-6).float()
    e_sil = _distance_transform_err(sil_recon, sil_gt)

    logits = -float(alpha) * e_xyz - float(gamma) * e_sil - float(delta) * e_rgb
    w_star = torch.exp(logits).clamp(0.0, 1.0)
    # Common background has zero residual and would otherwise become target=1.
    # Restrict soft validity supervision to pixels touched by either silhouette.
    w_star = w_star * torch.maximum(sil_recon, sil_gt)
    return w_star, sil_gt


def _pose_err(pred, gt):
    return F.l1_loss(pred["trans"], gt["trans"]) + F.l1_loss(pred["rot"], gt["rot"])


def _load_mesh_keep_visual(mesh_path: str):
    """
    Load mesh while preserving texture/visual metadata as much as possible.
    `force='mesh'` frequently strips or degrades visual info for OBJ+MTL.
    """
    m = trimesh.load(mesh_path, process=False)
    if isinstance(m, trimesh.Scene):
        geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError(f"no trimesh geometry in scene: {mesh_path}")
        # Prefer single-geometry path to retain original visual/material.
        if len(geoms) == 1:
            m = geoms[0]
        else:
            # Fallback for multi-geometry assets.
            m = trimesh.util.concatenate(geoms)
    if not isinstance(m, trimesh.Trimesh):
        # Last resort conversion.
        m = trimesh.load(mesh_path, force="mesh", process=False)
    return m


def _load_deferred_sample_io(sample: Dict[str, Any]) -> Dict[str, Any]:
    if "rgb" in sample and "mask" in sample and "depth" in sample and "K" in sample:
        return sample
    required = ("rgb_path", "mask_path", "depth_path", "K_path")
    if not all(k in sample for k in required):
        return sample

    mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {sample['mask_path']}")
    min_mask_pixels = int(sample.get("min_mask_pixels", 1))
    valid_pixels = int(np.count_nonzero(mask > 0))
    if valid_pixels < min_mask_pixels:
        raise RuntimeError(f"mask has too few valid pixels ({valid_pixels}) in {sample['mask_path']}")

    rgb = cv2.imread(sample["rgb_path"], cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read rgb: {sample['rgb_path']}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
    mask_bin = (mask > 0).astype(np.uint8)

    depth_path = sample["depth_path"]
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"failed to read depth: {depth_path}")
        depth = depth.astype(np.float32)
    if depth.size == 0:
        raise RuntimeError(f"empty depth map: {depth_path}")
    if np.nanmax(depth) > 100.0:
        depth = depth / max(float(sample.get("depth_scale", 1000.0)), 1.0)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    if depth.shape[:2] != mask_bin.shape[:2]:
        raise RuntimeError(
            f"depth/mask shape mismatch: depth={depth.shape} mask={mask_bin.shape} file={depth_path}"
        )

    loaded = dict(sample)
    loaded["rgb"] = rgb
    loaded["mask"] = mask_bin
    loaded["depth"] = depth.astype(np.float32)
    loaded["K"] = np.loadtxt(sample["K_path"], dtype=np.float32).reshape(3, 3)
    return loaded


@dataclass
class LossOutput:
    loss: torch.Tensor
    geom_validity_loss: float
    pose_utility_loss: float
    aux_bce_loss: float
    pred_geom_mean: float
    pred_pose_mean: float
    high_weight_area_ratio: float


def _concat_pose_data(pose_data_list):
    return BatchPoseData(
        rgbAs=torch.cat([pd.rgbAs for pd in pose_data_list], dim=0),
        rgbBs=torch.cat([pd.rgbBs for pd in pose_data_list], dim=0),
        depthAs=torch.cat([pd.depthAs for pd in pose_data_list], dim=0),
        depthBs=torch.cat([pd.depthBs for pd in pose_data_list], dim=0),
        normalAs=None,
        normalBs=None,
        poseA=torch.cat([pd.poseA for pd in pose_data_list], dim=0),
        poseB=None,
        xyz_mapAs=torch.cat([pd.xyz_mapAs for pd in pose_data_list], dim=0),
        xyz_mapBs=torch.cat([pd.xyz_mapBs for pd in pose_data_list], dim=0),
        tf_to_crops=torch.cat([pd.tf_to_crops for pd in pose_data_list], dim=0),
        Ks=torch.cat([pd.Ks for pd in pose_data_list], dim=0),
        mesh_diameters=torch.cat([pd.mesh_diameters for pd in pose_data_list], dim=0),
    )


def _compute_batch_loss(
    samples,
    model,
    cfg,
    dataset,
    glctx,
    iterations: int,
    alpha: float,
    gamma: float,
    delta: float,
    vm_weight: float,
    aux_bce_weight: float,
    pose_utility_scale: float,
):
    sample_cache = []
    for sample in samples:
        sample = _load_deferred_sample_io(sample)
        gt_mesh = _load_mesh_keep_visual(sample["gt_mesh_path"])
        gt_tensors = make_mesh_tensors(gt_mesh)
        recon_mesh_paths = sample.get("recon_mesh_paths", None)
        if not recon_mesh_paths:
            legacy_path = sample.get("recon_mesh_path", None)
            if legacy_path:
                recon_mesh_paths = [legacy_path]
            else:
                recon_mesh_paths = [sample["gt_mesh_path"]]
        recon_entries = []
        for p in recon_mesh_paths:
            recon_mesh = _load_mesh_keep_visual(p)
            recon_tensors = make_mesh_tensors(recon_mesh)
            recon_entries.append(
                {
                    "recon_tensors": recon_tensors,
                    "recon_diameter": estimate_mesh_diameter_from_tensors(recon_tensors, fallback=1.0),
                }
            )
        pose = sample["init_pose"].astype(np.float32)
        sample_cache.append(
            {
                "recon_entries": recon_entries,
                "gt_tensors": gt_tensors,
                "rgb": sample["rgb"],
                "obs_mask": sample["mask"].astype(np.uint8),
                "depth": sample["depth"].astype(np.float32),
                "K": sample["K"],
                "final_pose": torch.as_tensor(pose[None], dtype=torch.float, device="cuda"),
                "prev_mask": None,
            }
        )

    bs = len(sample_cache)
    total_loss = torch.zeros([], device="cuda")
    geom_losses = []
    pose_losses = []
    aux_losses = []
    geom_means = []
    pose_means = []
    high_ratios = []

    for _ in range(iterations):
        pd_recon_list = []
        pd_gt_list = []
        A_list, B_list, A_gt_list = [], [], []
        for s in sample_cache:
            for recon_entry in s["recon_entries"]:
                pd_recon = render_crop_pair(
                    recon_entry["recon_tensors"],
                    s["final_pose"][0].data.cpu().numpy(),
                    s["rgb"],
                    s["obs_mask"],
                    s["depth"],
                    s["K"],
                    cfg,
                    dataset,
                    glctx,
                    mesh_diameter_override=recon_entry["recon_diameter"],
                )
                pd_gt = render_crop_pair(
                    s["gt_tensors"],
                    s["final_pose"][0].data.cpu().numpy(),
                    s["rgb"],
                    s["obs_mask"],
                    s["depth"],
                    s["K"],
                    cfg,
                    dataset,
                    glctx,
                    tf_to_crops_override=pd_recon.tf_to_crops,
                    mesh_diameter_override=recon_entry["recon_diameter"],
                )
                A_i = torch.cat([pd_recon.rgbAs, pd_recon.xyz_mapAs], dim=1).float()
                B_i = torch.cat([pd_recon.rgbBs, pd_recon.xyz_mapBs], dim=1).float()
                A_gt_i = torch.cat([pd_gt.rgbAs, pd_gt.xyz_mapAs], dim=1).float()
                prev_mask = s["prev_mask"]
                if prev_mask is not None:
                    if prev_mask.shape[-2:] != A_i.shape[-2:]:
                        prev_mask = F.interpolate(prev_mask, size=A_i.shape[-2:], mode="bilinear", align_corners=False)
                    prev_mask = prev_mask.clamp(0.0, 1.0)
                    gate = _validity_mask_gate(prev_mask, vm_weight=vm_weight)
                    A_i = A_i * gate
                    A_gt_i = A_gt_i * gate
                pd_recon_list.append(pd_recon)
                pd_gt_list.append(pd_gt)
                A_list.append(A_i)
                B_list.append(B_i)
                A_gt_list.append(A_gt_i)

        A = torch.cat(A_list, dim=0)
        B = torch.cat(B_list, dim=0)
        A_gt = torch.cat(A_gt_list, dim=0)
        pd_recon = _concat_pose_data(pd_recon_list)
        pd_gt = _concat_pose_data(pd_gt_list)

        out = model(A, B)
        out_gt = model(A_gt, B)

        pred_geom = out["geom_validity"]
        pred_pose = out["pose_utility"]
        pred_geom_logits = out["geom_logits"]
        w_final = out["validity_mask"]
        w_star, gt_mask = _build_soft_validity_target(
            pd_recon,
            pd_gt,
            alpha=alpha,
            gamma=gamma,
            delta=delta,
            xyz_normalized=bool(cfg.get("normalize_xyz", False)),
        )

        # Pose utility target from "masking this region hurts pose estimate" surrogate.
        with torch.no_grad():
            base_pose_err = _pose_err(out, out_gt)
            A_drop = A * (1.0 - w_star)
            out_drop = model(A_drop, B)
            drop_pose_err = _pose_err(out_drop, out_gt)
            utility_scale = torch.sigmoid((drop_pose_err - base_pose_err) * float(pose_utility_scale))
            pose_target = (w_star * utility_scale).detach().clamp(0.0, 1.0)

        geom_validity_loss = F.smooth_l1_loss(pred_geom, w_star)
        pose_utility_loss = F.smooth_l1_loss(pred_pose, pose_target)
        aux_bce_loss = F.binary_cross_entropy_with_logits(pred_geom_logits.float(), gt_mask.float())

        obs_rgb = pd_recon.rgbBs
        residual_recon = pd_recon.rgbAs - obs_rgb
        residual_gt = pd_gt.rgbAs - obs_rgb
        loss_same_view = F.l1_loss(w_final * residual_recon, gt_mask * residual_gt)
        loss_pose_consistency = _pose_err(out, out_gt)
        loss_reg = torch.mean(w_final)

        loss = (
            geom_validity_loss
            + pose_utility_loss
            + float(aux_bce_weight) * aux_bce_loss
            + 0.25 * loss_same_view
            + loss_pose_consistency
            + 0.05 * loss_reg
        )

        total_loss = total_loss + loss
        w_final_detached = w_final.detach()
        offset = 0
        for bi in range(bs):
            k = len(sample_cache[bi]["recon_entries"])
            if k <= 0:
                sample_cache[bi]["prev_mask"] = None
                continue
            sample_cache[bi]["prev_mask"] = w_final_detached[offset : offset + k].mean(dim=0, keepdim=True)
            offset += k
        geom_losses.append(float(geom_validity_loss.item()))
        pose_losses.append(float(pose_utility_loss.item()))
        aux_losses.append(float(aux_bce_loss.item()))
        geom_means.append(float(pred_geom.mean().item()))
        pose_means.append(float(pred_pose.mean().item()))
        high_ratios.append(float((w_star > 0.7).float().mean().item()))

        if cfg["rot_rep"] == "axis_angle":
            rot_delta = so3_exp_map(torch.tanh(out["rot"]) * cfg.get("rot_normalizer", 1.0)).permute(0, 2, 1)
        else:
            rot_delta = rotation_6d_to_matrix(out["rot"]).permute(0, 2, 1)
        trans_delta = out["trans"]
        final_pose_all = torch.cat([s["final_pose"] for s in sample_cache], dim=0)
        expanded_final_pose = []
        for bi in range(bs):
            k = max(1, len(sample_cache[bi]["recon_entries"]))
            expanded_final_pose.append(final_pose_all[bi : bi + 1].repeat(k, 1, 1))
        expanded_final_pose = torch.cat(expanded_final_pose, dim=0)
        expanded_final_pose = egocentric_delta_pose_to_pose(
            expanded_final_pose, trans_delta=trans_delta, rot_mat_delta=rot_delta
        )
        offset = 0
        for bi in range(bs):
            k = max(1, len(sample_cache[bi]["recon_entries"]))
            sample_cache[bi]["final_pose"] = expanded_final_pose[offset : offset + k].mean(dim=0, keepdim=True)
            offset += k

    return LossOutput(
        loss=total_loss,
        geom_validity_loss=float(np.mean(geom_losses)),
        pose_utility_loss=float(np.mean(pose_losses)),
        aux_bce_loss=float(np.mean(aux_losses)),
        pred_geom_mean=float(np.mean(geom_means)),
        pred_pose_mean=float(np.mean(pose_means)),
        high_weight_area_ratio=float(np.mean(high_ratios)),
    )


def _init_phase_meter():
    return {
        "loss": 0.0,
        "geom_validity_loss": 0.0,
        "pose_utility_loss": 0.0,
        "aux_bce_loss": 0.0,
        "pred_geom_mean": 0.0,
        "pred_pose_mean": 0.0,
        "high_weight_area_ratio": 0.0,
        "steps": 0.0,
    }


def _add_meter(meter, out: LossOutput):
    meter["loss"] += float(out.loss.item())
    meter["geom_validity_loss"] += float(out.geom_validity_loss)
    meter["pose_utility_loss"] += float(out.pose_utility_loss)
    meter["aux_bce_loss"] += float(out.aux_bce_loss)
    meter["pred_geom_mean"] += float(out.pred_geom_mean)
    meter["pred_pose_mean"] += float(out.pred_pose_mean)
    meter["high_weight_area_ratio"] += float(out.high_weight_area_ratio)
    meter["steps"] += 1.0


def _finalize_meter_local(meter):
    steps = max(1.0, meter["steps"])
    out = {}
    for k, v in meter.items():
        if k == "steps":
            out[k] = float(v)
        else:
            out[k] = float(v) / steps
    return out


def _reduce_meter_global(local_mean, local_steps: float, device):
    global_steps = _reduce_sum_scalar(local_steps, device=device)
    if global_steps <= 0:
        return {k: 0.0 for k in local_mean}, 0
    global_mean = {}
    for k, v in local_mean.items():
        if k == "steps":
            continue
        global_sum = _reduce_sum_scalar(float(v) * float(local_steps), device=device)
        global_mean[k] = global_sum / global_steps
    return global_mean, int(global_steps)


def _run_one_epoch_ddp(
    loader,
    model,
    optimizer,
    scaler,
    cfg,
    dataset,
    glctx,
    args,
    is_train: bool,
    epoch_idx: int,
    device: torch.device,
):
    _set_model_mode(model, is_train=is_train)
    phase = "train" if is_train else "val"
    meter = _init_phase_meter()
    use_tqdm = (tqdm is not None) and (not args.disable_tqdm) and _is_main_process()
    progress = tqdm(enumerate(loader), total=len(loader), desc=f"epoch {epoch_idx+1}/{args.epochs} {phase}") if use_tqdm else enumerate(loader)

    if _is_main_process():
        print(f"[{phase}] epoch {epoch_idx+1}/{args.epochs} start, loader_len={len(loader)}")

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, batch_samples in progress:
            if i == 0 and _is_main_process():
                print(f"[{phase}] first batch fetched, batch_size={len(batch_samples)}")
            merged = _compute_batch_loss(
                samples=batch_samples,
                model=model,
                cfg=cfg,
                dataset=dataset,
                glctx=glctx,
                iterations=args.iterations,
                alpha=args.soft_alpha,
                gamma=args.soft_gamma,
                delta=args.soft_delta,
                vm_weight=args.vm_weight,
                aux_bce_weight=args.aux_bce_weight,
                pose_utility_scale=args.pose_utility_scale,
            )
            total_loss = merged.loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            _add_meter(meter, merged)

            if use_tqdm:
                progress.set_postfix(
                    loss=f"{float(total_loss.item()):.6f}",
                    geom=f"{merged.geom_validity_loss:.4f}",
                    pose=f"{merged.pose_utility_loss:.4f}",
                    bce=f"{merged.aux_bce_loss:.4f}",
                )
            if (i + 1) % args.log_every == 0 and _is_main_process():
                local_avg = _finalize_meter_local(meter)
                msg = (
                    f"[epoch {epoch_idx+1}/{args.epochs}] {phase} step {i+1}/{len(loader)} "
                    f"loss={local_avg['loss']:.6f} "
                    f"geom_validity_loss={local_avg['geom_validity_loss']:.6f} "
                    f"pose_utility_loss={local_avg['pose_utility_loss']:.6f} "
                    f"aux_bce_loss={local_avg['aux_bce_loss']:.6f} "
                    f"pred_geom_mean={local_avg['pred_geom_mean']:.6f} "
                    f"pred_pose_mean={local_avg['pred_pose_mean']:.6f} "
                    f"high_weight_area_ratio={local_avg['high_weight_area_ratio']:.6f}"
                )
                if use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)

    local_mean = _finalize_meter_local(meter)
    global_mean, global_steps = _reduce_meter_global(local_mean, meter["steps"], device=device)
    return global_mean, global_steps


def _save_history_json(history, out_path):
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _exit_log_path(args):
    raw = str(getattr(args, "exit_log_jsonl_path", "") or "").strip()
    if raw:
        return raw
    return os.path.join(args.out_dir, "train_ddp_exit_log.jsonl")


def _log_exit_event(args, reason: str, detail: str = "", epoch: Optional[int] = None, exc_text: str = ""):
    try:
        rank, world_size = _ddp_info()
    except Exception:
        rank, world_size = int(getattr(args, "rank", 0)), int(getattr(args, "world_size", 1))
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "pid": int(os.getpid()),
        "rank": int(rank),
        "world_size": int(world_size),
        "local_rank": int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0))),
        "reason": str(reason),
        "detail": str(detail),
        "epoch": None if epoch is None else int(epoch),
        "out_dir": str(getattr(args, "out_dir", "")),
        "vm_weight": float(getattr(args, "vm_weight", 0.75)),
    }
    if exc_text:
        payload["exception"] = str(exc_text)

    msg = (
        f"[exit] rank={payload['rank']}/{payload['world_size']} "
        f"local_rank={payload['local_rank']} reason={payload['reason']}"
    )
    if detail:
        msg += f" detail={detail}"
    print(msg, flush=True)

    path = _exit_log_path(args)
    try:
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[exit] failed to write exit log {path}: {e}", flush=True)


def _make_sam3d_part_dataset(**kwargs):
    try:
        supported = set(inspect.signature(Sam3DPartTrainDataset.__init__).parameters.keys())
        supported.discard("self")
    except Exception:
        supported = set(kwargs.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print(
            "[dataset][compat] Sam3DPartTrainDataset does not support args; "
            f"dropped={dropped}",
            flush=True,
        )
    return Sam3DPartTrainDataset(**filtered)


def _make_prebuild_dataset(
    args,
    root,
    split_name: str,
    rebuild_records_index: bool = False,
    log_prefix: str = "prebuild",
):
    if _is_main_process():
        print(
            f"[{log_prefix}] initializing dataset split={split_name} root={root} "
            f"recon_model={args.recon_model}",
            flush=True,
        )
    ds = _make_sam3d_part_dataset(
        dataset_root=root,
        cache_root=args.recon_cache_root,
        cache_split=split_name,
        sam3d_project_root=args.sam3d_project_root,
        sam3d_notebook_root=args.sam3d_notebook_root,
        sam3d_config_path=args.sam3d_config_path,
        min_mask_pixels=args.min_mask_pixels,
        seed=args.seed,
        rebuild_recon=args.force_rebuild_recon,
        allow_recon_write=True,
        strict_mode=True,
        fallback_to_gt_mesh_on_recon_fail=False,
        depth_scale=args.depth_scale,
        recon_min_views_per_part=args.recon_min_views_per_part,
        recon_max_views_per_part=args.recon_max_views_per_part,
        recon_rot_threshold_deg=args.recon_rot_threshold_deg,
        recon_trans_threshold=args.recon_trans_threshold,
        force_resample_recon=args.force_resample_recon,
        use_real_depth_pointmap=args.use_real_depth_pointmap,
        recon_model=args.recon_model,
        recon_view_density_scale=args.recon_view_density_scale,
        hunyuan_model_path=args.hunyuan_model_path,
        hunyuan_subfolder=args.hunyuan_subfolder,
        hunyuan_num_inference_steps=args.hunyuan_num_inference_steps,
        hunyuan_octree_resolution=args.hunyuan_octree_resolution,
        hunyuan_guidance_scale=args.hunyuan_guidance_scale,
        instantmesh_root=args.instantmesh_root,
        instantmesh_config_path=args.instantmesh_config_path,
        instantmesh_diffusion_model=args.instantmesh_diffusion_model,
        instantmesh_dino_model=args.instantmesh_dino_model, 
        instantmesh_unet_path=args.instantmesh_unet_path,
        instantmesh_model_path=args.instantmesh_model_path,
        instantmesh_diffusion_steps=args.instantmesh_diffusion_steps,
        instantmesh_scale=args.instantmesh_scale,
        instantmesh_view=args.instantmesh_view,
        instantmesh_foreground_ratio=args.instantmesh_foreground_ratio,
        instantmesh_export_texmap=args.instantmesh_export_texmap,
        rebuild_records_index=rebuild_records_index,
    )
    if _is_main_process():
        print(
            f"[{log_prefix}] dataset ready split={split_name} "
            f"records={len(ds.records)} parts={len(ds.part_keys)} "
            f"recon_model={args.recon_model}",
            flush=True,
        )
    return ds


def _run_dataset_index_only(args):
    train_root, val_root = _resolve_train_val_roots(args.dataset_root, args.train_subdir, args.val_subdir)
    split = str(args.dataset_index_split).lower()
    print(
        "[dataset-index] CPU-only index build requested; "
        "this does not initialize DDP/CUDA or load reconstruction models.",
        flush=True,
    )
    print(
        f"[dataset-index] cache_root={args.recon_cache_root} split={split} "
        f"force_rebuild={args.force_rebuild_dataset_index}",
        flush=True,
    )
    if split in ("both", "train"):
        _make_prebuild_dataset(
            args,
            train_root,
            "train",
            rebuild_records_index=args.force_rebuild_dataset_index,
            log_prefix="dataset-index",
        )
    if split in ("both", "val"):
        _make_prebuild_dataset(
            args,
            val_root,
            "val",
            rebuild_records_index=args.force_rebuild_dataset_index,
            log_prefix="dataset-index",
        )
    print(f"[dataset-index] ready for split={split}.", flush=True)
    return "dataset_index_ready"


def _prebuild_part_worker(worker_idx, args, root, split_name, part_keys, local_rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
    random.seed(int(args.seed) + int(worker_idx))
    np.random.seed(int(args.seed) + int(worker_idx))
    torch.manual_seed(int(args.seed) + int(worker_idx))
    torch.cuda.manual_seed_all(int(args.seed) + int(worker_idx))

    ds = _make_prebuild_dataset(args, root, split_name)
    total = len(part_keys)
    print(
        f"[prebuild-worker] split={split_name} worker={worker_idx} "
        f"pid={os.getpid()} local_rank={local_rank} parts={total}",
        flush=True,
    )
    for i, part_key in enumerate(part_keys):
        ds.ensure_recon_cache_for_part(tuple(part_key))
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(
                f"[prebuild-worker] split={split_name} worker={worker_idx} "
                f"done={i + 1}/{total}",
                flush=True,
            )


def _run_prebuild_phase(args, train_root, val_root, rank, world_size, local_rank):
    if not args.prebuild_recon:
        return

    recon_workers_per_rank = _parse_per_rank_value(
        args.recon_num_workers_per_rank,
        args.recon_num_workers,
        world_size,
    )
    recon_workers_per_rank = _cap_gpu_recon_workers_per_rank(
        recon_workers_per_rank,
        args.recon_model,
    )
    local_recon_workers = max(1, int(recon_workers_per_rank[rank]))

    def _run_one(root, name):
        ds = _make_prebuild_dataset(args, root, name)
        part_keys = ds.part_keys[rank::world_size]
        if _is_main_process():
            print(
                f"[prebuild] split={name} samples={len(ds)} parts={len(ds.part_keys)} "
                f"rank={rank}/{world_size} force_rebuild={args.force_rebuild_recon} "
                f"force_resample={args.force_resample_recon} "
                f"recon_workers_per_rank={recon_workers_per_rank}"
            )
        if local_recon_workers <= 1 or len(part_keys) <= 1:
            for part_key in part_keys:
                ds.ensure_recon_cache_for_part(part_key)
        else:
            chunks = [part_keys[i::local_recon_workers] for i in range(local_recon_workers)]
            chunks = [c for c in chunks if c]
            ctx = mp.get_context("spawn")
            procs = []
            for worker_idx, chunk in enumerate(chunks):
                p = ctx.Process(
                    target=_prebuild_part_worker,
                    args=(worker_idx, args, root, name, chunk, local_rank),
                    daemon=False,
                )
                p.start()
                procs.append(p)
            failed = []
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    failed.append((p.pid, p.exitcode))
            if failed:
                raise RuntimeError(f"prebuild workers failed for split={name}: {failed}")
        _sync_barrier()

    prebuild_split = str(args.prebuild_split).lower()
    if prebuild_split in ("both", "train"):
        _run_one(train_root, "train")
    if prebuild_split in ("both", "val"):
        _run_one(val_root, "val")
    if _is_main_process():
        print(f"[prebuild] reconstruction cache ready for split={prebuild_split}.")


def train_ddp(args):
    if getattr(args, "build_dataset_index_only", False):
        return _run_dataset_index_only(args)

    args.vm_weight = _safe_vm_weight(getattr(args, "vm_weight", 0.75))
    vm_weight_tag = _format_vm_weight_tag(args.vm_weight)
    if not getattr(args, "disable_vm_weight_naming", False):
        out_dir_name = Path(str(args.out_dir)).name
        if vm_weight_tag not in out_dir_name:
            args.out_dir = str(Path(str(args.out_dir)) / vm_weight_tag)

    rank, world_size, local_rank = _setup_distributed(args)
    device = torch.device(f"cuda:{local_rank}")
    _seed_everything(args.seed, rank)

    args.sam3d_config_path = _resolve_sam3d_config_path(args.sam3d_config_path, args.sam3d_project_root)
    train_root, val_root = _resolve_train_val_roots(args.dataset_root, args.train_subdir, args.val_subdir)
    cfg_path = _resolve_refiner_config_path(args.config)
    cfg = OmegaConf.load(cfg_path)
    if "use_BN" not in cfg:
        cfg["use_BN"] = False
    if "c_in" not in cfg:
        cfg["c_in"] = 6
    if "crop_ratio" not in cfg:
        cfg["crop_ratio"] = 1.2
    if "input_resize" not in cfg:
        cfg["input_resize"] = [160, 160]
    if "rot_rep" not in cfg:
        cfg["rot_rep"] = "axis_angle"
    if "trans_rep" not in cfg:
        cfg["trans_rep"] = "tracknet"
    if "enable_ref_coord" not in cfg:
        cfg["enable_ref_coord"] = bool(getattr(args, "enable_ref_coord", False))

    per_rank_bs = _parse_per_rank_value(args.batch_size_per_rank, args.batch_size, world_size)
    per_rank_workers = _parse_per_rank_value(args.num_workers_per_rank, args.num_workers, world_size)
    local_bs = per_rank_bs[rank]
    local_workers = per_rank_workers[rank]

    if _is_main_process():
        print(f"[dist] rank={rank} world_size={world_size} local_rank={local_rank}")
        print(f"[recon] model={args.recon_model}")
        if args.recon_model in ("sam3d", "sam3d_tsdf", "sam3d_tsdf_dmesh", "all"):
            print(f"[load] sam3d config: {args.sam3d_config_path}")
        if args.recon_model in ("hunyuan3d", "hunyuan3d_tsdf", "hunyuan3d_tsdf_dmesh", "all"):
            print(
                f"[load] hunyuan model path: {args.hunyuan_model_path} "
                f"subfolder={args.hunyuan_subfolder} "
                f"steps={args.hunyuan_num_inference_steps} "
                f"octree={args.hunyuan_octree_resolution} "
                f"guidance={args.hunyuan_guidance_scale}"
            )
        if args.recon_model in ("instantmesh", "instantmesh_tsdf", "instantmesh_tsdf_dmesh", "all"):
            print(f"[load] instantmesh root: {args.instantmesh_root}")
            print(f"[load] instantmesh config: {args.instantmesh_config_path}")
            print(f"[load] instantmesh diffusion model: {args.instantmesh_diffusion_model}")
            print(f"[load] instantmesh dino model: {args.instantmesh_dino_model}")
            print(f"[load] instantmesh unet: {args.instantmesh_unet_path}")
            print(f"[load] instantmesh model: {args.instantmesh_model_path}")
        print(f"[load] train root: {train_root}")
        print(f"[load] val root: {val_root}")
        print(f"[load] config: {cfg_path}")
        print(f"[data] batch_size_per_rank={per_rank_bs}")
        print(f"[data] num_workers_per_rank={per_rank_workers}")
        print(f"[train] vm_weight={args.vm_weight:.3f} ({vm_weight_tag})")
        print(f"[save] out_dir={args.out_dir}")

    _run_prebuild_phase(
        args,
        train_root=train_root,
        val_root=val_root,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    _sync_barrier()
    if getattr(args, "prebuild_only", False):
        if _is_main_process():
            print("[prebuild] prebuild-only requested; skip training.")
        _cleanup_distributed()
        return

    dataset = PoseRefinePairH5Dataset(cfg=cfg, h5_file="", mode="test")
    model = RefineNet(cfg=cfg, c_in=int(cfg["c_in"])).to(device).train()
    base_ckpt = _resolve_refiner_base_ckpt()
    _load_refiner_backbone_only(model, base_ckpt)

    _enable_trainable_params(
        model,
        train_pose_backbone=args.train_pose_backbone,
        train_adapters=args.train_adapters,
    )
    start_epoch = 0
    if args.auto_resume:
        latest_ckpt = _find_latest_train_ckpt_for_vm_weight(args.out_dir, vm_weight_tag)
        start_epoch = _load_training_ckpt_if_available(model, latest_ckpt)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer, lr_groups = _build_optimizer(ddp_model.module, base_lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if _is_main_process():
        print("[train] optimizer groups:")
        for g in lr_groups:
            print(f"  - {g['name']}: lr={g['lr']} params={len(g['params'])}")

    ds_train = _make_sam3d_part_dataset(
        dataset_root=train_root,
        cache_root=args.recon_cache_root,
        cache_split=args.train_subdir,
        sam3d_project_root=args.sam3d_project_root,
        sam3d_notebook_root=args.sam3d_notebook_root,
        sam3d_config_path=args.sam3d_config_path,
        min_mask_pixels=args.min_mask_pixels,
        seed=args.seed,
        rebuild_recon=False,
        allow_recon_write=False,
        strict_mode=True,
        fallback_to_gt_mesh_on_recon_fail=False,
        depth_scale=args.depth_scale,
        recon_min_views_per_part=args.recon_min_views_per_part,
        recon_max_views_per_part=args.recon_max_views_per_part,
        recon_rot_threshold_deg=args.recon_rot_threshold_deg,
        recon_trans_threshold=args.recon_trans_threshold,
        force_resample_recon=False,
        use_real_depth_pointmap=args.use_real_depth_pointmap,
        recon_model=args.recon_model,
        recon_view_density_scale=args.recon_view_density_scale,
        hunyuan_model_path=args.hunyuan_model_path,
        hunyuan_subfolder=args.hunyuan_subfolder,
        hunyuan_num_inference_steps=args.hunyuan_num_inference_steps,
        hunyuan_octree_resolution=args.hunyuan_octree_resolution,
        hunyuan_guidance_scale=args.hunyuan_guidance_scale,
        instantmesh_root=args.instantmesh_root,
        instantmesh_config_path=args.instantmesh_config_path,
        instantmesh_diffusion_model=args.instantmesh_diffusion_model,
        instantmesh_dino_model=args.instantmesh_dino_model,
        instantmesh_unet_path=args.instantmesh_unet_path,
        instantmesh_model_path=args.instantmesh_model_path,
        instantmesh_diffusion_steps=args.instantmesh_diffusion_steps,
        instantmesh_scale=args.instantmesh_scale,
        instantmesh_view=args.instantmesh_view,
        instantmesh_foreground_ratio=args.instantmesh_foreground_ratio,
        instantmesh_export_texmap=args.instantmesh_export_texmap,
        defer_sample_io=True,
    )
    ds_val = _make_sam3d_part_dataset(
        dataset_root=val_root,
        cache_root=args.recon_cache_root,
        cache_split=args.val_subdir,
        sam3d_project_root=args.sam3d_project_root,
        sam3d_notebook_root=args.sam3d_notebook_root,
        sam3d_config_path=args.sam3d_config_path,
        min_mask_pixels=args.min_mask_pixels,
        seed=args.seed,
        rebuild_recon=False,
        allow_recon_write=False,
        strict_mode=True,
        fallback_to_gt_mesh_on_recon_fail=False,
        depth_scale=args.depth_scale,
        recon_min_views_per_part=args.recon_min_views_per_part,
        recon_max_views_per_part=args.recon_max_views_per_part,
        recon_rot_threshold_deg=args.recon_rot_threshold_deg,
        recon_trans_threshold=args.recon_trans_threshold,
        force_resample_recon=False,
        use_real_depth_pointmap=args.use_real_depth_pointmap,
        recon_model=args.recon_model,
        recon_view_density_scale=args.recon_view_density_scale,
        hunyuan_model_path=args.hunyuan_model_path,
        hunyuan_subfolder=args.hunyuan_subfolder,
        hunyuan_num_inference_steps=args.hunyuan_num_inference_steps,
        hunyuan_octree_resolution=args.hunyuan_octree_resolution,
        hunyuan_guidance_scale=args.hunyuan_guidance_scale,
        instantmesh_root=args.instantmesh_root,
        instantmesh_config_path=args.instantmesh_config_path,
        instantmesh_diffusion_model=args.instantmesh_diffusion_model,
        instantmesh_dino_model=args.instantmesh_dino_model,
        instantmesh_unet_path=args.instantmesh_unet_path,
        instantmesh_model_path=args.instantmesh_model_path,
        instantmesh_diffusion_steps=args.instantmesh_diffusion_steps,
        instantmesh_scale=args.instantmesh_scale,
        instantmesh_view=args.instantmesh_view,
        instantmesh_foreground_ratio=args.instantmesh_foreground_ratio,
        instantmesh_export_texmap=args.instantmesh_export_texmap,
        defer_sample_io=True,
    )

    train_sampler = DistributedAlignedBatchSampler(
        dataset_size=len(ds_train),
        batch_sizes=per_rank_bs,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=args.seed,
    )
    val_sampler = DistributedAlignedBatchSampler(
        dataset_size=len(ds_val),
        batch_sizes=per_rank_bs,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        seed=args.seed + 777,
    )
    loader_train = DataLoader(
        ds_train,
        batch_sampler=train_sampler,
        num_workers=int(local_workers),
        pin_memory=args.pin_memory,
        persistent_workers=(int(local_workers) > 0 and args.persistent_workers),
        collate_fn=collate_list,
    )
    loader_val = DataLoader(
        ds_val,
        batch_sampler=val_sampler,
        num_workers=int(local_workers),
        pin_memory=args.pin_memory,
        persistent_workers=False,
        collate_fn=collate_list,
    )
    if _is_main_process():
        print(
            f"[data] train samples={len(ds_train)} val samples={len(ds_val)} "
            f"local_bs(rank0)={per_rank_bs[0]} local_workers(rank0)={per_rank_workers[0]}"
        )

    glctx = dr.RasterizeCudaContext()
    history = {
        "train": {
            "loss": [],
            "geom_validity_loss": [],
            "pose_utility_loss": [],
            "aux_bce_loss": [],
            "pred_geom_mean": [],
            "pred_pose_mean": [],
            "high_weight_area_ratio": [],
        },
        "val": {
            "loss": [],
            "geom_validity_loss": [],
            "pose_utility_loss": [],
            "aux_bce_loss": [],
            "pred_geom_mean": [],
            "pred_pose_mean": [],
            "high_weight_area_ratio": [],
        },
    }
    best_val = float("inf")
    bad_epochs = 0
    best_ckpt_path = os.path.join(args.out_dir, f"train_best_by_val_{vm_weight_tag}.pth")
    os.makedirs(args.out_dir, exist_ok=True)
    exit_reason = ""
    exit_detail = ""
    last_completed_epoch = int(start_epoch)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_mean, train_steps_global = _run_one_epoch_ddp(
            loader=loader_train,
            model=ddp_model,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            dataset=dataset,
            glctx=glctx,
            args=args,
            is_train=True,
            epoch_idx=epoch,
            device=device,
        )
        val_mean, val_steps_global = _run_one_epoch_ddp(
            loader=loader_val,
            model=ddp_model,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            dataset=dataset,
            glctx=glctx,
            args=args,
            is_train=False,
            epoch_idx=epoch,
            device=device,
        )

        if _is_main_process():
            payload = {
                "model": ddp_model.module.state_dict(),
                "mask_decoder": ddp_model.module.mask_decoder.state_dict(),
                "geom_head": ddp_model.module.geom_head.state_dict(),
                "pose_head": ddp_model.module.pose_head.state_dict(),
                "encodeAB_last2": {
                    "encodeAB.3": ddp_model.module.encodeAB[3].state_dict(),
                    "encodeAB.4": ddp_model.module.encodeAB[4].state_dict(),
                },
                "encodeAB_adapters": ddp_model.module.encodeAB_adapters.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": int(epoch + 1),
                "vm_weight": float(args.vm_weight),
            }
            if getattr(ddp_model.module, "enable_ref_coord", False):
                payload["coord_head"] = ddp_model.module.coord_head.state_dict()
                payload["coord_conf_head"] = ddp_model.module.coord_conf_head.state_dict()
            ckpt_path = os.path.join(args.out_dir, f"train_epoch_{epoch+1:03d}_{vm_weight_tag}.pth")
            latest_path = os.path.join(args.out_dir, f"train_latest_{vm_weight_tag}.pth")
            torch.save(payload, ckpt_path)
            torch.save(payload, latest_path)

            if val_mean["loss"] + args.early_stop_min_delta < best_val:
                best_val = val_mean["loss"]
                bad_epochs = 0
                torch.save(payload, best_ckpt_path)
                print(f"[save] best-by-val updated: {best_ckpt_path} (val_loss={best_val:.6f})")
            else:
                bad_epochs += 1

            for k in history["train"].keys():
                history["train"][k].append(float(train_mean[k]))
                history["val"][k].append(float(val_mean[k]))
            _save_history_json(history, args.loss_history_json_path)

            print(
                f"[epoch-summary] epoch={epoch+1} "
                f"train_loss={train_mean['loss']:.6f} (global_steps={train_steps_global}) "
                f"val_loss={val_mean['loss']:.6f} (global_steps={val_steps_global}) "
                f"train_geom_validity_loss={train_mean['geom_validity_loss']:.6f} "
                f"train_pose_utility_loss={train_mean['pose_utility_loss']:.6f} "
                f"train_aux_bce_loss={train_mean['aux_bce_loss']:.6f} "
                f"train_pred_geom_mean={train_mean['pred_geom_mean']:.6f} "
                f"train_pred_pose_mean={train_mean['pred_pose_mean']:.6f} "
                f"train_high_weight_area_ratio={train_mean['high_weight_area_ratio']:.6f} "
                f"best_val={best_val:.6f} bad_epochs={bad_epochs}/{args.early_stop_patience}"
            )

            should_stop = 1 if bad_epochs >= args.early_stop_patience else 0
            if should_stop:
                exit_detail = (
                    f"early_stop at epoch={epoch + 1}, best_val={best_val:.6f}, "
                    f"bad_epochs={bad_epochs}/{args.early_stop_patience}"
                )
        else:
            should_stop = 0

        stop_t = torch.tensor([should_stop], device=device, dtype=torch.int32)
        if _is_dist():
            torch.distributed.broadcast(stop_t, src=0)
        last_completed_epoch = int(epoch + 1)
        if int(stop_t.item()) == 1:
            exit_reason = "early_stop"
            if not exit_detail:
                exit_detail = f"early_stop broadcast after epoch={epoch + 1}"
            break

    if not exit_reason:
        if start_epoch >= args.epochs:
            exit_reason = "already_completed"
            exit_detail = f"start_epoch={start_epoch} >= epochs={args.epochs}"
        else:
            exit_reason = "completed"
            exit_detail = f"finished epochs {start_epoch + 1}..{args.epochs}"
    _log_exit_event(args, exit_reason, detail=exit_detail, epoch=last_completed_epoch)
    _cleanup_distributed()
    return exit_reason


def _build_parser():
    parser = argparse.ArgumentParser("Train RefineNet dual-head validity decoder with DDP")
    parser.add_argument(
        "--train-config",
        type=str,
        default=TRAIN_DDP_DEFAULT_CONFIG_PATH,
        help="Training hyper-parameter config yaml. Defaults to learning/training/configs/train_ddp.yaml.",
    )
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT, help="Root of dataset_train.")
    parser.add_argument("--train-subdir", type=str, default="train")
    parser.add_argument("--val-subdir", type=str, default="val")
    parser.add_argument("--recon-cache-root", type=str, default=DEFAULT_RECON_CACHE_ROOT)
    parser.add_argument("--sam3d-project-root", type=str, default=DEFAULT_SAM3D_PROJECT_ROOT)
    parser.add_argument("--sam3d-notebook-root", type=str, default=DEFAULT_SAM3D_NOTEBOOK_ROOT)
    parser.add_argument("--sam3d-config-path", type=str, default=None)
    parser.add_argument("--min-mask-pixels", type=int, default=32)
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth integer scale, e.g. 1000 for mm->m.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Refiner config yaml. If omitted, auto-load ref_pose/weights/2023-10-28-18-33-37/config.yml.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"{DEFAULT_REF_POSE_ROOT}/learning/weights/refine_validmask_ddp",
    )
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1, help="Fallback local batch when per-rank list is not set.")
    parser.add_argument("--batch-size-per-rank", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0, help="Fallback local workers when per-rank list is not set.")
    parser.add_argument("--num-workers-per-rank", type=str, default="")
    parser.add_argument("--recon-num-workers", type=int, default=0)
    parser.add_argument("--recon-num-workers-per-rank", type=str, default="")
    parser.add_argument("--recon-persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--build-dataset-index-only",
        action="store_true",
        help="Build/load the Sam3DPartTrainDataset records index on CPU and exit before DDP/CUDA setup.",
    )
    parser.add_argument(
        "--dataset-index-split",
        type=str,
        choices=["both", "train", "val"],
        default="both",
        help="Which split to index when --build-dataset-index-only is set.",
    )
    parser.add_argument(
        "--force-rebuild-dataset-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rescan masks and rewrite the dataset records index instead of loading an existing index.",
    )
    parser.add_argument("--prebuild-recon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prebuild-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run reconstruction prebuild for the requested split/model and exit before training.",
    )
    parser.add_argument(
        "--recon-model",
        type=str,
        default="sam3d",
        choices=[
            "sam3d",
            "sam3d_tsdf",
            "sam3d_tsdf_dmesh",
            "hunyuan3d",
            "hunyuan3d_tsdf",
            "hunyuan3d_tsdf_dmesh",
            "instantmesh",
            "instantmesh_tsdf",
            "instantmesh_tsdf_dmesh",
            "all",
        ],
        help=(
            "Which reconstruction model cache to use/build. Use 'all' to mix "
            "base, TSDF, and TSDF+DLMesh caches for SAM3D, Hunyuan3D, and InstantMesh."
        ),
    )
    parser.add_argument(
        "--recon-view-density-scale",
        type=float,
        default=1.5,
        help="Increase selected reconstruction views density (>1 means denser).",
    )
    parser.add_argument(
        "--prebuild-split",
        type=str,
        choices=["both", "train", "val"],
        default="both",
        help=(
            "Which split to reconstruct during prebuild. Use 'val' to rebuild only "
            "dataset_root/val into recon-cache-root and leave train untouched."
        ),
    )
    parser.add_argument("--force-rebuild-recon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-resample-recon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use-real-depth-pointmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dataset depth/K to provide SAM3D pointmaps and avoid loading MoGe during reconstruction.",
    )
    parser.add_argument("--recon-min-views-per-part", type=int, default=3)
    parser.add_argument("--recon-max-views-per-part", type=int, default=5)
    parser.add_argument("--recon-rot-threshold-deg", type=float, default=15.0)
    parser.add_argument("--recon-trans-threshold", type=float, default=0.05)
    parser.add_argument("--hunyuan-model-path", type=str, default=DEFAULT_HUNYUAN_MODEL_PATH)
    parser.add_argument("--hunyuan-subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--hunyuan-num-inference-steps", type=int, default=50)
    parser.add_argument("--hunyuan-octree-resolution", type=int, default=384)
    parser.add_argument("--hunyuan-guidance-scale", type=float, default=5.5)
    parser.add_argument("--instantmesh-root", type=str, default=DEFAULT_INSTANTMESH_ROOT)
    parser.add_argument("--instantmesh-config-path", type=str, default=DEFAULT_INSTANTMESH_CONFIG_PATH)
    parser.add_argument("--instantmesh-diffusion-model", type=str, default="sudo-ai/zero123plus-v1.2")
    parser.add_argument("--instantmesh-dino-model", type=str, default="")
    parser.add_argument("--instantmesh-unet-path", type=str, default="")
    parser.add_argument("--instantmesh-model-path", type=str, default="")
    parser.add_argument("--instantmesh-diffusion-steps", type=int, default=75)
    parser.add_argument("--instantmesh-scale", type=float, default=1.0)
    parser.add_argument("--instantmesh-view", type=int, default=6, choices=[4, 6])
    parser.add_argument("--instantmesh-foreground-ratio", type=float, default=0.85)
    parser.add_argument("--instantmesh-export-texmap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--iterations", type=int, default=5, help="Dual-optimization iterations per sample.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--train-pose-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also fine-tune encodeAB[3:5]. Default is false to keep the pretrained refiner stable.",
    )
    parser.add_argument(
        "--train-adapters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train residual encodeAB adapters. They are zero-initialized; keep disabled until mask-only training is stable.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)
    parser.add_argument("--loss-history-json-path", type=str, default=f"{DEFAULT_REF_POSE_ROOT}/train_refine_validity_mask_ddp_loss_history.json")
    parser.add_argument(
        "--exit-log-jsonl-path",
        type=str,
        default="",
        help="Append one JSON line per rank on exit. Default: <out-dir>/train_ddp_exit_log.jsonl.",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Soft validity target: w*=exp(-alpha*e_xyz - gamma*e_sil - delta*e_rgb)
    parser.add_argument("--soft-alpha", type=float, default=8.0)
    parser.add_argument("--soft-gamma", type=float, default=2.0)
    parser.add_argument("--soft-delta", type=float, default=1.0)
    parser.add_argument(
        "--vm-weight",
        type=float,
        default=0.75,
        help="Validity-mask gating weight w in gate=(1-w)+w*vm. Original behavior is w=0.75.",
    )
    parser.add_argument("--aux-bce-weight", type=float, default=0.02, help="Very low weight auxiliary BCE supervision.")
    parser.add_argument("--pose-utility-scale", type=float, default=10.0)
    parser.add_argument(
        "--enable-ref-coord",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional reference-coordinate auxiliary heads. Default off keeps current training I/O unchanged.",
    )
    parser.add_argument(
        "--disable-vm-weight-naming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable vm-weight suffix naming for checkpoints/output directory.",
    )

    # torchrun-compatible fallback args
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    return parser


def parse_args():
    parser = _build_parser()
    pre_args, _ = parser.parse_known_args()
    cfg_defaults = _load_train_config_defaults(pre_args.train_config)
    valid_keys = {a.dest for a in parser._actions if getattr(a, "dest", None) not in (None, "help")}
    filtered = {k: v for k, v in cfg_defaults.items() if k in valid_keys}
    parser.set_defaults(**filtered)
    args = parser.parse_args()
    args.vm_weight = _safe_vm_weight(args.vm_weight)
    return args


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    try:
        train_ddp(args)
    except KeyboardInterrupt:
        _log_exit_event(args, "keyboard_interrupt", detail="received KeyboardInterrupt")
        _cleanup_distributed()
        raise
    except SystemExit as e:
        _log_exit_event(args, "system_exit", detail=f"code={e.code}")
        _cleanup_distributed()
        raise
    except BaseException:
        exc_text = traceback.format_exc()
        _log_exit_event(
            args,
            "exception",
            detail="unhandled exception",
            exc_text=exc_text,
        )
        _cleanup_distributed()
        raise


# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 train_ddp.py --prebuild-recon --batch-size 6 --num-workers 6 --recon-num-workers 6
# torchrun --nproc_per_node=8 train_ddp.py --no-prebuild-recon --batch-size-per-rank 64,64,64,64,64,64,64,64 --num-workers-per-rank 16,16,16,16,16,16,16,16 --pin-memory --persistent-workers
# torchrun --nproc_per_node=8 train_ddp.py --train-config learning/training/configs/train_ddp.yaml --vm-weight 0.85
