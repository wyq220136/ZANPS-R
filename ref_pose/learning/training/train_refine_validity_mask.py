import argparse
import os
import sys
import glob
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import kornia
import nvdiffrast.torch as dr
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../../")
sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan")
sys.path.append("/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/ref_pose")

from Utils import *  # noqa
from learning.models.refine_network import RefineNet
from learning.datasets.h5_dataset import PoseRefinePairH5Dataset
from learning.datasets.pose_dataset import BatchPoseData
from learning.datasets.sam3d_part_dataset import Sam3DPartTrainDataset

DEFAULT_REF_POSE_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/ref_pose"
DEFAULT_DATASET_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
DEFAULT_SAM3D_PROJECT_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects"
DEFAULT_SAM3D_NOTEBOOK_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects/notebook"
DEFAULT_RECON_CACHE_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_recon_cache"


def collate_single(batch):
    if len(batch) != 1:
        raise ValueError(
            "This trainer currently supports batch_size=1 because each sample performs "
            "mesh loading, rendering, and iterative pose refinement independently. "
            "Use --batch-size 1."
        )
    return batch[0]


def _resolve_refiner_base_ckpt():
    """
    Match ref_pose original loading convention:
      ref_pose/weights/2023-10-28-18-33-37/model_best.pth
    """
    run_name = "2023-10-28-18-33-37"
    model_name = "model_best.pth"
    ckpt_path = os.path.join(code_dir, "weights", run_name, model_name)
    ckpt_path = os.path.normpath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"refiner base checkpoint not found: {ckpt_path}")
    return ckpt_path


def _resolve_refiner_config_path(config_arg: str | None):
    if config_arg:
        if not os.path.exists(config_arg):
            raise FileNotFoundError(f"config not found: {config_arg}")
        return config_arg
    run_name = "2023-10-28-18-33-37"
    cfg_path = os.path.join(code_dir, "weights", run_name, "config.yml")
    cfg_path = os.path.normpath(cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"default refiner config not found: {cfg_path}")
    return cfg_path


def _resolve_sam3d_config_path(config_arg: str | None, sam3d_project_root: str):
    if config_arg:
        if not os.path.exists(config_arg):
            raise FileNotFoundError(f"SAM3D config not found: {config_arg}")
        return os.path.normpath(config_arg)

    project_root = os.path.normpath(sam3d_project_root)
    preferred = os.path.join(project_root, "checkpoints", "hf", "pipeline.yaml")
    if os.path.exists(preferred):
        return os.path.normpath(preferred)

    pattern = os.path.join(project_root, "checkpoints", "*", "pipeline.yaml")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        if len(candidates) > 1:
            print(
                "[warn] multiple SAM3D pipeline configs found; using the first sorted candidate. "
                "Pass --sam3d-config-path explicitly if this is not the checkpoint you want. "
                f"candidates={candidates}",
                flush=True,
            )
        return os.path.normpath(candidates[0])

    raise FileNotFoundError(
        "SAM3D pipeline config not found. "
        "Please pass --sam3d-config-path explicitly (e.g., sam-3d-objects/checkpoints/<tag>/pipeline.yaml)."
    )


def _find_latest_mask_decoder_ckpt(out_dir: str):
    if not os.path.isdir(out_dir):
        return None
    candidates = sorted(
        glob.glob(os.path.join(out_dir, "mask_decoder_epoch_*.pth")),
        key=os.path.getmtime,
    )
    if candidates:
        return candidates[-1]
    latest_path = os.path.join(out_dir, "mask_decoder_latest.pth")
    if os.path.exists(latest_path):
        return latest_path
    return None


def _safe_vm_weight(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _format_vm_weight_tag(vm_weight: float) -> str:
    return f"vmw{_safe_vm_weight(vm_weight):.3f}"


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        return ckpt_obj["model"]
    return ckpt_obj


def _load_refiner_backbone_only(model, base_ckpt_path: str):
    """
    Load original RefineNet weights except mask_decoder.*.
    """
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid checkpoint format: {base_ckpt_path}")

    filtered = {k: v for k, v in state.items() if not k.startswith("mask_decoder.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[load] base ckpt (backbone only): {base_ckpt_path}")
    print(f"[load] backbone missing keys: {len(missing)} (expected includes mask_decoder.*)")
    print(f"[load] backbone unexpected keys: {len(unexpected)}")


def _load_mask_decoder_if_available(model, mask_decoder_ckpt: str):
    if not mask_decoder_ckpt:
        return
    if not os.path.exists(mask_decoder_ckpt):
        return
    ckpt = torch.load(mask_decoder_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "mask_decoder" in ckpt:
        state = ckpt["mask_decoder"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        m = ckpt["model"]
        state = {k[len("mask_decoder."):]: v for k, v in m.items() if k.startswith("mask_decoder.")}
    elif isinstance(ckpt, dict):
        # Accept direct mask-decoder state dict.
        state = ckpt
    else:
        raise RuntimeError(f"invalid mask_decoder checkpoint format: {mask_decoder_ckpt}")

    missing, unexpected = model.mask_decoder.load_state_dict(state, strict=False)
    print(f"[load] mask_decoder ckpt: {mask_decoder_ckpt}")
    print(f"[load] mask_decoder missing keys: {len(missing)}")
    print(f"[load] mask_decoder unexpected keys: {len(unexpected)}")


def estimate_mesh_diameter_from_tensors(mesh_tensors, fallback: float = 1.0) -> float:
    """
    Compute mesh diameter from mesh_tensors['pos'] robustly.
    """
    pos = mesh_tensors.get("pos", None)
    if pos is None:
        return float(fallback)
    if not torch.is_tensor(pos) or pos.numel() < 3:
        return float(fallback)
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    diameter = torch.linalg.norm(extent).item()
    if (not np.isfinite(diameter)) or diameter <= 1e-8:
        return float(fallback)
    return float(diameter)


def render_crop_pair(mesh_tensors, pose, rgb, obs_mask, depth, K, cfg, dataset, glctx, tf_to_crops_override=None):
    H, W = obs_mask.shape[:2]
    pose_t = torch.as_tensor(pose[None], dtype=torch.float, device="cuda")
    mesh_diameter = estimate_mesh_diameter_from_tensors(mesh_tensors, fallback=1.0)
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
    xyz_obs = depth2xyzmap_batch(
        torch.as_tensor(depth, dtype=torch.float, device="cuda")[None],
        torch.as_tensor(K, dtype=torch.float, device="cuda")[None],
        zfar=np.inf,
    ).permute(0, 3, 1, 2)
    mask_obs = torch.as_tensor(obs_mask.astype(np.float32), dtype=torch.float, device="cuda")[None, None]
    rgb_obs = kornia.geometry.transform.warp_perspective(
        rgb_obs, tf_to_crops, dsize=cfg["input_resize"], mode="bilinear", align_corners=False
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
    pose_data = BatchPoseData(
        rgbAs=rgb_r,
        rgbBs=rgb_obs,
        depthAs=depth_r[..., None].permute(0, 3, 1, 2),
        depthBs=None,
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


def _resolve_train_val_roots(dataset_root: str, train_subdir: str, val_subdir: str):
    dataset_root = os.path.normpath(dataset_root)
    train_root = os.path.join(dataset_root, train_subdir)
    val_root = os.path.join(dataset_root, val_subdir)
    if os.path.isdir(train_root) and os.path.isdir(val_root):
        return os.path.normpath(train_root), os.path.normpath(val_root)
    raise FileNotFoundError(
        f"train/val split not found under {dataset_root}. "
        f"Expected: {train_root} and {val_root}. "
        "Please run split_dataset_train_val.py first."
    )


def _compute_sample_loss(
    batch,
    model,
    cfg,
    dataset,
    glctx,
    iterations: int,
    min_mask_pixels: int,
    same_view_loss_weight: float,
    vm_weight: float,
):
    recon_mesh = trimesh.load(batch["recon_mesh_path"], force="mesh")
    gt_mesh = trimesh.load(batch["gt_mesh_path"], force="mesh")
    recon_tensors = make_mesh_tensors(recon_mesh)
    gt_tensors = make_mesh_tensors(gt_mesh)
    pose = batch["init_pose"].astype(np.float32)
    rgb = batch["rgb"]
    obs_mask = batch["mask"].astype(np.uint8)
    if int(np.count_nonzero(obs_mask)) < min_mask_pixels:
        return None
    depth = batch["depth"].astype(np.float32)
    K = batch["K"]

    prev_mask = None
    final_pose = torch.as_tensor(pose[None], dtype=torch.float, device="cuda")
    total_loss = torch.zeros([], device="cuda")

    for _ in range(iterations):
        pd_recon = render_crop_pair(
            recon_tensors, final_pose[0].data.cpu().numpy(), rgb, obs_mask, depth, K, cfg, dataset, glctx
        )
        pd_gt = render_crop_pair(
            gt_tensors,
            final_pose[0].data.cpu().numpy(),
            rgb,
            obs_mask,
            depth,
            K,
            cfg,
            dataset,
            glctx,
            tf_to_crops_override=pd_recon.tf_to_crops,
        )

        A = torch.cat([pd_recon.rgbAs, pd_recon.xyz_mapAs], dim=1).float()
        B = torch.cat([pd_recon.rgbBs, pd_recon.xyz_mapBs], dim=1).float()
        A_gt = torch.cat([pd_gt.rgbAs, pd_gt.xyz_mapAs], dim=1).float()
        if prev_mask is not None:
            if prev_mask.shape[-2:] != A.shape[-2:]:
                prev_mask = F.interpolate(prev_mask, size=A.shape[-2:], mode="bilinear", align_corners=False)
            prev_mask = prev_mask.clamp(0.0, 1.0)
            gate = (1.0 - float(vm_weight)) + float(vm_weight) * prev_mask
            A = A * gate
            A_gt = A_gt * gate

        with torch.cuda.amp.autocast(enabled=True):
            out = model(A, B)
            out_gt = model(A_gt, B)
            vm = out["validity_mask"]
            gt_mask = (pd_gt.depthAs > 1e-6).float()
            loss_shape = F.l1_loss(vm * pd_recon.rgbAs, gt_mask * pd_gt.rgbAs)
            obs_rgb = pd_recon.rgbBs
            residual_recon = pd_recon.rgbAs - obs_rgb
            residual_gt = pd_gt.rgbAs - obs_rgb
            loss_same_view = F.l1_loss(vm * residual_recon, gt_mask * residual_gt)
            loss_pose_consistency = F.l1_loss(out["trans"], out_gt["trans"]) + F.l1_loss(out["rot"], out_gt["rot"])
            loss_reg = torch.mean(vm)
        # BCE is not autocast-safe for probability inputs; force FP32 outside autocast.
        with torch.cuda.amp.autocast(enabled=False):
            loss_mask = F.binary_cross_entropy(vm.float(), gt_mask.float())
        with torch.cuda.amp.autocast(enabled=True):
            loss = (
                loss_mask
                + 0.5 * loss_shape
                + same_view_loss_weight * loss_same_view
                + loss_pose_consistency
                + 0.05 * loss_reg
            )

        total_loss = total_loss + loss
        prev_mask = vm.detach()

        if cfg["rot_rep"] == "axis_angle":
            rot_delta = so3_exp_map(torch.tanh(out["rot"]) * cfg.get("rot_normalizer", 1.0)).permute(0, 2, 1)
        else:
            rot_delta = rotation_6d_to_matrix(out["rot"]).permute(0, 2, 1)
        trans_delta = out["trans"]
        final_pose = egocentric_delta_pose_to_pose(final_pose, trans_delta=trans_delta, rot_mat_delta=rot_delta)

    return total_loss


def _run_one_epoch(
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
):
    if is_train:
        model.mask_decoder.train()
    else:
        model.mask_decoder.eval()

    running = 0.0
    valid_steps = 0
    phase = "train" if is_train else "val"
    use_tqdm = (tqdm is not None) and (not args.disable_tqdm)
    if use_tqdm:
        progress = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"epoch {epoch_idx+1}/{args.epochs} {phase}",
            dynamic_ncols=True,
            leave=True,
        )
    else:
        progress = enumerate(loader)
    print(f"[{phase}] epoch {epoch_idx+1}/{args.epochs} start, loader_len={len(loader)}")
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, batch in progress:
            t_step_start = time.time()
            if i == 0:
                print(f"[{phase}] first batch fetched, start computing loss")
            total_loss = _compute_sample_loss(
                batch=batch,
                model=model,
                cfg=cfg,
                dataset=dataset,
                glctx=glctx,
                iterations=args.iterations,
                min_mask_pixels=args.min_mask_pixels,
                same_view_loss_weight=args.same_view_loss_weight,
                vm_weight=float(np.clip(float(args.vm_weight), 0.0, 1.0)),
            )
            if total_loss is None:
                continue

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            valid_steps += 1
            running += float(total_loss.item())
            mean_so_far = running / max(1, valid_steps)
            if use_tqdm:
                progress.set_postfix(loss=f"{float(total_loss.item()):.6f}", avg=f"{mean_so_far:.6f}")
            if (i + 1) % args.log_every == 0:
                msg = (
                    f"[epoch {epoch_idx+1}/{args.epochs}] {phase} "
                    f"step {i+1}/{len(loader)} loss={float(total_loss.item()):.6f} avg={mean_so_far:.6f}"
                )
                if use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)
            if i == 0:
                print(f"[{phase}] first optimization step done in {time.time() - t_step_start:.2f}s")

    mean_loss = running / max(1, valid_steps)
    return mean_loss, valid_steps


def _save_loss_plot_and_json(history, plot_path, json_path):
    plot_dir = os.path.dirname(plot_path)
    json_dir = os.path.dirname(json_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="val_loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Refine Validity Mask Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def train(args):
    args.vm_weight = _safe_vm_weight(getattr(args, "vm_weight", 0.75))
    vm_weight_tag = _format_vm_weight_tag(args.vm_weight)
    out_dir_name = os.path.basename(os.path.normpath(args.out_dir))
    if vm_weight_tag not in out_dir_name:
        args.out_dir = os.path.join(args.out_dir, vm_weight_tag)

    args.sam3d_config_path = _resolve_sam3d_config_path(args.sam3d_config_path, args.sam3d_project_root)
    print(f"[load] sam3d config: {args.sam3d_config_path}")
    train_root, val_root = _resolve_train_val_roots(args.dataset_root, args.train_subdir, args.val_subdir)
    print(f"[load] train root: {train_root}")
    print(f"[load] val root: {val_root}")
    cfg_path = _resolve_refiner_config_path(args.config)
    print(f"[load] config: {cfg_path}")
    print(f"[train] vm_weight={args.vm_weight:.3f} ({vm_weight_tag})")
    print(f"[save] out_dir={args.out_dir}")
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

    dataset = PoseRefinePairH5Dataset(cfg=cfg, h5_file="", mode="test")
    model = RefineNet(cfg=cfg, c_in=int(cfg["c_in"])).cuda().train()
    base_ckpt = _resolve_refiner_base_ckpt()
    _load_refiner_backbone_only(model, base_ckpt)

    mask_decoder_ckpt = _find_latest_mask_decoder_ckpt(args.out_dir) if args.auto_resume_mask_decoder else None
    _load_mask_decoder_if_available(model, mask_decoder_ckpt)

    # Freeze all pretrained FoundationPose-refiner weights.
    for _, p in model.named_parameters():
        p.requires_grad = False
    # Train only the newly added mask decoder head.
    for _, p in model.mask_decoder.named_parameters():
        p.requires_grad = True
    # Keep frozen modules deterministic (e.g., BN running stats).
    model.eval()
    model.mask_decoder.train()
    trainable_params = [p for p in model.mask_decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    print(f"[train] trainable parameter tensors: {len(trainable_params)} (mask_decoder only)")
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    ds_train = Sam3DPartTrainDataset(
        dataset_root=train_root,
        cache_root=args.recon_cache_root,
        sam3d_project_root=args.sam3d_project_root,
        sam3d_notebook_root=args.sam3d_notebook_root,
        sam3d_config_path=args.sam3d_config_path,
        min_mask_pixels=args.min_mask_pixels,
        seed=42,
        rebuild_recon=args.rebuild_recon,
        use_real_depth_pointmap=args.use_real_depth_pointmap,
    )
    ds_val = Sam3DPartTrainDataset(
        dataset_root=val_root,
        cache_root=args.recon_cache_root,
        sam3d_project_root=args.sam3d_project_root,
        sam3d_notebook_root=args.sam3d_notebook_root,
        sam3d_config_path=args.sam3d_config_path,
        min_mask_pixels=args.min_mask_pixels,
        seed=42,
        rebuild_recon=False,
        use_real_depth_pointmap=args.use_real_depth_pointmap,
    )
    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_single,
    )
    print(f"[data] train samples={len(ds_train)} val samples={len(ds_val)}")
    glctx = dr.RasterizeCudaContext()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    bad_epochs = 0
    best_ckpt_path = os.path.join(args.out_dir, f"mask_decoder_best_by_val_{vm_weight_tag}.pth")

    for epoch in range(args.epochs):
        train_loss, train_steps = _run_one_epoch(
            loader=loader_train,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            dataset=dataset,
            glctx=glctx,
            args=args,
            is_train=True,
            epoch_idx=epoch,
        )
        val_loss, val_steps = _run_one_epoch(
            loader=loader_val,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            dataset=dataset,
            glctx=glctx,
            args=args,
            is_train=False,
            epoch_idx=epoch,
        )

        os.makedirs(args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"mask_decoder_epoch_{epoch+1:03d}_{vm_weight_tag}.pth")
        latest_path = os.path.join(args.out_dir, f"mask_decoder_latest_{vm_weight_tag}.pth")
        payload = {
            "mask_decoder": model.mask_decoder.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": int(epoch + 1),
            "vm_weight": float(args.vm_weight),
        }
        torch.save(payload, ckpt_path)
        torch.save(payload, latest_path)
        if val_loss + args.early_stop_min_delta < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(payload, best_ckpt_path)
            print(f"[save] best-by-val updated: {best_ckpt_path} (val_loss={best_val:.6f})")
        else:
            bad_epochs += 1

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        _save_loss_plot_and_json(history, args.loss_plot_path, args.loss_history_json_path)

        print(
            f"[epoch-summary] epoch={epoch+1} "
            f"train_loss={train_loss:.6f} (steps={train_steps}) "
            f"val_loss={val_loss:.6f} (steps={val_steps}) "
            f"best_val={best_val:.6f} bad_epochs={bad_epochs}/{args.early_stop_patience}"
        )
        print(f"[save] {ckpt_path}")
        print(f"[save] {latest_path}")
        print(f"[save] loss plot: {args.loss_plot_path}")

        if bad_epochs >= args.early_stop_patience:
            print(
                f"[early-stop] no val improvement for {args.early_stop_patience} epochs "
                f"(min_delta={args.early_stop_min_delta}). stop at epoch {epoch+1}."
            )
            break


def parse_args():
    parser = argparse.ArgumentParser("Train RefineNet with validity-mask decoder")
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT, help="Root of dataset_train.")
    parser.add_argument("--train-subdir", type=str, default="train")
    parser.add_argument("--val-subdir", type=str, default="val")
    parser.add_argument("--recon-cache-root", type=str, default=DEFAULT_RECON_CACHE_ROOT)
    parser.add_argument("--sam3d-project-root", type=str, default=DEFAULT_SAM3D_PROJECT_ROOT)
    parser.add_argument("--sam3d-notebook-root", type=str, default=DEFAULT_SAM3D_NOTEBOOK_ROOT)
    parser.add_argument("--sam3d-config-path", type=str, default=None)
    parser.add_argument("--rebuild-recon", action="store_true", help="Rebuild SAM3D reconstruction cache.")
    parser.add_argument(
        "--use-real-depth-pointmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dataset depth/K to provide SAM3D pointmaps and avoid loading MoGe during reconstruction.",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=32)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Refiner config yaml. If omitted, auto-load ref_pose/weights/2023-10-28-18-33-37/config.yml.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"{DEFAULT_REF_POSE_ROOT}/learning/weights/refine_validmask",
    )
    parser.add_argument(
        "--auto-resume-mask-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto load latest mask_decoder checkpoint from out-dir if present.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5, help="Dual-optimization iterations per sample.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--same-view-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--vm-weight",
        type=float,
        default=0.75,
        help="Validity-mask gating weight w in gate=(1-w)+w*vm. Original behavior is w=0.75.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)
    parser.add_argument(
        "--loss-plot-path",
        type=str,
        default=f"{DEFAULT_REF_POSE_ROOT}/train_refine_validity_mask_loss.png",
    )
    parser.add_argument(
        "--loss-history-json-path",
        type=str,
        default=f"{DEFAULT_REF_POSE_ROOT}/train_refine_validity_mask_loss_history.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
