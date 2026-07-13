import argparse
import glob
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp


SCRIPT_DIR = Path(__file__).resolve().parent
REF_POSE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]

for _p in (REPO_ROOT, REF_POSE_ROOT):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from learning.datasets.sam3d_part_dataset import Sam3DPartTrainDataset  # noqa: E402


DEFAULT_DATASET_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train"
DEFAULT_SAM3D_PROJECT_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects"
DEFAULT_SAM3D_NOTEBOOK_ROOT = (
    "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects/notebook"
)
DEFAULT_RECON_CACHE_ROOT = "/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train_recon_cache"
DEFAULT_HUNYUAN_MODEL_PATH = str(REPO_ROOT / "Hunyuan3D-2.1" / "ckpts")
DEFAULT_INSTANTMESH_ROOT = str(REPO_ROOT / "InstantMesh")
DEFAULT_INSTANTMESH_CONFIG_PATH = str(Path(DEFAULT_INSTANTMESH_ROOT) / "configs" / "instant-mesh-large.yaml")
DEFAULT_TRAIN_CONFIG = str(SCRIPT_DIR / "configs" / "train_ddp.yaml")


def _load_train_config_defaults(config_path: str):
    path = str(config_path or "").strip()
    if not path or not os.path.exists(path):
        return {}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        data = OmegaConf.to_container(cfg, resolve=True)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[config][warn] OmegaConf load failed for {path}: {exc}", flush=True)
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[config][warn] YAML load failed for {path}: {exc}", flush=True)
        return {}


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
        "Please pass --sam3d-config-path explicitly."
    )


def _resolve_train_val_roots(dataset_root: str, train_subdir: str, val_subdir: str):
    dataset_root = os.path.normpath(dataset_root)
    train_root = os.path.join(dataset_root, train_subdir)
    val_root = os.path.join(dataset_root, val_subdir)
    if os.path.isdir(train_root) and os.path.isdir(val_root):
        return os.path.normpath(train_root), os.path.normpath(val_root)
    raise FileNotFoundError(
        f"train/val split not found under {dataset_root}. "
        f"Expected: {train_root} and {val_root}."
    )


def _parse_int_list(raw: str):
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
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


def _torchrun_info(args):
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _seed_everything(seed: int, worker_offset: int = 0):
    final_seed = int(seed) + int(worker_offset)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def _make_prebuild_dataset(args, root, split_name: str, log_prefix: str = "prebuild"):
    print(
        f"[{log_prefix}] initializing dataset split={split_name} root={root} "
        f"recon_model={args.recon_model}",
        flush=True,
    )
    dataset_kwargs = {
        "dataset_root": root,
        "cache_root": args.recon_cache_root,
        "cache_split": split_name,
        "sam3d_project_root": args.sam3d_project_root,
        "sam3d_notebook_root": args.sam3d_notebook_root,
        "sam3d_config_path": args.sam3d_config_path,
        "min_mask_pixels": args.min_mask_pixels,
        "seed": args.seed,
        "rebuild_recon": args.force_rebuild_recon,
        "allow_recon_write": True,
        "strict_mode": True,
        "fallback_to_gt_mesh_on_recon_fail": False,
        "depth_scale": args.depth_scale,
        "recon_min_views_per_part": args.recon_min_views_per_part,
        "recon_max_views_per_part": args.recon_max_views_per_part,
        "recon_rot_threshold_deg": args.recon_rot_threshold_deg,
        "recon_trans_threshold": args.recon_trans_threshold,
        "force_resample_recon": args.force_resample_recon,
        "use_real_depth_pointmap": args.use_real_depth_pointmap,
        "recon_model": args.recon_model,
        "recon_view_density_scale": args.recon_view_density_scale,
        "hunyuan_model_path": args.hunyuan_model_path,
        "hunyuan_subfolder": args.hunyuan_subfolder,
        "hunyuan_num_inference_steps": args.hunyuan_num_inference_steps,
        "hunyuan_octree_resolution": args.hunyuan_octree_resolution,
        "hunyuan_guidance_scale": args.hunyuan_guidance_scale,
        "instantmesh_root": args.instantmesh_root,
        "instantmesh_config_path": args.instantmesh_config_path,
        "instantmesh_diffusion_model": args.instantmesh_diffusion_model,
        "instantmesh_dino_model": args.instantmesh_dino_model,
        "instantmesh_unet_path": args.instantmesh_unet_path,
        "instantmesh_model_path": args.instantmesh_model_path,
        "instantmesh_diffusion_steps": args.instantmesh_diffusion_steps,
        "instantmesh_scale": args.instantmesh_scale,
        "instantmesh_view": args.instantmesh_view,
        "instantmesh_foreground_ratio": args.instantmesh_foreground_ratio,
        "instantmesh_export_texmap": args.instantmesh_export_texmap,
        "rebuild_records_index": args.force_rebuild_dataset_index,
    }
    try:
        supported = set(inspect.signature(Sam3DPartTrainDataset.__init__).parameters.keys())
        supported.discard("self")
    except Exception:
        supported = set(dataset_kwargs.keys())
    filtered_kwargs = {k: v for k, v in dataset_kwargs.items() if k in supported}
    dropped = sorted(set(dataset_kwargs.keys()) - set(filtered_kwargs.keys()))
    if dropped:
        print(
            "[dataset][compat] Sam3DPartTrainDataset does not support args; "
            f"dropped={dropped}",
            flush=True,
        )
    ds = Sam3DPartTrainDataset(**filtered_kwargs)
    print(
        f"[{log_prefix}] dataset ready split={split_name} "
        f"records={len(ds.records)} parts={len(ds.part_keys)} recon_model={args.recon_model}",
        flush=True,
    )
    return ds


def _prebuild_part_worker(worker_idx, args, root, split_name, part_keys, local_rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
    _seed_everything(args.seed, worker_offset=worker_idx)
    ds = _make_prebuild_dataset(args, root, split_name, log_prefix=f"prebuild-worker-{worker_idx}")
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
                f"[prebuild-worker] split={split_name} worker={worker_idx} done={i + 1}/{total}",
                flush=True,
            )


def _run_split(args, root, split_name: str, rank: int, world_size: int, local_rank: int, workers_per_rank: list[int]):
    ds = _make_prebuild_dataset(args, root, split_name)
    part_keys = ds.part_keys[rank::world_size]
    local_workers = max(1, int(workers_per_rank[rank]))
    print(
        f"[prebuild] split={split_name} samples={len(ds)} parts={len(ds.part_keys)} "
        f"rank={rank}/{world_size} local_rank={local_rank} local_parts={len(part_keys)} "
        f"force_rebuild={args.force_rebuild_recon} force_resample={args.force_resample_recon} "
        f"workers_per_rank={workers_per_rank}",
        flush=True,
    )
    if local_workers <= 1 or len(part_keys) <= 1:
        for i, part_key in enumerate(part_keys):
            ds.ensure_recon_cache_for_part(part_key)
            if (i + 1) % 10 == 0 or (i + 1) == len(part_keys):
                print(f"[prebuild] split={split_name} rank={rank} done={i + 1}/{len(part_keys)}", flush=True)
        return

    chunks = [part_keys[i::local_workers] for i in range(local_workers)]
    chunks = [c for c in chunks if c]
    ctx = mp.get_context("spawn")
    procs = []
    for worker_idx, chunk in enumerate(chunks):
        p = ctx.Process(
            target=_prebuild_part_worker,
            args=(worker_idx, args, root, split_name, chunk, local_rank),
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
        raise RuntimeError(f"prebuild workers failed for split={split_name}: {failed}")


def run_prebuild(args):
    if not args.prebuild_recon:
        print("[prebuild] --no-prebuild-recon set; nothing to do.", flush=True)
        return

    rank, world_size, local_rank = _torchrun_info(args)
    _seed_everything(args.seed, worker_offset=rank)
    args.sam3d_config_path = _resolve_sam3d_config_path(args.sam3d_config_path, args.sam3d_project_root)
    train_root, val_root = _resolve_train_val_roots(args.dataset_root, args.train_subdir, args.val_subdir)

    workers_per_rank = _parse_per_rank_value(args.recon_num_workers_per_rank, args.recon_num_workers, world_size)
    workers_per_rank = [max(1, int(v)) for v in workers_per_rank]
    print(
        "[prebuild] reconstruction-only entrypoint; FoundationPose/Utils/mycpp are not imported here.",
        flush=True,
    )
    print(
        f"[dist] rank={rank} world_size={world_size} local_rank={local_rank} "
        f"recon_model={args.recon_model}",
        flush=True,
    )
    print(f"[load] train root: {train_root}", flush=True)
    print(f"[load] val root: {val_root}", flush=True)
    if args.recon_model in ("sam3d", "sam3d_tsdf", "sam3d_tsdf_dmesh", "all"):
        print(f"[load] sam3d config: {args.sam3d_config_path}", flush=True)
    if args.recon_model in ("hunyuan3d", "hunyuan3d_tsdf", "hunyuan3d_tsdf_dmesh", "all"):
        print(
            f"[load] hunyuan model path: {args.hunyuan_model_path} "
            f"subfolder={args.hunyuan_subfolder} steps={args.hunyuan_num_inference_steps} "
            f"octree={args.hunyuan_octree_resolution} guidance={args.hunyuan_guidance_scale}",
            flush=True,
        )
    if args.recon_model in ("instantmesh", "instantmesh_tsdf", "instantmesh_tsdf_dmesh", "all"):
        print(f"[load] instantmesh root: {args.instantmesh_root}", flush=True)
        print(f"[load] instantmesh config: {args.instantmesh_config_path}", flush=True)
        print(f"[load] instantmesh diffusion model: {args.instantmesh_diffusion_model}", flush=True)
        print(f"[load] instantmesh dino model: {args.instantmesh_dino_model}", flush=True)
        print(f"[load] instantmesh unet: {args.instantmesh_unet_path}", flush=True)
        print(f"[load] instantmesh model: {args.instantmesh_model_path}", flush=True)

    split = str(args.prebuild_split).lower()
    t0 = time.time()
    if split in ("both", "train"):
        _run_split(args, train_root, "train", rank, world_size, local_rank, workers_per_rank)
    if split in ("both", "val"):
        _run_split(args, val_root, "val", rank, world_size, local_rank, workers_per_rank)
    print(
        f"[prebuild] reconstruction cache ready split={split} rank={rank}/{world_size} "
        f"seconds={time.time() - t0:.2f}",
        flush=True,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        "Prebuild reconstruction cache without importing FoundationPose/Utils/mycpp."
    )
    parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-subdir", type=str, default="train")
    parser.add_argument("--val-subdir", type=str, default="val")
    parser.add_argument("--recon-cache-root", type=str, default=DEFAULT_RECON_CACHE_ROOT)
    parser.add_argument("--sam3d-project-root", type=str, default=DEFAULT_SAM3D_PROJECT_ROOT)
    parser.add_argument("--sam3d-notebook-root", type=str, default=DEFAULT_SAM3D_NOTEBOOK_ROOT)
    parser.add_argument("--sam3d-config-path", type=str, default=None)
    parser.add_argument("--min-mask-pixels", type=int, default=32)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--recon-num-workers", type=int, default=1)
    parser.add_argument("--recon-num-workers-per-rank", type=str, default="")
    parser.add_argument("--prebuild-recon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prebuild-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prebuild-split", type=str, choices=["both", "train", "val"], default="both")
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
    )
    parser.add_argument("--recon-view-density-scale", type=float, default=1.5)
    parser.add_argument("--force-rebuild-recon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-resample-recon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-rebuild-dataset-index", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-real-depth-pointmap", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()

    parser = _build_parser()
    cfg_defaults = _load_train_config_defaults(pre_args.train_config)
    valid_keys = {a.dest for a in parser._actions if getattr(a, "dest", None) not in (None, "help")}
    parser.set_defaults(**{k: v for k, v in cfg_defaults.items() if k in valid_keys})
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_prebuild(parse_args())
