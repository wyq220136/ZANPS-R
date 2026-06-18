import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HY3D_ROOT = os.path.join(REPO_ROOT, "Hunyuan3D-2.1")
HY3D_SHAPE_ROOT = os.path.join(HY3D_ROOT, "hy3dshape")
HY3D_LOCAL_CKPTS_ROOT = os.path.join(HY3D_ROOT, "ckpts")
HY3D_DEFAULT_SUBFOLDER = "hunyuan3d-dit-v2-1"


def default_hunyuan_model_path() -> str:
    return HY3D_LOCAL_CKPTS_ROOT


def _resolve_local_model_path(model_path: str, subfolder: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(model_path))
    config_path = os.path.join(expanded, subfolder, "config.yaml")
    ckpt_path = os.path.join(expanded, subfolder, "model.fp16.ckpt")
    safetensors_path = os.path.join(expanded, subfolder, "model.fp16.safetensors")
    if os.path.exists(config_path) and (os.path.exists(ckpt_path) or os.path.exists(safetensors_path)):
        return expanded
    raise FileNotFoundError(
        "Local Hunyuan3D checkpoint is missing. Expected files under "
        f"{os.path.join(expanded, subfolder)}: config.yaml and model.fp16.ckpt "
        "or model.fp16.safetensors."
    )


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


def _extract_part_id(mask_path: str, fallback_idx: int) -> int:
    base = os.path.splitext(os.path.basename(mask_path))[0]
    nums = re.findall(r"\d+", base)
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            pass
    return int(fallback_idx)


def _load_hunyuan_modules():
    if HY3D_SHAPE_ROOT not in sys.path:
        sys.path.insert(0, HY3D_SHAPE_ROOT)
    if HY3D_ROOT not in sys.path:
        sys.path.insert(0, HY3D_ROOT)

    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception:
        pass

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    return Hunyuan3DDiTFlowMatchingPipeline


@dataclass
class HunyuanReconstructor:
    model_path: str = default_hunyuan_model_path()
    subfolder: str = HY3D_DEFAULT_SUBFOLDER
    num_inference_steps: int = 50
    octree_resolution: int = 384
    guidance_scale: float = 5.5

    def __post_init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        self.model_path = _resolve_local_model_path(self.model_path, self.subfolder)
        pipeline_cls = _load_hunyuan_modules()
        self.pipeline = pipeline_cls.from_pretrained(self.model_path, subfolder=self.subfolder)

    def reconstruct_part(self, image_rgba: Image.Image, out_obj_path: str):
        mesh = self.pipeline(
            image=image_rgba,
            num_inference_steps=int(self.num_inference_steps),
            octree_resolution=int(self.octree_resolution),
            guidance_scale=float(self.guidance_scale),
        )[0]
        os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
        mesh.export(out_obj_path)


def _prepare_rgba_from_mask(rgb_bgr: np.ndarray, mask_gray: np.ndarray):
    mask = (mask_gray > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad_x = max(2, int((x1 - x0 + 1) * 0.05))
    pad_y = max(2, int((y1 - y0 + 1) * 0.05))
    h, w = mask.shape
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)

    crop_rgb = rgb_bgr[y0 : y1 + 1, x0 : x1 + 1, :]
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

    crop_rgba = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2RGBA)
    crop_rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)
    return Image.fromarray(crop_rgba)


def parse_view_and_frame(frame_id: str, obj_name: str):
    prefix = f"{obj_name}_"
    if frame_id.startswith(prefix):
        tail = frame_id[len(prefix) :]
        m = re.fullmatch(r"(\d+)_(\d+)", tail)
        if m:
            return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d+", frame_id)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return 0, 0


def _frame_part_visibility_count(obj_dir: str, frame_id: str, min_mask_pixels: int = 64) -> int:
    mask_dir = os.path.join(obj_dir, "gt_mask", frame_id)
    if not os.path.isdir(mask_dir):
        return 0
    cnt = 0
    for n in os.listdir(mask_dir):
        if not n.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        p = os.path.join(mask_dir, n)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if int(np.count_nonzero(m > 0)) >= int(min_mask_pixels):
            cnt += 1
    return cnt


def select_best_frame_per_view(obj_dir: str, min_mask_pixels: int = 64):
    obj_name = os.path.basename(obj_dir.rstrip("/\\"))
    mask_root = os.path.join(obj_dir, "gt_mask")
    if not os.path.isdir(mask_root):
        return []
    frame_ids = sorted(
        [d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))],
        key=natural_sort_key,
    )
    grouped = {}
    for fid in frame_ids:
        view_id, frame_idx = parse_view_and_frame(fid, obj_name)
        vis_cnt = _frame_part_visibility_count(obj_dir, fid, min_mask_pixels=min_mask_pixels)
        grouped.setdefault(view_id, []).append((vis_cnt, frame_idx, fid))

    refs = []
    for view_id in sorted(grouped.keys()):
        cands = grouped[view_id]
        cands.sort(key=lambda x: (-x[0], x[1]))
        refs.append((view_id, cands[0][2]))
    return refs


def reconstruct_reference_views_hunyuan3d(reconstructor: HunyuanReconstructor, obj_dir: str, min_mask_pixels: int = 64):
    mask_root = os.path.join(obj_dir, "gt_mask")
    if not os.path.isdir(mask_root):
        print(f"[SKIP] {obj_dir}: gt_mask not found")
        return

    refs = select_best_frame_per_view(obj_dir, min_mask_pixels=min_mask_pixels)
    if not refs:
        print(f"[SKIP] {obj_dir}: no valid reference frame found")
        return

    models_root = os.path.join(obj_dir, "models")
    os.makedirs(models_root, exist_ok=True)

    for view_id, frame_id in refs:
        frame_mask_dir = os.path.join(mask_root, frame_id)
        if not os.path.isdir(frame_mask_dir):
            continue

        rgb_path = ""
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(obj_dir, "rgb", f"{frame_id}{ext}")
            if os.path.exists(cand):
                rgb_path = cand
                break
        if not rgb_path:
            print(f"[SKIP] {obj_dir}: missing rgb for frame {frame_id}")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            print(f"[SKIP] {obj_dir}: cannot read rgb {rgb_path}")
            continue

        view_model_dir = os.path.join(models_root, f"view_{view_id}")
        os.makedirs(view_model_dir, exist_ok=True)
        print(f"[RECON-HY3D] {os.path.basename(obj_dir)} view={view_id}, ref={frame_id} -> {view_model_dir}")

        mask_files = sorted(
            [
                os.path.join(frame_mask_dir, n)
                for n in os.listdir(frame_mask_dir)
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=natural_sort_key,
        )

        for part_idx, mask_path in enumerate(mask_files):
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                continue
            if int(np.count_nonzero(mask_gray > 0)) < int(min_mask_pixels):
                continue

            rgba = _prepare_rgba_from_mask(rgb_bgr, mask_gray)
            if rgba is None:
                continue

            part_id = _extract_part_id(mask_path, part_idx)
            model_dir = os.path.join(view_model_dir, f"model_{part_id:04d}")
            obj_path = os.path.join(model_dir, "model.obj")

            try:
                reconstructor.reconstruct_part(rgba, obj_path)
            except Exception as e:
                print(f"[WARN] Hunyuan3D failed for frame={frame_id} part={part_id}: {e}")

    print(f"[RECON-HY3D-DONE] {os.path.basename(obj_dir)} reference models ready")
