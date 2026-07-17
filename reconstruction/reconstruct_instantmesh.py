from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANTMESH_ROOT = REPO_ROOT / "InstantMesh"
DEFAULT_CONFIG = INSTANTMESH_ROOT / "configs" / "instant-mesh-large.yaml"


def default_instantmesh_root() -> str:
    return str(INSTANTMESH_ROOT)


def default_instantmesh_config_path() -> str:
    return str(DEFAULT_CONFIG)


def _ensure_instantmesh_imports(instantmesh_root: str):
    root = Path(instantmesh_root).resolve()
    for p in (root, root.parent):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def _resolve_checkpoint(path: str, instantmesh_root: Path) -> str:
    if not path:
        return ""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = instantmesh_root / p
    return str(p.resolve())


def _resize_foreground_rgba(image_rgba: Image.Image, ratio: float) -> Image.Image:
    image = np.asarray(image_rgba)
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError(f"expected RGBA image, got shape={image.shape}")
    alpha = np.where(image[..., 3] > 0)
    if len(alpha[0]) == 0:
        raise ValueError("empty RGBA foreground alpha")
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    fg = image[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    new_size = int(new_image.shape[0] / float(ratio))
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return Image.fromarray(new_image)


@dataclass
class InstantMeshReconstructor:
    instantmesh_root: str = default_instantmesh_root()
    config_path: str = default_instantmesh_config_path()
    diffusion_model: str = "sudo-ai/zero123plus-v1.2"
    dino_model: str = ""
    unet_path: str = ""
    model_path: str = ""
    diffusion_steps: int = 75
    seed: int = 42
    scale: float = 1.0
    view: int = 6
    foreground_ratio: float = 0.85
    export_texmap: bool = False

    def __post_init__(self):
        self.instantmesh_root = str(Path(self.instantmesh_root).resolve())
        instantmesh_root = _ensure_instantmesh_imports(self.instantmesh_root)

        from diffusers import EulerAncestralDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from src.utils.train_util import instantiate_from_config
        from zero123plus.pipeline import Zero123PlusPipeline

        config_path = Path(self.config_path)
        if not config_path.is_absolute():
            config_path = instantmesh_root / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"InstantMesh config not found: {config_path}")
        self.config_path = str(config_path.resolve())
        self.config = OmegaConf.load(self.config_path)
        self.config_name = config_path.stem
        self.infer_config = self.config.infer_config
        self.is_flexicubes = self.config_name.startswith("instant-mesh")
        self.device = torch.device("cuda")
        if str(self.dino_model or "").strip():
            self.config.model_config.params.encoder_model_name = _resolve_checkpoint(self.dino_model, instantmesh_root)
            print(f"[instantmesh] using DINO encoder: {self.config.model_config.params.encoder_model_name}", flush=True)

        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            torch.cuda.manual_seed_all(int(self.seed))

        print(f"[instantmesh] loading diffusion model: {self.diffusion_model}", flush=True)
        self.pipeline = Zero123PlusPipeline.from_pretrained(
            self.diffusion_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="trailing",
        )

        explicit_unet_path = bool(str(self.unet_path or "").strip())
        unet_path = self.unet_path or str(self.infer_config.get("unet_path", ""))
        unet_path = _resolve_checkpoint(unet_path, instantmesh_root)
        if os.path.exists(unet_path):
            unet_ckpt_path = unet_path
        elif explicit_unet_path:
            raise FileNotFoundError(f"InstantMesh UNet checkpoint not found: {unet_path}")
        else:
            unet_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="diffusion_pytorch_model.bin",
                repo_type="model",
            )
        state_dict = torch.load(unet_ckpt_path, map_location="cpu")
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline = self.pipeline.to(self.device)

        print("[instantmesh] loading reconstruction model", flush=True)
        self.model = instantiate_from_config(self.config.model_config)
        explicit_model_path = bool(str(self.model_path or "").strip())
        model_path = self.model_path or str(self.infer_config.get("model_path", ""))
        model_path = _resolve_checkpoint(model_path, instantmesh_root)
        if os.path.exists(model_path):
            model_ckpt_path = model_path
        elif explicit_model_path:
            raise FileNotFoundError(f"InstantMesh reconstruction checkpoint not found: {model_path}")
        else:
            model_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename=f"{self.config_name.replace('-', '_')}.ckpt",
                repo_type="model",
            )
        ckpt = torch.load(model_ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        if self.is_flexicubes:
            self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

        from src.utils.camera_util import get_zero123plus_input_cameras

        self.input_cameras = get_zero123plus_input_cameras(
            batch_size=1,
            radius=4.0 * float(self.scale),
        ).to(self.device)
        print("[instantmesh] reconstructor is ready", flush=True)

    def _prepare_input(self, image_rgba: Image.Image) -> Image.Image:
        if image_rgba.mode != "RGBA":
            image_rgba = image_rgba.convert("RGBA")
        return _resize_foreground_rgba(image_rgba, float(self.foreground_ratio))

    def reconstruct_part(self, image_rgba: Image.Image, out_obj_path: str):
        from src.utils.mesh_util import save_obj, save_obj_with_mtl

        os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
        image = self._prepare_input(image_rgba)
        with torch.no_grad():
            output_image = self.pipeline(
                image,
                num_inference_steps=int(self.diffusion_steps),
            ).images[0]

            images = np.asarray(output_image, dtype=np.float32) / 255.0
            images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
            images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
            images = images.unsqueeze(0).to(self.device)
            images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

            input_cameras = self.input_cameras
            if int(self.view) == 4:
                indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
                images = images[:, indices]
                input_cameras = input_cameras[:, indices]

            planes = self.model.forward_planes(images, input_cameras)
            mesh_out = self.model.extract_mesh(
                planes,
                use_texture_map=bool(self.export_texmap),
                **self.infer_config,
            )
            if self.export_texmap:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    out_obj_path,
                )
            else:
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, out_obj_path)
        return out_obj_path
