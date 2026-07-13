#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf


def _walk(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(item, name)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk(item, f"{prefix}[{idx}]")
    else:
        yield prefix, value


def _looks_like_checkpoint(value):
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return any(token in lower for token in (".pth", ".pt", ".ckpt", ".safetensors"))


def _resolve_path(raw, config_path):
    if not isinstance(raw, str) or not raw:
        return None
    expanded = os.path.expanduser(os.path.expandvars(raw))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (Path(config_path).parent / path).resolve()


def inspect_config(path):
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=False)
    refs = []
    arch_hints = []
    for key, value in _walk(data):
        if _looks_like_checkpoint(value):
            resolved = _resolve_path(value, path)
            refs.append((key, value, resolved, bool(resolved and resolved.exists())))
        if isinstance(value, str):
            lower = value.lower()
            if any(token in lower for token in ("vitb", "vitl", "vith", "dinov2", "embedder", "fuser")):
                arch_hints.append((key, value))
        elif isinstance(value, int) and key.split(".")[-1] in {"embed_dim", "hidden_size", "num_layers", "depth"}:
            arch_hints.append((key, value))
    return refs, arch_hints


def main():
    parser = argparse.ArgumentParser("List SAM3D pipeline configs and referenced checkpoint files.")
    parser.add_argument(
        "--sam3d-project-root",
        default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/sam-3d-objects",
    )
    args = parser.parse_args()

    root = Path(args.sam3d_project_root).expanduser().resolve()
    configs = sorted(root.glob("checkpoints/*/pipeline.yaml"))
    if not configs:
        raise FileNotFoundError(f"No pipeline.yaml found under {root / 'checkpoints'}")

    for config in configs:
        print(f"\n[config] {config}")
        try:
            refs, arch_hints = inspect_config(config)
        except Exception as exc:
            print(f"  [error] failed to load: {exc}")
            continue
        if arch_hints:
            print("  [arch hints]")
            for key, value in arch_hints[:20]:
                print(f"    {key}: {value}")
        if refs:
            print("  [checkpoint refs]")
            for key, raw, resolved, exists in refs:
                status = "exists" if exists else "missing"
                print(f"    {key}: {raw}")
                print(f"      -> {resolved} ({status})")
        else:
            print("  [checkpoint refs] none found")


if __name__ == "__main__":
    main()
