from pathlib import Path
import sys

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse

from reconstruct_hunyuan3d import default_hunyuan_model_path
from recon_dmesh_common import add_dmesh_args, run_dmesh_object
from recon_tsdf_common import add_tsdf_args, run_tsdf_object
from recon_utils import DatasetObject, add_common_args, run_object_pipeline


METHOD = "hunyuan3d_tsdf_dmesh"
BASE_METHOD = "hunyuan3d"
TSDF_METHOD = "hunyuan3d_tsdf"


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace):
    tsdf_summary = run_tsdf_object(obj, args, BASE_METHOD, TSDF_METHOD)
    dmesh_summary = run_dmesh_object(obj, args, TSDF_METHOD, METHOD)
    dmesh_summary["tsdf_stage"] = tsdf_summary
    return dmesh_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run Hunyuan3D + TSDF fusion + legacy dmesh-branch DLMesh refinement using shared caches.",
        conflict_handler="resolve",
    )
    add_common_args(parser, METHOD)
    add_tsdf_args(parser)
    add_dmesh_args(parser)
    parser.add_argument("--model-path", type=str, default=default_hunyuan_model_path())
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--alignment-samples", type=int, default=50000)
    parser.add_argument("--alignment-seed", type=int, default=2026)
    parser.add_argument("--min-alignment-points", type=int, default=200)
    parser.add_argument("--alignment-icp-iters", type=int, default=30)
    parser.add_argument("--alignment-trim-quantile", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    run_object_pipeline(parse_args(), METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/recon_hunyuan3d_tsdf_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
#   python reconstruction/recon_hunyuan3d_tsdf_dmesh.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --gpus 0,1 --num-workers 2 --mode multi_image --coord-dir /shared/recon_coord/hunyuan3d_tsdf_dmesh --reset-coord
#
# Pipeline:
#   1. Reads/builds <work-root>/hunyuan3d.
#   2. Writes/reuses <work-root>/hunyuan3d_tsdf.
#   3. Runs remesh + PyTorch3D pose optimization + DLMesh refinement from <work-root>/hunyuan3d_tsdf into <work-root>/hunyuan3d_tsdf_dmesh.
#
# Key parameters:
#   --work-root: shared cache root for all stages.
#   --build-base-if-missing: run Hunyuan3D first if its shared cache is absent before TSDF.
#   --voxel-length/--sdf-trunc: TSDF resolution and truncation.
#   --dlmesh-*: controls remesh + PyTorch3D pose optimization + DLMesh geometry refinement.
#   --dmesh-*: deprecated aliases retained for old commands; the real DMesh repo is not called.
#   --copy-base-as-placeholder: dry-run only; explicitly copies TSDF mesh without DLMesh optimization.
