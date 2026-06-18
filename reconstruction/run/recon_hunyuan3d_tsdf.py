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
from recon_tsdf_common import add_tsdf_args, run_tsdf_object
from recon_utils import DatasetObject, add_common_args, run_object_pipeline


METHOD = "hunyuan3d_tsdf"
BASE_METHOD = "hunyuan3d"


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace):
    return run_tsdf_object(obj, args, BASE_METHOD, METHOD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Hunyuan3D + TSDF fusion using shared Hunyuan3D cache.")
    add_common_args(parser, METHOD)
    add_tsdf_args(parser)
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
#   python reconstruction/recon_hunyuan3d_tsdf.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --num-workers 4
#   python reconstruction/recon_hunyuan3d_tsdf.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --num-workers 16 --mode multi_image --coord-dir /shared/recon_coord/hunyuan3d_tsdf --reset-coord
#
# Key parameters:
#   --work-root: shared cache root. Reads <work-root>/hunyuan3d and writes <work-root>/hunyuan3d_tsdf.
#   --build-base-if-missing: run Hunyuan3D first if its shared cache is absent.
#   --voxel-length/--sdf-trunc: TSDF resolution and truncation.
