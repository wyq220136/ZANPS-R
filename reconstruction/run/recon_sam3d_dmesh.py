from pathlib import Path
import sys

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse

from recon_dmesh_common import add_dmesh_args, run_dmesh_object
from recon_utils import DatasetObject, add_common_args, run_object_pipeline


METHOD = "sam3d_dmesh"
BASE_METHOD = "sam3d"


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace):
    return run_dmesh_object(obj, args, BASE_METHOD, METHOD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run SAM3D + legacy dmesh-branch DLMesh refinement using shared SAM3D cache.")
    add_common_args(parser, METHOD)
    add_dmesh_args(parser)
    return parser.parse_args()


def main() -> None:
    run_object_pipeline(parse_args(), METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/recon_sam3d_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
#   python reconstruction/recon_sam3d_dmesh.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --gpus 0,1 --num-workers 2 --mode multi_image --coord-dir /shared/recon_coord/sam3d_dmesh --reset-coord
#
# Key parameters:
#   --work-root: shared cache root. Reads <work-root>/sam3d and writes <work-root>/sam3d_dmesh.
#   --build-base-if-missing: run SAM3D first if its shared cache is absent.
#   --dlmesh-*: controls remesh + PyTorch3D pose optimization + DLMesh geometry refinement.
#   --dmesh-*: deprecated aliases retained for old commands; the real DMesh repo is not called.
#   --copy-base-as-placeholder: dry-run only; explicitly copies SAM3D mesh without DLMesh optimization.
