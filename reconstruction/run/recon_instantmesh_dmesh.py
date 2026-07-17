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
from run.recon_instantmesh import add_instantmesh_args


METHOD = "instantmesh_dmesh"
BASE_METHOD = "instantmesh"


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace):
    return run_dmesh_object(obj, args, BASE_METHOD, METHOD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run InstantMesh + legacy dmesh-branch DLMesh refinement using shared InstantMesh cache.")
    add_common_args(parser, METHOD)
    add_dmesh_args(parser)
    add_instantmesh_args(parser)
    return parser.parse_args()


def main() -> None:
    run_object_pipeline(parse_args(), METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/run/recon_instantmesh_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
#
# Key parameters:
#   --work-root: shared cache root. Reads <work-root>/instantmesh and writes <work-root>/instantmesh_dmesh.
#   --build-base-if-missing: run InstantMesh first if its shared cache is absent.
#   --instantmesh-*: base reconstruction parameters used when building the cache.
#   --dlmesh-*: controls remesh + PyTorch3D pose optimization + DLMesh geometry refinement.
