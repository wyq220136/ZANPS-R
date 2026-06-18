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
from recon_tsdf_common import add_tsdf_args, run_tsdf_object
from recon_utils import DatasetObject, add_common_args, run_object_pipeline


METHOD = "sam3d_tsdf_dmesh"
BASE_METHOD = "sam3d"
TSDF_METHOD = "sam3d_tsdf"


def reconstruct_object(obj: DatasetObject, args: argparse.Namespace):
    tsdf_summary = run_tsdf_object(obj, args, BASE_METHOD, TSDF_METHOD)
    dmesh_summary = run_dmesh_object(obj, args, TSDF_METHOD, METHOD)
    dmesh_summary["tsdf_stage"] = tsdf_summary
    return dmesh_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run SAM3D + TSDF fusion + legacy dmesh-branch DLMesh refinement using shared caches.",
        conflict_handler="resolve",
    )
    add_common_args(parser, METHOD)
    add_tsdf_args(parser)
    add_dmesh_args(parser)
    return parser.parse_args()


def main() -> None:
    run_object_pipeline(parse_args(), METHOD, reconstruct_object)


if __name__ == "__main__":
    main()


# Usage:
#   python reconstruction/recon_sam3d_tsdf_dmesh.py --data-root dataset_train --split val --work-root reconstruction_runs --objects bottle_3517 --gpus 0 --num-workers 1
#   python reconstruction/recon_sam3d_tsdf_dmesh.py --data-root /data/dataset_train --split val --work-root /shared/recon_runs --object-source all --gpus 0,1 --num-workers 2 --mode multi_image --coord-dir /shared/recon_coord/sam3d_tsdf_dmesh --reset-coord
#
# Pipeline:
#   1. Reads/builds <work-root>/sam3d.
#   2. Writes/reuses <work-root>/sam3d_tsdf.
#   3. Runs remesh + PyTorch3D pose optimization + DLMesh refinement from <work-root>/sam3d_tsdf into <work-root>/sam3d_tsdf_dmesh.
#
# Key parameters:
#   --work-root: shared cache root for all stages.
#   --build-base-if-missing: run SAM3D first if its shared cache is absent before TSDF.
#   --voxel-length/--sdf-trunc: TSDF resolution and truncation.
#   --dlmesh-*: controls remesh + PyTorch3D pose optimization + DLMesh geometry refinement.
#   --dmesh-*: deprecated aliases retained for old commands; the real DMesh repo is not called.
#   --copy-base-as-placeholder: dry-run only; explicitly copies TSDF mesh without DLMesh optimization.
