from pathlib import Path
import sys
import copy

RECON_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = RECON_ROOT / "tools"
for _p in (RECON_ROOT, TOOLS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse

from recon_dmesh_common import add_dmesh_args, run_dmesh_object
from recon_part_postprocess_common import add_partcut_args, run_partcut_object
from recon_tsdf_common import add_tsdf_args, run_tsdf_object
from recon_utils import add_common_args, run_object_pipeline
from run.recon_instantmesh import add_instantmesh_args


METHOD = "instantmesh_partcut_tsdf_dmesh"
BASE_METHOD = "instantmesh"
PARTCUT_METHOD = "instantmesh_partcut"
TSDF_METHOD = "instantmesh_partcut_tsdf"


def _stage_args(args: argparse.Namespace, method: str) -> argparse.Namespace:
    out = copy.copy(args)
    # This pipeline must reuse existing base reconstruction results. Do not let
    # the TSDF helper try to build the synthetic partcut cache as a base method.
    out.build_base_if_missing = False
    if str(getattr(args, "coord_dir", "") or "").strip():
        out.coord_dir = str(Path(args.coord_dir).resolve() / method)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run InstantMesh base cache -> reference-only part cut -> TSDF -> DLMesh.",
        conflict_handler="resolve",
    )
    add_common_args(parser, METHOD)
    add_partcut_args(parser)
    add_tsdf_args(parser)
    add_dmesh_args(parser)
    # Kept for CLI compatibility with existing InstantMesh commands. This new
    # pipeline only reuses the existing instantmesh cache and does not rebuild it.
    add_instantmesh_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[stage 1/3] reference-only part cut from InstantMesh base results")
    run_object_pipeline(
        _stage_args(args, PARTCUT_METHOD),
        PARTCUT_METHOD,
        lambda obj, stage_args: run_partcut_object(obj, stage_args, BASE_METHOD, PARTCUT_METHOD),
    )
    print("[stage 2/3] TSDF refinement from part-cut InstantMesh results")
    run_object_pipeline(
        _stage_args(args, TSDF_METHOD),
        TSDF_METHOD,
        lambda obj, stage_args: run_tsdf_object(obj, stage_args, PARTCUT_METHOD, TSDF_METHOD),
    )
    print("[stage 3/3] DLMesh refinement from part-cut TSDF results")
    run_object_pipeline(
        _stage_args(args, METHOD),
        METHOD,
        lambda obj, stage_args: run_dmesh_object(obj, stage_args, TSDF_METHOD, METHOD),
    )


if __name__ == "__main__":
    main()
