#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

conda activate sam3d
cd "${PROJ_ROOT}/sam-3d-objects"
cd diff-gaussian-rasterization
pip install -e . --no-build-isolation
cd ..
cd nvdiffrast
pip install -e . --no-build-isolation
cd ..
cd pytorch3d
pip install -e . --no-build-isolation
cd ..
cd "${PROJ_ROOT}"

conda activate hunyuan
cd "${PROJ_ROOT}/Hunyuan3D-2.1"
cd nvdiffrast
pip install -e . --no-build-isolation
cd ..
cd pytorch3d
pip install -e . --no-build-isolation
cd ..
cd "${PROJ_ROOT}"

conda activate instantmesh
cd "${PROJ_ROOT}/baseline/Any6D/instantmesh"
cd nvdiffrast
pip install -e . --no-build-isolation
cd ..
cd pytorch3d
pip install -e . --no-build-isolation
cd ..
cd "${PROJ_ROOT}"
