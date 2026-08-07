#!/usr/bin/env bash
# Turn a saved docker image of the pinned FIDESlib environment into
# kimon/env/fideslib.sif (apptainer run mode).
#
#   kimon/env/build_sif.sh archive /path/to/fideslib.tar   # docker save output
#   kimon/env/build_sif.sh daemon  dnagpt-fideslib:786c-asymfix2
#
# READ THIS BEFORE USING IT ON LONESTAR6
# --------------------------------------
# docker/Dockerfile.fideslib is FROM nvidia/cuda:13.0.1-...  CUDA 13.0 requires
# an NVIDIA driver >= 580. Lonestar6's gpu-a100 nodes run driver 570.x, which
# tops out at CUDA 12.8, so kernels from a CUDA-13 image cannot launch there
# (`CUDA driver version is insufficient for CUDA runtime version`) unless the
# image also ships the cuda-compat-13-0 forward-compatibility package.
# On TACC use the native path instead:
#   kimon/env/clone_fideslib.sh && kimon/env/build_fideslib_native.sh
#   RUN_MODE=native kimon/env/build_binaries.sh
# and pass RUN_MODE=native to every run.
#
# This script therefore exists for hosts whose driver is >= 580, and to verify a
# received fideslib.tar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly SIF="${KIMON_SIF:-${SCRIPT_DIR}/fideslib.sif}"
readonly MODE="${1:-}"
readonly SOURCE="${2:-}"

if [[ -z "${MODE}" || -z "${SOURCE}" ]]; then
  echo "usage: $0 {archive|daemon} {TAR_PATH|DOCKER_TAG}" >&2
  exit 2
fi
if [[ -e "${SIF}" ]]; then
  echo "[FATAL] refusing to overwrite ${SIF}; delete it first" >&2
  exit 2
fi
command -v apptainer >/dev/null 2>&1 || {
  echo "[FATAL] apptainer not on PATH (module load tacc-apptainer)" >&2; exit 2; }

case "${MODE}" in
  archive)
    [[ -f "${SOURCE}" ]] || { echo "[FATAL] no such tar: ${SOURCE}" >&2; exit 2; }
    apptainer build "${SIF}" "docker-archive://${SOURCE}"
    ;;
  daemon)
    command -v docker >/dev/null 2>&1 || {
      echo "[FATAL] docker not available; use the archive mode" >&2; exit 2; }
    apptainer build "${SIF}" "docker-daemon://${SOURCE}"
    ;;
  *)
    echo "[FATAL] mode must be archive or daemon" >&2; exit 2 ;;
esac

echo "[ok] ${SIF}"
echo "[check] FIDESlib commit inside the image:"
apptainer exec "${SIF}" git -C /opt/FIDESlib rev-parse HEAD
echo "[check] CUDA runtime vs this host's driver:"
apptainer exec --nv "${SIF}" nvidia-smi --query-gpu=driver_version --format=csv,noheader || true
apptainer exec "${SIF}" bash -lc 'nvcc --version | tail -2' || true
