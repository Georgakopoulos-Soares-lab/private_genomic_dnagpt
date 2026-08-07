#!/usr/bin/env bash
# Build + install the pinned FIDESlib (and its patched OpenFHE) natively.
#
# Why native instead of the docker/apptainer image: docker/Dockerfile.fideslib
# is based on CUDA 13.0, which requires an NVIDIA driver >= 580. Lonestar6's
# gpu-a100 nodes run driver 570.x (CUDA 12.8), so a CUDA-13 image cannot launch
# kernels there. This script keeps every scientifically load-bearing pin --
# FIDESlib commit 786c7600 and docker/patches/fideslib-asymmetric-chebyshev.patch
# -- and changes only the CUDA toolkit to the host-matched 12.8.
#
# Usage (from the repo root, inside a GPU node or a build node):
#   source kimon/env/tacc_env.sh
#   kimon/env/build_fideslib_native.sh
#
# Env knobs: FIDESLIB_ROOT, FIDESLIB_PREFIX, FIDESLIB_ARCH (default 80 = A100),
#            BUILD_JOBS.
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly SRC="${FIDESLIB_ROOT:-${SCRIPT_DIR}/fideslib-src}"
readonly PREFIX="${FIDESLIB_PREFIX:-${SCRIPT_DIR}/fideslib-install}"
readonly PATCH="${REPO}/docker/patches/fideslib-asymmetric-chebyshev.patch"
# NOTE: TACC exports OMP_NUM_THREADS=1, and GNU coreutils `nproc` honours it, so
# `nproc` reports 1 on a 32-core node. Use the allocation's real core count.
_cores="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
readonly JOBS="${BUILD_JOBS:-$(( _cores > 24 ? 24 : _cores ))}"
# FIDESlib's own CMake takes a bare SM number (80), unlike the DNAGPT targets
# which accept CMake's "80-real" form.
readonly ARCH_BARE="${FIDESLIB_ARCH_BARE:-80}"

if [[ ! -d "${SRC}/.git" ]]; then
  echo "[FATAL] no FIDESlib checkout at ${SRC}; run kimon/env/clone_fideslib.sh first" >&2
  exit 2
fi
actual="$(git -C "${SRC}" rev-parse HEAD)"
if [[ "${actual}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[FATAL] nvcc not on PATH; source kimon/env/tacc_env.sh first" >&2
  exit 2
fi

# --- the asymmetric-Chebyshev correction (required; see docker/README.md) ----
if git -C "${SRC}" apply --check "${PATCH}" 2>/dev/null; then
  git -C "${SRC}" apply "${PATCH}"
  echo "[patch] applied fideslib-asymmetric-chebyshev.patch"
elif git -C "${SRC}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "[patch] already applied"
else
  echo "[FATAL] patch neither applies nor is applied; checkout is dirty" >&2
  exit 2
fi

# FIDESlib's CMakeLists hardcodes CUDA_PATH=/usr/local/cuda when undefined, so
# the module's toolkit root has to be passed explicitly.
CUDA_PATH="${CUDA_PATH:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "[cuda] CUDA_PATH=${CUDA_PATH}"

# --- step 1: OpenFHE (the FIDESlib-patched 1.5.1 fork) ----------------------
# Upstream drives this from CMake via
#   execute_process(COMMAND ./build.sh ... WORKING_DIRECTORY "../deps")
# guarded by -DFIDESLIB_INSTALL_OPENFHE=ON. Under CMake 4.x that relative
# COMMAND is resolved against the cmake process's cwd rather than
# WORKING_DIRECTORY and fails with "no such file or directory", so we invoke the
# same deps/build.sh directly and then configure FIDESlib with
# FIDESLIB_INSTALL_OPENFHE=OFF. Same script, same pinned OpenFHE tag
# (fideslib-ref-v1.5.1.1) and same deps/fideslib-ref-1.5.1.1.patch.
if [[ -f "${PREFIX}/lib/OpenFHE/OpenFHEConfig.cmake" ]] ||
   [[ -f "${PREFIX}/lib/cmake/OpenFHE/OpenFHEConfig.cmake" ]]; then
  echo "[openfhe] already installed under ${PREFIX}"
else
  echo "[openfhe] building via deps/build.sh -> ${PREFIX}"
  ( cd "${SRC}/deps" && ./build.sh "${PREFIX}" )
fi

# --- step 2: FIDESlib -------------------------------------------------------
# FIDESlib overrides CMAKE_INSTALL_PREFIX with its own FIDESLIB_INSTALL_PREFIX
# (CMakeLists.txt:181), so setting CMAKE_INSTALL_PREFIX alone is ignored.
rm -f "${SRC}/build/CMakeCache.txt"
cmake -S "${SRC}" -B "${SRC}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_PATH="${CUDA_PATH}" \
  -DFIDESLIB_INSTALL_PREFIX="${PREFIX}" \
  -DOPENFHE_INSTALL_PREFIX="${PREFIX}" \
  -DFIDESLIB_ARCH="${ARCH_BARE}" \
  -DFIDESLIB_INSTALL_OPENFHE=OFF \
  -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
  -DFIDESLIB_COMPILE_TESTS=OFF
cmake --build "${SRC}/build" -j "${JOBS}"
cmake --install "${SRC}/build"

echo "[ok] fideslib installed under ${PREFIX}"
ls -la "${PREFIX}/share/fideslib/cmake" 2>/dev/null || true
