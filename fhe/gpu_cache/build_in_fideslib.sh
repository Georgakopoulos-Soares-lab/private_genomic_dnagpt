#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_CACHE_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly JOBS="${BUILD_JOBS:-2}"

if [[ ! -d "${FIDES_ROOT}/.git" ]]; then
  echo "[FATAL] FIDESlib git checkout not found at ${FIDES_ROOT}" >&2
  exit 2
fi
actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi

cmake \
  -S "${SCRIPT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFIDESLIB_PACKAGE_DIR="${FIDESLIB_PACKAGE_DIR:-/usr/local/share/fideslib/cmake}" \
  -DFIDESLIB_ARCH="${FIDESLIB_ARCH:-80-real}" \
  -DCMAKE_CUDA_ARCHITECTURES="${FIDESLIB_ARCH:-80-real}"
cmake --build "${BUILD_DIR}" --target dnagpt_fides_cache --parallel "${JOBS}"

echo "[build] ${BUILD_DIR}/dnagpt_fides_cache"
