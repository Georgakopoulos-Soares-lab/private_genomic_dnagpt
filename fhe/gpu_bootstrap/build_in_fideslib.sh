#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_PATCH_SHA=81b6f6d8f466c67bc4e764c3f43d38ad2f6caec44b807e8fb916fc8b1063e295
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REFRESH_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly JOBS="${BUILD_JOBS:-2}"
readonly PATCH_FILE="${SCRIPT_DIR}/../../docker/patches/fideslib-asymmetric-chebyshev.patch"

if [[ ! -d "${FIDES_ROOT}/.git" ]]; then
  echo "[FATAL] FIDESlib checkout not found at ${FIDES_ROOT}" >&2
  exit 2
fi

actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi
actual_patch_sha="$(sha256sum "${PATCH_FILE}" | cut -d' ' -f1)"
if [[ "${actual_patch_sha}" != "${EXPECTED_PATCH_SHA}" ]]; then
  echo "[FATAL] backend patch SHA ${actual_patch_sha}; expected ${EXPECTED_PATCH_SHA}" >&2
  exit 2
fi
if ! git -C "${FIDES_ROOT}" apply --reverse --check "${PATCH_FILE}"; then
  echo "[FATAL] exact repository backend patch is not applied to ${FIDES_ROOT}" >&2
  exit 2
fi

cmake \
  -S "${SCRIPT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFIDESLIB_PACKAGE_DIR="${FIDESLIB_PACKAGE_DIR:-/usr/local/share/fideslib/cmake}" \
  -DCMAKE_CUDA_ARCHITECTURES="${FIDESLIB_ARCH:-80-real}"
cmake --build "${BUILD_DIR}" --target dnagpt_fides_refresh --parallel "${JOBS}"

echo "[build] ${BUILD_DIR}/dnagpt_fides_refresh"
