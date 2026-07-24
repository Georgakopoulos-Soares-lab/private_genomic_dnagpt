#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_PATCH_SHA=81b6f6d8f466c67bc4e764c3f43d38ad2f6caec44b807e8fb916fc8b1063e295
readonly EXPECTED_ACTIVATION_SHA=829b2b4b62b6315f8d93a33562cb4a4d82935c363b4ebfa9a56a8dd29edfe63f
readonly EXPECTED_BYTES=12288
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REFRESH_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly PATCH_FILE="${SCRIPT_DIR}/../../docker/patches/fideslib-asymmetric-chebyshev.patch"

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU MODE ACTIVATION_F64 OUTPUT_JSON" >&2
  echo "MODE: native-gpu | cpu-iterative-interop" >&2
  exit 2
fi

readonly GPU="$1"
readonly MODE="$2"
readonly ACTIVATION="$3"
readonly OUTPUT="$4"
readonly BINARY="${BUILD_DIR}/dnagpt_fides_refresh"

if [[ "${MODE}" != "native-gpu" && "${MODE}" != "cpu-iterative-interop" ]]; then
  echo "[FATAL] unsupported mode: ${MODE}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] ${BINARY} missing; run build_in_fideslib.sh first" >&2
  exit 2
fi
if [[ ! -f "${ACTIVATION}" ]]; then
  echo "[FATAL] activation fixture missing: ${ACTIVATION}" >&2
  exit 2
fi
if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "[FATAL] repository FIDESlib patch missing: ${PATCH_FILE}" >&2
  exit 2
fi
actual_bytes="$(wc -c < "${ACTIVATION}" | tr -d ' ')"
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" ]]; then
  echo "[FATAL] activation fixture is ${actual_bytes} bytes; expected ${EXPECTED_BYTES}" >&2
  exit 2
fi

actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi

source_sha="$(sha256sum "${SCRIPT_DIR}/src/dnagpt_fides_refresh.cpp" | cut -d' ' -f1)"
activation_sha="$(sha256sum "${ACTIVATION}" | cut -d' ' -f1)"
patch_sha="$(sha256sum "${PATCH_FILE}" | cut -d' ' -f1)"
if [[ "${activation_sha}" != "${EXPECTED_ACTIVATION_SHA}" ]]; then
  echo "[FATAL] activation SHA ${activation_sha}; expected ${EXPECTED_ACTIVATION_SHA}" >&2
  exit 2
fi
if [[ "${patch_sha}" != "${EXPECTED_PATCH_SHA}" ]]; then
  echo "[FATAL] backend patch SHA ${patch_sha}; expected ${EXPECTED_PATCH_SHA}" >&2
  exit 2
fi
if ! git -C "${FIDES_ROOT}" apply --reverse --check "${PATCH_FILE}"; then
  echo "[FATAL] exact repository backend patch is not applied to ${FIDES_ROOT}" >&2
  exit 2
fi

exec "${BINARY}" \
  --gpu "${GPU}" \
  --mode "${MODE}" \
  --activation "${ACTIVATION}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}" \
  --source-sha256 "${source_sha}" \
  --activation-sha256 "${activation_sha}" \
  --backend-patch-sha256 "${patch_sha}"
