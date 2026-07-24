#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_TOY_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU OUTPUT_JSON" >&2
  exit 2
fi

readonly GPU="$1"
readonly OUTPUT="$2"
readonly BINARY="${BUILD_DIR}/toy_dnagpt_fides"

if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] ${BINARY} missing; run build_in_fideslib.sh first" >&2
  exit 2
fi

actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi

source_sha="$(sha256sum "${SCRIPT_DIR}/src/toy_dnagpt_fides.cpp" | cut -d' ' -f1)"
fixture_sha="$(sha256sum "${SCRIPT_DIR}/src/toy_fixture.hpp" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}" \
  --source-sha256 "${source_sha}" \
  --fixture-sha256 "${fixture_sha}"
