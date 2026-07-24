#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_MANIFEST=3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c
readonly EXPECTED_RANGE_CONTROL=b148e30b42c430405a0a2c41401ab295701e0e1c3655ef00f06984ff9f8faba2
readonly EXPECTED_IMAGE_EVIDENCE='dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4'
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_MULTIBLOCK_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly RANGE_CONTROL_FILE="${FIDES_RANGE_CONTROL_FILE:-${SCRIPT_DIR}/../range_control/results/range_control_public_fixture_optimized_v2_PASS.json}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU FIXTURE_DIR OUTPUT_JSON" >&2
  exit 2
fi

readonly GPU="$1"
readonly FIXTURE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/dnagpt_two_block_fides"
readonly IMAGE_EVIDENCE="${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}"

if [[ "$(basename "${OUTPUT}")" != *"_blocks0_1_refresh_"* ]]; then
  echo "[FATAL] two-block evidence filename must contain _blocks0_1_refresh_" >&2
  exit 2
fi
if [[ "${IMAGE_EVIDENCE}" != "${EXPECTED_IMAGE_EVIDENCE}" ]]; then
  echo "[FATAL] container evidence must be ${EXPECTED_IMAGE_EVIDENCE}" >&2
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
if [[ ! -f "${FIXTURE_DIR}/manifest.json" ]]; then
  echo "[FATAL] fixture manifest missing: ${FIXTURE_DIR}/manifest.json" >&2
  exit 2
fi

actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi

(
  cd "${FIXTURE_DIR}"
  sha256sum --check --status "${SCRIPT_DIR}/fixture.sha256"
)
manifest_sha="$(sha256sum "${FIXTURE_DIR}/manifest.json" | cut -d' ' -f1)"
if [[ "${manifest_sha}" != "${EXPECTED_MANIFEST}" ]]; then
  echo "[FATAL] fixture manifest ${manifest_sha}; expected ${EXPECTED_MANIFEST}" >&2
  exit 2
fi
range_control_sha="$(sha256sum "${RANGE_CONTROL_FILE}" | cut -d' ' -f1)"
if [[ "${range_control_sha}" != "${EXPECTED_RANGE_CONTROL}" ]]; then
  echo "[FATAL] range-control contract ${range_control_sha}; expected ${EXPECTED_RANGE_CONTROL}" >&2
  exit 2
fi

source_sha="$(sha256sum "${SCRIPT_DIR}/src/dnagpt_two_block_fides.cpp" | cut -d' ' -f1)"
contract_sha="$(sha256sum "${SCRIPT_DIR}/fixture.sha256" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --range-control-sha256 "${range_control_sha}" \
  --container-image "${IMAGE_EVIDENCE}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}" \
  --source-sha256 "${source_sha}" \
  --fixture-contract-sha256 "${contract_sha}"
