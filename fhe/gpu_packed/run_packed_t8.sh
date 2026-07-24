#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_MANIFEST=75ce745757a8141df42054612d54a0837b5f0c2e7a771cafd57fc5055e054162
readonly EXPECTED_PUBLIC_CONTRACT=a913d3b8e3365525279b313a3dfb8ae863a6f7c0bf1c6d222e8f6886a853a0f4
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_PACKED_T8_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU FIXTURE_DIR OUTPUT_JSON" >&2
  exit 2
fi
readonly GPU="$1"
readonly FIXTURE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/dnagpt_packed_t8_fides"

if [[ "$(basename "${OUTPUT}")" != *"_packed_t8_"* ]]; then
  echo "[FATAL] evidence filename must contain _packed_t8_" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable output: ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] binary missing: ${BINARY}" >&2
  exit 2
fi
if [[ "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" == "UNSPECIFIED" ]]; then
  echo "[FATAL] immutable FIDES_CONTAINER_IMAGE identity is required" >&2
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
contract_sha="$(sha256sum "${SCRIPT_DIR}/public_t8_contract.json" | cut -d' ' -f1)"
if [[ "${manifest_sha}" != "${EXPECTED_MANIFEST}" ||
      "${contract_sha}" != "${EXPECTED_PUBLIC_CONTRACT}" ]]; then
  echo "[FATAL] fixture manifest or public contract identity mismatch" >&2
  exit 2
fi
source_sha="$(sha256sum "${SCRIPT_DIR}/src/dnagpt_packed_t8_fides.cpp" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --public-contract-sha256 "${contract_sha}" \
  --source-sha256 "${source_sha}" \
  --container-image "${FIDES_CONTAINER_IMAGE}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}"
