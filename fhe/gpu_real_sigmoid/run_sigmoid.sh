#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_MANIFEST=8d20a2841ee29a7a386171f1bd173b7189144359ce8f91ea5eb21cc60c0a78fe
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SIGMOID_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU GATE FIXTURE_DIR OUTPUT_JSON" >&2
  echo "GATE is one of: ln1, attention, full" >&2
  exit 2
fi

readonly GPU="$1"
readonly GATE="$2"
readonly FIXTURE_DIR="$3"
readonly OUTPUT="$4"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_sigmoid"

if [[ "${GATE}" != "ln1" && "${GATE}" != "attention" && "${GATE}" != "full" ]]; then
  echo "[FATAL] gate must be ln1, attention, or full" >&2
  exit 2
fi
if [[ "$(basename "${OUTPUT}")" != *"_sigmoid13_"* ]]; then
  echo "[FATAL] optimized evidence filename must contain _sigmoid13_" >&2
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

source_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_sigmoid.cpp" | cut -d' ' -f1)"
contract_sha="$(sha256sum "${SCRIPT_DIR}/fixture.sha256" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --gate "${GATE}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}" \
  --source-sha256 "${source_sha}" \
  --fixture-contract-sha256 "${contract_sha}"
