#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_MANIFEST=3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c
readonly EXPECTED_PARENT_SOURCE=d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df
readonly EXPECTED_DEPTH12_SOURCE=234e3659eb73685619fede1b95dcf637c72f5b504e54a3631dfb1ab4ce0d5397
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU FIXTURE_DIR OUTPUT_JSON" >&2
  echo "OUTPUT_JSON filename must contain _scheme_b_, _two_block_refresh_, and _depth12_" >&2
  exit 2
fi

readonly GPU="$1"
readonly FIXTURE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_scheme_b_two_block_refresh_depth12"

if [[ "$(basename "${OUTPUT}")" != *"_scheme_b_"* ]] || \
   [[ "$(basename "${OUTPUT}")" != *"_two_block_refresh_"* ]] || \
   [[ "$(basename "${OUTPUT}")" != *"_depth12_"* ]]; then
  echo "[FATAL] depth-12 evidence filename is missing a required identity token" >&2
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
  sha256sum --check --status "${SCRIPT_DIR}/fixture_two_block_t2.sha256"
)
manifest_sha="$(sha256sum "${FIXTURE_DIR}/manifest.json" | cut -d' ' -f1)"
if [[ "${manifest_sha}" != "${EXPECTED_MANIFEST}" ]]; then
  echo "[FATAL] fixture manifest ${manifest_sha}; expected ${EXPECTED_MANIFEST}" >&2
  exit 2
fi

parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b.cpp" | cut -d' ' -f1)"
if [[ "${parent_sha}" != "${EXPECTED_PARENT_SOURCE}" ]]; then
  echo "[FATAL] frozen Scheme B parent ${parent_sha}; expected ${EXPECTED_PARENT_SOURCE}" >&2
  exit 2
fi
depth12_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_depth12.cpp" | cut -d' ' -f1)"
if [[ "${depth12_sha}" != "${EXPECTED_DEPTH12_SOURCE}" ]]; then
  echo "[FATAL] depth-12 single-block source ${depth12_sha}; expected ${EXPECTED_DEPTH12_SOURCE}" >&2
  exit 2
fi

source_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_two_block_refresh_depth12.cpp" | cut -d' ' -f1)"
contract_sha="$(sha256sum "${SCRIPT_DIR}/fixture_two_block_t2.sha256" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}" \
  --source-sha256 "${source_sha}" \
  --frozen-source-sha256 "${depth12_sha}" \
  --fixture-contract-sha256 "${contract_sha}"
