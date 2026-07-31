#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_SOURCE=738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235
readonly EXPECTED_PARENT=65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365
readonly EXPECTED_FROZEN=70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f
readonly EXPECTED_SCHEDULE=c6b221f365ba6326f615c5554458d7bd092990d23c4ba0d5106ca7577cb7c3aa
readonly EXPECTED_MANIFEST=d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6
readonly EXPECTED_FIXTURE_CONTRACT=061d53bd25bbcaf75d4c12067ec77f032c3ba0e35300a0a6168c2ce15f3672db
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU FIXTURE_DIR OUTPUT_JSON" >&2
  exit 2
fi

readonly GPU="$1"
readonly FIXTURE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536"

if [[ "$(basename "${OUTPUT}")" != *"_scheme_b_"* ]] || \
   [[ "$(basename "${OUTPUT}")" != *"_simd_full_t103_depth9_digits3_ring65536_"* ]]; then
  echo "[FATAL] depth-9/digits-3/ring-65536 SIMD evidence filename must contain _scheme_b_ and _simd_full_t103_depth9_digits3_ring65536_" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] ${BINARY} missing; build the SIMD target first" >&2
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
  sha256sum --check --status "${SCRIPT_DIR}/fixture_t103.sha256"
)
manifest_sha="$(sha256sum "${FIXTURE_DIR}/manifest.json" | cut -d' ' -f1)"
if [[ "${manifest_sha}" != "${EXPECTED_MANIFEST}" ]]; then
  echo "[FATAL] fixture manifest ${manifest_sha}; expected ${EXPECTED_MANIFEST}" >&2
  exit 2
fi
source_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536.cpp" | cut -d' ' -f1)"
if [[ "${source_sha}" != "${EXPECTED_SOURCE}" ]]; then
  echo "[FATAL] depth-9/digits-3/ring-65536 full SIMD source ${source_sha}; expected ${EXPECTED_SOURCE}" >&2
  exit 2
fi
parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536.cpp" | cut -d' ' -f1)"
if [[ "${parent_sha}" != "${EXPECTED_PARENT}" ]]; then
  echo "[FATAL] frozen parent ${parent_sha}; expected ${EXPECTED_PARENT}" >&2
  exit 2
fi
frozen_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_general_attention_t103.cpp" | cut -d' ' -f1)"
if [[ "${frozen_sha}" != "${EXPECTED_FROZEN}" ]]; then
  echo "[FATAL] frozen T103 source ${frozen_sha}; expected ${EXPECTED_FROZEN}" >&2
  exit 2
fi
schedule_sha="$(sha256sum "${SCRIPT_DIR}/src/simd_t103_schedule.hpp" | cut -d' ' -f1)"
if [[ "${schedule_sha}" != "${EXPECTED_SCHEDULE}" ]]; then
  echo "[FATAL] SIMD schedule ${schedule_sha}; expected ${EXPECTED_SCHEDULE}" >&2
  exit 2
fi
contract_sha="$(sha256sum "${SCRIPT_DIR}/fixture_t103.sha256" | cut -d' ' -f1)"
if [[ "${contract_sha}" != "${EXPECTED_FIXTURE_CONTRACT}" ]]; then
  echo "[FATAL] fixture contract ${contract_sha}; expected ${EXPECTED_FIXTURE_CONTRACT}" >&2
  exit 2
fi
exec "${BINARY}" \
  --gpu "${GPU}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --parent-source-sha256 "${parent_sha}" \
  --frozen-source-sha256 "${frozen_sha}" \
  --schedule-sha256 "${schedule_sha}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --fixture-contract-sha256 "${contract_sha}" \
  --source-sha256 "${source_sha}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}"
