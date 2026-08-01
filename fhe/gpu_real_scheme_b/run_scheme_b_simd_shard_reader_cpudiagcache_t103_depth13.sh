#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_SHARD_READER_PARENT=3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89
readonly EXPECTED_STRUCTURAL_PARENT=4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6
readonly EXPECTED_SCHEDULE_PARENT=6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
readonly EXPECTED_CPUDIAGCACHE_PARENT=686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5
readonly EXPECTED_MANIFEST=d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU STATE_DIR FIXTURE_DIR OUTPUT_JSON PART_LIST" >&2
  echo "PART_LIST is a comma-separated subset of {query,key,value}" >&2
  echo "OUTPUT_JSON filename must contain _scheme_b_, _qkvshard_, and _cpudiagcache_" >&2
  exit 2
fi

readonly GPU="$1"
readonly STATE_DIR="$2"
readonly FIXTURE_DIR="$3"
readonly OUTPUT="$4"
readonly PART_LIST="$5"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13"

output_name="$(basename "${OUTPUT}")"
if [[ "${output_name}" != *"_scheme_b_"* ]] || \
   [[ "${output_name}" != *"_qkvshard_"* ]] || \
   [[ "${output_name}" != *"_cpudiagcache_"* ]]; then
  echo "[FATAL] shard-reader-cpudiagcache evidence filename must contain _scheme_b_, _qkvshard_, and _cpudiagcache_" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -d "${STATE_DIR}" ]]; then
  echo "[FATAL] state directory missing (shard-writer phase must run first): ${STATE_DIR}" >&2
  exit 2
fi
if [[ ! -f "${STATE_DIR}/writer_timing.txt" ]]; then
  echo "[FATAL] ${STATE_DIR}/writer_timing.txt missing -- shard-writer phase did not complete" >&2
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
  sha256sum --check --status "${SCRIPT_DIR}/fixture_t103_qkvshard.sha256"
)
manifest_sha="$(sha256sum "${FIXTURE_DIR}/manifest.json" | cut -d' ' -f1)"
if [[ "${manifest_sha}" != "${EXPECTED_MANIFEST}" ]]; then
  echo "[FATAL] fixture manifest ${manifest_sha}; expected ${EXPECTED_MANIFEST}" >&2
  exit 2
fi

source_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp" | cut -d' ' -f1)"
shard_reader_parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp" | cut -d' ' -f1)"
if [[ "${shard_reader_parent_sha}" != "${EXPECTED_SHARD_READER_PARENT}" ]]; then
  echo "[FATAL] shard-reader parent ${shard_reader_parent_sha}; expected ${EXPECTED_SHARD_READER_PARENT}" >&2
  exit 2
fi
structural_parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_serialize_reader.cpp" | cut -d' ' -f1)"
if [[ "${structural_parent_sha}" != "${EXPECTED_STRUCTURAL_PARENT}" ]]; then
  echo "[FATAL] structural parent ${structural_parent_sha}; expected ${EXPECTED_STRUCTURAL_PARENT}" >&2
  exit 2
fi
schedule_parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp" | cut -d' ' -f1)"
if [[ "${schedule_parent_sha}" != "${EXPECTED_SCHEDULE_PARENT}" ]]; then
  echo "[FATAL] Token-SIMD schedule parent ${schedule_parent_sha}; expected ${EXPECTED_SCHEDULE_PARENT}" >&2
  exit 2
fi
cpudiagcache_parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp" | cut -d' ' -f1)"
if [[ "${cpudiagcache_parent_sha}" != "${EXPECTED_CPUDIAGCACHE_PARENT}" ]]; then
  echo "[FATAL] CPU-diagonal-cache parent ${cpudiagcache_parent_sha}; expected ${EXPECTED_CPUDIAGCACHE_PARENT}" >&2
  exit 2
fi
contract_sha="$(sha256sum "${SCRIPT_DIR}/fixture_t103_qkvshard.sha256" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --state-dir "${STATE_DIR}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --part "${PART_LIST}" \
  --backend-commit "${actual_commit}" \
  --shard-reader-parent-sha256 "${shard_reader_parent_sha}" \
  --structural-parent-sha256 "${structural_parent_sha}" \
  --schedule-parent-sha256 "${schedule_parent_sha}" \
  --cpudiagcache-parent-sha256 "${cpudiagcache_parent_sha}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --fixture-contract-sha256 "${contract_sha}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-Brev GPU}" \
  --source-sha256 "${source_sha}"
