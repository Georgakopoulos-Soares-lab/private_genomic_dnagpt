#!/usr/bin/env bash
# Runner for the T=103 SIMD 12-block + GSR-head end-to-end driver, CONFIG-3
# "all-optimizations" variant
# (real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache). An
# additive fork of run_scheme_b_all_blocks_head_t103_simd.sh: it swaps the
# binary, the driver source, and the per-block source it verifies -- the
# cpudiagcache block (--cpudiagcache-source-sha256) instead of the plain
# depth-13 block (--depth13-source-sha256). Same fixture, same two-block
# refresh design pin. The binary fail-closes if any passed SHA/commit differs
# from its embedded PINNED_* values, so this script computes the live hashes
# and passes them through.
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU FIXTURE_DIR OUTPUT_JSON" >&2
  echo "OUTPUT_JSON filename must contain _scheme_b_ and _all_blocks_head_t103_cpudiagcache_" >&2
  exit 2
fi

readonly GPU="$1"
readonly FIXTURE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache"
readonly DRIVER_SRC="${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache.cpp"
readonly CPUDIAGCACHE_SRC="${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"
readonly TWO_BLOCK_SRC="${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_two_block_refresh.cpp"
readonly FIXTURE_CONTRACT="${SCRIPT_DIR}/fixture_all_blocks_head_t103.sha256"

if [[ "$(basename "${OUTPUT}")" != *"_scheme_b_"* ]] || \
   [[ "$(basename "${OUTPUT}")" != *"_all_blocks_head_t103_cpudiagcache_"* ]]; then
  echo "[FATAL] e2e evidence filename must contain _scheme_b_ and _all_blocks_head_t103_cpudiagcache_" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] ${BINARY} missing; run build_in_fideslib.sh (or kimon/env/build_binaries.sh) first" >&2
  exit 2
fi
if [[ ! -f "${FIXTURE_DIR}/manifest.json" ]]; then
  echo "[FATAL] fixture manifest missing: ${FIXTURE_DIR}/manifest.json" >&2
  exit 2
fi
if [[ ! -f "${FIXTURE_CONTRACT}" ]]; then
  echo "[FATAL] fixture contract missing: ${FIXTURE_CONTRACT}" >&2
  exit 2
fi

actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi

# Verify every fixture file against the contract before spending GPU hours.
(
  cd "${FIXTURE_DIR}"
  sha256sum --check --status "${FIXTURE_CONTRACT}"
) || { echo "[FATAL] fixture files do not match ${FIXTURE_CONTRACT}" >&2; exit 2; }

manifest_sha="$(sha256sum "${FIXTURE_DIR}/manifest.json" | cut -d' ' -f1)"
contract_sha="$(sha256sum "${FIXTURE_CONTRACT}" | cut -d' ' -f1)"
source_sha="$(sha256sum "${DRIVER_SRC}" | cut -d' ' -f1)"
cpudiagcache_sha="$(sha256sum "${CPUDIAGCACHE_SRC}" | cut -d' ' -f1)"
two_block_sha="$(sha256sum "${TWO_BLOCK_SRC}" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --fixture-dir "${FIXTURE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --cpudiagcache-source-sha256 "${cpudiagcache_sha}" \
  --two-block-refresh-source-sha256 "${two_block_sha}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --fixture-contract-sha256 "${contract_sha}" \
  --source-sha256 "${source_sha}" \
  --container-image "${FIDES_CONTAINER_IMAGE:-UNSPECIFIED}" \
  --environment "${FIDES_RUN_ENVIRONMENT:-TACC A100}"
