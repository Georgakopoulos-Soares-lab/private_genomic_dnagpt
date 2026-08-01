#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly EXPECTED_STRUCTURAL_PARENT=ea720f1b6c49524f1b49853b67622307fd55c9fb751317fee4b30c6750f69403
readonly EXPECTED_TOKEN_SIMD_PARENT=6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU STATE_DIR OUTPUT_JSON" >&2
  echo "OUTPUT_JSON filename must contain both _scheme_b_ and _qkvshard_" >&2
  exit 2
fi

readonly GPU="$1"
readonly STATE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13"

if [[ "$(basename "${OUTPUT}")" != *"_scheme_b_"* ]] || \
   [[ "$(basename "${OUTPUT}")" != *"_qkvshard_"* ]]; then
  echo "[FATAL] shard-writer evidence filename must contain both _scheme_b_ and _qkvshard_" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ -e "${STATE_DIR}" ]]; then
  echo "[FATAL] refusing to reuse an existing state directory (must be fresh per writer/reader-pair run): ${STATE_DIR}" >&2
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

source_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13.cpp" | cut -d' ' -f1)"
structural_parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_serialize_writer.cpp" | cut -d' ' -f1)"
if [[ "${structural_parent_sha}" != "${EXPECTED_STRUCTURAL_PARENT}" ]]; then
  echo "[FATAL] structural parent ${structural_parent_sha}; expected ${EXPECTED_STRUCTURAL_PARENT}" >&2
  exit 2
fi
token_simd_parent_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp" | cut -d' ' -f1)"
if [[ "${token_simd_parent_sha}" != "${EXPECTED_TOKEN_SIMD_PARENT}" ]]; then
  echo "[FATAL] Token-SIMD parameter parent ${token_simd_parent_sha}; expected ${EXPECTED_TOKEN_SIMD_PARENT}" >&2
  exit 2
fi

exec "${BINARY}" \
  --gpu "${GPU}" \
  --state-dir "${STATE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --structural-parent-sha256 "${structural_parent_sha}" \
  --token-simd-parent-sha256 "${token_simd_parent_sha}" \
  --source-sha256 "${source_sha}"
