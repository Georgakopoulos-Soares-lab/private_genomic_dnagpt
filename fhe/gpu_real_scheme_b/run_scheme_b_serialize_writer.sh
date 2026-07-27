#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU STATE_DIR OUTPUT_JSON" >&2
  echo "OUTPUT_JSON filename must contain both _scheme_b_ and _serialized_" >&2
  exit 2
fi

readonly GPU="$1"
readonly STATE_DIR="$2"
readonly OUTPUT="$3"
readonly BINARY="${BUILD_DIR}/real_dnagpt_fides_scheme_b_serialize_writer"

if [[ "$(basename "${OUTPUT}")" != *"_scheme_b_"* ]] || \
   [[ "$(basename "${OUTPUT}")" != *"_serialized_"* ]]; then
  echo "[FATAL] serialize-writer evidence filename must contain both _scheme_b_ and _serialized_" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi
if [[ -e "${STATE_DIR}" ]]; then
  echo "[FATAL] refusing to reuse an existing state directory (must be fresh per writer/reader pair): ${STATE_DIR}" >&2
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

source_sha="$(sha256sum "${SCRIPT_DIR}/src/real_dnagpt_fides_scheme_b_serialize_writer.cpp" | cut -d' ' -f1)"

exec "${BINARY}" \
  --gpu "${GPU}" \
  --state-dir "${STATE_DIR}" \
  --output "${OUTPUT}" \
  --backend-commit "${actual_commit}" \
  --source-sha256 "${source_sha}"
