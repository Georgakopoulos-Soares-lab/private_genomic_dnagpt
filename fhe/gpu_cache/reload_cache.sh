#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU CACHE_DIR CLIENT_ORACLE_KEY OUTPUT_JSON MANIFEST_SHA IMAGE@sha256:DIGEST SOURCE_SHA" >&2
  exit 2
fi

readonly GPU="$1"
readonly CACHE_DIR="$2"
readonly CLIENT_ORACLE_KEY="$3"
readonly OUTPUT="$4"
readonly MANIFEST_SHA="$5"
readonly IMAGE="$6"
readonly SOURCE_SHA="$7"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly BUILD_DIR="${FIDES_CACHE_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly BINARY="${BUILD_DIR}/dnagpt_fides_cache"

if [[ ! -x "${BINARY}" || ! -f "${CLIENT_ORACLE_KEY}" ]]; then
  echo "[FATAL] binary or external client oracle key missing" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing to overwrite immutable evidence: ${OUTPUT}" >&2
  exit 2
fi

if [[ ! "${MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ || ! "${SOURCE_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[FATAL] manifest/source SHA-256 arguments are malformed" >&2
  exit 2
fi

exec "${BINARY}" reload \
  --gpu "${GPU}" \
  --cache-dir "${CACHE_DIR}" \
  --oracle-key "${CLIENT_ORACLE_KEY}" \
  --output "${OUTPUT}" \
  --manifest-sha256 "${MANIFEST_SHA}" \
  --container-image "${IMAGE}" \
  --source-sha256 "${SOURCE_SHA}"
