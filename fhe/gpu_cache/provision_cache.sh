#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU CACHE_DIR CLIENT_ORACLE_KEY METADATA_JSON" >&2
  exit 2
fi

readonly GPU="$1"
readonly CACHE_DIR="$2"
readonly CLIENT_ORACLE_KEY="$3"
readonly METADATA="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly BUILD_DIR="${FIDES_CACHE_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly BINARY="${BUILD_DIR}/dnagpt_fides_cache"
CLIENT_PARENT="$(dirname "${CLIENT_ORACLE_KEY}")"
readonly CLIENT_PARENT

if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] ${BINARY} missing; run build_in_fideslib.sh first" >&2
  exit 2
fi
if [[ ! -d "${CLIENT_PARENT}" ]]; then
  echo "[FATAL] client oracle parent directory missing" >&2
  exit 2
fi
if [[ -e "${CACHE_DIR}" || -e "${CLIENT_ORACLE_KEY}" || -e "${METADATA}" ]]; then
  echo "[FATAL] refusing existing cache, client key, or metadata path" >&2
  exit 2
fi

exec "${BINARY}" provision \
  --gpu "${GPU}" \
  --cache-dir "${CACHE_DIR}" \
  --oracle-key-output "${CLIENT_ORACLE_KEY}" \
  --metadata-output "${METADATA}"
