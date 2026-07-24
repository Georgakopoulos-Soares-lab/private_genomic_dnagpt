#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PHYSICAL_GPU PROJECT_ROOT NEW_CACHE_DIR OUTPUT_JSON IMAGE_TAG" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
exec "${SCRIPT_DIR}/run_brev_host.sh" "$@"
