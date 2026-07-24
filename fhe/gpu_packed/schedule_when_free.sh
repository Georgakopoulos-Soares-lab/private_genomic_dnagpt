#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG RUN_TAG" >&2
  exit 2
fi
readonly REMOTE_ROOT="$1"
readonly SOURCE_SUBDIR="$2"
readonly FIXTURE_SUBDIR="$3"
readonly IMAGE_TAG="$4"
readonly RUN_TAG="$5"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REQUIRED_STABLE_POLLS=2
readonly POLL_SECONDS="${POLL_SECONDS:-30}"

declare -A stable=()
while true; do
  mapfile -t rows < <(
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  process_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)"
  for row in "${rows[@]}"; do
    IFS=',' read -r gpu uuid memory utilization <<<"${row}"
    gpu="${gpu// /}"
    uuid="${uuid// /}"
    memory="${memory// /}"
    utilization="${utilization// /}"
    processes="$(grep -Fxc "${uuid}" <<<"${process_uuids}" || true)"
    if (( memory < 100 && utilization == 0 && processes == 0 )); then
      stable["${gpu}"]=$(( ${stable["${gpu}"]:-0} + 1 ))
      if (( stable["${gpu}"] >= REQUIRED_STABLE_POLLS )); then
        exec "${SCRIPT_DIR}/launch_brev_packed_t8.sh" \
          "${gpu}" "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${FIXTURE_SUBDIR}" \
          "${IMAGE_TAG}" "${RUN_TAG}"
      fi
    else
      stable["${gpu}"]=0
    fi
  done
  sleep "${POLL_SECONDS}"
done
