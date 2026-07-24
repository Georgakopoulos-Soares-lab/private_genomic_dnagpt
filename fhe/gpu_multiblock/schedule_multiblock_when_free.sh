#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU_CSV REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG RUN_TAG MAX_POLLS" >&2
  exit 2
fi

readonly GPU_CSV="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly FIXTURE_SUBDIR="$4"
readonly IMAGE_TAG="$5"
readonly RUN_TAG="$6"
readonly MAX_POLLS="$7"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LAUNCHER="${SCRIPT_DIR}/launch_brev_multiblock.sh"
readonly POLL_SECONDS=30
readonly REQUIRED_STABLE_POLLS=2
CURRENT_LOCK=""

cleanup_lock() {
  if [[ -n "${CURRENT_LOCK}" ]]; then
    rmdir "${CURRENT_LOCK}" 2>/dev/null || true
  fi
}
trap cleanup_lock EXIT

if [[ ! "${MAX_POLLS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FATAL] MAX_POLLS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "${LAUNCHER}" ]]; then
  echo "[FATAL] launcher is not executable: ${LAUNCHER}" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<<"${GPU_CSV}"
declare -A stable_polls
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] invalid physical GPU index: ${gpu}" >&2
    exit 2
  fi
  stable_polls["${gpu}"]=0
done

gpu_is_free() {
  local gpu="$1"
  local memory_mib utilization_pct gpu_uuid compute_process_count
  read -r memory_mib utilization_pct < <(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${gpu}" | tr ',' ' '
  )
  gpu_uuid="$(
    nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu}"
  )"
  compute_process_count="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
      grep -Fxc "${gpu_uuid}" || true
  )"
  echo "gpu=${gpu} mem_mib=${memory_mib} util_pct=${utilization_pct} compute_processes=${compute_process_count}"
  (( memory_mib < 100 && utilization_pct < 10 && compute_process_count == 0 ))
}

for ((poll = 1; poll <= MAX_POLLS; poll++)); do
  echo "poll=${poll}/${MAX_POLLS} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for gpu in "${GPUS[@]}"; do
    lock_dir="/tmp/dnagpt-fhe-gpu-${gpu}.lock"
    if [[ -d "${lock_dir}" ]]; then
      echo "gpu=${gpu} reserved_by_dnagpt_scheduler=true"
      stable_polls["${gpu}"]=0
      continue
    fi
    if gpu_is_free "${gpu}"; then
      stable_polls["${gpu}"]=$((stable_polls["${gpu}"] + 1))
    else
      stable_polls["${gpu}"]=0
    fi
    if (( stable_polls["${gpu}"] >= REQUIRED_STABLE_POLLS )); then
      if ! mkdir "${lock_dir}"; then
        stable_polls["${gpu}"]=0
        continue
      fi
      CURRENT_LOCK="${lock_dir}"
      if ! gpu_is_free "${gpu}"; then
        rmdir "${CURRENT_LOCK}"
        CURRENT_LOCK=""
        stable_polls["${gpu}"]=0
        continue
      fi
      echo "stable_free_gpu=${gpu}; launching ${RUN_TAG}"
      if "${LAUNCHER}" \
        "${gpu}" \
        "${REMOTE_ROOT}" \
        "${SOURCE_SUBDIR}" \
        "${FIXTURE_SUBDIR}" \
        "${IMAGE_TAG}" \
        "${RUN_TAG}"; then
        readonly RUN_DONE="${REMOTE_ROOT}/${SOURCE_SUBDIR}/fhe/gpu_multiblock/${RUN_TAG}.done"
        while [[ ! -e "${RUN_DONE}" ]]; do
          sleep "${POLL_SECONDS}"
        done
        exit 0
      fi
      echo "launcher_refused; resuming clean-GPU search"
      rmdir "${CURRENT_LOCK}"
      CURRENT_LOCK=""
      stable_polls["${gpu}"]=0
    fi
  done
  sleep "${POLL_SECONDS}"
done

echo "[FATAL] no GPU stayed process-free for ${REQUIRED_STABLE_POLLS} polls" >&2
exit 4
