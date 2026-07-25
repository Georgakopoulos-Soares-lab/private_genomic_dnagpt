#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 GPU_CSV REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG RUN_TAG MAX_POLLS GPU_COUNT" >&2
  echo "  GPU_CSV: candidate physical GPU pool to search, e.g. '0,1,2,3,4,5,6,7'" >&2
  echo "  GPU_COUNT: how many GPUs to reserve and pass to one run, e.g. 2" >&2
  exit 2
fi

readonly GPU_CSV="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly FIXTURE_SUBDIR="$4"
readonly IMAGE_TAG="$5"
readonly RUN_TAG="$6"
readonly MAX_POLLS="$7"
readonly GPU_COUNT="$8"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LAUNCHER="${SCRIPT_DIR}/launch_brev_multiblock.sh"
readonly POLL_SECONDS=30
readonly REQUIRED_STABLE_POLLS=2
CURRENT_LOCKS=()

cleanup_locks() {
  for lock_dir in "${CURRENT_LOCKS[@]:-}"; do
    [[ -n "${lock_dir}" ]] && rmdir "${lock_dir}" 2>/dev/null || true
  done
}
trap cleanup_locks EXIT

if [[ ! "${MAX_POLLS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FATAL] MAX_POLLS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FATAL] GPU_COUNT must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "${LAUNCHER}" ]]; then
  echo "[FATAL] launcher is not executable: ${LAUNCHER}" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<<"${GPU_CSV}"
if (( GPU_COUNT > ${#GPUS[@]} )); then
  echo "[FATAL] GPU_COUNT (${GPU_COUNT}) exceeds candidate pool size (${#GPUS[@]})" >&2
  exit 2
fi
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
  ready_gpus=()
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
      ready_gpus+=("${gpu}")
    fi
  done

  if (( ${#ready_gpus[@]} >= GPU_COUNT )); then
    selected=("${ready_gpus[@]:0:GPU_COUNT}")
    acquired=()
    ok=1
    for gpu in "${selected[@]}"; do
      lock_dir="/tmp/dnagpt-fhe-gpu-${gpu}.lock"
      if ! mkdir "${lock_dir}"; then
        ok=0
        break
      fi
      acquired+=("${lock_dir}")
    done
    if (( ok )); then
      recheck_ok=1
      for gpu in "${selected[@]}"; do
        gpu_is_free "${gpu}" || recheck_ok=0
      done
      if (( recheck_ok )); then
        CURRENT_LOCKS=("${acquired[@]}")
        selected_csv="$(IFS=,; echo "${selected[*]}")"
        echo "stable_free_gpus=${selected_csv}; launching ${RUN_TAG}"
        if "${LAUNCHER}" \
          "${selected_csv}" \
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
      else
        echo "recheck_failed; resuming clean-GPU search"
      fi
    else
      echo "lock_race_lost; resuming clean-GPU search"
    fi
    for lock_dir in "${acquired[@]:-}"; do
      [[ -n "${lock_dir}" ]] && rmdir "${lock_dir}" 2>/dev/null || true
    done
    CURRENT_LOCKS=()
    for gpu in "${selected[@]}"; do
      stable_polls["${gpu}"]=0
    done
  fi
  sleep "${POLL_SECONDS}"
done

echo "[FATAL] never found ${GPU_COUNT} simultaneously process-free GPUs for ${REQUIRED_STABLE_POLLS} polls" >&2
exit 4
