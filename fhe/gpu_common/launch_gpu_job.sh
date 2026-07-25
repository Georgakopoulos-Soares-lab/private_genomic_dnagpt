#!/usr/bin/env bash
# Generic single-job GPU launcher template. Sources capacity_lib.sh for the
# battle-tested preflight check, then backgrounds an arbitrary command on a
# specific physical GPU with overwrite-protected LOG/DONE/OUTPUT files.
#
# This is the reusable core extracted from fhe/gpu_real_scheme_b's launch
# script: inject "whatever we want to run" as the trailing command instead
# of copy-pasting the capacity/preflight logic into a new script each time.
#
# Usage:
#   launch_gpu_job.sh PHYSICAL_GPU LOG_FILE DONE_FILE OUTPUT_FILE -- CMD...
#
#   PHYSICAL_GPU   physical GPU index to preflight-check and launch on
#   LOG_FILE       stdout+stderr of CMD is redirected here
#   DONE_FILE      written with CMD's exit status once it finishes (marker
#                  file an orchestrator can poll for)
#   OUTPUT_FILE    expected result artifact (e.g. an evidence JSON); only
#                  used for the pre-launch overwrite guard, never written by
#                  this script itself -- CMD is responsible for producing it.
#                  Pass "" to skip the overwrite guard for this path.
#   CMD...         the actual command to run, e.g. a `docker run ...`
#                  invocation or a plain script call. Run via `sh -c` so
#                  shell metacharacters in CMD are honored; quote args
#                  needing preservation yourself.
#
# Exit codes:
#   0   job launched (backgrounded); caller should poll DONE_FILE
#   2   usage error, or refusing to overwrite an existing LOG/DONE/OUTPUT
#   3   preflight confirmed the GPU is occupied; refusing to launch
#
# Example:
#   gpu="$(source .../capacity_lib.sh; gpucap_wait_for_capacity)"
#   ./launch_gpu_job.sh "${gpu}" /tmp/my_job.log /tmp/my_job.done \
#     /tmp/my_job_output.json -- \
#     docker run --rm --gpus "device=${gpu}" -v /data:/work myimage:tag \
#       /work/run_my_experiment.sh --tag my_run_20260101
set -uo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./capacity_lib.sh
source "${LIB_DIR}/capacity_lib.sh"

if [[ $# -lt 5 || "$4" == "--" ]]; then
  echo "usage: $0 PHYSICAL_GPU LOG_FILE DONE_FILE OUTPUT_FILE -- CMD..." >&2
  exit 2
fi

PHYSICAL_GPU="$1"
LOG_FILE="$2"
DONE_FILE="$3"
OUTPUT_FILE="$4"
shift 4
if [[ "${1:-}" != "--" ]]; then
  echo "usage: $0 PHYSICAL_GPU LOG_FILE DONE_FILE OUTPUT_FILE -- CMD..." >&2
  exit 2
fi
shift

if [[ $# -eq 0 ]]; then
  echo "[FATAL] no CMD supplied after --" >&2
  exit 2
fi

gpucap_preflight_confirm_gpu "${PHYSICAL_GPU}"
preflight_rc=$?
gpucap_log "preflight gpu=${PHYSICAL_GPU} confirmed=${GPUCAP_PREFLIGHT_CONFIRMED} note='${GPUCAP_PREFLIGHT_NOTE}'"
if [[ "${preflight_rc}" -ne 0 ]]; then
  exit "${preflight_rc}"
fi

for path in "${LOG_FILE}" "${DONE_FILE}" "${OUTPUT_FILE}"; do
  if [[ -n "${path}" && -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${DONE_FILE}")"

export GPUCAP_PREFLIGHT_NOTE GPUCAP_PREFLIGHT_CONFIRMED PHYSICAL_GPU
nohup sh -c '
  "$@" >"'"${LOG_FILE}"'" 2>&1
  status=$?
  printf "%s\n" "${status}" >"'"${DONE_FILE}"'"
' sh "$@" >/dev/null 2>&1 &

echo "run_pid=$! log=${LOG_FILE} done=${DONE_FILE} output=${OUTPUT_FILE:-<none>} preflight_note='${GPUCAP_PREFLIGHT_NOTE}'"
