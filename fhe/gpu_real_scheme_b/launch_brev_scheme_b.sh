#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../gpu_common/capacity_lib.sh
source "${SCRIPT_DIR}/../gpu_common/capacity_lib.sh"

if [[ $# -ne 7 ]]; then
  echo "usage: $0 PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG GATE RUN_TAG" >&2
  exit 2
fi

readonly PHYSICAL_GPU="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly FIXTURE_SUBDIR="$4"
readonly IMAGE_TAG="$5"
readonly GATE="$6"
readonly RUN_TAG="$7"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly LOG="${SOURCE_DIR}/${RUN_TAG}.run.log"
readonly DONE="${SOURCE_DIR}/${RUN_TAG}.done"
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"

if [[ "${GATE}" != "ln1" && "${GATE}" != "attention" && "${GATE}" != "full" ]]; then
  echo "[FATAL] gate must be ln1, attention, or full" >&2
  exit 2
fi
if [[ "${RUN_TAG}" != *"_scheme_b_"* ]]; then
  echo "[FATAL] Scheme B evidence tag must contain _scheme_b_" >&2
  exit 2
fi

# See gpu_common/capacity_lib.sh header for the full incident history behind
# this preflight: nvidia-smi on this shared 255-core, 8-tenant host has been
# observed to transiently emit a garbled/error response or empty output
# during acute concurrent-load bursts (both on indexed `-i N` queries and,
# during sharper bursts, even the unindexed bulk form). Correctness of the
# encrypted computation does not depend on this check -- only our own
# wall-clock timing measurement does, already caveated in provenance -- so
# gpucap_preflight_confirm_gpu fails closed only on a *confirmed* occupied
# reading and otherwise soft-proceeds, trusting the caller's recent
# idle-GPU determination rather than losing a narrow capacity window to a
# monitoring-tool blind spot.
gpucap_preflight_confirm_gpu "${PHYSICAL_GPU}"
if [[ "${GPUCAP_PREFLIGHT_CONFIRMED}" -eq 1 ]]; then
  compute_process_count="$(gpucap_compute_process_count "${GPUCAP_UUID}")"
  echo "preflight_mem_mib=${GPUCAP_MEM_MIB} preflight_util_pct=${GPUCAP_UTIL_PCT} compute_processes=${compute_process_count}"
  # This shared host has small (~500-700 MiB) idle CUDA-context processes
  # present on every physical GPU at all times (observed across all 8 devices
  # on 2026-07-25); a nonzero compute-process count is therefore not a reliable
  # occupancy signal here and is logged for information only.
fi

mkdir -p "${SOURCE_DIR}/evidence"
for path in "${LOG}" "${DONE}" "${OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done

# Record whether the preflight occupancy reading was actually confirmed, so
# the evidence provenance honestly reflects whether this run's timing could
# be contention-affected in a way we couldn't verify pre-launch.
readonly PREFLIGHT_NOTE="${GPUCAP_PREFLIGHT_NOTE}"

export PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR
export IMAGE_TAG GATE RUN_TAG LOG DONE PREFLIGHT_NOTE
nohup sh -c '
  docker run --rm \
    --gpus "device=${PHYSICAL_GPU}" \
    -v "${REMOTE_ROOT}:/work" \
    -w "/work/${SOURCE_SUBDIR}" \
    -e "FIDES_REAL_SCHEME_B_BUILD_DIR=/work/${SOURCE_SUBDIR}/build" \
    -e "FIDES_CONTAINER_IMAGE=${IMAGE_TAG}" \
    -e "FIDES_RUN_ENVIRONMENT=Brev A100-SXM4-80GB physical GPU ${PHYSICAL_GPU} [gpu] (${PREFLIGHT_NOTE})" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    "/work/${SOURCE_SUBDIR}/run_scheme_b.sh" \
    0 \
    "${GATE}" \
    "/work/${FIXTURE_SUBDIR}" \
    "/work/${SOURCE_SUBDIR}/evidence/${RUN_TAG}.json" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
' >/dev/null 2>&1 &

echo "run_pid=$! log=${LOG} output=${OUTPUT}"
