#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../gpu_common/capacity_lib.sh
source "${SCRIPT_DIR}/../gpu_common/capacity_lib.sh"

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR STATE_SUBDIR FIXTURE_SUBDIR PART_LIST" >&2
  echo "PART_LIST is a comma-separated subset of {query,key,value}" >&2
  exit 2
fi

readonly PHYSICAL_GPU="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly STATE_SUBDIR="$4"
readonly FIXTURE_SUBDIR="$5"
readonly PART_LIST="$6"
readonly RUN_TAG="${SCHEME_B_SHARD_READER_RUN_TAG:?set SCHEME_B_SHARD_READER_RUN_TAG (must contain _scheme_b_, _qkvshard_, and _cpudiagcache_)}"
readonly IMAGE_TAG="${SCHEME_B_IMAGE:?set SCHEME_B_IMAGE}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly LOG="${SOURCE_DIR}/${RUN_TAG}.run.log"
readonly DONE="${SOURCE_DIR}/${RUN_TAG}.done"
readonly VRAM_LOG="${SOURCE_DIR}/${RUN_TAG}.vram.log"
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"
readonly CONTAINER_NAME="dnagpt-${RUN_TAG}"

if [[ "${RUN_TAG}" != *"_scheme_b_"* ]] || [[ "${RUN_TAG}" != *"_qkvshard_"* ]] || \
   [[ "${RUN_TAG}" != *"_cpudiagcache_"* ]]; then
  echo "[FATAL] shard-reader-cpudiagcache evidence tag must contain _scheme_b_, _qkvshard_, and _cpudiagcache_" >&2
  exit 2
fi
if [[ ! -d "${REMOTE_ROOT}/${STATE_SUBDIR}" ]]; then
  echo "[FATAL] state directory missing (shard-writer phase must run and complete first): ${REMOTE_ROOT}/${STATE_SUBDIR}" >&2
  exit 2
fi

# Per-GPU-only preflight (no host load-average gate here -- that gate is
# reserved for the wait_and_run_*.sh orchestrators). Correctness of the
# encrypted computation does not depend on this check -- only our own
# wall-clock timing measurement does, already caveated in provenance. Each
# of the two concurrent shard workers calls this independently for its own
# PHYSICAL_GPU, exactly as every existing single-GPU gate already does.
gpucap_preflight_confirm_gpu "${PHYSICAL_GPU}"
if [[ "${GPUCAP_PREFLIGHT_CONFIRMED}" -eq 1 ]]; then
  compute_process_count="$(gpucap_compute_process_count "${GPUCAP_UUID}")"
  echo "preflight_mem_mib=${GPUCAP_MEM_MIB} preflight_util_pct=${GPUCAP_UTIL_PCT} compute_processes=${compute_process_count}"
fi

mkdir -p "${SOURCE_DIR}/evidence"
for path in "${LOG}" "${DONE}" "${VRAM_LOG}" "${OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[FATAL] refusing to reuse container name ${CONTAINER_NAME}" >&2
  exit 2
fi

readonly PREFLIGHT_NOTE="${GPUCAP_PREFLIGHT_NOTE}"

export PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR STATE_SUBDIR FIXTURE_SUBDIR PART_LIST
export IMAGE_TAG RUN_TAG LOG DONE VRAM_LOG PREFLIGHT_NOTE CONTAINER_NAME
# The single-quoted script is intentionally expanded by the detached shell.
# shellcheck disable=SC2016
nohup sh -c '
  (
    while [ ! -f "${DONE}" ]; do
      date -u "+%Y-%m-%dT%H:%M:%SZ"
      printf "physical_gpu=%s container=%s\n" "${PHYSICAL_GPU}" "${CONTAINER_NAME}"
      docker top "${CONTAINER_NAME}" -eo pid,args 2>/dev/null || true
      nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
      nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 5
    done
  ) >"${VRAM_LOG}" 2>&1 &
  sampler_pid=$!

  docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${PHYSICAL_GPU}" \
    -v "${REMOTE_ROOT}:/work" \
    -w "/work/${SOURCE_SUBDIR}" \
    -e "FIDES_REAL_SCHEME_B_BUILD_DIR=/work/${SOURCE_SUBDIR}/build" \
    -e "FIDES_CONTAINER_IMAGE=${IMAGE_TAG}" \
    -e "FIDES_RUN_ENVIRONMENT=Brev A100-SXM4-80GB physical GPU ${PHYSICAL_GPU} [gpu] (${PREFLIGHT_NOTE})" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    "/work/${SOURCE_SUBDIR}/run_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.sh" \
    0 \
    "/work/${STATE_SUBDIR}" \
    "/work/${FIXTURE_SUBDIR}" \
    "/work/${SOURCE_SUBDIR}/evidence/${RUN_TAG}.json" \
    "${PART_LIST}" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
  wait "${sampler_pid}"
' >/dev/null 2>&1 &
disown

echo "run_pid=$! log=${LOG} vram_log=${VRAM_LOG} output=${OUTPUT}"
