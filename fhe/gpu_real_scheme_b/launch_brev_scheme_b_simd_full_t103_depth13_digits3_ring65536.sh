#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=../gpu_common/capacity_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../gpu_common/capacity_lib.sh"

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG RUN_TAG" >&2
  exit 2
fi

readonly PHYSICAL_GPU="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly FIXTURE_SUBDIR="$4"
readonly IMAGE_TAG="$5"
readonly RUN_TAG="$6"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly LOG="${SOURCE_DIR}/${RUN_TAG}.run.log"
readonly DONE="${SOURCE_DIR}/${RUN_TAG}.done"
readonly VRAM_LOG="${SOURCE_DIR}/${RUN_TAG}.vram.log"
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"
readonly CONTAINER_NAME="dnagpt-${RUN_TAG}"

if [[ "${RUN_TAG}" != *"_scheme_b_"* ]] || \
   [[ "${RUN_TAG}" != *"_simd_full_t103_depth13_digits3_ring65536_"* ]]; then
  echo "[FATAL] depth-13/digits-3/ring-65536 SIMD tag must contain _scheme_b_ and _simd_full_t103_depth13_digits3_ring65536_" >&2
  exit 2
fi

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

export PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR
export IMAGE_TAG RUN_TAG LOG DONE VRAM_LOG PREFLIGHT_NOTE
export CONTAINER_NAME
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
    "/work/${SOURCE_SUBDIR}/run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh" \
    0 \
    "/work/${FIXTURE_SUBDIR}" \
    "/work/${SOURCE_SUBDIR}/evidence/${RUN_TAG}.json" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
  wait "${sampler_pid}"
' >/dev/null 2>&1 &
disown

echo "run_pid=$! log=${LOG} vram_log=${VRAM_LOG} output=${OUTPUT}"
