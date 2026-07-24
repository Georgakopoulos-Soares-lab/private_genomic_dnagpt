#!/usr/bin/env bash
set -euo pipefail

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
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"

if [[ "${RUN_TAG}" != *"_packed_t8_"* ]]; then
  echo "[FATAL] run tag must contain _packed_t8_" >&2
  exit 2
fi

read -r memory_mib utilization_pct < <(
  nvidia-smi \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    -i "${PHYSICAL_GPU}" | tr ',' ' '
)
readonly GPU_UUID="$(
  nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${PHYSICAL_GPU}"
)"
readonly COMPUTE_PROCESS_COUNT="$(
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
    grep -Fxc "${GPU_UUID}" || true
)"
echo "preflight_mem_mib=${memory_mib} preflight_util_pct=${utilization_pct} compute_processes=${COMPUTE_PROCESS_COUNT}"
if (( memory_mib >= 100 || utilization_pct != 0 || COMPUTE_PROCESS_COUNT != 0 )); then
  echo "[FATAL] physical GPU ${PHYSICAL_GPU} is not process-free; refusing launch" >&2
  exit 3
fi

mkdir -p "${SOURCE_DIR}/evidence"
for path in "${LOG}" "${DONE}" "${OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done

readonly IMAGE_EVIDENCE="${IMAGE_TAG}@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4"
export PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR
export IMAGE_TAG IMAGE_EVIDENCE RUN_TAG LOG DONE
nohup sh -c '
  docker run --rm \
    --gpus "device=${PHYSICAL_GPU}" \
    -v "${REMOTE_ROOT}:/work" \
    -w "/work/${SOURCE_SUBDIR}" \
    -e "FIDES_PACKED_T8_BUILD_DIR=/work/${SOURCE_SUBDIR}/build" \
    -e "FIDES_CONTAINER_IMAGE=${IMAGE_EVIDENCE}" \
    -e "FIDES_RUN_ENVIRONMENT=Brev A100-SXM4-80GB physical GPU ${PHYSICAL_GPU} [gpu]" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    "/work/${SOURCE_SUBDIR}/run_packed_t8.sh" \
    0 \
    "/work/${FIXTURE_SUBDIR}" \
    "/work/${SOURCE_SUBDIR}/evidence/${RUN_TAG}.json" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
' >/dev/null 2>&1 &
echo "run_pid=$! log=${LOG} output=${OUTPUT}"
