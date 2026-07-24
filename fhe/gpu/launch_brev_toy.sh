#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PHYSICAL_GPU REMOTE_SOURCE_DIR IMAGE_TAG IMAGE_EVIDENCE RUN_TAG" >&2
  exit 2
fi

readonly PHYSICAL_GPU="$1"
readonly SOURCE_DIR="$2"
readonly IMAGE_TAG="$3"
readonly IMAGE_EVIDENCE="$4"
readonly RUN_TAG="$5"
readonly LOG="${SOURCE_DIR}/${RUN_TAG}.run.log"
readonly DONE="${SOURCE_DIR}/${RUN_TAG}.done"
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"

read -r memory_mib utilization_pct < <(
  nvidia-smi \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    -i "${PHYSICAL_GPU}" | tr ',' ' '
)
echo "preflight_mem_mib=${memory_mib} preflight_util_pct=${utilization_pct}"
if (( memory_mib >= 10000 || utilization_pct >= 10 )); then
  echo "[FATAL] physical GPU ${PHYSICAL_GPU} is occupied; refusing to launch" >&2
  exit 3
fi

mkdir -p "${SOURCE_DIR}/evidence"
for path in "${LOG}" "${DONE}" "${OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done

export PHYSICAL_GPU SOURCE_DIR IMAGE_TAG IMAGE_EVIDENCE RUN_TAG LOG DONE OUTPUT
nohup sh -c '
  docker run --rm \
    --gpus "device=${PHYSICAL_GPU}" \
    -v "${SOURCE_DIR}:/work" \
    -w /work \
    -e FIDES_TOY_BUILD_DIR=/work/build_asymfix2 \
    -e "FIDES_CONTAINER_IMAGE=${IMAGE_EVIDENCE}" \
    -e "FIDES_RUN_ENVIRONMENT=Brev A100-SXM4-80GB physical GPU ${PHYSICAL_GPU} [gpu]" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    /work/run_toy.sh \
    0 \
    "/work/evidence/${RUN_TAG}.json" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
' >/dev/null 2>&1 &

echo "run_pid=$! log=${LOG} output=${OUTPUT}"
