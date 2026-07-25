#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PHYSICAL_GPU_CSV REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG RUN_TAG" >&2
  echo "  PHYSICAL_GPU_CSV: one or more comma-separated physical GPU indices, e.g. '4' or '4,5'" >&2
  exit 2
fi

readonly PHYSICAL_GPU_CSV="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly FIXTURE_SUBDIR="$4"
readonly IMAGE_TAG="$5"
readonly RUN_TAG="$6"
readonly MODULE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}/fhe/gpu_multiblock"
readonly LOG="${MODULE_DIR}/${RUN_TAG}.run.log"
readonly DONE="${MODULE_DIR}/${RUN_TAG}.done"
readonly OUTPUT="${MODULE_DIR}/evidence/${RUN_TAG}.json"

if [[ "${RUN_TAG}" != *"_blocks0_1_refresh_"* ]]; then
  echo "[FATAL] two-block evidence tag must contain _blocks0_1_refresh_" >&2
  exit 2
fi

IFS=',' read -r -a PHYSICAL_GPUS <<<"${PHYSICAL_GPU_CSV}"
if [[ ${#PHYSICAL_GPUS[@]} -eq 0 ]]; then
  echo "[FATAL] PHYSICAL_GPU_CSV must list at least one GPU" >&2
  exit 2
fi
for physical_gpu in "${PHYSICAL_GPUS[@]}"; do
  if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] invalid physical GPU index: ${physical_gpu}" >&2
    exit 2
  fi
  read -r memory_mib utilization_pct < <(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${physical_gpu}" | tr ',' ' '
  )
  gpu_uuid="$(
    nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${physical_gpu}"
  )"
  compute_process_count="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
      grep -Fxc "${gpu_uuid}" || true
  )"
  echo "preflight_gpu=${physical_gpu} mem_mib=${memory_mib} util_pct=${utilization_pct} compute_processes=${compute_process_count}"
  if (( memory_mib >= 10000 || utilization_pct >= 10 || compute_process_count != 0 )); then
    echo "[FATAL] physical GPU ${physical_gpu} is occupied; refusing to launch" >&2
    exit 3
  fi
done

mkdir -p "${MODULE_DIR}/evidence"
for path in "${LOG}" "${DONE}" "${OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${REMOTE_ROOT}/${FIXTURE_SUBDIR}/manifest.json" ]]; then
  echo "[FATAL] fixture manifest missing: ${REMOTE_ROOT}/${FIXTURE_SUBDIR}/manifest.json" >&2
  exit 2
fi

# Docker's --gpus device=A,B,... remaps the requested physical GPUs to
# sequential container-local CUDA indices 0..N-1, in the order listed. The
# binary must therefore be given "0,1,...,N-1", not the physical indices.
container_gpu_list="0"
for ((i = 1; i < ${#PHYSICAL_GPUS[@]}; i++)); do
  container_gpu_list="${container_gpu_list},${i}"
done

readonly IMAGE_EVIDENCE="${IMAGE_TAG}@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4"
export PHYSICAL_GPU_CSV REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR
export IMAGE_TAG IMAGE_EVIDENCE RUN_TAG LOG DONE container_gpu_list
# The extra literal quotes around device=... are required by Docker's own
# --gpus value parser: a bare comma-separated device list (device=5,6) is
# misread as two top-level flag fields and fails with "cannot set both Count
# and DeviceIDs on device request". Wrapping it in literal double quotes
# (device="5,6" -> "device=5,6") tells Docker to treat the whole list as one
# value. Confirmed against Docker 29.3.1 / nvidia-container-toolkit 1.18.1.
nohup sh -c '
  docker run --rm \
    --gpus "\"device=${PHYSICAL_GPU_CSV}\"" \
    -v "${REMOTE_ROOT}:/work" \
    -w "/work/${SOURCE_SUBDIR}/fhe/gpu_multiblock" \
    -e "FIDES_MULTIBLOCK_BUILD_DIR=/work/${SOURCE_SUBDIR}/fhe/gpu_multiblock/build" \
    -e "FIDES_CONTAINER_IMAGE=${IMAGE_EVIDENCE}" \
    -e "FIDES_RUN_ENVIRONMENT=Brev A100-SXM4-80GB physical GPUs ${PHYSICAL_GPU_CSV} [gpu]" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    "/work/${SOURCE_SUBDIR}/fhe/gpu_multiblock/run_two_block.sh" \
    "${container_gpu_list}" \
    "/work/${FIXTURE_SUBDIR}" \
    "/work/${SOURCE_SUBDIR}/fhe/gpu_multiblock/evidence/${RUN_TAG}.json" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
' >/dev/null 2>&1 &

echo "run_pid=$! log=${LOG} output=${OUTPUT} physical_gpus=${PHYSICAL_GPU_CSV} container_gpus=${container_gpu_list}"
