#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../gpu_common/capacity_lib.sh
source "${SCRIPT_DIR}/../gpu_common/capacity_lib.sh"

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR STATE_SUBDIR" >&2
  exit 2
fi

readonly PHYSICAL_GPU="$1"
readonly REMOTE_ROOT="$2"
readonly SOURCE_SUBDIR="$3"
readonly STATE_SUBDIR="$4"
readonly RUN_TAG="${SCHEME_B_SERIALIZE_WRITER_RUN_TAG:?set SCHEME_B_SERIALIZE_WRITER_RUN_TAG (must contain _scheme_b_ and _serialized_)}"
readonly IMAGE_TAG="${SCHEME_B_IMAGE:?set SCHEME_B_IMAGE}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly LOG="${SOURCE_DIR}/${RUN_TAG}.run.log"
readonly DONE="${SOURCE_DIR}/${RUN_TAG}.done"
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"

if [[ "${RUN_TAG}" != *"_scheme_b_"* ]] || [[ "${RUN_TAG}" != *"_serialized_"* ]]; then
  echo "[FATAL] serialize-writer evidence tag must contain both _scheme_b_ and _serialized_" >&2
  exit 2
fi

# See gpu_common/capacity_lib.sh and launch_brev_scheme_b.sh for the full
# incident history behind this preflight (shared-host nvidia-smi reliability
# issue). Correctness does not depend on it -- the writer never touches the
# GPU at all (no LoadContext/GenCryptoContextGPU call, see the C++ source
# header comment); this preflight only informs the timing note recorded in
# the evidence JSON.
gpucap_preflight_confirm_gpu "${PHYSICAL_GPU}"
if [[ "${GPUCAP_PREFLIGHT_CONFIRMED}" -eq 1 ]]; then
  compute_process_count="$(gpucap_compute_process_count "${GPUCAP_UUID}")"
  echo "preflight_mem_mib=${GPUCAP_MEM_MIB} preflight_util_pct=${GPUCAP_UTIL_PCT} compute_processes=${compute_process_count}"
fi

mkdir -p "${SOURCE_DIR}/evidence"
for path in "${LOG}" "${DONE}" "${OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done
if [[ -e "${REMOTE_ROOT}/${STATE_SUBDIR}" ]]; then
  echo "[FATAL] refusing to reuse an existing state directory: ${REMOTE_ROOT}/${STATE_SUBDIR}" >&2
  exit 2
fi

readonly PREFLIGHT_NOTE="${GPUCAP_PREFLIGHT_NOTE}"

export PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR STATE_SUBDIR
export IMAGE_TAG RUN_TAG LOG DONE PREFLIGHT_NOTE
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
    "/work/${SOURCE_SUBDIR}/run_scheme_b_serialize_writer.sh" \
    0 \
    "/work/${STATE_SUBDIR}" \
    "/work/${SOURCE_SUBDIR}/evidence/${RUN_TAG}.json" \
    >"${LOG}" 2>&1
  status=$?
  printf "%s\n" "${status}" >"${DONE}"
' >/dev/null 2>&1 &

echo "run_pid=$! log=${LOG} output=${OUTPUT}"
