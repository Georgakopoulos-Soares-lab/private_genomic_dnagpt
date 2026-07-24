#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PHYSICAL_GPU PROJECT_ROOT NEW_CACHE_DIR OUTPUT_JSON IMAGE_TAG" >&2
  exit 2
fi

readonly PHYSICAL_GPU="$1"
readonly IMAGE_TAG="$5"
PROJECT_ROOT="$(realpath "$2")"
CACHE_DIR="$(realpath -m "$3")"
OUTPUT="$(realpath -m "$4")"
readonly PROJECT_ROOT CACHE_DIR OUTPUT
readonly MODULE_DIR="${PROJECT_ROOT}/fhe/gpu_cache"
readonly REAL_SOURCE="${PROJECT_ROOT}/fhe/gpu_real/src/real_dnagpt_fides.cpp"
readonly MODULE_SOURCE="${MODULE_DIR}/src/dnagpt_fides_cache.cpp"
readonly BINARY="${MODULE_DIR}/build/dnagpt_fides_cache"
readonly IMAGE_DIGEST="sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4"
readonly IMAGE_EVIDENCE="${IMAGE_TAG}@${IMAGE_DIGEST}"
CACHE_PARENT="$(dirname "${CACHE_DIR}")"
CACHE_NAME="$(basename "${CACHE_DIR}")"
OUTPUT_PARENT="$(dirname "${OUTPUT}")"
OUTPUT_NAME="$(basename "${OUTPUT}")"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CLIENT_DIR="$(mktemp -d /tmp/dnagpt-fides-cache-client.XXXXXX)"
readonly CACHE_PARENT CACHE_NAME OUTPUT_PARENT OUTPUT_NAME
readonly HOST_UID HOST_GID CLIENT_DIR
readonly PROVISION_METADATA="${CLIENT_DIR}/provision-metadata.json"

cleanup_client_material() {
  case "${CLIENT_DIR}" in
    /tmp/dnagpt-fides-cache-client.*)
      find "${CLIENT_DIR}" -type f -delete
      rmdir "${CLIENT_DIR}" 2>/dev/null || true
      ;;
    *)
      echo "[FATAL] refusing unsafe client-material cleanup target" >&2
      return 1
      ;;
  esac
}
trap cleanup_client_material EXIT

if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ || "${IMAGE_TAG}" == *"@"* ]]; then
  echo "[FATAL] physical GPU must be numeric and IMAGE_TAG must not include a digest" >&2
  exit 2
fi
if [[ ! -d "${MODULE_DIR}" || ! -f "${REAL_SOURCE}" ]]; then
  echo "[FATAL] PROJECT_ROOT does not contain the GPU cache/real sources" >&2
  exit 2
fi
if [[ ! -d "${CACHE_PARENT}" || ! -w "${CACHE_PARENT}" ||
      ! -d "${OUTPUT_PARENT}" || ! -w "${OUTPUT_PARENT}" ]]; then
  echo "[FATAL] cache/output parent directory missing" >&2
  exit 2
fi
if [[ -e "${CACHE_DIR}" || -e "${OUTPUT}" ]]; then
  echo "[FATAL] refusing existing immutable cache or evidence path" >&2
  exit 2
fi
case "${OUTPUT}" in
  "${CACHE_DIR}"|"${CACHE_DIR}"/*)
    echo "[FATAL] evidence output must be outside the public cache" >&2
    exit 2
    ;;
esac

actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
if [[ "${actual_image_id}" != "${IMAGE_DIGEST}" ]]; then
  echo "[FATAL] image ${actual_image_id}; expected ${IMAGE_DIGEST}" >&2
  exit 2
fi

check_gpu_free() {
  local memory_mib utilization_pct gpu_uuid compute_process_count
  read -r memory_mib utilization_pct < <(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${PHYSICAL_GPU}" | tr ',' ' '
  )
  gpu_uuid="$(
    nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${PHYSICAL_GPU}"
  )"
  compute_process_count="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
      grep -Fxc "${gpu_uuid}" || true
  )"
  echo "preflight_mem_mib=${memory_mib} preflight_util_pct=${utilization_pct} compute_processes=${compute_process_count}"
  if (( memory_mib >= 10000 || utilization_pct >= 10 ||
        compute_process_count != 0 )); then
    echo "[FATAL] physical GPU ${PHYSICAL_GPU} is occupied" >&2
    exit 3
  fi
}

# Host Python is an audit/provenance tool only. The pinned image has no Python.
python3 "${MODULE_DIR}/rotation_contract.py" \
  --source "${REAL_SOURCE}" \
  --check-cpp "${MODULE_DIR}/include/generated_rotation_contract.hpp"
python3 "${MODULE_DIR}/verify_static.py"

# Build inside the immutable image; CMake and the build script are Python-free.
docker run --rm \
  -v "${PROJECT_ROOT}:/repo" \
  -w /repo \
  --entrypoint /bin/bash \
  "${IMAGE_TAG}" \
  -lc "/repo/fhe/gpu_cache/build_in_fideslib.sh && chown -R ${HOST_UID}:${HOST_GID} /repo/fhe/gpu_cache/build"
if [[ ! -x "${BINARY}" ]]; then
  echo "[FATAL] cache gate binary was not produced" >&2
  exit 2
fi

check_gpu_free

# Process 1: provision. Client material and public cache are distinct mounts.
docker run --rm \
  --gpus "device=${PHYSICAL_GPU}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${PROJECT_ROOT}:/repo:ro" \
  -v "${CACHE_PARENT}:/public-cache-parent" \
  -v "${CLIENT_DIR}:/client" \
  --entrypoint /repo/fhe/gpu_cache/build/dnagpt_fides_cache \
  "${IMAGE_TAG}" \
  provision \
  --gpu 0 \
  --cache-dir "/public-cache-parent/${CACHE_NAME}" \
  --oracle-key-output /client/oracle-client.bin \
  --metadata-output /client/provision-metadata.json

# Seal and verify on the Brev host, never inside the Python-free image.
python3 "${MODULE_DIR}/cache_contract.py" seal \
  --cache-dir "${CACHE_DIR}" \
  --real-source "${REAL_SOURCE}" \
  --module-source "${MODULE_SOURCE}" \
  --image "${IMAGE_EVIDENCE}" \
  --provision-metadata "${PROVISION_METADATA}"
MANIFEST_SHA="$(
  python3 "${MODULE_DIR}/cache_contract.py" verify \
    --cache-dir "${CACHE_DIR}" \
    --real-source "${REAL_SOURCE}" \
    --module-source "${MODULE_SOURCE}" \
    --image "${IMAGE_EVIDENCE}" \
    --print-sha-only
)"
SOURCE_SHA="$(sha256sum "${MODULE_SOURCE}" | cut -d' ' -f1)"
readonly MANIFEST_SHA SOURCE_SHA

check_gpu_free

# Process 2: a clean container reload. The public cache is read-only; client
# material is a separate read-only mount and is loaded only for the final oracle
# decrypt after the encrypted evaluator has completed.
docker run --rm \
  --gpus "device=${PHYSICAL_GPU}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${PROJECT_ROOT}:/repo:ro" \
  -v "${CACHE_DIR}:/public-cache:ro" \
  -v "${CLIENT_DIR}:/client:ro" \
  -v "${OUTPUT_PARENT}:/evidence-parent" \
  --entrypoint /repo/fhe/gpu_cache/build/dnagpt_fides_cache \
  "${IMAGE_TAG}" \
  reload \
  --gpu 0 \
  --cache-dir /public-cache \
  --oracle-key /client/oracle-client.bin \
  --output "/evidence-parent/${OUTPUT_NAME}" \
  --manifest-sha256 "${MANIFEST_SHA}" \
  --container-image "${IMAGE_EVIDENCE}" \
  --source-sha256 "${SOURCE_SHA}"

python3 "${MODULE_DIR}/cache_contract.py" verify \
  --cache-dir "${CACHE_DIR}" \
  --real-source "${REAL_SOURCE}" \
  --module-source "${MODULE_SOURCE}" \
  --image "${IMAGE_EVIDENCE}"
echo "[pass] immutable evidence: ${OUTPUT}"
