#!/usr/bin/env bash
# Waits for a low-load CPU window, builds only the SIMD micro-gate target
# without reserving a GPU, then hands off to the capacity-gated matched pair.
set -uo pipefail

readonly REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
readonly SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_general_attention_v1/gpu_real_scheme_b}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
readonly PAIR_TAG="${SCHEME_B_SIMD_PAIR_TAG:?set SCHEME_B_SIMD_PAIR_TAG}"
readonly LOAD_THRESHOLD="${SCHEME_B_SIMD_BUILD_LOAD_THRESHOLD:-250}"
readonly POLL_SECONDS="${SCHEME_B_SIMD_BUILD_POLL_SECONDS:-60}"
readonly MAX_WAIT_SECONDS="${SCHEME_B_SIMD_BUILD_MAX_WAIT_SECONDS:-259200}"
readonly BUILD_LOG="${SOURCE_DIR}/${PAIR_TAG}.build.log"
readonly BUILD_DONE="${SOURCE_DIR}/${PAIR_TAG}.build.done"
readonly TARGET=real_dnagpt_fides_scheme_b_simd_linear_t103

if [[ "${PAIR_TAG}" != *"_scheme_b_"* ]] || \
   [[ "${PAIR_TAG}" != *"_simd_linear_t103_"* ]]; then
  echo "[FATAL] pair tag must contain _scheme_b_ and _simd_linear_t103_" >&2
  exit 2
fi
for path in "${BUILD_LOG}" "${BUILD_DONE}"; do
  if [[ -e "${path}" ]]; then
    echo "[FATAL] refusing to overwrite ${path}" >&2
    exit 2
  fi
done

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_build_capacity() {
  local start_ts now_ts load1 load1_int
  start_ts="$(date +%s)"
  while true; do
    load1="$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}')"
    load1="${load1// /}"
    load1_int="${load1%.*}"
    log "build poll: load1=${load1} threshold=${LOAD_THRESHOLD}"
    if (( load1_int < LOAD_THRESHOLD )); then
      return 0
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > MAX_WAIT_SECONDS )); then
      log "FATAL: no build-capacity window within ${MAX_WAIT_SECONDS}s"
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
}

build_target() {
  log "building ${TARGET} without GPU allocation"
  docker run --rm \
    -v "${REMOTE_ROOT}:/work" \
    -w "/work/${SOURCE_SUBDIR}" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc "
      cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DFIDESLIB_PACKAGE_DIR=/usr/local/share/fideslib/cmake \
        -DFIDESLIB_ARCH=80-real \
        -DCMAKE_CUDA_ARCHITECTURES=80-real &&
      cmake --build build --target ${TARGET} --parallel 2
    " >"${BUILD_LOG}" 2>&1
  local status=$?
  printf '%s\n' "${status}" >"${BUILD_DONE}"
  if [[ "${status}" -ne 0 ]]; then
    log "FATAL: SIMD target build failed with exit ${status}"
    return "${status}"
  fi
  log "SIMD target build passed"
}

main() {
  wait_for_build_capacity || exit $?
  build_target || exit $?
  log "handing off to capacity-gated packed/serial-control pair"
  "${SOURCE_DIR}/wait_and_run_scheme_b_simd_linear_t103.sh"
}

main "$@"
