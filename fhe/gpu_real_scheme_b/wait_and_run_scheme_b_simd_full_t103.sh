#!/usr/bin/env bash
# Capacity-gated matched pair for the T=103 Token-SIMD linear micro-gate.
# The packed gate runs first. The serial control runs only after packed passes.
set -uo pipefail

readonly REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
readonly SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_general_attention_v1/gpu_real_scheme_b}"
readonly FIXTURE_SUBDIR="${SCHEME_B_FIXTURE_SUBDIR:-real_fixture/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0}"
readonly IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
readonly PACKED_TAG="${SCHEME_B_SIMD_PACKED_RUN_TAG:?set SCHEME_B_SIMD_PACKED_RUN_TAG}"
readonly SERIAL_TAG="${SCHEME_B_SIMD_SERIAL_RUN_TAG:?set SCHEME_B_SIMD_SERIAL_RUN_TAG}"
readonly PAIR_TAG="${SCHEME_B_SIMD_PAIR_TAG:?set SCHEME_B_SIMD_PAIR_TAG}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/${PAIR_TAG}.orchestrator.log"

# shellcheck source=../gpu_common/capacity_lib.sh
# shellcheck disable=SC1091
source "${REMOTE_ROOT}/gpu_common/capacity_lib.sh"

for tag_mode in "${PACKED_TAG}:packed" "${SERIAL_TAG}:serial_control"; do
  tag="${tag_mode%%:*}"
  mode="${tag_mode#*:}"
  if [[ "${tag}" != *"_scheme_b_"* ]] || \
     [[ "${tag}" != *"_simd_linear_t103_"* ]] || \
     [[ "${tag}" != *"_${mode}_"* ]]; then
    echo "[FATAL] bad ${mode} tag ${tag}" >&2
    exit 2
  fi
done
if [[ -e "${ORCHESTRATOR_LOG}" ]]; then
  echo "[FATAL] refusing to reuse orchestrator log ${ORCHESTRATOR_LOG}" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" |
    tee -a "${ORCHESTRATOR_LOG}"
}

run_mode() {
  local tag="$1" mode="$2" gpu done_file output_file rc
  done_file="${SOURCE_DIR}/${tag}.done"
  output_file="${SOURCE_DIR}/evidence/${tag}.json"
  if [[ -f "${output_file}" ]]; then
    log "skip: ${mode} evidence already exists"
    return 0
  fi
  while true; do
    if ! gpu="$(gpucap_wait_for_capacity)"; then
      log "FATAL: no capacity found for ${mode}"
      return 1
    fi
    log "launching SIMD mode=${mode} tag=${tag} gpu=${gpu}"
    if ! "${SOURCE_DIR}/launch_brev_scheme_b_simd_linear_t103.sh" \
        "${gpu}" "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${FIXTURE_SUBDIR}" \
        "${IMAGE}" "${tag}" "${mode}" >>"${ORCHESTRATOR_LOG}" 2>&1; then
      log "launch refused for ${mode}; returning to capacity wait"
      sleep 30
      continue
    fi
    while [[ ! -f "${done_file}" ]]; do
      sleep 15
    done
    rc="$(sed -n '1p' "${done_file}")"
    log "mode=${mode} process exit status=${rc}"
    if [[ "${rc}" -ne 0 ]]; then
      return "${rc}"
    fi
    return 0
  done
}

main() {
  log "SIMD matched-pair orchestrator starting (pid $$)"
  if ! run_mode "${PACKED_TAG}" packed; then
    log "FATAL: packed SIMD gate failed; serial control will not run"
    exit 1
  fi
  log "packed SIMD gate passed; proceeding to matched serial control"
  if ! run_mode "${SERIAL_TAG}" serial_control; then
    log "FATAL: serial-control SIMD gate failed"
    exit 1
  fi
  log "SCHEME B SIMD LINEAR T103 MATCHED PAIR COMPLETE"
}

main "$@"
