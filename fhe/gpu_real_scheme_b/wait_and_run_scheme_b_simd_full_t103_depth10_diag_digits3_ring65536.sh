#!/usr/bin/env bash
# Capacity-gated T=103 Token-SIMD depth-10-diag/digits-3/ring-65536
# complete block-0 gate.
set -uo pipefail

readonly REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
readonly SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_general_attention_v1/gpu_real_scheme_b}"
readonly FIXTURE_SUBDIR="${SCHEME_B_FIXTURE_SUBDIR:-real_fixture/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0}"
readonly IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
readonly RUN_TAG="${SCHEME_B_SIMD_FULL_DEPTH10_DIAG_DIGITS3_RING65536_RUN_TAG:?set SCHEME_B_SIMD_FULL_DEPTH10_DIAG_DIGITS3_RING65536_RUN_TAG}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/${RUN_TAG}.orchestrator.log"

# shellcheck source=../gpu_common/capacity_lib.sh
# shellcheck disable=SC1091
source "${REMOTE_ROOT}/gpu_common/capacity_lib.sh"

if [[ "${RUN_TAG}" != *"_scheme_b_"* ]] || \
   [[ "${RUN_TAG}" != *"_simd_full_t103_depth10_diag_digits3_ring65536_"* ]]; then
  echo "[FATAL] bad depth-10-diag/digits-3/ring-65536 SIMD full-block tag ${RUN_TAG}" >&2
  exit 2
fi
if [[ -e "${ORCHESTRATOR_LOG}" ]]; then
  echo "[FATAL] refusing to reuse orchestrator log ${ORCHESTRATOR_LOG}" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" |
    tee -a "${ORCHESTRATOR_LOG}"
}

run_gate() {
  local gpu done_file output_file rc
  done_file="${SOURCE_DIR}/${RUN_TAG}.done"
  output_file="${SOURCE_DIR}/evidence/${RUN_TAG}.json"
  if [[ -f "${output_file}" ]]; then
    log "skip: full-block evidence already exists"
    return 0
  fi
  while true; do
    if ! gpu="$(gpucap_wait_for_capacity)"; then
      log "FATAL: no GPU capacity found"
      return 1
    fi
    log "launching depth-10-diag/digits-3/ring-65536 SIMD full block tag=${RUN_TAG} gpu=${gpu}"
    if ! "${SOURCE_DIR}/launch_brev_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh" \
        "${gpu}" "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${FIXTURE_SUBDIR}" \
        "${IMAGE}" "${RUN_TAG}" >>"${ORCHESTRATOR_LOG}" 2>&1; then
      log "launch refused; returning to capacity wait"
      sleep 30
      continue
    fi
    while [[ ! -f "${done_file}" ]]; do
      sleep 15
    done
    rc="$(sed -n '1p' "${done_file}")"
    log "depth-10-diag/digits-3/ring-65536 SIMD full-block process exit status=${rc}"
    if [[ "${rc}" -ne 0 ]]; then
      return "${rc}"
    fi
    return 0
  done
}

main() {
  log "depth-10-diag/digits-3/ring-65536 SIMD full-block orchestrator starting (pid $$)"
  if ! run_gate; then
    log "FATAL: SIMD full-block gate failed"
    exit 1
  fi
  log "SCHEME B SIMD FULL T103 DEPTH10 DIAG DIGITS3 RING65536 COMPLETE"
}

main "$@"
