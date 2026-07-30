#!/usr/bin/env bash
# Capacity-gated orchestrator for the T=2 Scheme B block-0 -> block-1
# full-hidden-state refresh correctness gate.
set -uo pipefail

REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_two_block_refresh_v1}"
FIXTURE_SUBDIR="${SCHEME_B_FIXTURE_SUBDIR:-real_fixture/gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0}"
IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
RUN_TAG="${SCHEME_B_TWO_BLOCK_REFRESH_RUN_TAG:?set SCHEME_B_TWO_BLOCK_REFRESH_RUN_TAG (must contain _scheme_b_ and _two_block_refresh_)}"
readonly REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE RUN_TAG
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly DONE="${SOURCE_DIR}/${RUN_TAG}.done"
readonly OUTPUT="${SOURCE_DIR}/evidence/${RUN_TAG}.json"
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/${RUN_TAG}.orchestrator.log"

# shellcheck source=../gpu_common/capacity_lib.sh
source "${REMOTE_ROOT}/gpu_common/capacity_lib.sh"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" |
    tee -a "${ORCHESTRATOR_LOG}"
}

run_two_block_refresh() {
  local gpu="$1"
  log "launching two-block refresh tag=${RUN_TAG} gpu=${gpu}"
  if ! "${SOURCE_DIR}/launch_brev_scheme_b_two_block_refresh.sh" \
      "${gpu}" "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${FIXTURE_SUBDIR}" \
      "${IMAGE}" "${RUN_TAG}" >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for tag=${RUN_TAG}; will retry capacity wait"
    return 99
  fi

  while [[ ! -f "${DONE}" ]]; do
    sleep 15
  done
  local status
  status="$(sed -n '1p' "${DONE}")"
  log "tag=${RUN_TAG} process exit status=${status}"
  return "${status}"
}

main() {
  if [[ -f "${OUTPUT}" ]]; then
    exit 0
  fi
  if [[ -e "${ORCHESTRATOR_LOG}" ]]; then
    echo "[FATAL] refusing to reuse orchestrator log ${ORCHESTRATOR_LOG}" >&2
    exit 2
  fi

  log "two-block refresh orchestrator starting (pid $$) tag=${RUN_TAG}"
  local gpu rc
  while true; do
    if ! gpu="$(gpucap_wait_for_capacity)"; then
      log "FATAL: no capacity found within configured wait"
      exit 1
    fi
    run_two_block_refresh "${gpu}"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      sleep 30
      continue
    fi
    if [[ "${rc}" -ne 0 ]]; then
      log "FATAL: two-block refresh failed with exit ${rc}; manual investigation required"
      exit "${rc}"
    fi
    log "SCHEME B TWO-BLOCK REFRESH RUN PASSED"
    break
  done
}

main "$@"
