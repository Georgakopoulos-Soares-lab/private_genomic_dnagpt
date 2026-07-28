#!/usr/bin/env bash
# Orchestrator for the general causal-attention prototype
# (real_dnagpt_fides_scheme_b_general_attention): waits for host load + GPU
# capacity, then runs one evaluation of the requested gate (ln1, attention,
# or full) against the real T=3 fixture unattended.
#
# Mirrors wait_and_run_scheme_b_lintransform.sh's capacity-wait logic
# exactly (same LOAD_THRESHOLD/POLL_SECONDS/MAX_WAIT_SECONDS defaults, same
# stderr-only log() inside wait_for_capacity -- see
# wait_and_run_scheme_b.sh's header comment for the 2026-07-25 incident this
# avoids). Runs directly on the Brev host (not inside a container) since it
# needs host-level `uptime`/`nvidia-smi` and drives
# launch_brev_scheme_b_general_attention.sh, which itself launches the
# actual GPU work in a detached `docker run`.
#
# Safe to re-run: skips if the evidence JSON already exists, and never
# overwrites an existing log/done/output file
# (launch_brev_scheme_b_general_attention.sh is fail-closed on that
# already).
set -uo pipefail

readonly REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
readonly SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_general_attention_v1/gpu_real_scheme_b}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly FIXTURE_SUBDIR="${SCHEME_B_FIXTURE_SUBDIR:-real_fixture/gsr_pos0_block0_t3_d63353abdc1a_52d046d1fcf0}"
readonly IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
readonly RUN_TAG="${SCHEME_B_GENERAL_ATTENTION_RUN_TAG:?set SCHEME_B_GENERAL_ATTENTION_RUN_TAG (must contain _scheme_b_ and _general_attention_)}"
readonly GATE="${SCHEME_B_GENERAL_ATTENTION_GATE:-attention}"

readonly LOAD_THRESHOLD="${LOAD_THRESHOLD:-250}"
readonly POLL_SECONDS="${POLL_SECONDS:-60}"
readonly MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-259200}" # give up after 72h
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/scheme_b_general_attention_orchestrator.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${ORCHESTRATOR_LOG}"
}

pick_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    while IFS=',' read -r idx mem util; do
      idx="${idx// /}"; mem="${mem// /}"; util="${util// /}"
      if (( mem < 10000 && util < 10 )); then
        echo "${idx}"
        break
      fi
    done
}

# See wait_and_run_scheme_b.sh for why log() inside this function MUST go to
# stderr only: this function's stdout is captured by the caller as the GPU
# index, and tee'd poll lines leaking onto stdout corrupted that capture in
# a real 2026-07-25 incident (exit 125, "unresolvable CDI devices").
wait_for_capacity() {
  local start_ts gpu load1 load1_int now_ts
  start_ts=$(date +%s)
  while true; do
    load1="$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}')"
    load1="${load1// /}"
    load1_int="${load1%.*}"
    gpu="$(pick_idle_gpu)"
    log "poll: load1=${load1} threshold=${LOAD_THRESHOLD} idle_gpu=${gpu:-none}" >&2
    if [[ -n "${gpu}" ]] && (( load1_int < LOAD_THRESHOLD )); then
      echo "${gpu}"
      return 0
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts > MAX_WAIT_SECONDS )); then
      log "giving up: no capacity within ${MAX_WAIT_SECONDS}s" >&2
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
}

run_general_attention() {
  local tag="$1" gpu="$2"
  local done_file="${SOURCE_DIR}/${tag}.done"
  log "launching general-attention run tag=${tag} gpu=${gpu} gate=${GATE}"
  if ! "${SOURCE_DIR}/launch_brev_scheme_b_general_attention.sh" "${gpu}" "${REMOTE_ROOT}" \
      "${SOURCE_SUBDIR}" "${FIXTURE_SUBDIR}" "${IMAGE}" "${tag}" "${GATE}" \
      >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for tag=${tag}; will retry capacity wait"
    return 99
  fi
  while [[ ! -f "${done_file}" ]]; do
    sleep 15
  done
  local status
  status="$(cat "${done_file}")"
  log "tag=${tag} process exit status=${status}"
  return "${status}"
}

main() {
  log "general-attention orchestrator starting (pid $$) tag=${RUN_TAG} gate=${GATE}"
  if [[ -f "${SOURCE_DIR}/evidence/${RUN_TAG}.json" ]]; then
    log "skip: evidence already exists at ${RUN_TAG}.json"
    exit 0
  fi
  local gpu rc
  while true; do
    if ! gpu="$(wait_for_capacity)"; then
      log "FATAL: giving up, no capacity found"
      exit 1
    fi
    run_general_attention "${RUN_TAG}" "${gpu}"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      sleep 30
      continue
    fi
    if [[ "${rc}" -ne 0 ]]; then
      log "FATAL: general-attention run failed with exit ${rc}; stopping (manual investigation needed)"
      exit "${rc}"
    fi
    log "general-attention run PASSED"
    break
  done
  log "SCHEME B GENERAL ATTENTION RUN COMPLETE"
}

main "$@"
