#!/usr/bin/env bash
# Orchestrator: waits for host load + GPU capacity, then runs the remaining
# Scheme B real-weight D=768/T=2 gates (attention, full) unattended.
#
# Runs directly on the Brev host (not inside a container) since it needs
# host-level `uptime`/`nvidia-smi` and drives launch_brev_scheme_b.sh, which
# itself launches the actual GPU work in a detached `docker run`.
#
# Safe to re-run: it skips any gate that already has evidence JSON, and it
# never overwrites an existing log/done/output file (launch_brev_scheme_b.sh
# is fail-closed on that already).
set -uo pipefail

readonly REMOTE_ROOT="/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724"
readonly SOURCE_SUBDIR="gpu_real_scheme_b_v1/fhe/gpu_real_scheme_b"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly FIXTURE_SUBDIR="real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0"
readonly IMAGE="dnagpt-fideslib:786c-asymfix2"
readonly DATE_TAG="20260725"

# 1-minute load average threshold below which we consider the host usable.
# This machine has 255 cores; a load average this high still leaves the GPU
# itself free, but CPU-bound rotation-key generation gets starved above it
# (observed: an attention-gate key generation stalled 8.5+ minutes at
# load average ~1000-1127, versus ~26s for the equivalent Scheme A gate on an
# unloaded host). Adjust via LOAD_THRESHOLD env var if needed.
readonly LOAD_THRESHOLD="${LOAD_THRESHOLD:-250}"
readonly POLL_SECONDS="${POLL_SECONDS:-60}"
readonly MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-259200}" # give up after 72h
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/scheme_b_orchestrator.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${ORCHESTRATOR_LOG}"
}

# Prints the index of the first physical GPU with memory.used < 10000 MiB and
# utilization < 10%, or nothing if none qualify.
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

# Blocks until load average is below threshold AND a GPU is idle. Echoes the
# chosen GPU index on success; returns 1 after MAX_WAIT_SECONDS with nothing.
wait_for_capacity() {
  local start_ts gpu load1 load1_int now_ts
  start_ts=$(date +%s)
  while true; do
    load1="$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}')"
    load1="${load1// /}"
    load1_int="${load1%.*}"
    gpu="$(pick_idle_gpu)"
    log "poll: load1=${load1} threshold=${LOAD_THRESHOLD} idle_gpu=${gpu:-none}"
    if [[ -n "${gpu}" ]] && (( load1_int < LOAD_THRESHOLD )); then
      echo "${gpu}"
      return 0
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts > MAX_WAIT_SECONDS )); then
      log "giving up: no capacity within ${MAX_WAIT_SECONDS}s"
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
}

# Launches one gate and blocks until its .done file appears. Returns:
#   0        gate ran and passed
#   5        gate ran and failed the oracle gate (matches the binary's own
#            metrics.passed?0:5 exit convention)
#   99       launch was refused (GPU became busy between our check and the
#            launch, or another fail-closed guard tripped) -- caller should
#            retry from wait_for_capacity
run_gate() {
  local gate="$1" tag="$2" gpu="$3"
  local done_file="${SOURCE_DIR}/${tag}.done"
  log "launching gate=${gate} tag=${tag} gpu=${gpu}"
  if ! "${SOURCE_DIR}/launch_brev_scheme_b.sh" "${gpu}" "${REMOTE_ROOT}" \
      "${SOURCE_SUBDIR}" "${FIXTURE_SUBDIR}" "${IMAGE}" "${gate}" "${tag}" \
      >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for gate=${gate}; will retry capacity wait"
    return 99
  fi
  while [[ ! -f "${done_file}" ]]; do
    sleep 15
  done
  local status
  status="$(cat "${done_file}")"
  log "gate=${gate} tag=${tag} process exit status=${status}"
  return "${status}"
}

main() {
  log "orchestrator starting (pid $$)"
  local specs=(
    "attention:fhe_fides_real_d768_t2_attention_scheme_b_a100_${DATE_TAG}"
    "full:fhe_fides_real_d768_t2_block0_scheme_b_a100_${DATE_TAG}"
  )
  local spec gate tag gpu rc
  for spec in "${specs[@]}"; do
    gate="${spec%%:*}"
    tag="${spec##*:}"
    if [[ -f "${SOURCE_DIR}/evidence/${tag}.json" ]]; then
      log "skip gate=${gate}: evidence already exists at ${tag}.json"
      continue
    fi
    while true; do
      if ! gpu="$(wait_for_capacity)"; then
        log "FATAL: giving up on gate=${gate}, no capacity found"
        exit 1
      fi
      run_gate "${gate}" "${tag}" "${gpu}"
      rc=$?
      if [[ "${rc}" -eq 99 ]]; then
        sleep 30
        continue
      fi
      if [[ "${rc}" -ne 0 ]]; then
        log "FATAL: gate=${gate} failed with exit ${rc}; stopping (manual investigation needed)"
        exit "${rc}"
      fi
      log "gate=${gate} PASSED"
      break
    done
  done
  log "ALL SCHEME B GATES COMPLETE"
}

main "$@"
