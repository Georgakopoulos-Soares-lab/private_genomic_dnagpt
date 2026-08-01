#!/usr/bin/env bash
# Orchestrator for the COMBINED-lever correctness micro-gate: 2-GPU
# process-per-GPU sharding (Stage 1, Q/K/V split) PLUS the CPU-side
# diagonal-vector cache, both on top of the same passing Token-SIMD B=8
# T=103 depth-13/digits-3/ring-65536 base. Forked from
# wait_and_run_scheme_b_simd_shard_qkv_t103_depth13.sh: the writer phase is
# UNCHANGED (reused verbatim via the existing
# launch_brev_scheme_b_simd_shard_writer_t103_depth13.sh -- the cache lever
# only touches the reader's matmul(), so the writer needs no fork at all);
# only the reader phase is switched to the new cpudiagcache-carrying
# reader's launch script.
#
# Waits for host capacity, launches the (unchanged) Token-SIMD-parameter-
# matched shard writer, waits for it to finish, then waits for capacity on
# TWO distinct physical GPUs simultaneously and launches two independent
# shard-reader-cpudiagcache workers concurrently -- worker A with
# `--part query,key`, worker B with `--part value` -- both pointed
# read-only at the same state directory, each with its own independent
# per-process diagonal cache (no cross-process cache sharing; see the new
# reader source's file header comment). Deletes the state directory
# afterward regardless of pass/fail (it contains a serialized secret-key
# file).
#
# Safe to re-run: skips any phase whose evidence JSON already exists.
set -uo pipefail

readonly REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
readonly SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_general_attention_v1/gpu_real_scheme_b}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly FIXTURE_SUBDIR="${SCHEME_B_FIXTURE_SUBDIR:-real_fixture/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0}"
readonly IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
readonly WRITER_TAG="${SCHEME_B_SHARD_WRITER_RUN_TAG:?set SCHEME_B_SHARD_WRITER_RUN_TAG (must contain _scheme_b_ and _qkvshard_)}"
readonly READER_A_TAG="${SCHEME_B_SHARD_READER_A_RUN_TAG:?set SCHEME_B_SHARD_READER_A_RUN_TAG (must contain _scheme_b_, _qkvshard_, and _cpudiagcache_)}"
readonly READER_B_TAG="${SCHEME_B_SHARD_READER_B_RUN_TAG:?set SCHEME_B_SHARD_READER_B_RUN_TAG (must contain _scheme_b_, _qkvshard_, and _cpudiagcache_)}"
readonly READER_A_PARTS="${SCHEME_B_SHARD_READER_A_PARTS:-query,key}"
readonly READER_B_PARTS="${SCHEME_B_SHARD_READER_B_PARTS:-value}"
readonly STATE_SUBDIR="${SCHEME_B_SHARD_STATE_SUBDIR:-${SOURCE_SUBDIR}/state/${WRITER_TAG}}"

readonly LOAD_THRESHOLD="${LOAD_THRESHOLD:-250}"
readonly POLL_SECONDS="${POLL_SECONDS:-60}"
readonly MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-259200}" # give up after 72h
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/${WRITER_TAG}.cpudiagcache_orchestrator.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${ORCHESTRATOR_LOG}"
}

# Prints up to $1 distinct idle GPU indices (mem<10000, util<10), one per
# line, in ascending index order. May print fewer than requested.
pick_idle_gpus() {
  local want="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F', *' -v want="${want}" '
      { gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if ($2 + 0 < 10000 && $3 + 0 < 10) { print $1; count++ }
        if (count >= want) exit
      }'
}

# See wait_and_run_scheme_b_serialize.sh for why log() inside this function
# MUST go to stderr only: this function's stdout is captured by the caller
# as the newline-separated GPU index list.
wait_for_capacity() {
  local need="$1"
  local start_ts gpus gpu_count load1 load1_int now_ts
  start_ts=$(date +%s)
  while true; do
    load1="$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}')"
    load1="${load1// /}"
    load1_int="${load1%.*}"
    gpus="$(pick_idle_gpus "${need}")"
    gpu_count=0
    if [[ -n "${gpus}" ]]; then
      gpu_count="$(printf '%s\n' "${gpus}" | grep -c .)"
    fi
    log "poll: load1=${load1} threshold=${LOAD_THRESHOLD} need=${need} idle_gpus_found=${gpu_count} (${gpus//$'\n'/,})" >&2
    if (( gpu_count >= need )) && (( load1_int < LOAD_THRESHOLD )); then
      printf '%s\n' "${gpus}"
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

run_writer_phase() {
  local gpu="$1"
  local done_file="${SOURCE_DIR}/${WRITER_TAG}.done"
  log "launching writer phase (unchanged writer) tag=${WRITER_TAG} gpu=${gpu}"
  if ! SCHEME_B_SHARD_WRITER_RUN_TAG="${WRITER_TAG}" SCHEME_B_IMAGE="${IMAGE}" \
      "${SOURCE_DIR}/launch_brev_scheme_b_simd_shard_writer_t103_depth13.sh" "${gpu}" \
      "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${STATE_SUBDIR}" \
      >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for writer tag=${WRITER_TAG}; will retry capacity wait"
    return 99
  fi
  while [[ ! -f "${done_file}" ]]; do
    sleep 15
  done
  local rc
  rc="$(cat "${done_file}")"
  log "writer tag=${WRITER_TAG} process exit status=${rc}"
  return "${rc}"
}

# Launches BOTH shard-reader-cpudiagcache workers back-to-back (each
# launch script itself detaches into its own nohup'd docker run and returns
# immediately, so this is not a blocking pair) and then waits for both
# .done files.
run_reader_phase_pair() {
  local gpu_a="$1" gpu_b="$2"
  local done_a="${SOURCE_DIR}/${READER_A_TAG}.done"
  local done_b="${SOURCE_DIR}/${READER_B_TAG}.done"

  log "launching reader worker A (cpudiagcache) tag=${READER_A_TAG} gpu=${gpu_a} parts=${READER_A_PARTS}"
  if ! SCHEME_B_SHARD_READER_RUN_TAG="${READER_A_TAG}" SCHEME_B_IMAGE="${IMAGE}" \
      "${SOURCE_DIR}/launch_brev_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.sh" "${gpu_a}" \
      "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${STATE_SUBDIR}" "${FIXTURE_SUBDIR}" \
      "${READER_A_PARTS}" \
      >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for reader worker A tag=${READER_A_TAG}; will retry capacity wait"
    return 99
  fi

  log "launching reader worker B (cpudiagcache) tag=${READER_B_TAG} gpu=${gpu_b} parts=${READER_B_PARTS}"
  if ! SCHEME_B_SHARD_READER_RUN_TAG="${READER_B_TAG}" SCHEME_B_IMAGE="${IMAGE}" \
      "${SOURCE_DIR}/launch_brev_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.sh" "${gpu_b}" \
      "${REMOTE_ROOT}" "${SOURCE_SUBDIR}" "${STATE_SUBDIR}" "${FIXTURE_SUBDIR}" \
      "${READER_B_PARTS}" \
      >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for reader worker B tag=${READER_B_TAG} (worker A already launched -- leaving it to finish, not killing it); will retry capacity wait for B only after A completes"
    while [[ ! -f "${done_a}" ]]; do
      sleep 15
    done
    log "reader worker A tag=${READER_A_TAG} process exit status=$(cat "${done_a}") (worker B never launched this round)"
    return 99
  fi

  while [[ ! -f "${done_a}" || ! -f "${done_b}" ]]; do
    sleep 15
  done
  local rc_a rc_b
  rc_a="$(cat "${done_a}")"
  rc_b="$(cat "${done_b}")"
  log "reader worker A tag=${READER_A_TAG} process exit status=${rc_a}"
  log "reader worker B tag=${READER_B_TAG} process exit status=${rc_b}"
  if [[ "${rc_a}" -ne 0 || "${rc_b}" -ne 0 ]]; then
    return 1
  fi
  return 0
}

cleanup_state_dir() {
  # Secret-key hygiene: the state directory contains a serialized secret
  # key. Delete the whole directory once both reader workers have run,
  # regardless of pass/fail -- never log the secret-key filename itself,
  # just that cleanup happened.
  local state_dir_abs="${REMOTE_ROOT}/${STATE_SUBDIR}"
  if [[ -d "${state_dir_abs}" ]]; then
    rm -rf "${state_dir_abs}"
    log "state directory removed (secret-key hygiene)"
  fi
}

main() {
  log "shard-qkv-cpudiagcache orchestrator starting (pid $$) writer_tag=${WRITER_TAG} reader_a_tag=${READER_A_TAG} reader_b_tag=${READER_B_TAG}"
  if [[ -f "${SOURCE_DIR}/evidence/${READER_A_TAG}.json" && \
        -f "${SOURCE_DIR}/evidence/${READER_B_TAG}.json" ]]; then
    log "skip: both reader-worker evidence files already exist"
    exit 0
  fi

  local gpus gpu rc

  if [[ ! -f "${SOURCE_DIR}/evidence/${WRITER_TAG}.json" ]]; then
    while true; do
      if ! gpus="$(wait_for_capacity 1)"; then
        log "FATAL: giving up, no capacity found for writer phase"
        exit 1
      fi
      gpu="$(printf '%s\n' "${gpus}" | head -n1)"
      run_writer_phase "${gpu}"
      rc=$?
      if [[ "${rc}" -eq 99 ]]; then
        sleep 30
        continue
      fi
      if [[ "${rc}" -ne 0 ]]; then
        log "FATAL: writer phase failed with exit ${rc}; stopping (manual investigation needed)"
        cleanup_state_dir
        exit "${rc}"
      fi
      log "writer phase DONE"
      break
    done
  else
    log "skip: writer evidence already exists at ${WRITER_TAG}.json"
  fi

  if [[ -f "${SOURCE_DIR}/evidence/${READER_A_TAG}.json" && \
        -f "${SOURCE_DIR}/evidence/${READER_B_TAG}.json" ]]; then
    log "skip: both reader-worker evidence files already exist (writer phase was re-run/skip only)"
    cleanup_state_dir
    log "SCHEME B TOKEN-SIMD SHARD-QKV+CPUDIAGCACHE COMBINED MICRO-GATE COMPLETE"
    exit 0
  fi

  while true; do
    if ! gpus="$(wait_for_capacity 2)"; then
      log "FATAL: giving up, no 2-GPU capacity found for reader phase"
      cleanup_state_dir
      exit 1
    fi
    local gpu_a gpu_b
    gpu_a="$(printf '%s\n' "${gpus}" | sed -n '1p')"
    gpu_b="$(printf '%s\n' "${gpus}" | sed -n '2p')"
    run_reader_phase_pair "${gpu_a}" "${gpu_b}"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      sleep 30
      continue
    fi
    if [[ "${rc}" -ne 0 ]]; then
      log "FATAL: at least one reader worker failed (see per-worker exit statuses above); stopping (manual investigation needed)"
      cleanup_state_dir
      exit "${rc}"
    fi
    log "both reader workers PASSED"
    break
  done

  cleanup_state_dir
  log "SCHEME B TOKEN-SIMD SHARD-QKV+CPUDIAGCACHE COMBINED MICRO-GATE COMPLETE"
}

main "$@"
