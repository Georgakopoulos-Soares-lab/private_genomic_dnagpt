#!/usr/bin/env bash
# Orchestrator for the CROSS-PROCESS context/key serialization prototype:
# waits for host capacity, launches the writer phase (builds context/keys,
# serializes to a fresh state directory, exits without evaluating), waits
# for it to finish, then launches the reader phase (a genuinely separate
# `docker run` / OS process that deserializes everything from the state
# directory, calls LoadContext, and evaluates the full gate against the
# unchanged 4e-2 oracle). Deletes the state directory afterward regardless
# of pass/fail -- it contains a serialized secret-key file
# (~/.agents/policies/secrets.md; see the writer/reader C++ source header
# comments for the full hygiene rationale).
#
# Mirrors wait_and_run_scheme_b_cached.sh's capacity-wait logic exactly (same
# LOAD_THRESHOLD/POLL_SECONDS/MAX_WAIT_SECONDS defaults, same stderr-only
# log() inside wait_for_capacity -- see wait_and_run_scheme_b.sh's header
# comment for the 2026-07-25 incident this avoids). Runs directly on the
# Brev host (not inside a container) since it needs host-level
# `uptime`/`nvidia-smi` and drives launch_brev_scheme_b_serialize_writer.sh/
# _reader.sh, which themselves launch the actual work in detached
# `docker run` invocations.
#
# Safe to re-run: skips if the reader's evidence JSON already exists.
set -uo pipefail

readonly REMOTE_ROOT="${SCHEME_B_REMOTE_ROOT:-/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724}"
readonly SOURCE_SUBDIR="${SCHEME_B_SOURCE_SUBDIR:-gpu_real_scheme_b_serialized_v1/fhe/gpu_real_scheme_b}"
readonly SOURCE_DIR="${REMOTE_ROOT}/${SOURCE_SUBDIR}"
readonly FIXTURE_SUBDIR="${SCHEME_B_FIXTURE_SUBDIR:-real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0}"
readonly IMAGE="${SCHEME_B_IMAGE:-dnagpt-fideslib:786c-asymfix2}"
readonly WRITER_TAG="${SCHEME_B_SERIALIZE_WRITER_RUN_TAG:?set SCHEME_B_SERIALIZE_WRITER_RUN_TAG (must contain _scheme_b_ and _serialized_)}"
readonly READER_TAG="${SCHEME_B_SERIALIZE_READER_RUN_TAG:?set SCHEME_B_SERIALIZE_READER_RUN_TAG (must contain _scheme_b_ and _serialized_)}"
readonly STATE_SUBDIR="${SCHEME_B_SERIALIZE_STATE_SUBDIR:-${SOURCE_SUBDIR}/state/${READER_TAG}}"

readonly LOAD_THRESHOLD="${LOAD_THRESHOLD:-250}"
readonly POLL_SECONDS="${POLL_SECONDS:-60}"
readonly MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-259200}" # give up after 72h
readonly ORCHESTRATOR_LOG="${SOURCE_DIR}/scheme_b_serialize_orchestrator.log"

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
# index, and a real 2026-07-25 incident corrupted that capture when poll
# lines leaked onto stdout.
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

run_phase() {
  local phase="$1" tag="$2" gpu="$3"
  local done_file="${SOURCE_DIR}/${tag}.done"
  log "launching ${phase} phase tag=${tag} gpu=${gpu}"
  if [[ "${phase}" == "writer" ]]; then
    if ! SCHEME_B_SERIALIZE_WRITER_RUN_TAG="${tag}" SCHEME_B_IMAGE="${IMAGE}" \
        "${SOURCE_DIR}/launch_brev_scheme_b_serialize_writer.sh" "${gpu}" "${REMOTE_ROOT}" \
        "${SOURCE_SUBDIR}" "${STATE_SUBDIR}" \
        >>"${ORCHESTRATOR_LOG}" 2>&1; then
      log "launch refused for ${phase} tag=${tag}; will retry capacity wait"
      return 99
    fi
  else
    if ! SCHEME_B_SERIALIZE_READER_RUN_TAG="${tag}" SCHEME_B_IMAGE="${IMAGE}" \
        "${SOURCE_DIR}/launch_brev_scheme_b_serialize_reader.sh" "${gpu}" "${REMOTE_ROOT}" \
        "${SOURCE_SUBDIR}" "${STATE_SUBDIR}" "${FIXTURE_SUBDIR}" \
        >>"${ORCHESTRATOR_LOG}" 2>&1; then
      log "launch refused for ${phase} tag=${tag}; will retry capacity wait"
      return 99
    fi
  fi
  while [[ ! -f "${done_file}" ]]; do
    sleep 15
  done
  local status
  status="$(cat "${done_file}")"
  log "${phase} tag=${tag} process exit status=${status}"
  return "${status}"
}

cleanup_state_dir() {
  # Secret-key hygiene: the state directory contains a serialized secret
  # key. Delete the whole directory once the reader phase has run,
  # regardless of pass/fail -- never log the secret-key filename itself,
  # just that cleanup happened.
  local state_dir_abs="${REMOTE_ROOT}/${STATE_SUBDIR}"
  if [[ -d "${state_dir_abs}" ]]; then
    rm -rf "${state_dir_abs}"
    log "state directory removed (secret-key hygiene)"
  fi
}

main() {
  log "serialize orchestrator starting (pid $$) writer_tag=${WRITER_TAG} reader_tag=${READER_TAG}"
  if [[ -f "${SOURCE_DIR}/evidence/${READER_TAG}.json" ]]; then
    log "skip: reader evidence already exists at ${READER_TAG}.json"
    exit 0
  fi

  local gpu rc

  if [[ ! -f "${SOURCE_DIR}/evidence/${WRITER_TAG}.json" ]]; then
    while true; do
      if ! gpu="$(wait_for_capacity)"; then
        log "FATAL: giving up, no capacity found for writer phase"
        exit 1
      fi
      run_phase writer "${WRITER_TAG}" "${gpu}"
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

  while true; do
    if ! gpu="$(wait_for_capacity)"; then
      log "FATAL: giving up, no capacity found for reader phase"
      cleanup_state_dir
      exit 1
    fi
    run_phase reader "${READER_TAG}" "${gpu}"
    rc=$?
    if [[ "${rc}" -eq 99 ]]; then
      sleep 30
      continue
    fi
    if [[ "${rc}" -ne 0 ]]; then
      log "FATAL: reader phase failed with exit ${rc}; stopping (manual investigation needed)"
      cleanup_state_dir
      exit "${rc}"
    fi
    log "reader phase PASSED"
    break
  done

  cleanup_state_dir
  log "SCHEME B SERIALIZE (CROSS-PROCESS) RUN COMPLETE"
}

main "$@"
