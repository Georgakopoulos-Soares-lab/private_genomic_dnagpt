#!/usr/bin/env bash
# Generic unattended orchestrator TEMPLATE. Copy this file into a new
# experiment directory, edit the "JOBS TO RUN" section below, and it will
# poll for host-load + GPU capacity and launch each job in turn using the
# same battle-tested capacity_lib.sh logic validated for Scheme B on
# 2026-07-25 (see fhe/gpu_real_scheme_b/wait_and_run_scheme_b.sh, the
# concrete instance this template was generalized from).
#
# Safe to re-run: it skips any job whose SKIP_IF_EXISTS path already exists,
# and launch_gpu_job.sh never overwrites an existing log/done/output file.
#
# Run directly on the target host (not inside a container) since it needs
# host-level `uptime`/`nvidia-smi`.
set -uo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./capacity_lib.sh
source "${LIB_DIR}/capacity_lib.sh"

# --- tunables (override via env before running, or edit these defaults) ---
export GPUCAP_LOAD_THRESHOLD="${GPUCAP_LOAD_THRESHOLD:-250}"
export GPUCAP_POLL_SECONDS="${GPUCAP_POLL_SECONDS:-60}"
export GPUCAP_MAX_WAIT_SECONDS="${GPUCAP_MAX_WAIT_SECONDS:-259200}" # 72h
readonly ORCHESTRATOR_LOG="${ORCHESTRATOR_LOG:-$(pwd)/orchestrator.log}"

log() {
  gpucap_log "$*" | tee -a "${ORCHESTRATOR_LOG}"
}

# ============================================================================
# JOBS TO RUN -- edit this section for your experiment. Each job needs:
#   name           short label for logging
#   skip_if_exists path checked at start; if it exists, the job is skipped
#   log/done       paths launch_gpu_job.sh will write to
#   output         expected result artifact (empty string "" to skip guard)
#   cmd            array: the actual command to run (e.g. a `docker run...`)
#
# Example (mirrors fhe/gpu_real_scheme_b's real usage):
#
#   add_job "attention" \
#     "/data/.../evidence/my_attention_run.json" \
#     "/data/.../my_attention_run.run.log" \
#     "/data/.../my_attention_run.done" \
#     "/data/.../evidence/my_attention_run.json" \
#     -- docker run --rm --gpus "device=\${gpu}" -v /data:/work myimage:tag \
#        /work/run_my_experiment.sh attention /work/fixture out.json
#
# JOBS is an array of pipe-free bash arrays; use add_job to populate it
# safely (avoids fragile string-splitting of commands containing spaces).
declare -a JOB_NAMES=()
declare -a JOB_SKIP_IF_EXISTS=()
declare -a JOB_LOG=()
declare -a JOB_DONE=()
declare -a JOB_OUTPUT=()
declare -a JOB_CMDS=() # newline-joined, one command array serialized per job

add_job() {
  local name="$1" skip="$2" logf="$3" donef="$4" output="$5"
  shift 5
  if [[ "${1:-}" != "--" ]]; then
    echo "add_job: expected -- before CMD" >&2
    exit 2
  fi
  shift
  JOB_NAMES+=("${name}")
  JOB_SKIP_IF_EXISTS+=("${skip}")
  JOB_LOG+=("${logf}")
  JOB_DONE+=("${donef}")
  JOB_OUTPUT+=("${output}")
  # Serialize the command array with NUL separators into one string per job.
  local joined
  joined="$(printf '%s\0' "$@")"
  JOB_CMDS+=("${joined}")
}

# vvv EDIT BELOW: replace with your own job(s) vvv
# add_job "example" "" "/tmp/example.log" "/tmp/example.done" "" -- \
#   echo "replace me with a real command"
# ^^^ EDIT ABOVE ^^^
# ============================================================================

run_job() {
  local idx="$1" gpu="$2"
  local name="${JOB_NAMES[$idx]}" logf="${JOB_LOG[$idx]}"
  local donef="${JOB_DONE[$idx]}" output="${JOB_OUTPUT[$idx]}"
  local -a cmd=()
  while IFS= read -r -d '' part; do cmd+=("${part}"); done <<<"${JOB_CMDS[$idx]}"

  log "launching job=${name} tag_gpu=${gpu}"
  if ! "${LIB_DIR}/launch_gpu_job.sh" "${gpu}" "${logf}" "${donef}" "${output}" \
      -- "${cmd[@]}" >>"${ORCHESTRATOR_LOG}" 2>&1; then
    log "launch refused for job=${name}; will retry capacity wait"
    return 99
  fi
  while [[ ! -f "${donef}" ]]; do
    sleep 15
  done
  local status
  status="$(cat "${donef}")"
  log "job=${name} process exit status=${status}"
  return "${status}"
}

main() {
  log "orchestrator starting (pid $$), ${#JOB_NAMES[@]} job(s) configured"
  if [[ "${#JOB_NAMES[@]}" -eq 0 ]]; then
    log "FATAL: no jobs configured -- edit the JOBS TO RUN section before running"
    exit 2
  fi
  local i gpu rc
  for ((i = 0; i < ${#JOB_NAMES[@]}; i++)); do
    if [[ -n "${JOB_SKIP_IF_EXISTS[$i]}" && -e "${JOB_SKIP_IF_EXISTS[$i]}" ]]; then
      log "skip job=${JOB_NAMES[$i]}: ${JOB_SKIP_IF_EXISTS[$i]} already exists"
      continue
    fi
    while true; do
      if ! gpu="$(gpucap_wait_for_capacity 2>>"${ORCHESTRATOR_LOG}")"; then
        log "FATAL: giving up on job=${JOB_NAMES[$i]}, no capacity found"
        exit 1
      fi
      run_job "${i}" "${gpu}"
      rc=$?
      if [[ "${rc}" -eq 99 ]]; then
        sleep 30
        continue
      fi
      if [[ "${rc}" -ne 0 ]]; then
        log "FATAL: job=${JOB_NAMES[$i]} failed with exit ${rc}; stopping (manual investigation needed)"
        exit "${rc}"
      fi
      log "job=${JOB_NAMES[$i]} PASSED"
      break
    done
  done
  log "ALL JOBS COMPLETE"
}

main "$@"
