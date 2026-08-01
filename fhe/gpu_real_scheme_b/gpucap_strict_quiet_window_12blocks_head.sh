#!/usr/bin/env bash
# Stricter quiet-window capacity gate for the T=103 12-block+head Scheme B
# Token-SIMD run (real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head).
#
# Why this exists instead of reusing gpu_common/capacity_lib.sh's
# gpucap_wait_for_capacity() directly: that function is well-suited to the
# single-block gates it was built for (minutes to a couple of hours each),
# where a single instantaneous poll showing host load1 below
# GPUCAP_LOAD_THRESHOLD=250 and one idle GPU is an acceptable bar. This run
# is a genuinely different risk class: a realistic 15-45 hour, single
# uninterrupted job on shared 8-A100 infrastructure this project does not
# control. Launching it on a momentary lull inside a much bigger contention
# spike (observed today: this host's load1 swinging between ~22 and
# ~1123+ within minutes) would very likely mean most of a multi-day job
# runs contended, or is killed by an operator reclaiming the host. A much
# stricter, SUSTAINED bar is warranted, and per project convention (never
# edit a shared file other concurrent gates depend on) it must live in its
# own new file.
#
# This script is a design/wrapper only: importable and directly runnable,
# but not wired into any auto-launching orchestrator, and not invoked by
# this session. It must be reviewed and explicitly invoked by a human
# before it ever gates a real launch.
#
# Stricter criteria (both required, both SUSTAINED across
# STRICT_CONSECUTIVE_POLLS consecutive polls spaced STRICT_POLL_SECONDS
# apart -- not a single instantaneous reading):
#
#   1. Host 1-minute load average stays below STRICT_LOAD_THRESHOLD
#      (default 50) on every one of the required consecutive polls. Why 50,
#      not the shared default of 250: this host's observed idle baseline
#      today was ~15-25; 250 is calibrated to tolerate routine multi-tenant
#      noise for a short gate, not to signal genuine multi-day-safe quiet.
#      50 sits comfortably above the observed idle floor while rejecting
#      any early ramp of another tenant's job. Any single poll at or above
#      the threshold resets the consecutive-quiet streak to zero -- a job
#      that is 4/5 of the way to a quiet window and then sees one bad
#      reading must start the count over, not launch on a near-miss.
#
#   2. ZERO nvidia compute-application processes on ALL physical GPUs on
#      the host (not just the one GPU this job would use). The shared
#      library's gpucap_compute_process_count() is explicitly documented
#      as informational-only there ("this host has small always-on idle
#      CUDA-context processes on every GPU, so a nonzero count is not a
#      reliable occupancy signal") -- but that judgment was calibrated for
#      short gates on one already-mem/util-idle GPU. For a 15-45 hour
#      exclusive commitment, this script requires the nvidia-smi
#      --query-compute-apps bulk listing (over ALL GPUs, unindexed, same
#      reliability lesson as the shared library's own bulk-query
#      preference) to be completely EMPTY, on every required consecutive
#      poll. A host that is not just "our target GPU is idle" but
#      genuinely has no compute process running anywhere is a much
#      stronger signal that no other tenant is about to start a large job
#      that will contend for host-wide CPU, PCIe, and memory-bandwidth
#      resources for the run's entire multi-day duration.
#
# On top of both sustained criteria, the existing per-GPU memory/utilization
# idle thresholds (GPUCAP_MEM_THRESHOLD_MIB / GPUCAP_UTIL_THRESHOLD_PCT from
# the shared library, left at their defaults) are still required for the
# specific GPU this script selects, each poll.
#
# Deliberate divergence from the shared library's fail-OPEN philosophy on a
# glitchy nvidia-smi reading: gpucap_preflight_confirm_gpu() soft-proceeds
# if nvidia-smi returns non-numeric output after retries, reasoning that
# occupancy-check unreliability is a timing risk, not a correctness risk,
# for a short gate. For a 15-45 hour exclusive commitment the calculus
# differs: this script fails CLOSED (resets the streak, keeps polling) on
# any glitched reading, on either criterion, on any poll.
#
# Usage:
#   source "gpu_common/capacity_lib.sh"   # (for gpucap_log only)
#   source "gpu_real_scheme_b/gpucap_strict_quiet_window_12blocks_head.sh"
#   gpu="$(gpucap_strict_wait_for_quiet_window)"
#   # then still call the shared library's gpucap_preflight_confirm_gpu
#   # immediately before the actual docker run, as every other gate does.
#
# All thresholds are overridable via environment variables so this file
# does not need editing to tune the bar; see the defaults below.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
# shellcheck source=../gpu_common/capacity_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../gpu_common/capacity_lib.sh"

: "${STRICT_LOAD_THRESHOLD:=50}"          # host 1-min load average ceiling
: "${STRICT_CONSECUTIVE_POLLS:=5}"        # required consecutive quiet polls
: "${STRICT_POLL_SECONDS:=60}"            # spacing between polls
: "${STRICT_MAX_WAIT_SECONDS:=604800}"     # give up after this long (default 7d)

# Prints the number of nvidia compute-application processes bound to ANY
# GPU on the host (bulk, unindexed query -- same reliability lesson as
# gpucap_pick_idle_gpu). Prints a negative number (-1) if nvidia-smi itself
# did not return parseable output, so the caller can distinguish "0 real
# processes" from "unreliable reading" and fail closed on the latter.
gpucap_strict_host_wide_compute_process_count() {
  local raw nvidia_smi_status
  raw="$(timeout "${GPUCAP_NVIDIA_SMI_TIMEOUT}" nvidia-smi \
    --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>&1)"
  nvidia_smi_status=$?
  if (( nvidia_smi_status != 0 )); then
    echo "-1"
    return
  fi
  if [[ -z "${raw}" ]]; then
    echo "0"
    return
  fi
  # A successful call with no processes prints nothing at all; a
  # successful call with N processes prints N lines. Any line that does
  # not look like "digits, gpu-uuid-string" is treated as an unreliable
  # glitch, not a real process count.
  local bad_lines
  bad_lines="$(printf '%s\n' "${raw}" | grep -cv '^[0-9][0-9]*, GPU-')"
  if (( bad_lines > 0 )); then
    echo "-1"
    return
  fi
  printf '%s\n' "${raw}" | wc -l | tr -d ' '
}

# Blocks until BOTH strict criteria have held for STRICT_CONSECUTIVE_POLLS
# consecutive polls. Echoes the chosen idle GPU index on success (selected
# via the shared library's existing per-GPU mem/util idle check, applied
# fresh on the final qualifying poll). Returns 1 after
# STRICT_MAX_WAIT_SECONDS with nothing echoed.
gpucap_strict_wait_for_quiet_window() {
  local start_ts now_ts load1 load1_int host_procs streak=0 gpu=""
  start_ts=$(date +%s)
  gpucap_log "strict quiet-window gate starting: load_threshold=${STRICT_LOAD_THRESHOLD} consecutive_polls_required=${STRICT_CONSECUTIVE_POLLS} poll_seconds=${STRICT_POLL_SECONDS}" >&2
  while true; do
    load1="$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}')"
    load1="${load1// /}"
    load1_int="${load1%.*}"
    host_procs="$(gpucap_strict_host_wide_compute_process_count)"
    gpu="$(gpucap_pick_idle_gpu)"

    local load_ok=0 procs_ok=0
    if [[ "${load1_int}" =~ ^[0-9]+$ ]] && (( load1_int < STRICT_LOAD_THRESHOLD )); then
      load_ok=1
    fi
    if [[ "${host_procs}" =~ ^[0-9]+$ ]] && (( host_procs == 0 )); then
      procs_ok=1
    fi

    if (( load_ok == 1 && procs_ok == 1 && ${#gpu} > 0 )); then
      streak=$((streak + 1))
    else
      if (( streak > 0 )); then
        gpucap_log "quiet streak reset at ${streak}/${STRICT_CONSECUTIVE_POLLS} (load1=${load1} host_wide_compute_processes=${host_procs} idle_gpu=${gpu:-none})" >&2
      fi
      streak=0
    fi
    gpucap_log "poll: load1=${load1} (threshold ${STRICT_LOAD_THRESHOLD}) host_wide_compute_processes=${host_procs} idle_gpu=${gpu:-none} streak=${streak}/${STRICT_CONSECUTIVE_POLLS}" >&2

    if (( streak >= STRICT_CONSECUTIVE_POLLS )); then
      gpucap_log "quiet window confirmed: ${STRICT_CONSECUTIVE_POLLS} consecutive clean polls, selecting gpu=${gpu}" >&2
      echo "${gpu}"
      return 0
    fi

    now_ts=$(date +%s)
    if (( now_ts - start_ts > STRICT_MAX_WAIT_SECONDS )); then
      gpucap_log "giving up: no sustained quiet window within ${STRICT_MAX_WAIT_SECONDS}s" >&2
      return 1
    fi
    sleep "${STRICT_POLL_SECONDS}"
  done
}
