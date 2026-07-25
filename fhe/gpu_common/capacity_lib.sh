#!/usr/bin/env bash
# Shared, battle-tested GPU-capacity-detection + safe-launch library for the
# shared multi-tenant Brev host (255 cores, 8x A100). Extracted 2026-07-25
# from fhe/gpu_real_scheme_b/{launch_brev_scheme_b.sh,wait_and_run_scheme_b.sh}
# after two real incidents where naive nvidia-smi occupancy checks missed
# genuine capacity windows:
#   1. `nvidia-smi -i <index>` (single-GPU indexed query) transiently returns
#      a garbled/error response ("Invalid PCI bus id format...") under acute
#      multi-tenant driver contention. The unindexed bulk query form
#      (`--query-gpu=... ` over all GPUs, no `-i`) is far more reliable.
#   2. Even the bulk form can occasionally glitch during a sharp load spike.
#      Since correctness of an FHE computation does not depend on GPU
#      occupancy (only our own wall-clock timing measurement does, which is
#      already caveated as potentially contention-affected), a launcher
#      should not burn an entire capacity window refusing to launch just
#      because its *own monitoring* hiccuped. It should fail closed only on
#      a *confirmed* valid reading that shows real occupancy, and otherwise
#      proceed, trusting a very recent capacity determination made by the
#      caller (e.g. gpucap_wait_for_capacity's own idle-GPU pick).
#
# Usage: source this file, then call the gpucap_* functions below. Every
# tunable is overridable via environment variable so callers do not need to
# edit this file.
#
#   source ".../gpu_common/capacity_lib.sh"
#   gpu="$(gpucap_wait_for_capacity)"                # blocks until capacity
#   gpucap_preflight_confirm_gpu "${gpu}" || exit $?  # re-check just before launch
#   echo "confirmed=${GPUCAP_PREFLIGHT_CONFIRMED} note=${GPUCAP_PREFLIGHT_NOTE}"

set -uo pipefail

: "${GPUCAP_MEM_THRESHOLD_MIB:=10000}"     # GPU considered occupied at/above this
: "${GPUCAP_UTIL_THRESHOLD_PCT:=10}"       # GPU considered occupied at/above this
: "${GPUCAP_LOAD_THRESHOLD:=250}"          # host 1-min load average ceiling
: "${GPUCAP_POLL_SECONDS:=60}"             # capacity poll interval
: "${GPUCAP_MAX_WAIT_SECONDS:=259200}"     # give up after this long (default 72h)
: "${GPUCAP_PREFLIGHT_ATTEMPTS:=5}"        # nvidia-smi retry attempts before soft-proceeding
: "${GPUCAP_PREFLIGHT_SLEEP_SECONDS:=3}"   # sleep between preflight retry attempts
: "${GPUCAP_NVIDIA_SMI_TIMEOUT:=15}"       # per-call timeout guard (seconds)

gpucap_log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

# Prints the index of the first physical GPU with memory.used below
# GPUCAP_MEM_THRESHOLD_MIB and utilization below GPUCAP_UTIL_THRESHOLD_PCT,
# or nothing if none qualify. Always uses the unindexed bulk query form.
gpucap_pick_idle_gpu() {
  timeout "${GPUCAP_NVIDIA_SMI_TIMEOUT}" nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null |
    while IFS=',' read -r idx mem util; do
      idx="${idx// /}"; mem="${mem// /}"; util="${util// /}"
      if [[ "${mem}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] &&
         (( mem < GPUCAP_MEM_THRESHOLD_MIB && util < GPUCAP_UTIL_THRESHOLD_PCT )); then
        echo "${idx}"
        break
      fi
    done
}

# Blocks until host load1 is below GPUCAP_LOAD_THRESHOLD AND a GPU is idle.
# Echoes the chosen GPU index on success; returns 1 after
# GPUCAP_MAX_WAIT_SECONDS with nothing echoed. Logs one line per poll via
# gpucap_log (redirect/tee this yourself if you want it persisted).
gpucap_wait_for_capacity() {
  local start_ts gpu load1 load1_int now_ts
  start_ts=$(date +%s)
  while true; do
    load1="$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}')"
    load1="${load1// /}"
    load1_int="${load1%.*}"
    gpu="$(gpucap_pick_idle_gpu)"
    gpucap_log "poll: load1=${load1} threshold=${GPUCAP_LOAD_THRESHOLD} idle_gpu=${gpu:-none}" >&2
    if [[ -n "${gpu}" ]] && (( load1_int < GPUCAP_LOAD_THRESHOLD )); then
      echo "${gpu}"
      return 0
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts > GPUCAP_MAX_WAIT_SECONDS )); then
      gpucap_log "giving up: no capacity within ${GPUCAP_MAX_WAIT_SECONDS}s" >&2
      return 1
    fi
    sleep "${GPUCAP_POLL_SECONDS}"
  done
}

# Re-confirms a specific physical GPU is idle right before launching.
# Sets (non-readonly, caller may inspect): GPUCAP_MEM_MIB, GPUCAP_UTIL_PCT,
# GPUCAP_UUID, GPUCAP_PREFLIGHT_CONFIRMED (1/0), GPUCAP_PREFLIGHT_NOTE
# (human-readable, safe to embed in run provenance).
# Returns 0 if it is safe to launch (confirmed idle, or unconfirmed and
# soft-proceeding), or 3 if a *confirmed* reading shows real occupancy
# (fail closed -- caller should refuse to launch).
gpucap_preflight_confirm_gpu() {
  local physical_gpu="$1"
  local mem="" util="" uuid="" confirmed=0 attempt
  for ((attempt = 1; attempt <= GPUCAP_PREFLIGHT_ATTEMPTS; attempt++)); do
    read -r mem util uuid <<<"$(
      timeout "${GPUCAP_NVIDIA_SMI_TIMEOUT}" nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu,uuid \
        --format=csv,noheader,nounits 2>&1 |
        awk -F', *' -v idx="${physical_gpu}" '$1==idx {print $2, $3, $4}'
    )"
    if [[ "${mem}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ && -n "${uuid}" ]]; then
      confirmed=1
      break
    fi
    gpucap_log "[WARN] preflight nvidia-smi query attempt ${attempt} returned non-numeric/missing output for GPU ${physical_gpu} (mem='${mem}' util='${util}' uuid='${uuid}'); retrying" >&2
    sleep "${GPUCAP_PREFLIGHT_SLEEP_SECONDS}"
  done

  GPUCAP_MEM_MIB="${mem}"
  GPUCAP_UTIL_PCT="${util}"
  GPUCAP_UUID="${uuid}"
  GPUCAP_PREFLIGHT_CONFIRMED="${confirmed}"

  if [[ "${confirmed}" -eq 1 ]]; then
    if (( mem >= GPUCAP_MEM_THRESHOLD_MIB || util >= GPUCAP_UTIL_THRESHOLD_PCT )); then
      GPUCAP_PREFLIGHT_NOTE="preflight_confirmed_occupied mem=${mem}MiB util=${util}%"
      gpucap_log "[FATAL] physical GPU ${physical_gpu} is occupied (confirmed reading: mem=${mem}MiB util=${util}%); refusing to launch" >&2
      return 3
    fi
    GPUCAP_PREFLIGHT_NOTE="preflight_confirmed mem=${mem}MiB util=${util}%"
    return 0
  fi

  # nvidia-smi itself stayed uncooperative after retrying. This affects only
  # the reliability of *this specific* pre-launch double-check, not the
  # correctness of the computation -- soft-proceed on the caller's already
  # recent idle-GPU determination rather than losing the window.
  GPUCAP_PREFLIGHT_NOTE="preflight_UNCONFIRMED(nvidia-smi glitch after ${GPUCAP_PREFLIGHT_ATTEMPTS} attempts, proceeded on caller's recent capacity check)"
  gpucap_log "[WARN] preflight nvidia-smi query for GPU ${physical_gpu} did not return a confirmed reading after ${GPUCAP_PREFLIGHT_ATTEMPTS} attempts; proceeding unconfirmed (performance-only risk, not a correctness risk)" >&2
  return 0
}

# Optional informational helper: counts nvidia compute-apps processes bound
# to a given GPU UUID. Not used for the fail-closed decision (this host has
# small always-on idle CUDA-context processes on every GPU, so a nonzero
# count is not a reliable occupancy signal here) -- purely for logging.
gpucap_compute_process_count() {
  local gpu_uuid="$1"
  timeout "${GPUCAP_NVIDIA_SMI_TIMEOUT}" nvidia-smi \
    --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
    grep -Fxc "${gpu_uuid}" || true
}
