#!/usr/bin/env bash
# Launch a kimon config .sbatch as an UNINTERRUPTIBLE, TACC-native (no SLURM)
# background job, with GPU / RAM / runtime telemetry and a live heartbeat.
#
#   kimon/env/run_detached.sh <config-name> <path-to.sbatch>
#   e.g. kimon/env/run_detached.sh config3_all_opts kimon/configs/config3_all_opts/block.sbatch
#
# Uninterruptible: the job runs under `setsid nohup` in its own session, so it
# survives SSH/idev-shell disconnects and this process exiting. It keeps running
# on the node until it finishes or the idev allocation ends.
#
# GPUs: uses every GPU visible in this allocation (CUDA_VISIBLE_DEVICES, or all
# of `nvidia-smi -L` if unset). NOTE: config3 all-opts is single-process /
# single-GPU by design; multi-GPU *sharding* is config2 and needs >=2 physical
# GPUs -- on a 1-GPU node there is nothing to shard.
#
# Evidence (under kimon/logs/<config>/detached_<stamp>/):
#   telemetry.csv  - iso_time,elapsed_s,gpu_mem_used_mib,gpu_util_pct,
#                    target_rss_mib,target_vmhwm_mib,host_load1  (every 5s)
#   heartbeat.log  - one HEARTBEAT line every ~30s + a final DONE line
#   run.out        - full stdout/stderr of the .sbatch (incl. [PASS]/[FAIL])
#   time.txt       - /usr/bin/time -v (peak RSS, wall, CPU) of the whole job
#   launch.env     - provenance (node, GPU, fixture, contract mode, git rev)
#   pid, done      - runner PGID and exit-code marker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly CONFIG="${1:?usage: run_detached.sh <config-name> <sbatch>}"
readonly SBATCH="${2:?usage: run_detached.sh <config-name> <sbatch>}"
[[ -f "${REPO}/${SBATCH}" || -f "${SBATCH}" ]] || { echo "[FATAL] no such sbatch: ${SBATCH}" >&2; exit 2; }

# UTC stamp without Date.now-in-shell issues; date is fine here.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVID="${REPO}/kimon/logs/${CONFIG}/detached_${STAMP}"
mkdir -p "${EVID}"

# GPUs visible to this allocation.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VIS="${CUDA_VISIBLE_DEVICES}"
else
  VIS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
NGPU="$(awk -F, '{print NF}' <<<"${VIS}")"

{
  echo "config=${CONFIG}"
  echo "sbatch=${SBATCH}"
  echo "stamp_utc=${STAMP}"
  echo "node=$(hostname -f)"
  echo "cuda_visible_devices=${VIS}"
  echo "n_gpus_used=${NGPU}"
  echo "gpus=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | paste -sd';' -)"
  echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
  echo "git_rev=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo NA)"
  echo "run_mode=${RUN_MODE:-native}"
  echo "platform_contract=${KIMON_USE_PLATFORM_CONTRACT:-1}"
  echo "note=config3 all-opts is single-GPU by design; sharding (config2) needs >=2 GPUs"
} > "${EVID}/launch.env"

readonly DONE="${EVID}/done"
readonly TELE="${EVID}/telemetry.csv"
readonly HB="${EVID}/heartbeat.log"
echo "iso_time,elapsed_s,gpu_mem_used_mib,gpu_util_pct,target_rss_mib,target_vmhwm_mib,host_load1" > "${TELE}"

# ---- sampler: GPU + target-process RAM + host load, until 'done' appears -----
sampler() {
  local start now el pid rss hwm gmem gutil load
  start="$(date +%s)"
  local last_hb=0
  while [[ ! -f "${DONE}" ]]; do
    now="$(date +%s)"; el=$(( now - start ))
    # newest running config binary (the encrypted evaluator)
    pid="$(pgrep -n -f 'build/real_dnagpt_fides_scheme_b' 2>/dev/null || true)"
    rss=""; hwm=""
    if [[ -n "${pid}" && -r "/proc/${pid}/status" ]]; then
      rss="$(awk '/^VmRSS:/{printf "%.0f", $2/1024}' "/proc/${pid}/status" 2>/dev/null)"
      hwm="$(awk '/^VmHWM:/{printf "%.0f", $2/1024}' "/proc/${pid}/status" 2>/dev/null)"
    fi
    gmem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{if(NR)print s; else print "NA"}')"
    gutil="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd';' - || echo NA)"
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo NA)"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${el},${gmem},${gutil},${rss:-NA},${hwm:-NA},${load}" >> "${TELE}"
    if (( el - last_hb >= 30 )); then
      echo "HEARTBEAT elapsed=${el}s state=running gpu_mem=${gmem}MiB gpu_util=${gutil}% target_rss=${rss:-NA}MiB peak_rss=${hwm:-NA}MiB load=${load}" >> "${HB}"
      last_hb="${el}"
    fi
    sleep 5
  done
  local rc; rc="$(cat "${DONE}" 2>/dev/null || echo '?')"
  local total; total=$(( $(date +%s) - start ))
  echo "HEARTBEAT elapsed=${total}s state=DONE rc=${rc} peak_rss=$(awk -F, 'END{print $6}' "${TELE}")MiB" >> "${HB}"
}

# ---- the job: run the .sbatch under /usr/bin/time, mark done on exit ---------
job() {
  cd "${REPO}"
  export REPO_ROOT="${REPO}"
  export SLURM_SUBMIT_DIR="${REPO}"
  unset SLURM_JOB_ID            # so the run tag is a unique UTC stamp
  export RUN_MODE="${RUN_MODE:-native}"
  export CUDA_VISIBLE_DEVICES="${VIS}"
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/tacc_env.sh" >/dev/null 2>&1 || true
  /usr/bin/time -v -o "${EVID}/time.txt" bash "${SBATCH}" > "${EVID}/run.out" 2>&1
  echo "$?" > "${DONE}"
}

export -f sampler job
export EVID DONE TELE HB REPO SCRIPT_DIR SBATCH VIS RUN_MODE

# Detach both in one new session; sampler in background, job in foreground of
# the detached session, so 'done' is written exactly when the job exits.
setsid nohup bash -c 'sampler & job; wait' > "${EVID}/detached.log" 2>&1 &
RUNNER_PGID=$!
echo "${RUNNER_PGID}" > "${EVID}/pid"
disown || true

echo "[detached] config=${CONFIG} gpus=${VIS} (${NGPU})"
echo "[detached] evidence dir: ${EVID}"
echo "[detached] heartbeat:    ${HB}"
echo "[detached] live log:     ${EVID}/run.out"
echo "[detached] pid(pgid):    ${RUNNER_PGID}"
