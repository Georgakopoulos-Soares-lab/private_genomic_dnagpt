#!/usr/bin/env bash
#SBATCH -J kimon-c3-e2e-mon
#SBATCH -o kimon/logs/config3_all_opts/slurm-e2e-mon-%j.out
#SBATCH -e kimon/logs/config3_all_opts/slurm-e2e-mon-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH -A MCB25091
#
# Config 3 e2e (all 12 DNAGPT blocks + GSR head, T=103, 1 GPU, diagonal-vector
# cache per block), hardened for the real, resource-committing long run:
#   - RUN_MODE=native set explicitly (this host's FIDESlib is native, not the
#     apptainer .sif -- see kimon/env/tacc_env.sh).
#   - OMP thread-affinity knobs (OMP_NUM_THREADS/PROC_BIND/PLACES) applied --
#     the campaign's v1_affinity_only, accepted, ~20% wall-time win, zero
#     correctness risk (unlike the abandoned v2 concurrent-decrypt attempt).
#   - every log line timestamped with elapsed seconds, and a background
#     telemetry sampler (GPU mem/util, host RSS, load1, CPU-busy) runs for the
#     WHOLE job -- so a timeout/crash at hour 30 of an estimated ~5-6h run
#     still leaves a readable per-block record instead of an opaque loss.
#
# Calibration basis (A0 job 3355234, 2026-08-10, gpu-a100-dev, light
# contention, load1 peak ~32): 4 real completed blocks at 1511-1558s each
# (tight, ~3% spread) -> ~1520s/block -> 12-block projection ~18,240s
# (~5.1h) for block compute alone, before head/refresh overhead. This
# supersedes the earlier 15-45h estimate, which was based on the
# non-diagcache binary and/or heavily contaminated runs.
#
#   SUBMIT FROM A LOGIN NODE:
#     mkdir -p kimon/logs/config3_all_opts
#     sbatch kimon/configs/config3_all_opts/e2e_monitored.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export RUN_MODE=native
# TACC pre-sets OMP_NUM_THREADS=1, so a `:-` default never applies -- assign
# unconditionally. 32 matches the campaign's fastest validated sample.
export OMP_NUM_THREADS="${KIMON_OMP:-32}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# shellcheck disable=SC1091
source kimon/env/common.sh

CONFIG=config3_all_opts
JOB="${SLURM_JOB_ID:-$(run_stamp)}"
TAG="fhe_c3_allopts_e2e_monitored_scheme_b_all_blocks_head_t103_cpudiagcache_a100_${JOB}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

echo "=================== kimon config3 e2e (monitored, SLURM batch) ==================="
echo "job=${JOB} node=$(hostname -f) omp=${OMP_NUM_THREADS} proc_bind=${OMP_PROC_BIND} places=${OMP_PLACES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader | sed 's/^/gpu = /' || true
echo "===================================================================================="

# --- telemetry sampler: GPU0 mem/util, target RSS, host load, CPU-busy --------
# Runs for the whole job (not just calibration), so a mid-run failure still
# leaves a readable resource trace. Co-tenant watchdog: gpu-a100 nodes are
# single-GPU/node (unlike the shared 3-GPU gpu-a100-dev boxes), so index 1+
# should not even exist here; the check is kept for safety/portability.
readonly TICK=10
readonly TELE="${RUN_DIR}/telemetry.csv"
echo "iso_time,elapsed_s,gpu0_mem_used_mib,gpu0_util_pct,gpu_others_mem_mib,target_rss_mib,target_vmhwm_mib,host_load1,cpu_cores_busy" > "${TELE}"
(
  start="$(date +%s)"; prev_cpu=""; prev_t=""
  hz="$(getconf CLK_TCK 2>/dev/null || echo 100)"
  while :; do
    now="$(date +%s)"; el=$(( now - start ))
    pid="$(pgrep -n -f '12blocks_head_cpudiagcache' 2>/dev/null || true)"
    rss=NA; hwm=NA; cpub=NA
    if [[ -n "${pid}" && -r "/proc/${pid}/status" ]]; then
      rss="$(awk '/^VmRSS:/{printf "%.0f",$2/1024}' "/proc/${pid}/status" 2>/dev/null || echo NA)"
      hwm="$(awk '/^VmHWM:/{printf "%.0f",$2/1024}' "/proc/${pid}/status" 2>/dev/null || echo NA)"
      cur_cpu="$(awk '{print $14+$15}' "/proc/${pid}/stat" 2>/dev/null || echo '')"
      if [[ -n "${cur_cpu}" && -n "${prev_cpu}" && -n "${prev_t}" && "${now}" -gt "${prev_t}" ]]; then
        cpub="$(awk -v c="${cur_cpu}" -v p="${prev_cpu}" -v dt="$(( now - prev_t ))" -v hz="${hz}" \
                'BEGIN{printf "%.1f",(c-p)/hz/dt}')"
      fi
      prev_cpu="${cur_cpu}"; prev_t="${now}"
    fi
    gmem="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo NA)"
    gutil="$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo NA)"
    gother="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
              | awk -F', ' '$1!=0{s+=$2} END{printf "%d",s+0}' || echo NA)"
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo NA)"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${el},${gmem},${gutil},${gother},${rss},${hwm},${load},${cpub}" >> "${TELE}"
    sleep "${TICK}"
  done
) >/dev/null 2>&1 & SAMPLER=$!
trap 'kill "${SAMPLER}" 2>/dev/null || true' EXIT

# Every stdout/stderr line stamped with elapsed seconds -- readable per-block
# marginal cost even if the job is killed/times out mid-run.
t0="$(date +%s)"
set +e
run_scheme_b run_scheme_b_all_blocks_head_t103_cpudiagcache.sh \
    0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" 2>&1 \
  | stdbuf -oL awk -v t0="${t0}" '{printf "[%6d s] %s\n", systime()-t0, $0; fflush()}' \
  > "${RUN_DIR}/run.log"
RC="${PIPESTATUS[0]}"
set -e
WALL=$(( $(date +%s) - t0 ))
kill "${SAMPLER}" 2>/dev/null || true

echo "wall_clock_s=${WALL}"
echo "--- run.log (tail) ---"; tail -30 "${RUN_DIR}/run.log" || true

if [[ "${RC}" -eq 0 ]] && grep -q "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_PASS" "${RUN_DIR}/run.log"; then
  echo "[PASS] config3 e2e (monitored) -> ${OUT}"
  exit 0
else
  echo "[FAIL] config3 e2e (monitored) (rc=${RC}); see ${RUN_DIR}/run.log + ${TELE}"
  exit 1
fi
