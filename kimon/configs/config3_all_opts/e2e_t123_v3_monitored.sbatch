#!/usr/bin/env bash
#SBATCH -J kimon-c3-e2e-t123v3-mon
#SBATCH -o kimon/logs/config3_all_opts/slurm-e2e-t123v3-mon-%j.out
#SBATCH -e kimon/logs/config3_all_opts/slurm-e2e-t123v3-mon-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH -A MCB25091
#
# Config 3 e2e (all 12 DNAGPT blocks + GSR head, T=103, 1 GPU), T123_V3
# per-block lever: composes the T123_V3 per-block source (CPU-side
# diagonal-vector cache + Tier1 encode-once encoded-Plaintext templates with
# per-use clone + Tier2/3 OMP-parallel batched encode + post-LoadContext free
# of the CPU-side OpenFHE EvalMult/rotation key maps) instead of the plain
# cpudiagcache block used by e2e_monitored.sbatch. Additive fork of that
# script: same telemetry-sampler / timestamped-log / OMP-affinity pattern,
# only the runner script, tag, process-name grep, and PASS marker differ.
#
#   - RUN_MODE=native set explicitly (this host's FIDESlib is native, not the
#     apptainer .sif -- see kimon/env/tacc_env.sh).
#   - OMP thread-affinity knobs (OMP_NUM_THREADS/PROC_BIND/PLACES) applied --
#     the campaign's v1_affinity_only, accepted, ~20% wall-time win, zero
#     correctness risk (unlike the abandoned v2 concurrent-decrypt attempt);
#     T123_V3 stacks the encode-once/parallel-encode/free-host-keys levers on
#     top of this, per the single-block campaign (job 3345272 -> 3347729).
#   - every log line timestamped with elapsed seconds, and a background
#     telemetry sampler (GPU mem/util, host RSS, load1, CPU-busy) runs for the
#     WHOLE job -- so a mid-run timeout/crash still leaves a readable
#     per-block record instead of an opaque loss.
#
# Runtime estimate (NOT yet directly measured for this 12-block T123_V3
# driver -- extrapolated only): the single-block T123_V3 campaign measured
# 592-663s/block on gpu-a100-dev (job 3345272 affinity-only: 530s wall;
# job 3347729 v3_free_host_keys: same op schedule, RAM-focused, timing noisy
# but same order) vs the plain-cpudiagcache driver's real 12-block-run
# calibration of ~1520s/block (A0 job 3355234, 2026-08-10). Naive scaling
# (~625s/block x 12 =~ 7500s =~ 2.1h) plus head/refresh overhead suggests a
# ~2-2.5h full run -- i.e. this COULD fit gpu-a100-dev's 2h cap on a good
# day, but every prior "single-block time x 12" estimate for this project has
# undershot real 12-block runs by a large factor (contention, cumulative
# host RAM growth across blocks, refresh/decrypt overhead not present in the
# single-block gate), so gpu-a100-dev is NOT assumed safe here without a
# dedicated current-driver profile first (see CLAUDE.md's "dedicated
# current-block profile ... not an unprofiled multi-day run" rule).
# Defaulting to gpu-a100 (48h) for safety; switch the -p line (and drop -t if
# desired) to gpu-a100-dev only after a short calibration run confirms the
# 2-2.5h estimate on this driver specifically.
#
#   SUBMIT FROM A LOGIN NODE:
#     mkdir -p kimon/logs/config3_all_opts
#     sbatch kimon/configs/config3_all_opts/e2e_t123_v3_monitored.sbatch
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
TAG="fhe_c3_allopts_e2e_t123_v3_monitored_scheme_b_all_blocks_head_t103_cpudiagcache_t123_v3_a100_${JOB}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

echo "=================== kimon config3 e2e T123_V3 (monitored, SLURM batch) ==================="
echo "job=${JOB} node=$(hostname -f) omp=${OMP_NUM_THREADS} proc_bind=${OMP_PROC_BIND} places=${OMP_PLACES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader | sed 's/^/gpu = /' || true
echo "============================================================================================"

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
    pid="$(pgrep -n -f '12blocks_head_cpudiagcache_t123_v3' 2>/dev/null || true)"
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
run_scheme_b run_scheme_b_all_blocks_head_t103_cpudiagcache_t123_v3.sh \
    0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" 2>&1 \
  | stdbuf -oL awk -v t0="${t0}" '{printf "[%6d s] %s\n", systime()-t0, $0; fflush()}' \
  > "${RUN_DIR}/run.log"
RC="${PIPESTATUS[0]}"
set -e
WALL=$(( $(date +%s) - t0 ))
kill "${SAMPLER}" 2>/dev/null || true

echo "wall_clock_s=${WALL}"
echo "--- run.log (tail) ---"; tail -30 "${RUN_DIR}/run.log" || true

if [[ "${RC}" -eq 0 ]] && grep -q "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_T123_V3_PASS" "${RUN_DIR}/run.log"; then
  echo "[PASS] config3 e2e T123_V3 (monitored) -> ${OUT}"
  exit 0
else
  echo "[FAIL] config3 e2e T123_V3 (monitored) (rc=${RC}); see ${RUN_DIR}/run.log + ${TELE}"
  exit 1
fi
