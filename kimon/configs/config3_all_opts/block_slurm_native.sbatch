#!/usr/bin/env bash
# =============================================================================
# Config 3 (all-opts, 1 block) as a REAL SLURM batch job on TACC Lonestar6.
#
#   SUBMIT FROM A LOGIN NODE (you cannot sbatch from a compute/idev node):
#     sbatch /scratch/10899/kimopro/private_genomic_dnagpt/kimon/configs/config3_all_opts/block_slurm_native.sbatch
#
# Unlike the idev/detached approach, a batch job runs in its OWN allocation and
# survives login/compute-session disconnects and idev expiry -- it only ends
# when the encrypted block finishes (or hits the walltime below).
#
# Everything is self-contained: absolute REPO path (submit-dir-independent),
# native FIDESlib env, platform-tagged fixture pins, and GPU/RAM/runtime
# telemetry. All output lands on /scratch (readable from any node).
#
# What it runs: real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache
#   = SIMD(B=8) + depth-13/digits-3/ring-65536 + CPU diagonal-vector cache,
#   one full DNAGPT block at T=103. Single GPU by design (this partition gives
#   1 A100 per node; multi-GPU sharding is config2 and needs >=2 GPUs).
# =============================================================================
#SBATCH -J kimon-c3-block
#SBATCH -p gpu-a100-small
#SBATCH -A MCB25091
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/config3_all_opts/slurm/c3-%j.out
#SBATCH -e /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/config3_all_opts/slurm/c3-%j.err

set -euo pipefail

readonly REPO=/scratch/10899/kimopro/private_genomic_dnagpt
readonly JOB="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly EVID="${REPO}/kimon/logs/config3_all_opts/slurm/job_${JOB}"
mkdir -p "${EVID}"

cd "${REPO}"
export REPO_ROOT="${REPO}"
export RUN_MODE=native
# Force the platform-tagged fixture contracts + pins (this is the ls6 native path).
export KIMON_USE_PLATFORM_CONTRACT="${KIMON_USE_PLATFORM_CONTRACT:-1}"

# Native toolchain (gcc13.2/cuda12.8/python3.12) + FIDESlib LD_LIBRARY_PATH.
# shellcheck disable=SC1091
source "${REPO}/kimon/env/tacc_env.sh"

echo "=================== kimon config3 (SLURM batch) ==================="
echo "job_id       = ${JOB}"
echo "node         = $(hostname -f)"
echo "gpus_visible = ${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/gpu          = /'
echo "evidence_dir = ${EVID}"
echo "started_utc  = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==================================================================="

# ---- background telemetry sampler: GPU mem/util + target-proc RAM + load ----
readonly TELE="${EVID}/telemetry.csv"
echo "iso_time,elapsed_s,gpu_mem_used_mib,gpu_util_pct,target_rss_mib,target_vmhwm_mib,host_load1" > "${TELE}"
(
  start="$(date +%s)"
  while :; do
    el=$(( $(date +%s) - start ))
    pid="$(pgrep -n -f 'build/real_dnagpt_fides_scheme_b_simd_full_t103' 2>/dev/null || true)"
    rss=NA; hwm=NA
    if [[ -n "${pid}" && -r "/proc/${pid}/status" ]]; then
      rss="$(awk '/^VmRSS:/{printf "%.0f",$2/1024}' "/proc/${pid}/status" 2>/dev/null || echo NA)"
      hwm="$(awk '/^VmHWM:/{printf "%.0f",$2/1024}' "/proc/${pid}/status" 2>/dev/null || echo NA)"
    fi
    gmem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{if(NR)print s; else print "NA"}')"
    gutil="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd';' - || echo NA)"
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo NA)"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${el},${gmem},${gutil},${rss},${hwm},${load}" >> "${TELE}"
    sleep 5
  done
) &
SAMPLER=$!
trap 'kill "${SAMPLER}" 2>/dev/null || true' EXIT

# ---- the run: common.sh sets the platform-contract env + runs the binary -----
# shellcheck disable=SC1091
source "${REPO}/kimon/env/common.sh"

CONFIG=config3_all_opts
TAG="fhe_c3_allopts_block_t103_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_a100_slurm_${JOB}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

t0="$(date +%s)"
set +e
# Direct redirect (shell-opened -> run.log is guaranteed created). A previous
# `run_scheme_b ... | tee run.log` sometimes failed to persist the file because
# `run_scheme_b` is a function run in a pipeline subshell; the authoritative
# verdict below keys off the result JSON, not this log.
run_scheme_b run_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.sh \
  0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" > "${RUN_DIR}/run.log" 2>&1
RC=$?
set -e
WALL=$(( $(date +%s) - t0 ))
kill "${SAMPLER}" 2>/dev/null || true

# surface the run's tail into the SLURM -o console
echo "----- run.log (tail) -----"; tail -25 "${RUN_DIR}/run.log" 2>/dev/null || true

# ---- verdict: the binary's result JSON is authoritative ---------------------
PASSED=no
if [[ -f "${OUT}" ]] && grep -q '"passed"[[:space:]]*:[[:space:]]*true' "${OUT}"; then PASSED=yes; fi
MARKER=no
grep -q "REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_CPUDIAGCACHE_GATE_PASS" "${RUN_DIR}/run.log" 2>/dev/null && MARKER=yes

{
  echo "job_id            = ${JOB}"
  echo "node              = $(hostname -f)"
  echo "finished_utc      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code         = ${RC}"
  echo "gate              = $([[ "${RC}" -eq 0 && "${PASSED}" == yes ]] && echo PASS || echo FAIL)"
  echo "passed_json       = ${PASSED}    gate_marker_in_log = ${MARKER}"
  echo "result_json       = ${OUT}"
  echo "run_log           = ${RUN_DIR}/run.log"
  echo "--- accuracy (from result JSON) ---"
  grep -oE '"(global_rel_inf|worst_token_rel_inf|max_abs_error|tol|all_finite)"[^,}]*' "${OUT}" 2>/dev/null || true
  echo "--- reported timings (from result JSON) ---"
  grep -oE '"(context_keygen_load|encrypted_evaluation|client_boundary_seconds_total|final_decrypt)"[^,}]*' "${OUT}" 2>/dev/null || true
  echo "--- diag cache + op schedule (from result JSON) ---"
  grep -oE '"(diagonal_cache_hits|diagonal_cache_misses|matrix_products|explicit_ciphertext_plaintext_multiplications|explicit_rotations)"[^,}]*' "${OUT}" 2>/dev/null || true
  echo "--- timing/memory (measured this job) ---"
  echo "wall_clock_s      = ${WALL}"
  awk -F, 'NR>1{el=$2; if($3>gm)gm=$3; if($6!="NA"&&$6+0>rss)rss=$6}
           END{print "telemetry_elapsed_s = "el"\ngpu_mem_peak_mib    = "gm"\ntarget_rss_peak_mib = "rss" ("rss/1024" GiB)"}' "${TELE}"
} | tee "${EVID}/SUMMARY.txt"

echo "==================================================================="
if [[ "${RC}" -eq 0 && "${PASSED}" == yes ]]; then
  echo "[PASS] config3 block -> ${OUT}"
  echo "[evidence] ${EVID}/SUMMARY.txt"
  exit 0
else
  echo "[FAIL] config3 block (rc=${RC}, passed_json=${PASSED}); see ${RUN_DIR}/run.log, ${OUT}, ${EVID}/SUMMARY.txt"
  exit 1
fi
