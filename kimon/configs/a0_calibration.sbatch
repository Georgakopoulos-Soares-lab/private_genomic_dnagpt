#!/usr/bin/env bash
# =============================================================================
# A0 -- platform revalidation + fit calibration for gpu-a100-dev
# =============================================================================
# ONE QUESTION: does the 12-block + GSR-head end-to-end driver fit inside the
# gpu-a100-dev 2-hour wall on a single A100, and if not, how many blocks per
# job does it have to be sharded into?
#
# Why this job exists
# -------------------
#   qa100development: MaxWall=02:00:00, MaxJobsPU=1 (strictly serial, one job
#   at a time). A complete single T=103 block on c301 currently costs 530-728 s
#   (campaign.csv), so 12 blocks project to 1.8-2.4 h -- straddling the wall.
#   The e2e driver (real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_
#   cpudiagcache) is ALSO pre-T123: it has no Tier-1/2/3 encode optimizations,
#   so its per-block cost is unknown and probably worse than the T123 block.
#   It has never been run at T=103. Guessing costs 2 h of a serial queue;
#   measuring costs the same 2 h and returns a number.
#
# What it does (4 phases, all inside one 2 h allocation, single GPU)
#   1  attestation  -- node exclusivity, all GPUs idle, cores, env, and BOTH
#                      platform fixture contracts (block + all-blocks) verified
#                      on THIS node class. This is the "revalidation" half.
#   2  reference    -- one complete T123-v3 single block, same node, for a
#                      same-session reference point and an A1 variance seed.
#   3  e2e probe    -- the real 12-block+head driver under a hard timeout using
#                      every remaining second, with each stdout line stamped by
#                      elapsed wall time so per-block marginal cost is readable
#                      whether or not it finishes.
#   4  verdict      -- GO / NO-GO + required blocks-per-job shard size, written
#                      to verdict.json and SUMMARY.txt.
#
# Deliberately NOT here: the OMP thread sweep. campaign.csv shows omp=32
# (server 445 s) beating omp=128 (server 558-635 s), but confounded across
# nodes/binaries. That is a +/-30% question; fit is a yes/no question, and with
# MaxJobsPU=1 the fit question gets the whole allocation. Re-run this script
# with KIMON_A0_OMP_SWEEP=1 as a follow-up A0b job to settle OMP at matched
# node + matched binary.
#
#   SUBMIT FROM A LOGIN NODE:
#     sbatch /scratch/10899/kimopro/private_genomic_dnagpt/kimon/configs/a0_calibration.sbatch
#
#   Knobs (all optional):
#     KIMON_OMP=N            thread count for both phases      (default 32)
#     KIMON_A0_OMP_SWEEP=1   phase 2 becomes a 32/64/128 sweep, phase 3 skipped
#     KIMON_A0_SKIP_REF=1    skip phase 2, give phase 3 the whole window
#
#SBATCH -J kimon-a0-calib
#SBATCH -p gpu-a100-dev
#SBATCH -A MCB25091
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 02:00:00
#SBATCH -o /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/a0_calibration/slurm/a0-%j.out
#SBATCH -e /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/a0_calibration/slurm/a0-%j.err
set -euo pipefail

readonly REPO=/scratch/10899/kimopro/private_genomic_dnagpt
readonly JOB="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly EVID="${REPO}/kimon/logs/a0_calibration/a0_job_${JOB}"
mkdir -p "${EVID}"
cd "${REPO}"
export REPO_ROOT="${REPO}"

# Wall-clock budget. -t is 02:00:00; reserve the tail for phase 4 so the
# verdict is always written even if the e2e probe runs to its own timeout.
readonly T_START="$(date +%s)"
readonly WALL_LIMIT=7200
readonly TAIL_RESERVE=300          # phase 4 + teardown
readonly REF_TIMEOUT=1500          # phase 2 cap (block currently 530-728 s)
deadline() { echo $(( T_START + WALL_LIMIT - TAIL_RESERVE )); }
remaining() { echo $(( $(deadline) - $(date +%s) )); }

# shellcheck disable=SC1091
source "${REPO}/kimon/env/tacc_env.sh"

# TACC pre-sets OMP_NUM_THREADS=1, so a `:-` default NEVER applies. Assign
# unconditionally. Default 32: campaign.csv's fastest sample (v1_affinity_only,
# server 445.4 s) used 32, and 128 has never beaten it at matched conditions.
export OMP_NUM_THREADS="${KIMON_OMP:-32}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
NUMA_WRAP=()
if command -v numactl >/dev/null 2>&1; then NUMA_WRAP=(numactl --localalloc); fi

readonly SRC="${REPO}/fhe/gpu_real_scheme_b/src"
readonly BUILD="${REPO}/fhe/gpu_real_scheme_b/build"
readonly BLOCK_BIN="${BUILD}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3"
readonly E2E_RUNNER="${REPO}/fhe/gpu_real_scheme_b/run_scheme_b_all_blocks_head_t103_cpudiagcache.sh"
readonly E2E_BIN="${BUILD}/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache"
readonly FIX_BLOCK="${REPO}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
readonly FIX_E2E="${REPO}/checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
readonly CONTRACT_BLOCK="${REPO}/kimon/env/fixtures_ls6/fixture_t103.sha256"
readonly CONTRACT_E2E="${REPO}/kimon/env/fixtures_ls6/fixture_all_blocks_head_t103.sha256"

# =============================================================================
# PHASE 1 -- attestation + platform revalidation
# =============================================================================
readonly ATTEST="${EVID}/attestation.txt"
{
  echo "=========== A0 phase 1: attestation ==========="
  echo "job                 = ${JOB}"
  echo "node                = $(hostname -f)"
  echo "date_utc            = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "partition           = ${SLURM_JOB_PARTITION:-NA}"
  echo "--- CPU / exclusivity ---"
  echo "cpus_on_node        = ${SLURM_CPUS_ON_NODE:-NA}"
  echo "cpus_per_task       = ${SLURM_CPUS_PER_TASK:-NA}"
  echo "nproc_all           = $(nproc --all)"
  echo "affinity_cpus       = $(taskset -cp $$ 2>/dev/null | sed 's/.*: //' | head -1 || true)"
  echo "omp_num_threads     = ${OMP_NUM_THREADS}"
  echo "numactl             = ${NUMA_WRAP[*]:-none}"
  echo "host_load1          = $(awk '{print $1}' /proc/loadavg)"
  echo "mem_available_gib   = $(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo)"
  # Exclusive allocation (OverSubscribe=NO) is what makes timing trustworthy.
  # Any other user's process on this node invalidates every number below.
  echo "other_user_procs    = $(ps -eo user= | sort -u | grep -cvE "^(root|${USER}|nobody|dbus|polkitd|chrony|rpc|systemd|munge|nvidia|libstor|rpcuser|sssd)$" || true)"
  echo "--- GPUs (single-A100 constraint: only index 0 is used) ---"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader | sed 's/^/  gpu = /'
  echo "  NOTE: indices 1+ must read 0 MiB / 0% for the whole run -- they are"
  echo "        the proof that no co-tenant shared this node."
  echo "--- toolchain ---"
  echo "driver              = $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "nvcc                = $(nvcc --version 2>/dev/null | tail -1 || echo NA)"
  echo "fideslib_root       = ${FIDESLIB_ROOT:-NA}"
  echo "fideslib_commit     = $(git -C "${FIDESLIB_ROOT:-/nonexistent}" rev-parse HEAD 2>/dev/null || echo NA)"
  echo "fideslib_expected   = 786c7600fb2f16b724e0acf73df367b27b8afed6"
} | tee "${ATTEST}"

# Fixture contracts. This is the actual "revalidation": c301-* is different
# hardware from the v330-* nodes these platform contracts were generated on
# (128 real cores / 2 sockets, no hypervisor CPU masking vs 32 virtualized
# vCPUs). The float64 @-matmul oracle arrays already diverged once between
# Mac and v330; if they diverge again here, every downstream run is invalid
# until kimon/env/refresh_platform_contracts.sh is re-run on this node class.
CONTRACT_OK=yes
check_contract() {  # name dir contract
  local name="$1" dir="$2" contract="$3"
  if [[ ! -f "${dir}/manifest.json" ]]; then
    echo "  ${name}: FIXTURE MISSING (${dir})"; CONTRACT_OK=no; return
  fi
  if [[ ! -f "${contract}" ]]; then
    echo "  ${name}: CONTRACT MISSING (${contract})"; CONTRACT_OK=no; return
  fi
  if ( cd "${dir}" && sha256sum --check --status "${contract}" ); then
    echo "  ${name}: OK ($(wc -l < "${contract}") files vs $(basename "${contract}"))"
  else
    echo "  ${name}: MISMATCH vs $(basename "${contract}")  <-- platform drift"
    ( cd "${dir}" && sha256sum --check "${contract}" 2>&1 | grep -v ': OK$' | head -10 ) || true
    CONTRACT_OK=no
  fi
}
{
  echo "--- platform fixture contracts (revalidation on this node class) ---"
  check_contract "block_t103"           "${FIX_BLOCK}" "${CONTRACT_BLOCK}"
  check_contract "all_blocks_head_t103" "${FIX_E2E}"   "${CONTRACT_E2E}"
  echo "contracts_ok        = ${CONTRACT_OK}"
  echo "--- binaries ---"
  echo "block_bin_present   = $([[ -x "${BLOCK_BIN}" ]] && echo yes || echo NO)"
  echo "e2e_bin_present     = $([[ -x "${E2E_BIN}" ]] && echo yes || echo NO)"
  echo "e2e_has_t123        = no  (source is ..._12blocks_head_cpudiagcache.cpp;"
  echo "                          the Tier-1/2/3 encode work exists only on the"
  echo "                          single-block ..._t123_v{1,2,3} sources)"
} | tee -a "${ATTEST}"

if [[ "${CONTRACT_OK}" != yes ]]; then
  echo "[FATAL] platform contract drift on $(hostname -s). Regenerate with" >&2
  echo "        kimon/env/refresh_platform_contracts.sh ON THIS NODE CLASS," >&2
  echo "        then re-submit. Refusing to spend the allocation." >&2
  exit 2
fi

# =============================================================================
# telemetry sampler -- shared by phases 2 and 3
# =============================================================================
readonly TICK=5
# NOTE (job 3354534 post-mortem): the backgrounded subshell MUST have its own
# stdout/stderr redirected. It is started from inside `$(start_sampler ...)`,
# and a background child inherits the command-substitution pipe -- so without
# the `>/dev/null 2>&1` below, `$(...)` blocks forever waiting for that pipe to
# close, even though `echo $!` already ran. Job 3354534 burned its entire 2 h
# allocation here: the sampler wrote 1,389 telemetry rows while the script sat
# in the command substitution and never launched the binary (GPU 0 MiB/0% and
# no run.log for 7,170 s). Do not remove the redirection.
start_sampler() {  # pattern outfile -> prints sampler PID
  local pat="$1" out="$2"
  echo "iso_time,elapsed_s,gpu0_mem_used_mib,gpu0_util_pct,gpu_others_mem_mib,target_rss_mib,target_vmhwm_mib,host_load1,cpu_cores_busy" > "${out}"
  (
    local start prev_cpu="" prev_t="" hz
    start="$(date +%s)"; hz="$(getconf CLK_TCK 2>/dev/null || echo 100)"
    while :; do
      local now el pid rss hwm cpub cur_cpu gmem gutil gother load
      now="$(date +%s)"; el=$(( now - start ))
      pid="$(pgrep -n -f "${pat}" 2>/dev/null || true)"
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
      # co-tenant watchdog: everything except GPU 0 must stay at 0 MiB.
      gother="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', ' '$1!=0{s+=$2} END{printf "%d",s+0}' || echo NA)"
      load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo NA)"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${el},${gmem},${gutil},${gother},${rss},${hwm},${load},${cpub}" >> "${out}"
      sleep "${TICK}"
    done
  ) >/dev/null 2>&1 &          # <-- load-bearing; see NOTE above
  echo $!
}

# Watchdog: prove the target actually started. Job 3354534 sat idle for 2 h and
# only the telemetry revealed it. Deliberately NOT pgrep-based: the sampler runs
# `pgrep -f "${pat}"` every TICK seconds, so a pgrep-based watchdog matches the
# sampler's own pgrep command line and false-positives (observed in test). Use
# two signals that cannot self-match -- the run log growing, and GPU 0 memory
# leaving zero once FIDESlib allocates.
require_started() {  # logfile seconds label
  local log="$1" limit="$2" label="$3" i=0 gmem
  while (( i < limit )); do
    if [[ -s "${log}" ]]; then
      echo "    [watchdog] ${label} producing output after ${i}s"; return 0
    fi
    gmem="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
    if [[ "${gmem}" =~ ^[0-9]+$ ]] && (( gmem > 0 )); then
      echo "    [watchdog] ${label} allocated GPU memory (${gmem} MiB) after ${i}s"; return 0
    fi
    sleep 2; i=$(( i + 2 ))
  done
  echo "    [watchdog] WARNING: ${label} produced no output and no GPU allocation after ${limit}s" >&2
  return 1
}

jget() { grep -oE "\"$1\"[[:space:]]*:[[:space:]]*[0-9.eE+-]+" "$2" 2>/dev/null | grep -oE '[0-9.eE+-]+$' | head -1; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
# grep -c prints "0" AND exits 1 on no match, so `|| echo 0` would emit "0\n0"
# and corrupt the awk verdict. Take the first line and default in the shell.
count_matches() { local n; n="$(grep -ciE "$1" "$2" 2>/dev/null | head -1)"; echo "${n:-0}"; }

# =============================================================================
# PHASE 2 -- single-block reference (same node, same session)
# =============================================================================
REF_WALL=""; REF_SERVER=""; REF_CLIENT=""; REF_PASS=no
run_block_ref() {  # omp -> sets REF_*
  local omp="$1"
  local tag="a0_ref_block_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3_omp${omp}_job${JOB}"
  local dir="${EVID}/ref_omp${omp}"; mkdir -p "${dir}"
  local out="${dir}/${tag}.json"
  local tele="${dir}/telemetry.csv"

  echo "--- phase 2: single-block reference, OMP=${omp} (timeout ${REF_TIMEOUT}s) ---"
  OMP_NUM_THREADS="${omp}"
  local sampler; sampler="$(start_sampler 'cpudiagcache_t123_v3' "${tele}")"
  require_started "${dir}/run.log" 180 "block(omp=${omp})" &
  local watchdog=$!
  local t0; t0="$(date +%s)"
  set +e
  # -k 60: plain `timeout` only sends TERM and then waits forever if the CUDA
  # process ignores it. Follow up with KILL so the allocation is never lost.
  OMP_NUM_THREADS="${omp}" timeout -k 60 "${REF_TIMEOUT}" \
    "${NUMA_WRAP[@]}" "${BLOCK_BIN}" \
      --gpu 0 --fixture-dir "${FIX_BLOCK}" --output "${out}" \
      --backend-commit "786c7600fb2f16b724e0acf73df367b27b8afed6" \
      --parent-source-sha256   "$(sha_of "${SRC}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp")" \
      --frozen-source-sha256   "$(sha_of "${SRC}/real_dnagpt_fides_scheme_b_general_attention_t103.cpp")" \
      --schedule-sha256        "$(sha_of "${SRC}/simd_t103_schedule.hpp")" \
      --fixture-manifest-sha256 "$(sha_of "${FIX_BLOCK}/manifest.json")" \
      --fixture-contract-sha256 "$(sha_of "${CONTRACT_BLOCK}")" \
      --source-sha256          "$(sha_of "${SRC}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp")" \
      --container-image "native-fideslib-786c7600-cuda12.8-gcc13.2" \
      --environment "TACC Lonestar6 gpu-a100-dev $(hostname -s) A100 [gpu] A0_REF omp=${omp} [native FIDESlib, CUDA 12.8]" \
      > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  REF_WALL=$(( $(date +%s) - t0 ))
  kill "${sampler}" "${watchdog}" 2>/dev/null || true
  REF_PASS=no
  [[ -f "${out}" ]] && grep -q '"passed"[[:space:]]*:[[:space:]]*true' "${out}" && REF_PASS=yes
  REF_SERVER="$(jget server_linear_algebra_seconds "${out}")"
  REF_CLIENT="$(jget client_boundary_seconds_total "${out}")"
  echo "    omp=${omp} rc=${rc} pass=${REF_PASS} wall=${REF_WALL}s server=${REF_SERVER:-NA} client=${REF_CLIENT:-NA}"
  echo "omp=${omp},rc=${rc},pass=${REF_PASS},wall_s=${REF_WALL},server_s=${REF_SERVER:-NA},client_s=${REF_CLIENT:-NA}" \
    >> "${EVID}/ref_results.csv"
}

if [[ "${KIMON_A0_OMP_SWEEP:-0}" == "1" ]]; then
  echo "=========== A0 phase 2: OMP SWEEP (A0b mode; phase 3 skipped) ==========="
  for n in 32 64 128; do
    [[ "$(remaining)" -gt $(( REF_TIMEOUT + 120 )) ]] || { echo "    budget exhausted, stopping sweep"; break; }
    run_block_ref "${n}"
  done
elif [[ "${KIMON_A0_SKIP_REF:-0}" == "1" ]]; then
  echo "=========== A0 phase 2: SKIPPED (KIMON_A0_SKIP_REF=1) ==========="
else
  echo "=========== A0 phase 2: single-block reference ==========="
  run_block_ref "${OMP_NUM_THREADS}"
fi

# =============================================================================
# PHASE 3 -- e2e probe: the actual fit question
# =============================================================================
E2E_BUDGET=0; E2E_WALL=0; E2E_RC=NA; E2E_DONE=no; E2E_BLOCKS=0
if [[ "${KIMON_A0_OMP_SWEEP:-0}" == "1" ]]; then
  echo "=========== A0 phase 3: SKIPPED (OMP sweep mode) ==========="
else
  E2E_BUDGET="$(remaining)"
  echo "=========== A0 phase 3: e2e probe (budget ${E2E_BUDGET}s) ==========="
  if [[ "${E2E_BUDGET}" -lt 600 ]]; then
    echo "  [SKIP] under 600 s left; phase 2 consumed the window."
  else
    E2E_DIR="${EVID}/e2e_probe"; mkdir -p "${E2E_DIR}"
    E2E_TAG="a0_probe_scheme_b_all_blocks_head_t103_cpudiagcache_job${JOB}"
    E2E_OUT="${E2E_DIR}/${E2E_TAG}.json"
    E2E_TELE="${E2E_DIR}/telemetry.csv"
    # Point the runner at the LS6 platform contract (its default is the frozen
    # Mac-origin one, which does not match this node's float64 oracle arrays).
    export KIMON_FIXTURE_CONTRACT="${CONTRACT_E2E}"
    export FIDES_CONTAINER_IMAGE="native-fideslib-786c7600-cuda12.8-gcc13.2"
    export FIDES_RUN_ENVIRONMENT="TACC Lonestar6 gpu-a100-dev $(hostname -s) A100 [gpu] A0_E2E_PROBE omp=${OMP_NUM_THREADS} [native FIDESlib, CUDA 12.8]"
    sampler2="$(start_sampler '12blocks_head_cpudiagcache' "${E2E_TELE}")"
    require_started "${E2E_DIR}/run.log" 300 'e2e' &
    watchdog2=$!
    t0="$(date +%s)"
    # Stamp every stdout line with elapsed seconds. The driver's own progress
    # format is not relied on: whatever it prints, the stamps make per-block
    # marginal cost recoverable even if the probe is cut off mid-run.
    set +e
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" timeout -k 60 "${E2E_BUDGET}" \
      stdbuf -oL -eL "${E2E_RUNNER}" 0 "${FIX_E2E}" "${E2E_OUT}" 2>&1 \
      | stdbuf -oL awk -v t0="$(date +%s)" '{printf "[%6d s] %s\n", systime()-t0, $0; fflush()}' \
      > "${E2E_DIR}/run.log"
    E2E_RC="${PIPESTATUS[0]}"
    set -e
    E2E_WALL=$(( $(date +%s) - t0 ))
    kill "${sampler2}" "${watchdog2:-}" 2>/dev/null || true
    [[ -f "${E2E_OUT}" ]] && grep -q '"passed"[[:space:]]*:[[:space:]]*true' "${E2E_OUT}" && E2E_DONE=yes
    # Count completed blocks from the stamped log, format-agnostically.
    E2E_BLOCKS="$(count_matches 'block[^0-9]*(1[0-2]|[0-9]).*(done|complete|pass|refresh)' "${E2E_DIR}/run.log")"
    echo "  rc=${E2E_RC} wall=${E2E_WALL}s completed=${E2E_DONE} blocks_seen=${E2E_BLOCKS}"
    echo "  --- e2e probe log (tail) ---"; tail -30 "${E2E_DIR}/run.log" || true
  fi
fi

# =============================================================================
# PHASE 4 -- verdict
# =============================================================================
{
  echo "=========== A0 phase 4: verdict ==========="
  echo "job=${JOB} node=$(hostname -f) omp=${OMP_NUM_THREADS} numactl=${NUMA_WRAP[*]:-none}"
  echo "contracts_ok = ${CONTRACT_OK}"
  echo
  echo "--- phase 2: single-block reference ---"
  if [[ -f "${EVID}/ref_results.csv" ]]; then cat "${EVID}/ref_results.csv"; else echo "  (skipped)"; fi
  echo
  echo "--- phase 3: e2e probe ---"
  echo "  budget_s      = ${E2E_BUDGET}"
  echo "  wall_s        = ${E2E_WALL}"
  echo "  completed     = ${E2E_DONE}"
  echo "  exit_code     = ${E2E_RC}   (124 = hit the probe timeout, expected if NO-GO)"
  echo "  blocks_seen   = ${E2E_BLOCKS}"
  echo
  echo "--- FIT VERDICT (12 blocks + GSR head vs the 7200 s gpu-a100-dev wall) ---"
  awk -v done="${E2E_DONE}" -v wall="${E2E_WALL}" -v blocks="${E2E_BLOCKS}" \
      -v refwall="${REF_WALL:-0}" -v limit=7200 -v reserve=300 'BEGIN{
    usable = limit - reserve;
    if (done == "yes") {
      printf "  VERDICT = GO. Full e2e completed in %d s (%.2f h), %.0f%% of the wall.\n", wall, wall/3600.0, 100.0*wall/limit;
      printf "  A6 can run one example per job. Shard size = 12 blocks/job.\n";
    } else if (blocks + 0 >= 1) {
      per = wall / (blocks + 0);
      proj = per * 12;
      printf "  measured marginal cost = %.0f s/block (%d blocks in %d s)\n", per, blocks, wall;
      printf "  projected 12-block e2e = %.0f s (%.2f h)\n", proj, proj/3600.0;
      if (proj <= usable) {
        printf "  VERDICT = GO (projected, not observed). Re-run A0 with KIMON_A0_SKIP_REF=1\n";
        printf "            to confirm end-to-end inside one window before committing A6.\n";
      } else {
        nb = int(usable / per); if (nb < 1) nb = 1;
        printf "  VERDICT = NO-GO. Shard A6 into <= %d block(s)/job -> %d jobs per example,\n", nb, int((12 + nb - 1) / nb);
        printf "            chained through the client refresh boundary (already [V] PASS:\n";
        printf "            two_block_refresh + cross-process context/key serialization).\n";
      }
    } else if (refwall + 0 > 0) {
      printf "  VERDICT = INCONCLUSIVE. e2e produced no readable block markers.\n";
      printf "  Fallback bound from the phase-2 single block: %d s x 12 = %.0f s (%.2f h).\n", refwall, refwall*12.0, refwall*12.0/3600.0;
      printf "  NOTE: the e2e binary has NO T123 optimizations, so this is a LOWER bound;\n";
      printf "        the real e2e is expected to be slower per block, not faster.\n";
      printf "  Next: read e2e_probe/run.log for the driver progress format and fix the\n";
      printf "        block-marker grep in phase 3, then re-run with KIMON_A0_SKIP_REF=1.\n";
    } else {
      printf "  VERDICT = INCONCLUSIVE (no usable phase-2 or phase-3 measurement).\n";
    }
  }'
  echo
  echo "--- measurement-validity attestation (A1 prerequisite) ---"
  for t in "${EVID}"/*/telemetry.csv; do
    [[ -f "${t}" ]] || continue
    echo "  $(basename "$(dirname "${t}")"):"
    awk -F, 'NR>1{
        if($3>gm)gm=$3; if($5>go)go=$5; if($6!="NA"&&$6+0>rss)rss=$6;
        if($8!="NA"&&$8+0>ld)ld=$8; if($9!="NA"&&$9+0>cb)cb=$9; n++
      } END{
        printf "    gpu0_mem_peak=%d MiB  other_gpu_mem_peak=%d MiB  target_rss_peak=%.1f GiB\n", gm, go, rss/1024;
        printf "    host_load1_peak=%.1f  peak_cpu_cores_busy=%.1f  samples=%d\n", ld, cb, n;
        if (go+0 > 0) printf "    *** CO-TENANT DETECTED on GPU 1+ -- timing from this run is NOT clean ***\n";
      }' "${t}"
  done
  echo
  echo "evidence_dir = ${EVID}"
} | tee "${EVID}/SUMMARY.txt"

# machine-readable verdict for the next planning step
{
  printf '{\n'
  printf '  "job": "%s",\n'            "${JOB}"
  printf '  "node": "%s",\n'           "$(hostname -f)"
  printf '  "partition": "gpu-a100-dev",\n'
  printf '  "wall_limit_s": %d,\n'     "${WALL_LIMIT}"
  printf '  "omp_num_threads": %s,\n'  "${OMP_NUM_THREADS}"
  printf '  "contracts_ok": "%s",\n'   "${CONTRACT_OK}"
  printf '  "ref_block_wall_s": %s,\n' "${REF_WALL:-null}"
  printf '  "ref_block_passed": "%s",\n' "${REF_PASS}"
  printf '  "ref_server_s": %s,\n'     "${REF_SERVER:-null}"
  printf '  "ref_client_s": %s,\n'     "${REF_CLIENT:-null}"
  printf '  "e2e_budget_s": %s,\n'     "${E2E_BUDGET}"
  printf '  "e2e_wall_s": %s,\n'       "${E2E_WALL}"
  printf '  "e2e_completed": "%s",\n'  "${E2E_DONE}"
  printf '  "e2e_exit_code": "%s",\n'  "${E2E_RC}"
  printf '  "e2e_blocks_seen": %s\n'   "${E2E_BLOCKS}"
  printf '}\n'
} > "${EVID}/verdict.json"

echo "==============================================================="
echo "[A0 DONE] ${EVID}/SUMMARY.txt  +  ${EVID}/verdict.json"
# A0 is a calibration probe: a NO-GO is a valid, informative outcome, so the
# job exits 0 unless the platform itself failed revalidation (phase 1 exit 2).
exit 0
