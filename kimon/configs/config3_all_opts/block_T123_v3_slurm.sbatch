#!/usr/bin/env bash
# =============================================================================
# T123 v3 = T123 (Tier 1/2/3 encode optimizations) + freeing the CPU-side
# (OpenFHE) EvalMult/rotation key maps right after LoadContext, as a SLURM
# batch job on TACC Lonestar6. Same fixture / oracle / depth / packing /
# crypto-op schedule / client-boundary sequencing as t123 (sequential, no
# concurrency change -- v2's parallel-boundary attempt is abandoned, see
# kimon/logs/optimization_campaign/PLAN.md). Only HOW host RAM is managed
# changes:
#   parent (t123)  LoadContext copies the relin key + 80 rotation keys to
#                  the GPU, but OpenFHE's CPU-side copies (in its static
#                  eval-mult-key/rotation-key maps) stay resident in host
#                  RAM for the entire run, unused after LoadContext.
#   v3             right after LoadContext, call OpenFHE's own
#                  ClearEvalMultKeys()/ClearEvalAutomorphismKeys() to free
#                  those CPU-side copies immediately. Source-verified
#                  (LoadContext is the only reader of these maps in
#                  FIDESlib's non-vendored code) and validated on real GPU
#                  hardware by the micro-gate's TEST_I before being wired in.
#
#   SUBMIT FROM A LOGIN NODE:
#     sbatch /scratch/10899/kimopro/private_genomic_dnagpt/kimon/configs/config3_all_opts/block_T123_v3_slurm.sbatch
#
# Uses all cores (OMP_NUM_THREADS = cores on the node). Output on /scratch.
#SBATCH -J kimon-T123v3-block
#SBATCH -p gpu-a100-dev
#SBATCH -A MCB25091
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -o /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/config3_all_opts/slurm/T123v3-%j.out
#SBATCH -e /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/config3_all_opts/slurm/T123v3-%j.err
set -euo pipefail

readonly REPO=/scratch/10899/kimopro/private_genomic_dnagpt
readonly JOB="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly EVID="${REPO}/kimon/logs/config3_all_opts/slurm/T123v3_job_${JOB}"
mkdir -p "${EVID}"
cd "${REPO}"
export REPO_ROOT="${REPO}"
# shellcheck disable=SC1091
source "${REPO}/kimon/env/tacc_env.sh"

# --- Tier 2/3: use every core on the node for the parallel batched encode ----
# FORCE it -- TACC pre-sets OMP_NUM_THREADS=1, so a `:-` default never applies.
# Assign unconditionally (override with KIMON_OMP=N).
export OMP_NUM_THREADS="${KIMON_OMP:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}"
# Safe, numerics-neutral locality knobs (affect speed only, not results):
# pin OMP threads to cores for the parallel batched encode, and (if available)
# allocate host memory local to the running CPUs -> faster encode + faster
# host<->device (PCIe) staging. 1-GPU safe; no effect on the oracle. Carried
# over from v1 (accepted into the campaign as v1_affinity_only).
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
NUMA_WRAP=()
if command -v numactl >/dev/null 2>&1; then NUMA_WRAP=(numactl --localalloc); fi

readonly SRC="${REPO}/fhe/gpu_real_scheme_b/src"
readonly BUILD="${REPO}/fhe/gpu_real_scheme_b/build"
readonly BIN="${BUILD}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3"
readonly FIX="${REPO}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
readonly CONTRACT="${REPO}/kimon/env/fixtures_ls6/fixture_t103.sha256"   # platform-tagged
readonly TAG="fhe_T123v3_allopts_block_t103_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3_a100_slurm_${JOB}"
readonly RUN_DIR="${REPO}/kimon/logs/config3_all_opts/${TAG}"
mkdir -p "${RUN_DIR}"
readonly OUT="${RUN_DIR}/${TAG}.json"

[[ -x "${BIN}" ]] || { echo "[FATAL] T123 v3 binary missing: ${BIN} (build it: KIMON_TARGETS=real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3 kimon/env/build_binaries.sh)" >&2; exit 2; }
[[ -f "${FIX}/manifest.json" ]] || { echo "[FATAL] fixture missing: ${FIX}" >&2; exit 2; }

echo "=================== kimon T123 v3 (SLURM batch) ==================="
echo "job=${JOB} node=$(hostname -f) omp=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/gpu = /'
echo "==============================================================="

# integrity: the arrays the C++ consumes must match the platform contract
( cd "${FIX}" && sha256sum --check --status "${CONTRACT}" ) \
  || { echo "[FATAL] fixture arrays do not match platform contract ${CONTRACT}" >&2; exit 2; }

# sha args (must equal the binary's compiled pins: parent/frozen/schedule +
# platform fixture manifest/contract; source-sha is recorded only).
sha() { sha256sum "$1" | cut -d' ' -f1; }
manifest_sha="$(sha "${FIX}/manifest.json")"
contract_sha="$(sha "${CONTRACT}")"
parent_sha="$(sha "${SRC}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp")"
frozen_sha="$(sha "${SRC}/real_dnagpt_fides_scheme_b_general_attention_t103.cpp")"
schedule_sha="$(sha "${SRC}/simd_t103_schedule.hpp")"
source_sha="$(sha "${SRC}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp")"

# --- telemetry sampler: GPU mem/util + target RSS + host load + CPU-busy ------
# The extra cpu_cores_busy column (from /proc/<pid>/stat utime+stime deltas)
# lets us SEE the client-vs-server phases independently of the binary's own
# timers: server = GPU-active (high gpu_util); client boundary = GPU idle while
# the target process burns CPU (decrypt/re-encrypt). readonly TICK=5 s.
readonly TICK=5
readonly TELE="${EVID}/telemetry.csv"
echo "iso_time,elapsed_s,gpu_mem_used_mib,gpu_util_pct,target_rss_mib,target_vmhwm_mib,host_load1,cpu_cores_busy" > "${TELE}"
(
  start="$(date +%s)"
  prev_cpu=""; prev_t=""
  hz="$(getconf CLK_TCK 2>/dev/null || echo 100)"
  while :; do
    now="$(date +%s)"; el=$(( now - start ))
    pid="$(pgrep -n -f 'cpudiagcache_t123_v3' 2>/dev/null || true)"
    rss=NA; hwm=NA; cpub=NA
    if [[ -n "${pid}" && -r "/proc/${pid}/status" ]]; then
      rss="$(awk '/^VmRSS:/{printf "%.0f",$2/1024}' "/proc/${pid}/status" 2>/dev/null || echo NA)"
      hwm="$(awk '/^VmHWM:/{printf "%.0f",$2/1024}' "/proc/${pid}/status" 2>/dev/null || echo NA)"
      # utime(14)+stime(15) in clock ticks -> CPU-seconds; delta/interval = cores busy
      cur_cpu="$(awk '{print $14+$15}' "/proc/${pid}/stat" 2>/dev/null || echo '')"
      if [[ -n "${cur_cpu}" && -n "${prev_cpu}" && -n "${prev_t}" && "${now}" -gt "${prev_t}" ]]; then
        cpub="$(awk -v c="${cur_cpu}" -v p="${prev_cpu}" -v dt="$(( now - prev_t ))" -v hz="${hz}" \
                'BEGIN{printf "%.1f",(c-p)/hz/dt}')"
      fi
      prev_cpu="${cur_cpu}"; prev_t="${now}"
    fi
    # -i 0: this job only ever touches --gpu 0; on multi-GPU nodes (e.g.
    # gpu-a100-dev has 3 A100s per node) a node-wide sum/list would mix in
    # other GPUs -- possibly another job's, since that partition is shared.
    gmem="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo NA)"
    gutil="$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo NA)"
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo NA)"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${el},${gmem},${gutil},${rss},${hwm},${load},${cpub}" >> "${TELE}"
    sleep "${TICK}"
  done
) & SAMPLER=$!
trap 'kill "${SAMPLER}" 2>/dev/null || true' EXIT

t0="$(date +%s)"
set +e
"${NUMA_WRAP[@]}" "${BIN}" \
  --gpu 0 --fixture-dir "${FIX}" --output "${OUT}" \
  --backend-commit "786c7600fb2f16b724e0acf73df367b27b8afed6" \
  --parent-source-sha256 "${parent_sha}" \
  --frozen-source-sha256 "${frozen_sha}" \
  --schedule-sha256 "${schedule_sha}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --fixture-contract-sha256 "${contract_sha}" \
  --source-sha256 "${source_sha}" \
  --container-image "native-fideslib-786c7600-cuda12.8-gcc13.2" \
  --environment "TACC Lonestar6 gpu-a100-dev $(hostname -s) A100 [gpu] T123_V3 omp=${OMP_NUM_THREADS} [native FIDESlib, CUDA 12.8]" \
  > "${RUN_DIR}/run.log" 2>&1
RC=$?
set -e
WALL=$(( $(date +%s) - t0 ))
kill "${SAMPLER}" 2>/dev/null || true

echo "----- run.log (tail) -----"; tail -25 "${RUN_DIR}/run.log" 2>/dev/null || true

PASSED=no
[[ -f "${OUT}" ]] && grep -q '"passed"[[:space:]]*:[[:space:]]*true' "${OUT}" && PASSED=yes

# jget KEY -> numeric value from the result JSON (or empty).
jget() { grep -oE "\"$1\"[[:space:]]*:[[:space:]]*[0-9.eE+-]+" "${OUT}" 2>/dev/null | grep -oE '[0-9.eE+-]+$' | head -1; }
KEYGEN=$(jget context_keygen_load); ENCIN=$(jget encrypt_embedded_inputs)
EVAL=$(jget encrypted_evaluation); SRV=$(jget server_linear_algebra_seconds)
CLI=$(jget client_boundary_seconds_total); DEC=$(jget final_decrypt)
RT=$(jget round_trips); STILES=$(jget score_tile_decryptions); WTILES=$(jget weight_tile_encryptions)

{
  echo "job=${JOB} node=$(hostname -f) omp=${OMP_NUM_THREADS} numactl=${NUMA_WRAP[*]:-none} exit=${RC}"
  echo "gate = $([[ "${RC}" -eq 0 && "${PASSED}" == yes ]] && echo PASS || echo FAIL)  passed_json=${PASSED}"
  echo "result_json = ${OUT}"
  echo "wall_clock_s = ${WALL}"
  echo "--- accuracy (result JSON) ---"
  grep -oE '"(global_rel_inf|worst_token_rel_inf|max_abs_error|all_finite|tol)"[^,}]*' "${OUT}" 2>/dev/null || true

  echo "--- CLIENT vs SERVER work (binary timers) ---"
  awk -v k="${KEYGEN:-0}" -v e="${ENCIN:-0}" -v s="${SRV:-0}" -v c="${CLI:-0}" -v d="${DEC:-0}" 'BEGIN{
    cli=e+c+d; onl=s+cli;
    printf "  server_linear_algebra_s = %.1f\n", s;
    printf "  client_total_s          = %.1f  (encrypt_input %.2f + boundaries %.1f + final_decrypt %.2f)\n", cli, e, c, d;
    printf "  online_total_s          = %.1f\n", onl;
    if(onl>0){ printf "  server_share            = %.1f%%\n  client_share            = %.1f%%\n", 100*s/onl, 100*cli/onl; }
    printf "  keygen_one_time_s       = %.2f (amortized across queries)\n", k;
  }'
  echo "--- CLIENT boundary per stage (round-trip counts, JSON) ---"
  awk -v rt="${RT:-0}" -v st="${STILES:-0}" -v wt="${WTILES:-0}" -v c="${CLI:-0}" 'BEGIN{
    soft=st+wt; other=rt-soft;                 # other = LayerNorm(26)+GELU(13)
    printf "  total round-trips       = %d\n", rt;
    printf "  softmax/attention       = %d  (%d score-tile decrypts + %d weight-tile encrypts)\n", soft, st, wt;
    printf "  layernorm+gelu          = %d\n", other;
    if(rt>0){ printf "  softmax share of client = %.1f%%  (~%.1f s of %.1f s)\n", 100*soft/rt, c*soft/rt, c; }
  }'

  echo "--- independent phase cross-check (telemetry: GPU-active vs client-CPU) ---"
  # server phase ~ gpu_util high; client phase ~ gpu idle while target CPU busy.
  awk -F, -v tick="${TICK:-5}" 'NR>1{
      gu=$4+0; cb=($8=="NA"?0:$8+0);
      if(gu>=20) srv+=tick; else if(gu<5 && cb>=1.0) cli+=tick; else oth+=tick;
      if($3>gm)gm=$3; if($6!="NA"&&$6+0>rss)rss=$6; el=$2; if(cb>cbmax)cbmax=cb;
    } END{
      printf "  gpu-active(server) wall  = %d s\n  cpu-busy/gpu-idle(client) = %d s\n  other/overlap wall       = %d s\n", srv, cli, oth;
      printf "  gpu_mem_peak_mib = %d\n  target_rss_peak_mib = %d (%.1f GiB)\n  peak_client_cpu_cores = %.1f\n  telemetry_elapsed_s = %d\n", gm, rss, rss/1024, cbmax, el;
    }' "${TELE}"

  echo "--- v3 target: keygen-phase RSS (should drop right after LoadContext, not stay flat) ---"
  awk -F, 'NR>1 && $5!="NA"{ printf "  t=%ss rss=%sMiB\n", $2, $5 }' "${TELE}" | head -8
} | tee "${EVID}/SUMMARY.txt"

# Auto-record a normalized evidence row into the optimization campaign, so the
# before/after comparison survives across sessions with no manual step.
if [[ -f "${OUT}" ]]; then
  "${REPO}/.venv/bin/python" "${REPO}/kimon/analyze/campaign_metrics.py" \
    "${KIMON_CAMPAIGN_LABEL:-run_${JOB}}" "${OUT}" "${TELE}" "${RUN_DIR}/run.log" \
    2>/dev/null | tail -3 || true
fi

echo "==============================================================="
if [[ "${RC}" -eq 0 && "${PASSED}" == yes ]]; then
  echo "[PASS] T123 v3 block -> ${OUT}"; exit 0
else
  echo "[FAIL] T123 v3 block (rc=${RC}, passed_json=${PASSED}); see ${RUN_DIR}/run.log + ${EVID}/SUMMARY.txt"; exit 1
fi
