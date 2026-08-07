#!/usr/bin/env bash
# THROWAWAY v4 feasibility diagnostic job -- not tracked evidence, delete
# after use (this script + the _v4_diag.cpp source + CMakeLists block +
# output dir). Runs the diagnostic binary just long enough to print every
# weight's actual encode level (12 misses = 12 weights) then can be killed;
# we don't need the full 12-minute run, just the QKV+attn_proj+first-MLP
# misses, which happen within the first ~5 minutes.
#SBATCH -J kimon-v4diag
#SBATCH -p gpu-a100-dev
#SBATCH -A MCB25091
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:15:00
#SBATCH -o /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/v4diag-%j.out
#SBATCH -e /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/v4diag-%j.err
set -euo pipefail
readonly REPO=/scratch/10899/kimopro/private_genomic_dnagpt
cd "${REPO}"
export REPO_ROOT="${REPO}"
# shellcheck disable=SC1091
source "${REPO}/kimon/env/tacc_env.sh"
export OMP_NUM_THREADS="${KIMON_OMP:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}"

readonly BIN="${REPO}/fhe/gpu_real_scheme_b/build/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag"
readonly FIX="${REPO}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
readonly CONTRACT="${REPO}/kimon/env/fixtures_ls6/fixture_t103.sha256"
readonly OUT="/tmp/v4diag_${SLURM_JOB_ID}.json"

sha() { sha256sum "$1" | cut -d' ' -f1; }
SRC="${REPO}/fhe/gpu_real_scheme_b/src"
manifest_sha="$(sha "${FIX}/manifest.json")"
contract_sha="$(sha "${CONTRACT}")"
parent_sha="$(sha "${SRC}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp")"
frozen_sha="$(sha "${SRC}/real_dnagpt_fides_scheme_b_general_attention_t103.cpp")"
schedule_sha="$(sha "${SRC}/simd_t103_schedule.hpp")"
source_sha="$(sha "${SRC}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag.cpp")"

echo "node=$(hostname -f) omp=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${BIN}" \
  --gpu 0 --fixture-dir "${FIX}" --output "${OUT}" \
  --backend-commit "786c7600fb2f16b724e0acf73df367b27b8afed6" \
  --parent-source-sha256 "${parent_sha}" \
  --frozen-source-sha256 "${frozen_sha}" \
  --schedule-sha256 "${schedule_sha}" \
  --fixture-manifest-sha256 "${manifest_sha}" \
  --fixture-contract-sha256 "${contract_sha}" \
  --source-sha256 "${source_sha}" \
  --container-image "native-fideslib-786c7600-cuda12.8-gcc13.2" \
  --environment "v4diag $(hostname -s) omp=${OMP_NUM_THREADS}" \
  2>&1 | tee "/tmp/v4diag_${SLURM_JOB_ID}.log" &
RUNPID=$!

# We only need the 12 [diag] lines (all emitted within the first ~5 min,
# well before the full ~12 min run completes) -- poll and kill early once
# we see all 12, to save GPU time on a throwaway measurement.
for _ in $(seq 1 170); do
  sleep 5
  n=$(grep -c '^\[diag\]' "/tmp/v4diag_${SLURM_JOB_ID}.log" 2>/dev/null || echo 0)
  echo "progress: ${n}/12 weight levels captured"
  if [[ "${n}" -ge 12 ]]; then
    echo "=== all 12 weights captured, stopping early ==="
    kill "${RUNPID}" 2>/dev/null || true
    break
  fi
done
wait "${RUNPID}" 2>/dev/null || true

echo "=== FULL DIAG OUTPUT ==="
grep '^\[diag\]' "/tmp/v4diag_${SLURM_JOB_ID}.log"
