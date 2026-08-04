#!/usr/bin/env bash
#SBATCH -J kimon-c3-block
#SBATCH -o kimon/logs/config3_all_opts/slurm-block-%j.out
#SBATCH -e kimon/logs/config3_all_opts/slurm-block-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -A CHANGE_ME_ALLOCATION
#
# Config 3 = ALL single-process optimizations combined: B=8 Token-SIMD +
# depth-13/digits-3/ring-65536 + CPU-side diagonal-vector cache. ONE full
# block, T=103, 1 GPU. This is the fastest single-GPU config (accuracy-verified
# ~1e-9). See GUIDE.md. (The +sharding variant is block_sharded.sbatch.)
#
# Submit FROM THE REPO ROOT:
#   mkdir -p kimon/logs/config3_all_opts
#   sbatch kimon/configs/config3_all_opts/block.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load tacc-apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
source kimon/env/common.sh

CONFIG=config3_all_opts
TAG="fhe_c3_allopts_block_t103_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_a100_${SLURM_JOB_ID:-$(run_stamp)}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

run_scheme_b run_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.sh \
  0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" 2>&1 | tee "${RUN_DIR}/run.log"

if grep -q "REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_CPUDIAGCACHE_GATE_PASS" "${RUN_DIR}/run.log"; then
  echo "[PASS] config3 block -> ${OUT}"
else
  echo "[FAIL] config3 block; see ${RUN_DIR}/run.log"; exit 1
fi
