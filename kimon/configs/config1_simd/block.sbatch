#!/usr/bin/env bash
#SBATCH -J kimon-c1-block
#SBATCH -o kimon/logs/config1_simd/slurm-block-%j.out
#SBATCH -e kimon/logs/config1_simd/slurm-block-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -A CHANGE_ME_ALLOCATION
#
# Config 1 = SIMD batching (B=8 Token-SIMD), ONE full DNAGPT block, T=103,
# depth-13/digits-3/ring-65536. 1 GPU. See GUIDE.md.
#
# Submit FROM THE REPO ROOT:
#   mkdir -p kimon/logs/config1_simd
#   sbatch kimon/configs/config1_simd/block.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# Lmod's `module` function aborts under `set -u`; guard it (see kimon/env/tacc_env.sh).
{ set +u; module load tacc-apptainer 2>/dev/null || module load apptainer 2>/dev/null || true; set -u; }
source kimon/env/common.sh

CONFIG=config1_simd
TAG="fhe_c1_simd_block_t103_scheme_b_simd_full_t103_depth13_digits3_ring65536_a100_${SLURM_JOB_ID:-$(run_stamp)}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

# 1 GPU: the binary's --gpu 0 selects the first GPU visible in this allocation
# (SLURM controls visibility; no docker-style physical-index remap needed).
run_scheme_b run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh \
  0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" 2>&1 | tee "${RUN_DIR}/run.log"

if grep -q "REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_GATE_PASS" "${RUN_DIR}/run.log"; then
  echo "[PASS] config1 block -> ${OUT}"
else
  echo "[FAIL] config1 block; see ${RUN_DIR}/run.log"; exit 1
fi
