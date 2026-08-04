#!/usr/bin/env bash
#SBATCH -J kimon-c1-e2e
#SBATCH -o kimon/logs/config1_simd/slurm-e2e-%j.out
#SBATCH -e kimon/logs/config1_simd/slurm-e2e-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -A CHANGE_ME_ALLOCATION
#
# Config 1 = SIMD batching, END-TO-END: all 12 DNAGPT blocks + the GSR head,
# T=103, 1 GPU, with a client refresh between block pairs. This is the full
# encrypted forward pass. Est. 15.5-45.5 h -- request the max walltime and run
# on an uncontended node.
#
# Submit FROM THE REPO ROOT (after Setup in README):
#   mkdir -p kimon/logs/config1_simd
#   sbatch kimon/configs/config1_simd/e2e.sbatch          # edit -A first
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load tacc-apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
source kimon/env/common.sh

CONFIG=config1_simd
TAG="fhe_c1_simd_e2e_scheme_b_all_blocks_head_t103_a100_${SLURM_JOB_ID:-$(run_stamp)}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

run_scheme_b run_scheme_b_all_blocks_head_t103_simd.sh \
  0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" 2>&1 | tee "${RUN_DIR}/run.log"

if grep -q "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_SIMD_PASS" "${RUN_DIR}/run.log"; then
  echo "[PASS] config1 e2e -> ${OUT}"
else
  echo "[FAIL] config1 e2e; see ${RUN_DIR}/run.log"; exit 1
fi
