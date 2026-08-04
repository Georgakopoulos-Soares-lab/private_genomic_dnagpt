#!/usr/bin/env bash
#SBATCH -J kimon-c3-e2e
#SBATCH -o kimon/logs/config3_all_opts/slurm-e2e-%j.out
#SBATCH -e kimon/logs/config3_all_opts/slurm-e2e-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -A CHANGE_ME_ALLOCATION
#
# Config 3 = ALL single-process optimizations combined, END-TO-END: all 12
# DNAGPT blocks + the GSR head, T=103, 1 GPU, with a client refresh between
# block pairs -- composing the CPU-side diagonal-vector-cache per-block source
# instead of the plain depth-13 block. Same complete encrypted forward pass and
# accuracy target (~1e-9) as Config 1 e2e; only the per-block compute path
# differs (the diagonal cache removes redundant CPU-side plaintext-encoding
# work). Est. 15.5-45.5 h -- request the max walltime and run on an uncontended
# node (the diagonal cache is CPU-side, so a clean host matters for timing).
#
# Submit FROM THE REPO ROOT (after Setup in README):
#   mkdir -p kimon/logs/config3_all_opts
#   sbatch kimon/configs/config3_all_opts/e2e.sbatch          # edit -A first
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load tacc-apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
source kimon/env/common.sh

CONFIG=config3_all_opts
TAG="fhe_c3_allopts_e2e_scheme_b_all_blocks_head_t103_cpudiagcache_a100_${SLURM_JOB_ID:-$(run_stamp)}"
RUN_DIR="$(new_run_dir "${CONFIG}" "${TAG}")"
OUT="${RUN_DIR}/${TAG}.json"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

run_scheme_b run_scheme_b_all_blocks_head_t103_cpudiagcache.sh \
  0 "$(cpath "${FIX}")" "$(cpath "${OUT}")" 2>&1 | tee "${RUN_DIR}/run.log"

if grep -q "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_PASS" "${RUN_DIR}/run.log"; then
  echo "[PASS] config3 e2e -> ${OUT}"
else
  echo "[FAIL] config3 e2e; see ${RUN_DIR}/run.log"; exit 1
fi
