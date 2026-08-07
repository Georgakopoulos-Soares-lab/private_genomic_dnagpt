#!/usr/bin/env bash
#SBATCH -J kimon-c2-block
#SBATCH -o kimon/logs/config2_sharding/slurm-block-%j.out
#SBATCH -e kimon/logs/config2_sharding/slurm-block-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A CHANGE_ME_ALLOCATION
#
# Config 2 = GPU sharding of the attention Q/K/V PROJECTION step (not a full
# block), T=103, depth-13. Process-per-GPU: a CPU-only writer serializes
# context+keys+inputs to a state dir, then TWO reader processes run CONCURRENTLY
# on 2 GPUs -- worker A computes the query+key projections (--gpu 0), worker B
# the value projection (--gpu 1). No ciphertext crosses processes (Q/K/V are
# independent; readers do NOT run attention/GELU/MLP). Full sharded block/e2e is
# blocked by FIDESlib's lack of ciphertext serialization -- see GUIDE.md.
#
# The state dir holds a SERIALIZED SECRET KEY -- it is rm -rf'd on exit (trap).
#
# Submit FROM THE REPO ROOT (needs 2 of the node's 3 A100s):
#   mkdir -p kimon/logs/config2_sharding
#   sbatch kimon/configs/config2_sharding/block.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# Lmod's `module` function aborts under `set -u`; guard it (see kimon/env/tacc_env.sh).
{ set +u; module load tacc-apptainer 2>/dev/null || module load apptainer 2>/dev/null || true; set -u; }
source kimon/env/common.sh

CONFIG=config2_sharding
STAMP="${SLURM_JOB_ID:-$(run_stamp)}"
RUN_DIR="$(new_run_dir "${CONFIG}" "qkvshard_${STAMP}")"
FIX="${REPO_ROOT}/checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
require_fixture "${FIX}"

STATE_DIR="${RUN_DIR}/state"          # serialized secret key lives here
mkdir -p "${STATE_DIR}"
cleanup_state() { rm -rf "${STATE_DIR}" && echo "[hygiene] removed secret-key state dir"; }
trap cleanup_state EXIT

# tags MUST contain _scheme_b_ AND _qkvshard_ (the reader/writer scripts enforce it)
WTAG="fhe_c2_qkvshard_writer_scheme_b_a100_${STAMP}"
ATAG="fhe_c2_qkvshard_reader_querykey_scheme_b_a100_${STAMP}"
BTAG="fhe_c2_qkvshard_reader_value_scheme_b_a100_${STAMP}"

# Phase 1: writer (CPU-only; builds context/keys, serializes, exits).
run_scheme_b run_scheme_b_simd_shard_writer_t103_depth13.sh \
  0 "$(cpath "${STATE_DIR}")" "$(cpath "${RUN_DIR}/${WTAG}.json")" 2>&1 | tee "${RUN_DIR}/writer.log"

# Phase 2: two readers, CONCURRENT, on 2 distinct GPUs (logical 0 and 1).
run_scheme_b run_scheme_b_simd_shard_reader_t103_depth13.sh \
  0 "$(cpath "${STATE_DIR}")" "$(cpath "${FIX}")" "$(cpath "${RUN_DIR}/${ATAG}.json")" "query,key" \
  >"${RUN_DIR}/reader_querykey.log" 2>&1 &
pid_a=$!
run_scheme_b run_scheme_b_simd_shard_reader_t103_depth13.sh \
  1 "$(cpath "${STATE_DIR}")" "$(cpath "${FIX}")" "$(cpath "${RUN_DIR}/${BTAG}.json")" "value" \
  >"${RUN_DIR}/reader_value.log" 2>&1 &
pid_b=$!

rc=0
wait "${pid_a}" || rc=1
wait "${pid_b}" || rc=1

if grep -q "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_PASS" "${RUN_DIR}/reader_querykey.log" \
   && grep -q "REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_PASS" "${RUN_DIR}/reader_value.log" \
   && [[ "${rc}" -eq 0 ]]; then
  echo "[PASS] config2 block (both shards) -> ${RUN_DIR}"
else
  echo "[FAIL] config2 block; see ${RUN_DIR}/reader_*.log"; exit 1
fi
