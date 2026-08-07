#!/usr/bin/env bash
# Stage-2/3 GPU micro-gate for the T123 optimizations. Builds + runs a tiny
# standalone test on one A100; a few minutes total. Validates correctness/
# safety BEFORE committing to a full ~10 min block run.
#   Stage-2 (TEST_A/B/C/D): Tier-1 encoded-Plaintext clone reuse across
#     levels + Tier-2 parallel (OMP) encode -- validated the T123 driver.
#   Stage-3 / v2 (TEST_E/F/G): concurrent client-boundary Decrypt/Encrypt
#     from OMP threads on the SAME CryptoContext/keys -- validates the
#     optimization campaign's v2 (parallelize the 91 score-tile decrypts +
#     727 weight-tile encrypts, currently fully serial and GPU-idle).
#
#   SUBMIT FROM A LOGIN NODE:
#     sbatch /scratch/10899/kimopro/private_genomic_dnagpt/kimon/configs/microgate.sbatch
#
# PASS = TEST_A/B/D/E/F maxdiff ~0 (<1e-6), TEST_C/G OK with maxdiff_after
# ~0, MICROGATE_DONE, no crash.
#SBATCH -J kimon-microgate
#SBATCH -p gpu-a100-dev
#SBATCH -A MCB25091
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:20:00
#SBATCH -o /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/microgate/slurm/mg-%j.out
#SBATCH -e /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/microgate/slurm/mg-%j.err
set -euo pipefail
readonly REPO=/scratch/10899/kimopro/private_genomic_dnagpt
cd "${REPO}"
# shellcheck disable=SC1091
source "${REPO}/kimon/env/tacc_env.sh"
# FORCE the core count -- TACC pre-sets OMP_NUM_THREADS=1, so a `:-` default
# would never apply. Assign unconditionally (override with KIMON_OMP=N).
export OMP_NUM_THREADS="${KIMON_OMP:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}"
echo "node=$(hostname -f) omp=${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MG="${REPO}/fhe/gpu_real_scheme_b/microgate"
cmake -S "${MG}" -B "${MG}/build" -DCMAKE_BUILD_TYPE=Release \
  -DFIDESLIB_PACKAGE_DIR="${FIDESLIB_PACKAGE_DIR}" \
  -DOPENFHE_PACKAGE_DIR="${FIDESLIB_PREFIX}/lib/OpenFHE" \
  -DFIDESLIB_ARCH=80-real -DCMAKE_CUDA_ARCHITECTURES=80-real
cmake --build "${MG}/build" --target clone_microtest -j "${OMP_NUM_THREADS}"

echo "===== running micro-gate ====="
"${MG}/build/clone_microtest" 0
echo "===== micro-gate exit=$? ====="
