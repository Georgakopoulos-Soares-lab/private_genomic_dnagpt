#!/usr/bin/env bash
# =============================================================================
# Build the kimon binaries natively (no idev) -- batch-job wrapper around
# kimon/env/build_binaries.sh RUN_MODE=native.
#
# Why this exists: build_binaries.sh needs a GPU node (CMake probes the CUDA
# toolkit for the sm_80 targets) and RUN_MODE=native (this host's FIDESlib
# install is native, built for driver 570.x / CUDA 12.8 -- the apptainer path
# needs a .sif that was never built here and targets CUDA 13.0 anyway). This
# recompiles all 10 kimon target binaries against current source, which also
# picks up any pinned-hash drift (e.g. the 2026-08-10 stale-binary mismatch)
# automatically since the pins are recomputed from source at build time.
#
# The FIDESlib/OpenFHE native install already exists under
# kimon/env/fideslib-install/, so this is a relink/recompile of the DNAGPT
# targets only -- expected a few minutes, not the from-scratch 15-30 min the
# README quotes.
#
#   SUBMIT FROM A LOGIN NODE:
#     sbatch kimon/configs/build_binaries.sbatch
#
#SBATCH -J kimon-build
#SBATCH -p gpu-a100-dev
#SBATCH -A MCB25091
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -o /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/build/slurm-%j.out
#SBATCH -e /scratch/10899/kimopro/private_genomic_dnagpt/kimon/logs/build/slurm-%j.err
set -euo pipefail

readonly REPO=/scratch/10899/kimopro/private_genomic_dnagpt
cd "${REPO}"
mkdir -p "${REPO}/kimon/logs/build"

echo "job=${SLURM_JOB_ID:-NA} node=$(hostname -f) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/gpu = /' || true

RUN_MODE=native BUILD_JOBS="${BUILD_JOBS:-8}" "${REPO}/kimon/env/build_binaries.sh"

echo "=== built binaries ==="
ls -la "${REPO}/fhe/gpu_real_scheme_b/build" | grep real_dnagpt || true
echo "[BUILD DONE]"
