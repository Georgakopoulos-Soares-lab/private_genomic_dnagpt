#!/usr/bin/env bash
# Shared TACC (Lonestar6) environment for the kimon encrypted-DNAGPT runs.
#
# Source this from a bash shell before any build or run:
#   source kimon/env/tacc_env.sh
#
# It is idempotent and safe to source repeatedly. It does NOT load a container;
# the container path (apptainer) and the native path both use it for modules.
#
# IMPORTANT (Lmod): never pipe a `module` invocation (`module load x | tail`).
# The shell function is then evaluated in a subshell and the environment change
# is silently discarded. Redirect instead (`module load x >/dev/null 2>&1`).

# --- module environment -----------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
  for _init in /opt/apps/lmod/lmod/init/bash /usr/share/lmod/lmod/init/bash; do
    # shellcheck disable=SC1090
    [[ -f "${_init}" ]] && source "${_init}" && break
  done
  unset _init
fi

# CUDA 12.8 is chosen to match the Lonestar6 gpu-a100 driver (570.x exposes
# CUDA 12.8). The pinned docker image uses CUDA 13.0, which needs driver >=580
# and therefore cannot execute on this host without a compat layer. FIDESlib's
# commit and the asymmetric-Chebyshev patch stay pinned; only the CUDA toolkit
# differs, and that deviation must be recorded in every run's environment
# string.
KIMON_MODULES="${KIMON_MODULES:-gcc/13.2.0 cuda/12.8 cmake/4.1.1 python/3.12.11}"
if [[ "${KIMON_WANT_APPTAINER:-0}" == "1" ]]; then
  KIMON_MODULES="${KIMON_MODULES} tacc-apptainer/1.4.1"
fi
# Lmod's `module` function references unset variables, so it aborts the caller
# under `set -u` (nounset) -- even `|| true` can't save it. common.sh sources
# this file under `set -u`, so disable nounset around module ops and restore.
case "$-" in *u*) _kimon_had_u=1 ;; *) _kimon_had_u=0 ;; esac
set +u
# `sbatch --export=ALL` (or an interactive login shell) can hand this script
# an environment with a DIFFERENT python/gcc/cuda module already loaded (e.g.
# TACC's default intel19/python3). Lmod's family-conflict handling then makes
# `module load python/3.12.11` a silent no-op (stderr is suppressed below), so
# LD_LIBRARY_PATH never gains the module's lib dir and .venv's python fails
# with "cannot open shared object file: libpython3.12.so.1.0" later -- with no
# visible error, since that failure is itself inside a suppressed pipeline.
# `module purge` first makes the loaded set deterministic regardless of what
# the caller's shell had.
module purge >/dev/null 2>&1 || true
# shellcheck disable=SC2086
module load ${KIMON_MODULES} >/dev/null 2>&1 || true
[[ "${_kimon_had_u}" == "1" ]] && set -u
unset _kimon_had_u

# --- paths ------------------------------------------------------------------
KIMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KIMON_ENV_DIR
export REPO_ROOT="${REPO_ROOT:-$(cd "${KIMON_ENV_DIR}/../.." && pwd)}"

# FIDESlib native checkout + install prefix (both gitignored).
export FIDESLIB_ROOT="${FIDESLIB_ROOT:-${KIMON_ENV_DIR}/fideslib-src}"
export FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-${KIMON_ENV_DIR}/fideslib-install}"
export FIDESLIB_PACKAGE_DIR="${FIDESLIB_PACKAGE_DIR:-${FIDESLIB_PREFIX}/share/fideslib/cmake}"
export FIDESLIB_ARCH="${FIDESLIB_ARCH:-80-real}"
export FIDES_REAL_SCHEME_B_BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${REPO_ROOT}/fhe/gpu_real_scheme_b/build}"

# The installed FIDESlib/OpenFHE shared objects must be resolvable at run time.
export LD_LIBRARY_PATH="${FIDESLIB_PREFIX}/lib:${FIDESLIB_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${FIDESLIB_PREFIX}:${CMAKE_PREFIX_PATH:-}"

# Keep pip/cmake scratch off the small $HOME quota. Use TACC's $SCRATCH (the
# real Lustre scratch for this user) as the base -- NOT $(id -u), which under
# this harness is the container uid (903286) and yields an unwritable path.
_kimon_scratch="${SCRATCH:-$(cd "${REPO_ROOT}/.." && pwd)}"
export TMPDIR="${TMPDIR:-${_kimon_scratch}/kimon_tmp}"
mkdir -p "${TMPDIR}" 2>/dev/null || export TMPDIR=/tmp
unset _kimon_scratch

echo "[tacc_env] repo=${REPO_ROOT}"
echo "[tacc_env] fideslib_src=${FIDESLIB_ROOT}"
echo "[tacc_env] fideslib_prefix=${FIDESLIB_PREFIX}"
echo "[tacc_env] nvcc=$(command -v nvcc || echo MISSING) gcc=$(gcc -dumpversion 2>/dev/null || echo MISSING)"
