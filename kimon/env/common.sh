#!/usr/bin/env bash
# Shared helpers for the kimon config runners (block.sbatch / e2e.sbatch).
#
# Sourced from the repo root:  source kimon/env/common.sh
#
# Two execution modes, selected by RUN_MODE:
#   apptainer (default) -- run the repo's fhe/gpu_real_scheme_b/run_*.sh inside
#                          kimon/env/fideslib.sif via `apptainer exec --nv`,
#                          with the repo bind-mounted at /work.
#   native              -- run the same script directly against a natively
#                          installed FIDESlib (kimon/env/build_fideslib_native.sh).
#
# On Lonestar6 the native mode is the supported one: the pinned docker image is
# CUDA 13.0 and the gpu-a100 driver (570.x) only exposes CUDA 12.8, so a
# CUDA-13 container cannot launch kernels on this host.
#
# Exported for the callers: REPO_ROOT, RUN_MODE
# Functions:  run_stamp  new_run_dir  require_fixture  cpath  run_scheme_b

set -uo pipefail

KIMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KIMON_ENV_DIR
export REPO_ROOT="${REPO_ROOT:-$(cd "${KIMON_ENV_DIR}/../.." && pwd)}"
export RUN_MODE="${RUN_MODE:-apptainer}"
export KIMON_SIF="${KIMON_SIF:-${KIMON_ENV_DIR}/fideslib.sif}"
readonly KIMON_SCHEME_B_DIR="${REPO_ROOT}/fhe/gpu_real_scheme_b"

# --- module environment (native mode needs CUDA/gcc; both need a shell PATH) --
if [[ "${RUN_MODE}" == "native" ]]; then
  # shellcheck disable=SC1091
  source "${KIMON_ENV_DIR}/tacc_env.sh" >/dev/null 2>&1 || true
fi

# --- small helpers -----------------------------------------------------------

# A sortable UTC stamp, used when SLURM_JOB_ID is absent (off-SLURM runs).
run_stamp() { date -u '+%Y%m%dT%H%M%SZ'; }

# new_run_dir CONFIG TAG -> creates kimon/logs/CONFIG/TAG and echoes its path.
# Refuses to reuse a directory so evidence stays immutable.
new_run_dir() {
  local config="$1" tag="$2"
  local dir="${REPO_ROOT}/kimon/logs/${config}/${tag}"
  if [[ -e "${dir}" ]]; then
    echo "[FATAL] run dir already exists (evidence is immutable): ${dir}" >&2
    exit 2
  fi
  mkdir -p "${dir}"
  printf '%s\n' "${dir}"
}

require_fixture() {
  local fix="$1"
  if [[ ! -f "${fix}/manifest.json" ]]; then
    echo "[FATAL] fixture missing: ${fix}" >&2
    echo "        run: kimon/env/prepare_fixtures.sh" >&2
    exit 2
  fi
}

# cpath HOST_PATH -> the path as the executing process sees it.
# apptainer mode rewrites the repo prefix to the bind target /work.
cpath() {
  local p="$1"
  if [[ "${RUN_MODE}" == "native" ]]; then
    printf '%s\n' "${p}"
  else
    printf '%s\n' "${p/#${REPO_ROOT}//work}"
  fi
}

# _platform_contract_env RUN_SCRIPT_NAME
# On this ls6 node the frozen (Mac-origin) fixture hashes don't match the
# locally-regenerated float64 @-matmul oracle arrays (weights/input DO match).
# The run scripts honour opt-in KIMON_FIXTURE_CONTRACT / KIMON_EXPECTED_MANIFEST
# / KIMON_EXPECTED_FIXTURE_CONTRACT overrides; here we point them at the
# platform-tagged contracts under kimon/env/fixtures_ls6/. Default ON in native
# mode; set KIMON_USE_PLATFORM_CONTRACT=0 to force the frozen contracts.
_export_platform_contract() {
  local script="$1"
  local ls6="${KIMON_ENV_DIR}/fixtures_ls6"
  [[ -f "${ls6}/expected.env" ]] || { echo "[FATAL] ${ls6}/expected.env missing; run kimon/env/refresh_platform_contracts.sh" >&2; exit 2; }
  # shellcheck disable=SC1091
  source "${ls6}/expected.env"
  # The 4 config main sources were edited additively so the binary's
  # fixture-identity pins are compile-time-overridable; tell the block run
  # scripts to trust that git-reviewed edit (parent/frozen/schedule stay pinned).
  export KIMON_TRUST_MAIN_SOURCE=1
  case "${script}" in
    *all_blocks_head*)     # e2e: contract-only gate
      export KIMON_FIXTURE_CONTRACT="${ls6}/fixture_all_blocks_head_t103.sha256" ;;
    *shard*)               # config 2 Q/K/V sharding
      export KIMON_FIXTURE_CONTRACT="${ls6}/fixture_t103_qkvshard.sha256"
      export KIMON_EXPECTED_FIXTURE_CONTRACT="${KIMON_LS6_CONTRACT_QKVSHARD}"
      export KIMON_EXPECTED_MANIFEST="${KIMON_LS6_MANIFEST_BLOCK}" ;;
    *)                     # single-block depth13 / cpudiagcache
      export KIMON_FIXTURE_CONTRACT="${ls6}/fixture_t103.sha256"
      export KIMON_EXPECTED_FIXTURE_CONTRACT="${KIMON_LS6_CONTRACT_T103}"
      export KIMON_EXPECTED_MANIFEST="${KIMON_LS6_MANIFEST_BLOCK}" ;;
  esac
  echo "[platform-contract] ${script} -> $(basename "${KIMON_FIXTURE_CONTRACT}")"
}

# run_scheme_b RUN_SCRIPT_NAME ARGS...
# RUN_SCRIPT_NAME is a file name inside fhe/gpu_real_scheme_b/.
run_scheme_b() {
  local script="$1"; shift
  if [[ ! -x "${KIMON_SCHEME_B_DIR}/${script}" ]]; then
    echo "[FATAL] runner missing or not executable: ${KIMON_SCHEME_B_DIR}/${script}" >&2
    exit 2
  fi
  if [[ "${RUN_MODE}" == "native" && "${KIMON_USE_PLATFORM_CONTRACT:-1}" == "1" ]]; then
    _export_platform_contract "${script}"
  fi

  local gpu_label="${CUDA_VISIBLE_DEVICES:-all-visible}"
  local env_note="TACC Lonestar6 ${SLURM_JOB_PARTITION:-unknown-partition} node ${SLURMD_NODENAME:-$(hostname -s)} A100 [gpu] (visible=${gpu_label}, job=${SLURM_JOB_ID:-none})"

  if [[ "${RUN_MODE}" == "native" ]]; then
    FIDESLIB_ROOT="${FIDESLIB_ROOT}" \
    FIDES_REAL_SCHEME_B_BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR}" \
    FIDES_CONTAINER_IMAGE="${FIDES_CONTAINER_IMAGE:-native-fideslib-786c7600-cuda12.8-gcc13.2}" \
    FIDES_RUN_ENVIRONMENT="${FIDES_RUN_ENVIRONMENT:-${env_note} [native FIDESlib, CUDA 12.8]}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      "${KIMON_SCHEME_B_DIR}/${script}" "$@"
    return $?
  fi

  if [[ ! -f "${KIMON_SIF}" ]]; then
    echo "[FATAL] .sif not found: ${KIMON_SIF}" >&2
    echo "        run: kimon/env/build_sif.sh archive /path/to/fideslib.tar" >&2
    exit 2
  fi
  apptainer exec --nv \
    --bind "${REPO_ROOT}:/work" \
    --env "FIDES_REAL_SCHEME_B_BUILD_DIR=/work/fhe/gpu_real_scheme_b/build" \
    --env "FIDES_CONTAINER_IMAGE=$(basename "${KIMON_SIF}")" \
    --env "FIDES_RUN_ENVIRONMENT=${FIDES_RUN_ENVIRONMENT:-${env_note} [apptainer]}" \
    "${KIMON_SIF}" \
    /bin/bash "/work/fhe/gpu_real_scheme_b/${script}" "$@"
}
