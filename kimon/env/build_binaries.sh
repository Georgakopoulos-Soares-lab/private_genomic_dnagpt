#!/usr/bin/env bash
# Compile only the binaries the kimon configs need (not all ~30 repo targets).
#
# Needs a GPU node: the CUDA targets are compiled for sm_80 and CMake probes the
# toolkit, so run this inside `idev -p gpu-a100` (or gpu-a100-small).
#
# Usage from the repo root:
#   RUN_MODE=native kimon/env/build_binaries.sh        # native FIDESlib (TACC)
#   kimon/env/build_binaries.sh                        # inside kimon/env/fideslib.sif
#
# Env knobs: BUILD_JOBS (default 8), KIMON_TARGETS (override the target list).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly REPO="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
readonly MODE="${RUN_MODE:-apptainer}"
readonly JOBS="${BUILD_JOBS:-8}"

# The six binaries the three configs execute, plus the optional combined
# sharding+diagcache reader used by config3's block_sharded variant.
DEFAULT_TARGETS=(
  real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536
  real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache
  real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123
  real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2
  real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3
  real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head
  real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache
  real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3
  real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13
  real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13
  real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13
)
read -r -a TARGETS <<<"${KIMON_TARGETS:-${DEFAULT_TARGETS[*]}}"

build_native() {
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/tacc_env.sh"
  if [[ ! -f "${FIDESLIB_PACKAGE_DIR}/fideslibConfig.cmake" ]] &&
     [[ ! -f "${FIDESLIB_PACKAGE_DIR}/fideslib-config.cmake" ]]; then
    echo "[FATAL] no installed FIDESlib CMake package at ${FIDESLIB_PACKAGE_DIR}" >&2
    echo "        run: kimon/env/clone_fideslib.sh && kimon/env/build_fideslib_native.sh" >&2
    exit 2
  fi
  # The DNAGPT CMakeLists takes both package dirs as cache vars, defaulting to
  # the docker image's /usr/local. Point them at the native install prefix
  # (share/fideslib/cmake and lib/OpenFHE under FIDESLIB_PREFIX).
  local openfhe_pkg="${OPENFHE_PACKAGE_DIR:-${FIDESLIB_PREFIX}/lib/OpenFHE}"
  if [[ ! -f "${openfhe_pkg}/OpenFHEConfig.cmake" ]]; then
    echo "[FATAL] OpenFHE CMake package not at ${openfhe_pkg}" >&2
    exit 2
  fi
  # Platform-tagged fixture-identity pins (TACC ls6). The 4 config binaries pin
  # the fixture manifest/contract sha256 (frozen = Mac-origin). On this node the
  # regenerated fixtures are numerically correct but those hashes differ, so we
  # inject the platform pins at compile time via bare-hex -D (stringified in the
  # source). Frozen defaults are untouched; disable with KIMON_USE_PLATFORM_CONTRACT=0.
  local cuda_extra=""
  if [[ "${KIMON_USE_PLATFORM_CONTRACT:-1}" == "1" ]]; then
    local envf="${SCRIPT_DIR}/fixtures_ls6/expected.env"
    if [[ -f "${envf}" ]]; then
      # shellcheck disable=SC1090
      source "${envf}"
      cuda_extra="-DKIMON_PIN_MANIFEST=${KIMON_LS6_MANIFEST_BLOCK}"
      cuda_extra+=" -DKIMON_PIN_CONTRACT=${KIMON_LS6_CONTRACT_T103}"
      cuda_extra+=" -DKIMON_PIN_AB_MANIFEST=${KIMON_LS6_MANIFEST_MULTIBLOCK}"
      cuda_extra+=" -DKIMON_PIN_AB_CONTRACT=${KIMON_LS6_CONTRACT_ALLBLOCKS}"
      # The cpudiagcache binary pins its PARENT anchor = the depth13 source, which
      # the platform build edits additively. Inject the actual (edited) depth13
      # sha so that binary's parent check accepts the in-tree source.
      local depth13_src="${REPO}/fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
      local depth13_sha; depth13_sha="$(sha256sum "${depth13_src}" | cut -d' ' -f1)"
      cuda_extra+=" -DKIMON_PIN_CPUDIAG_PARENT=${depth13_sha}"
      # The 12-block+head drivers each pin their per-block parent source as a
      # whole. The cpudiagcache per-block source is itself additively edited
      # by the platform build (same reason as depth13_sha above), so its
      # driver's pin drifts too unless overridden the same way.
      local cpudiagcache_src="${REPO}/fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"
      local cpudiagcache_sha; cpudiagcache_sha="$(sha256sum "${cpudiagcache_src}" | cut -d' ' -f1)"
      cuda_extra+=" -DKIMON_PIN_CPUDIAGCACHE_FULL=${cpudiagcache_sha}"
      # The T123_V3 12-block+head driver pins its per-block parent (the
      # T123_V3 single-block source) the same way; that source is itself
      # additively edited by the platform build (same reason as above), so
      # its driver's pin drifts too unless overridden the same way.
      local cpudiagcache_t123_v3_src="${REPO}/fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp"
      local cpudiagcache_t123_v3_sha; cpudiagcache_t123_v3_sha="$(sha256sum "${cpudiagcache_t123_v3_src}" | cut -d' ' -f1)"
      cuda_extra+=" -DKIMON_PIN_CPUDIAGCACHE_T123_V3_FULL=${cpudiagcache_t123_v3_sha}"
      echo "[platform-pins] injecting ls6 fixture-identity pins + cpudiag parent=${depth13_sha:0:12} + cpudiagcache full=${cpudiagcache_sha:0:12} + cpudiagcache t123_v3 full=${cpudiagcache_t123_v3_sha:0:12} into the CUDA build"
    else
      echo "[warn] ${envf} missing; building with frozen fixture pins (runs will reject regenerated fixtures)" >&2
    fi
  fi
  cmake -S "${REPO}/fhe/gpu_real_scheme_b" -B "${FIDES_REAL_SCHEME_B_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFIDESLIB_PACKAGE_DIR="${FIDESLIB_PACKAGE_DIR}" \
    -DOPENFHE_PACKAGE_DIR="${openfhe_pkg}" \
    -DFIDESLIB_ARCH="${FIDESLIB_ARCH}" \
    -DCMAKE_CUDA_ARCHITECTURES="${FIDESLIB_ARCH}" \
    -DCMAKE_CUDA_FLAGS="${cuda_extra}"
  for t in "${TARGETS[@]}"; do
    echo "=== building ${t} ==="
    cmake --build "${FIDES_REAL_SCHEME_B_BUILD_DIR}" --target "${t}" --parallel "${JOBS}"
  done
  echo "[ok] binaries in ${FIDES_REAL_SCHEME_B_BUILD_DIR}"
  ls -la "${FIDES_REAL_SCHEME_B_BUILD_DIR}" | grep real_dnagpt || true
}

build_apptainer() {
  local sif="${KIMON_SIF:-${SCRIPT_DIR}/fideslib.sif}"
  [[ -f "${sif}" ]] || { echo "[FATAL] .sif not found: ${sif}" >&2; exit 2; }
  apptainer exec --nv --bind "${REPO}:/work" "${sif}" /bin/bash -lc "
    set -euo pipefail
    cd /work/fhe/gpu_real_scheme_b
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DFIDESLIB_PACKAGE_DIR=/usr/local/share/fideslib/cmake \
      -DFIDESLIB_ARCH=80-real -DCMAKE_CUDA_ARCHITECTURES=80-real
    for t in ${TARGETS[*]}; do
      echo \"=== building \${t} ===\"
      cmake --build build --target \"\${t}\" --parallel ${JOBS}
    done
  "
}

case "${MODE}" in
  native) build_native ;;
  apptainer) build_apptainer ;;
  *) echo "[FATAL] RUN_MODE must be native or apptainer (got ${MODE})" >&2; exit 2 ;;
esac
