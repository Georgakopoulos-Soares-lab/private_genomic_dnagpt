#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_COMMIT=786c7600fb2f16b724e0acf73df367b27b8afed6
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FIDES_ROOT="${FIDESLIB_ROOT:-/opt/FIDESlib}"
readonly BUILD_DIR="${FIDES_REAL_SCHEME_B_BUILD_DIR:-${SCRIPT_DIR}/build}"
readonly JOBS="${BUILD_JOBS:-2}"

if [[ ! -d "${FIDES_ROOT}/.git" ]]; then
  echo "[FATAL] FIDESlib git checkout not found at ${FIDES_ROOT}" >&2
  exit 2
fi

actual_commit="$(git -C "${FIDES_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "[FATAL] FIDESlib commit ${actual_commit}; expected ${EXPECTED_COMMIT}" >&2
  exit 2
fi

cmake \
  -S "${SCRIPT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFIDESLIB_PACKAGE_DIR="${FIDESLIB_PACKAGE_DIR:-/usr/local/share/fideslib/cmake}" \
  -DFIDESLIB_ARCH="${FIDESLIB_ARCH:-80-real}" \
  -DCMAKE_CUDA_ARCHITECTURES="${FIDESLIB_ARCH:-80-real}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_cached --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_serialize_writer --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_serialize_reader --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_profiled --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_diagcache --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_warmup --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_lintransform --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_general_attention --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_general_attention_t8 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_general_attention_t32 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_general_attention_t103 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_linear_t103 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth8 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_two_block_refresh --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_two_block_refresh_depth12 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_all_blocks_head_t2 --parallel "${JOBS}"
cmake --build "${BUILD_DIR}" --target real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head --parallel "${JOBS}"
# The process-separated MLP reader/merge targets are retained as negative
# implementation evidence but intentionally excluded here: the pinned FIDESlib
# exposes no Ciphertext serialization API, so those targets are known not to build.

echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_cached"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_serialize_writer"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_serialize_reader"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_profiled"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_diagcache"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_warmup"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_lintransform"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_general_attention"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_general_attention_t8"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_general_attention_t32"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_general_attention_t103"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_linear_t103"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth8"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_two_block_refresh"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_two_block_refresh_depth12"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_all_blocks_head_t2"
echo "[build] ${BUILD_DIR}/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head"
