cmake_minimum_required(VERSION 3.25.2)

project(dnagpt_fides_real_scheme_b_block LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
set(FIDESLIB_PACKAGE_DIR "/usr/local/share/fideslib/cmake"
    CACHE PATH "Installed pinned FIDESlib CMake package")
find_package(fideslib CONFIG REQUIRED
    PATHS "${FIDESLIB_PACKAGE_DIR}"
    NO_DEFAULT_PATH)

# real_dnagpt_fides_scheme_b_lintransform.cpp is the only source in this tree
# that includes FIDESlib's native CKKS/Ciphertext.cuh and CKKS/Plaintext.cuh
# directly (to call FIDESlib::CKKS::LinearTransform -- see
# docs/hybrid/tasks.md's 2026-07-28 scoping section). Those headers pull in
# openfhe.h, but fideslib::fideslib's exported INTERFACE_INCLUDE_DIRECTORIES
# only covers FIDESlib's own installed headers, not OpenFHE's -- confirmed by
# reading fideslibTargets.cmake and examples/bert-tiny/CMakeLists.txt (which
# adds OpenFHE's include dirs itself for the same reason). Every other
# .cpp in this tree only uses the OpenFHE-API-compatible <fideslib.hpp>
# wrapper and does not need this.
set(OPENFHE_PACKAGE_DIR "/usr/local/lib/OpenFHE"
    CACHE PATH "Installed pinned OpenFHE CMake package")
find_package(OpenFHE CONFIG REQUIRED
    PATHS "${OPENFHE_PACKAGE_DIR}"
    NO_DEFAULT_PATH)

add_executable(real_dnagpt_fides_scheme_b src/real_dnagpt_fides_scheme_b.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_cached src/real_dnagpt_fides_scheme_b_cached.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_cached.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_cached PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_cached PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_cached PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_serialize_writer src/real_dnagpt_fides_scheme_b_serialize_writer.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_serialize_writer.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_serialize_writer PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_serialize_writer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_serialize_writer PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_serialize_reader src/real_dnagpt_fides_scheme_b_serialize_reader.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_serialize_reader.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_serialize_reader PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_serialize_reader PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_serialize_reader PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13
    src/real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13
    src/real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Correctness-only micro-gate combining the two independently-passing
# levers: 2-GPU process-per-GPU Q/K/V sharding (Stage 1) PLUS the CPU-side
# diagonal-vector cache. Forks the shard reader, portin the cpudiagcache
# lever's matmul()/cached_packed_values() verbatim. Additive; touches no
# other target in this file.
add_executable(
    real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13
    src/real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_profiled src/real_dnagpt_fides_scheme_b_profiled.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_profiled.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_profiled PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_profiled PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_profiled PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_diagcache src/real_dnagpt_fides_scheme_b_diagcache.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_diagcache.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_diagcache PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_diagcache PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_diagcache PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_warmup src/real_dnagpt_fides_scheme_b_warmup.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_warmup.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_warmup PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_warmup PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_warmup PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_lintransform src/real_dnagpt_fides_scheme_b_lintransform.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_lintransform.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_lintransform PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_include_directories(real_dnagpt_fides_scheme_b_lintransform PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(real_dnagpt_fides_scheme_b_lintransform PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_lintransform PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_general_attention src/real_dnagpt_fides_scheme_b_general_attention.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_general_attention.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_general_attention PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_general_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_general_attention PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_general_attention_t8 src/real_dnagpt_fides_scheme_b_general_attention_t8.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_general_attention_t8.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_general_attention_t8 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_general_attention_t8 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_general_attention_t8 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_general_attention_t32 src/real_dnagpt_fides_scheme_b_general_attention_t32.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_general_attention_t32.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_general_attention_t32 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_general_attention_t32 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_general_attention_t32 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_general_attention_t103 src/real_dnagpt_fides_scheme_b_general_attention_t103.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_general_attention_t103.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_general_attention_t103 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_general_attention_t103 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_general_attention_t103 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_simd_linear_t103
    src/real_dnagpt_fides_scheme_b_simd_linear_t103.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_linear_t103.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_simd_linear_t103 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_simd_linear_t103 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_simd_linear_t103 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_simd_full_t103
    src/real_dnagpt_fides_scheme_b_simd_full_t103.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_simd_full_t103 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_simd_full_t103 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_simd_full_t103 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_simd_full_t103_depth8
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_simd_full_t103_depth8 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_simd_full_t103_depth8 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_two_block_refresh src/real_dnagpt_fides_scheme_b_two_block_refresh.cpp)
set_source_files_properties(src/real_dnagpt_fides_scheme_b_two_block_refresh.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_two_block_refresh PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_two_block_refresh PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_two_block_refresh PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_two_block_refresh_depth12
    src/real_dnagpt_fides_scheme_b_two_block_refresh_depth12.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_two_block_refresh_depth12.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_two_block_refresh_depth12 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_two_block_refresh_depth12 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_two_block_refresh_depth12 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(real_dnagpt_fides_scheme_b_all_blocks_head_t2
    src/real_dnagpt_fides_scheme_b_all_blocks_head_t2.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_all_blocks_head_t2.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(real_dnagpt_fides_scheme_b_all_blocks_head_t2 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(real_dnagpt_fides_scheme_b_all_blocks_head_t2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(real_dnagpt_fides_scheme_b_all_blocks_head_t2 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# T123: the all-opts block with Tier 1 (encode-once encoded-Plaintext templates
# + per-use clone), Tier 2/3 (multi-core OMP parallel batched encode). Additive
# fork of the cpudiagcache source; same fixture, oracle, and pins.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-fopenmp>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# T123 v2: OMP-parallel client boundary crossings (the 91 score-tile decrypts
# + 727 weight-tile encrypts, previously fully serial and GPU-idle during
# them). Same crypto-op schedule/oracle as t123; additive fork, t123 stays
# untouched as an immutable historical binary.
# ABANDONED (kimon/logs/optimization_campaign/PLAN.md): the micro-gate found
# an intermittent race in concurrent Decrypt/Encrypt on FIDESlib's shared
# CryptoContext (two different crash signatures across runs of unchanged
# test code). Kept as a documented negative result; do not run as a full
# driver or present as a validated optimization. v3 below forks from the
# clean t123 source instead.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-fopenmp>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# T123 v3: free the CPU-side (OpenFHE) EvalMult/rotation key maps right after
# LoadContext -- source-verified dead weight afterward (LoadContext is the
# only reader in FIDESlib's non-vendored code), validated by the micro-gate's
# TEST_I on real GPU hardware. Additive fork of the clean t123 source (NOT
# t123_v2, which is abandoned); needs OpenFHE's raw headers directly (see the
# source comment), which fideslib::fideslib does not propagate transitively
# -- same already-solved issue as real_dnagpt_fides_scheme_b_lintransform
# above.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_include_directories(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3 PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-fopenmp>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# THROWAWAY v4 feasibility diagnostic -- NOT an evidence-bearing binary, not
# added to build_binaries.sh, to be deleted (source + this CMake block) once
# the per-weight level data is captured. Prints each weight's actual encode
# level to stderr; otherwise identical to t123_v3.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_include_directories(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v4_diag
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Complete 12-block backbone + GSR classifier head, Scheme B Token-SIMD B=8,
# T=103 (full prompt). Composes the depth-13/digits-3/ring-65536 per-block
# source (unchanged, included) 12x with an adapted Token-SIMD inter-block
# refresh (design pinned from real_dnagpt_fides_scheme_b_two_block_refresh.cpp)
# and a new classifier-head evaluator. Additive; does not touch any other
# target in this file.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head
    src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Config-3 "all-optimizations" END-TO-END: the same complete 12-block backbone
# + GSR head as the target above, but composing the CPU-side diagonal-vector
# cache per-block source (..._cpudiagcache.cpp, included) instead of the plain
# depth-13 block. Composition logic is byte-for-byte the config-1 e2e driver's;
# only the #included per-block source differs. Additive; touches no other target.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache
    src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Config-3 "all-optimizations" END-TO-END, T123_V3 per-block lever: the same
# complete 12-block backbone + GSR head as the target above, but composing
# the T123_V3 per-block source (..._cpudiagcache_t123_v3.cpp, included) --
# encode-once encoded-Plaintext template cache (Tier1) + OMP-parallel batched
# encode (Tier2/3) + post-LoadContext free of the CPU-side OpenFHE
# EvalMult/rotation key maps (v3) -- instead of the plain cpudiagcache
# per-block source. Composition logic is byte-for-byte the config-3
# cpudiagcache e2e driver's, plus one addition mirrored from T123_V3's own
# main(): this driver's own context setup also frees the CPU-side key maps
# once, right after its own LoadContext. Additive; touches no other target.
# Needs the raw OpenFHE headers directly (same already-solved reason as the
# standalone T123_V3 single-block target above): fideslib::fideslib does not
# propagate OpenFHE's include dirs transitively.
add_executable(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3
    src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_include_directories(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3 PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-fopenmp>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# Historical negative prototype for process-separated MLP down-projection
# sharding. The pinned FIDESlib has no Ciphertext serialization API, so these
# targets are intentionally excluded from default builds. They remain
# explicitly addressable to reproduce the compile-time backend limitation.
add_executable(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13
    EXCLUDE_FROM_ALL
    src/real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13
    EXCLUDE_FROM_ALL
    src/real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13.cpp)
set_source_files_properties(
    src/real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13.cpp
    PROPERTIES LANGUAGE CUDA)
target_link_libraries(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13 PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_compile_options(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
