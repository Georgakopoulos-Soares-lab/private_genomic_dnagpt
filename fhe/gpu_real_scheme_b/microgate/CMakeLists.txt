cmake_minimum_required(VERSION 3.25.2)
project(clone_microtest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
set(FIDESLIB_PACKAGE_DIR "/usr/local/share/fideslib/cmake" CACHE PATH "")
find_package(fideslib CONFIG REQUIRED PATHS "${FIDESLIB_PACKAGE_DIR}" NO_DEFAULT_PATH)
set(OPENFHE_PACKAGE_DIR "/usr/local/lib/OpenFHE" CACHE PATH "")
find_package(OpenFHE CONFIG REQUIRED PATHS "${OPENFHE_PACKAGE_DIR}" NO_DEFAULT_PATH)

add_executable(clone_microtest clone_microtest.cpp)
set_source_files_properties(clone_microtest.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(clone_microtest PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)
# TEST_I (v3) includes <openfhe.h> directly (to call OpenFHE's own
# ClearEvalMultKeys/ClearEvalAutomorphismKeys) -- fideslib::fideslib's
# exported INTERFACE_INCLUDE_DIRECTORIES only covers FIDESlib's own headers,
# not OpenFHE's (same issue already solved for
# real_dnagpt_fides_scheme_b_lintransform in the parent CMakeLists.txt).
# Link-time symbols are already satisfied transitively via fideslib::fideslib.
target_include_directories(clone_microtest PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(clone_microtest PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
set_target_properties(clone_microtest PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
