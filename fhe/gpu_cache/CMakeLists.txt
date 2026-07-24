cmake_minimum_required(VERSION 3.25.2)

project(dnagpt_fides_public_cache LANGUAGES CXX CUDA)

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

set(GENERATED_ROTATION_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/include/generated_rotation_contract.hpp")

add_executable(
    dnagpt_fides_cache
    src/dnagpt_fides_cache.cpp
    "${GENERATED_ROTATION_HEADER}")
set_source_files_properties(
    src/dnagpt_fides_cache.cpp
    PROPERTIES LANGUAGE CUDA)
target_include_directories(
    dnagpt_fides_cache
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
    dnagpt_fides_cache
    PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)
target_compile_options(
    dnagpt_fides_cache PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(
    dnagpt_fides_cache PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
