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
