cmake_minimum_required(VERSION 3.25.2)

project(dnagpt_fides_refresh LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
set(OPENFHE_PACKAGE_DIR "/usr/local/lib/OpenFHE"
    CACHE PATH "Installed pinned OpenFHE CMake package")
find_package(OpenFHE CONFIG REQUIRED
    PATHS "${OPENFHE_PACKAGE_DIR}"
    NO_DEFAULT_PATH)
set(FIDESLIB_PACKAGE_DIR "/usr/local/share/fideslib/cmake"
    CACHE PATH "Installed pinned FIDESlib CMake package")
find_package(fideslib CONFIG REQUIRED
    PATHS "${FIDESLIB_PACKAGE_DIR}"
    NO_DEFAULT_PATH)

add_executable(dnagpt_fides_refresh src/dnagpt_fides_refresh.cpp)
set_source_files_properties(src/dnagpt_fides_refresh.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(dnagpt_fides_refresh PRIVATE
    fideslib::fideslib
    OPENFHEpke
    OPENFHEcore
    OpenMP::OpenMP_CXX)
target_include_directories(dnagpt_fides_refresh PRIVATE
    "${OpenFHE_INCLUDE}"
    "${OpenFHE_INCLUDE}/core"
    "${OpenFHE_INCLUDE}/pke"
    "${OpenFHE_INCLUDE}/binfhe")
target_compile_options(dnagpt_fides_refresh PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(dnagpt_fides_refresh PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
