cmake_minimum_required(VERSION 3.25.2)

project(dnagpt_fides_toy LANGUAGES CXX CUDA)

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

add_executable(toy_dnagpt_fides src/toy_dnagpt_fides.cpp)
set_source_files_properties(src/toy_dnagpt_fides.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(toy_dnagpt_fides PRIVATE
    fideslib::fideslib
    OpenMP::OpenMP_CXX)
target_include_directories(toy_dnagpt_fides PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(toy_dnagpt_fides PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
set_target_properties(toy_dnagpt_fides PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
