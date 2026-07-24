cmake_minimum_required(VERSION 3.25.2)
project(dnagpt_fides_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
find_package(
    fideslib CONFIG REQUIRED
    PATHS /usr/local/share/fideslib/cmake
    NO_DEFAULT_PATH)
add_executable(dnagpt_fides_probe asymmetric_cheb.cpp)
target_link_libraries(
    dnagpt_fides_probe
    PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)
