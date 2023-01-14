cmake_minimum_required(VERSION 3.11)

project(SimpleFFTW)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic)

enable_testing()

add_subdirectory(lib)
add_subdirectory(test)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/mylibConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
)
