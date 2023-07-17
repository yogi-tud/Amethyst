# convenient cuda clang support was added in 3.19
cmake_minimum_required(VERSION 3.21)
set(CLANG_VERSION "12" CACHE STRING "specify clang version")
option(USE_NVCC "use nvcc instead of clang" ON)
option(CLANG_LIBCPP "work around clang ICE on libstdc++ by using libc++ instead" OFF)
option(CUB_SUBMODULE "use cub submodule instead of find package" OFF)


if(NOT USE_NVCC)
    if("${CLANG_VERSION}" STREQUAL "")
        set(CLANG_VERSION_POSTFIX "")
    else()
        set(CLANG_VERSION_POSTFIX "-${CLANG_VERSION}")
    endif()

    set(CMAKE_CUDA_COMPILER "clang++${CLANG_VERSION_POSTFIX}")
    set(CMAKE_CXX_COMPILER "clang++${CLANG_VERSION_POSTFIX}")

    # force c++20
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # disable unknown cuda version warning
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Wall -Wextra -Wno-unused-parameter -Wfatal-errors")
endif()

# don't enable gnu extensions
set(CMAKE_CXX_EXTENSIONS OFF)

project(gpu_elemenstuffing LANGUAGES CXX CUDA)

if(CUB_SUBMODULE)
    list(APPEND INCLUDES "deps/cub")
else()
    find_package(CUB)
endif()

find_package(CUDAToolkit)

set(INCLUDES
    src
    deps/cub
)

set(SOURCES
    src/benchs/bench_binary_op_baseline.cu
    src/benchs/bench_binary_op_nogaps.cu
    src/benchs/bench_binary_op_slabs.cu
    src/benchs/bench_compressstore_slabs.cu
    src/benchs/bench_compressstore_baseline_cub.cu
    src/benchs/bench_filter_baseline.cu
    src/benchs/bench_filter_nogaps.cu
    src/benchs/bench_filter_slabs.cu
    src/benchs/bench_groupby_sm_array.cu
    src/benchs/bench_groupby_sort_cub.cu
    src/benchs/bench_hop_baseline_cub.cu
    src/benchs/bench_hop_nogaps.cu
    src/benchs/bench_hop_slabs.cu
    src/main.cu
)

add_executable(gpu_elementstuffing ${SOURCES})

set_property(TARGET gpu_elementstuffing PROPERTY CUDA_ARCHITECTURES 75 61)

target_include_directories(gpu_elementstuffing PRIVATE ${INCLUDES} ${CUB_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(gpu_elementstuffing PRIVATE CUDA::cublasLt)

#TODO: proper solution
target_compile_definitions(gpu_elementstuffing PUBLIC THRUST_IGNORE_CUB_VERSION_CHECK)

set_target_properties(gpu_elementstuffing PROPERTIES EXPORT_COMPILE_COMMANDS true)
