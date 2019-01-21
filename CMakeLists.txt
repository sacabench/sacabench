################################################################################
# Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed. "
    "Please remove ${CMAKE_BINARY_DIR}/CMakeCache.txt and the ${CMAKE_BINARY_DIR}/CMakeFiles/ directory. "
    "Then create a sub-directory and call `cmake ..`")
endif()

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(sacabench LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(SACA_ENABLE_CUDA ON)
endif()

if(SACA_ENABLE_CUDA)
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support, GPU algorithms will not be enabled")
endif()

# Compile all code with ccache, if it exists in the system
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

find_package(OpenMP REQUIRED)
if(APPLE)
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set(OpenMP_C_FLAG "-fopenmp=libomp -Wno-unused-command-line-argument")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(OpenMP_CXX_FLAG "-fopenmp=libomp -Wno-unused-command-line-argument")
    endif()
endif()

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Not so nice C++17 integration, as CMake version >= 3.8 is not available here
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

# Default build configuration to a "Release" build.
# The configuration affects things like compiler optimizations
# and debugging symbols in the binary.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif(NOT CMAKE_BUILD_TYPE)

message("Build type: ${CMAKE_BUILD_TYPE}")

option(SACA_WERROR "Turn GCC warnings into errors")

IF (WIN32)

set(SACA_BENCH_FLAGS "/std:c++17")
set(SACA_BENCH_DEBUG_FLAGS "")
set(SACA_BENCH_RELEASE_FLAGS "")

ELSE()
set(SACA_BENCH_FLAGS "-Wall;-pedantic;-Wpedantic;-Wextra;-Werror=return-type;-fdiagnostics-color=auto;-fopenmp")
if (SACA_WERROR)
    set(SACA_BENCH_FLAGS "${SACA_BENCH_FLAGS};-Werror;-pedantic-errors")
endif()
set(SACA_BENCH_DEBUG_FLAGS "-O0;-g;-ggdb;-DDEBUG;")
set(SACA_BENCH_RELEASE_FLAGS "-O3;-march=native;-DNDEBUG;-g;")

ENDIF()

# Enable OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add custom cmake extensions
include(cmakemodules/git_submodule_subdirectories.cmake)

set(BUILD_GMOCK OFF CACHE BOOL "")
set(BUILD_GTEST ON CACHE BOOL "")
git_submodule_subdirectory(external/gtest)
git_submodule_subdirectory(external/tudostats)
git_submodule_subdirectory(external/CLI11)
set(DATASET_ARGS "--delete_original;--min_prefix_size=200;--max_prefix_size=200" CACHE STRING "" FORCE)
git_submodule_subdirectory(external/datasets)

if(SACA_ENABLE_CUDA)
git_submodule(external/cub)
endif()

add_subdirectory(sacabench)
add_subdirectory(external/ips4o)

if(SACA_ENABLE_CUDA)
add_subdirectory(cuda)
endif()

add_subdirectory(external/reference_impls/gsaca)
add_subdirectory(external/reference_impls/deep-shallow)
add_subdirectory(external/reference_impls/divsufsort)
add_subdirectory(external/reference_impls/bpr)
add_subdirectory(external/reference_impls/dc3)
add_subdirectory(external/reference_impls/msufsort)
add_subdirectory(external/reference_impls/saca_k)
add_subdirectory(external/reference_impls/parallel_divsufsort)

if(SACA_ENABLE_CUDA)
set(CMAKE_USE_PTHREADS_INIT OFF CACHE BOOL "")
#set(gtest_disable_pthreads ON CACHE BOOL "")
endif()

add_subdirectory(test)
add_subdirectory(bigtest)

################################################################################
