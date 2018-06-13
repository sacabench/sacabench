################################################################################
# Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed. "
    "Create a sub-directory and call cmake ..")
endif()

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(sacabench)

# Compile all code with ccache, if it exists in the system
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

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

set(SACA_BENCH_FLAGS "-Wall;-pedantic;-Wextra;-Werror=return-type")
set(SACA_BENCH_DEBUG_FLAGS "-O0;-ggdb;-DDEBUG")
set(SACA_BENCH_RELEASE_FLAGS "-O3;-march=native;-DNDEBUG")

# Add custom cmake extensions
include(cmakemodules/git_submodule_subdirectories.cmake)

set(BUILD_GMOCK OFF CACHE BOOL "")
set(BUILD_GTEST ON CACHE BOOL "")
git_submodule_subdirectory(external/gtest)
git_submodule_subdirectory(external/tudostats)
git_submodule_subdirectory(external/CLI11)
add_subdirectory(sacabench)
add_subdirectory(test)

################################################################################
