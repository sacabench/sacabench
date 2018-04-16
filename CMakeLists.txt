################################################################################
# CMakeLists.txt
#
# Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
#
# All rights reserved. Published under the TBD license in the LICENSE file.
################################################################################

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(dsss)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Not so noce C++17 integration, as CMake version >= 3.8 is not available here
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

add_subdirectory(sacabench)

################################################################################
