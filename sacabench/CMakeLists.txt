################################################################################
# Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

# Create Header-only library containing all SACA implementations and utility
add_library(saca_bench_lib INTERFACE)

target_include_directories(saca_bench_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(saca_bench_lib INTERFACE
  $<$<CONFIG:Debug>:dl>
  tudocomp_stat
)

# Create executable that runs all registered SACAs
add_executable(sacabench
  saca/register.cpp
  sacabench.cpp
)

target_link_libraries(sacabench
  saca_bench_lib
  tudocomp_stat
  CLI11
)

target_compile_options(sacabench PUBLIC
  ${SACA_BENCH_FLAGS}
  $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
)

# Register all SACAs to generate an executable for each SACA
# TODO: If extra launchers are needed, use a single common
#       source file parametric over different different headers.
# set(SACA_BENCH_SACAS
#   saca/example1.cpp
#   saca/example2.cpp)
#
# foreach(SACA IN ITEMS ${SACA_BENCH_SACAS})
#   get_filename_component(NEW_NAME ${SACA} NAME_WE)
#   add_executable(${NEW_NAME}
#     ${SACA}
#   )
#   target_link_libraries(${NEW_NAME}
#     saca_bench_lib
#   )
#   target_compile_options(${NEW_NAME} PUBLIC
#     ${SACA_BENCH_FLAGS}
#     $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
#     $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
#   )
# endforeach(SACA)

################################################################################
