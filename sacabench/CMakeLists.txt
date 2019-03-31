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
  reference_gsaca
  reference_saca_deep_shallow
  divsufsort
  divsufsort64
  divsufsort_par
  divsufsort_par64
  parallel_divsufsort
  reference_bpr2_SuffixArray
  reference_bpr2_Util
  reference_bpr2_Standard
  msufsort
  reference_dc3
  reference_saca_k
  ${CMAKE_DL_LIBS}
  atomic
)

if(SACA_ENABLE_CUDA)
target_link_libraries(saca_bench_lib INTERFACE
  cuda_wrapper
)
endif()

# Create executable that runs all registered SACAs
add_executable(sacabench
  sacabench.cpp
)
target_link_libraries(sacabench
  saca_bench_lib
  CLI11
  atomic
  OpenMP::OpenMP_CXX
)
target_compile_options(sacabench PUBLIC
  ${SACA_BENCH_FLAGS}
  $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
)
configure_file(util/source_dir_constant.hpp.in
               source_dir_constant.hpp)
target_include_directories(sacabench PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# A debugging tool (legacy)
add_executable(sacabench-tools
  sacabench-tools.cpp
)
target_link_libraries(sacabench-tools
  CLI11
  "${OpenMP_CXX_FLAGS}"
  atomic
)
target_compile_options(sacabench-tools PUBLIC

  ${SACA_BENCH_FLAGS}
  $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
)

function(SACA_REGISTER header type)
    string(REGEX REPLACE
        "[^a-zA-Z0-9\\.\\-]"
        "_"
        safe_type
        "${type}")

    set(SACA_HEADER "${header}")
    set(SACA_TYPE "${type}")
    configure_file(saca/register.cpp.in "${CMAKE_CURRENT_BINARY_DIR}/${safe_type}.cpp")

    target_sources(sacabench PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/${safe_type}.cpp"
    )
endfunction(SACA_REGISTER)

include(saca/register.cmake)

################################################################################
