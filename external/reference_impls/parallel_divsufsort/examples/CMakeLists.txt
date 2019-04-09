add_executable(timing timing.cpp)
target_link_libraries (timing parallel_divsufsort lprange)
target_compile_options(timing PUBLIC
  ${SACA_BENCH_FLAGS}
  $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")
