include_directories(
        "${CMAKE_CURRENT_BINARY_DIR}"
        )
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/parallel.hpp.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/parallel.hpp"
	)
add_library(lprange parallel-range.cpp)
target_compile_options(lprange PUBLIC
  ${SACA_BENCH_FLAGS}
  $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
)
install(TARGETS lprange
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
