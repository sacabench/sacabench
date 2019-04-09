include_directories("${CMAKE_CURRENT_BINARY_DIR}/../include"
	"${CMAKE_CURRENT_BINARY_DIR}/../external/libprange/include"
	"${CMAKE_CURRENT_BINARY_DIR}"
)
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/parallel.hpp.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/parallel.hpp"
	)
file(GLOB libFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(parallel_divsufsort ${libFiles})
target_link_libraries(
    parallel_divsufsort
    lprange
)
target_compile_options(parallel_divsufsort PUBLIC
  ${SACA_BENCH_FLAGS}
  $<$<CONFIG:Debug>:${SACA_BENCH_DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${SACA_BENCH_RELEASE_FLAGS}>
)
install(TARGETS parallel_divsufsort
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
