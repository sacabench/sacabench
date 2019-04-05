add_executable(timing timing.cpp)
target_link_libraries (timing cilkrts parallel_divsufsort lprange)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")
