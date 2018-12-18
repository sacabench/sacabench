add_executable(PRangeLiteDemo main.cpp)
target_link_libraries (PRangeLiteDemo libprange cilkrts)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../include")
