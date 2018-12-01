# GTest.
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}
	"${CMAKE_CURRENT_BINARY_DIR}/../lib" // For parallel.h.
	"${CMAKE_CURRENT_SOURCE_DIR}/../lib" // For internal code in parallel-range.cpp.
	)

add_executable(runTests tests.cpp)
target_link_libraries (runTests libprange cilkrts ${GTEST_LIBRARIES} pthread)
