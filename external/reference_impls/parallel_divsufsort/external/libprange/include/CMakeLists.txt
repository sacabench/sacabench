configure_file("${CMAKE_CURRENT_SOURCE_DIR}/parallel-range.h" "${CMAKE_CURRENT_BINARY_DIR}/parallel-range.h" COPYONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/parallel-range.h" DESTINATION include)
