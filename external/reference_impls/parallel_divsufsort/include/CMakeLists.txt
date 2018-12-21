configure_file("${CMAKE_CURRENT_SOURCE_DIR}/parallel_divsufsort.h" "${CMAKE_CURRENT_BINARY_DIR}/parallel_divsufsort.h" COPYONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/parallel_divsufsort.h" DESTINATION include)
