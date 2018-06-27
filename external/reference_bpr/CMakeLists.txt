include(ExternalProject)

ExternalProject_Add(reference_bpr2
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/implementation"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/implementation"
    CONFIGURE_COMMAND cmake -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/implementation" "${CMAKE_CURRENT_BINARY_DIR}/implementation"
    COMMAND autoconf
    COMMAND ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/install
    BUILD_COMMAND make
    INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install"
)
