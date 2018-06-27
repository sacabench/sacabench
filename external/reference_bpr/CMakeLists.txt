include(ExternalProject)

ExternalProject_Add(reference_bpr2
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/implementation"
    CONFIGURE_COMMAND autoconf && ./configure
    BUILD_COMMAND "make"
)


