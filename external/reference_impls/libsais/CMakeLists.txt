add_library(
        reference_libsais
        libsais.c
        libsais64.c
)

target_include_directories(
        reference_libsais PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)