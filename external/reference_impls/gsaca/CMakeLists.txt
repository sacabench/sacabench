add_library(
        reference_gsaca
        gsaca.c
)

target_include_directories(
        reference_gsaca PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)