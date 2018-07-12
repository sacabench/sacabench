add_library(
        reference_saca_k INTERFACE
)

target_include_directories(
        reference_dc3 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
