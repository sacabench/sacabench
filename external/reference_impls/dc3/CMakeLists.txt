add_library(
        reference_dc3
        drittel.C
)

target_include_directories(
        reference_dc3 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

set(MB_DC3_GLOBAL_FLAGS "-w)

target_compile_options(
    reference_dc3 PRIVATE
    ${SACA_BENCH_FLAGS}
    ${MB_DC3_GLOBAL_FLAGS}
)
