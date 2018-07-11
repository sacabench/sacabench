add_library(
        msufsort
        InductionSort.cpp
        MSufSort.cpp
)

target_include_directories(
        msufsort PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

set(MB_MSUFSORT_GLOBAL_FLAGS "-w;-O3;-march=native;-mtune=native")

target_compile_options(
    msufsort PRIVATE
    ${MB_MSUFSORT_GLOBAL_FLAGS}
)
