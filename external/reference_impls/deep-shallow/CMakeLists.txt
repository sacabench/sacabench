cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_library(
    reference_saca_deep_shallow
    globals.c
    ds.c
    shallow.c
    deep2.c
    helped.c
    blind2.c
)

target_include_directories(
    reference_saca_deep_shallow PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

set(MB_DS_GLOBAL_FLAGS "-w;-g;-O3;-fomit-frame-pointer;-W;-Wall;-Winline;-DNDEBUG")

target_compile_options(
    reference_saca_deep_shallow PRIVATE
    ${MB_DS_GLOBAL_FLAGS}
)
