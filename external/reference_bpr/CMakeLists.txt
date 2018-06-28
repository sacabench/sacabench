include(ExternalProject)

ExternalProject_Add(reference_bpr2_external_project
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/implementation"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/implementation"
    # ------------------
    CONFIGURE_COMMAND cmake -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/implementation" "${CMAKE_CURRENT_BINARY_DIR}/implementation"
    COMMAND autoconf
    COMMAND ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/install
    # ------------------
    BUILD_COMMAND make
    # ------------------
    INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/install"
    # ------------------
    EXCLUDE_FROM_ALL 1
)
ExternalProject_Get_Property(reference_bpr2_external_project binary_dir)

function(reference_bpr2_lib lib_name path_name inner_libname)
    set(LIBRARY
        "${binary_dir}/src/${path_name}/.libs/lib${inner_libname}.a"
    )
    set(INCLUDE
        "${binary_dir}/src/${path_name}/"
    )
    file(MAKE_DIRECTORY "${INCLUDE}")

    add_library(${lib_name} IMPORTED STATIC GLOBAL)
    add_dependencies(${lib_name} reference_bpr2_external_project)
    set_target_properties(${lib_name} PROPERTIES
        "IMPORTED_LOCATION" "${LIBRARY}"
        #"IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
        "INTERFACE_INCLUDE_DIRECTORIES" "${INCLUDE}"
    )
endfunction()

reference_bpr2_lib(reference_bpr2_SuffixArray SuffixArray suffixarray)
reference_bpr2_lib(reference_bpr2_Util Util util)
reference_bpr2_lib(reference_bpr2_Standard Standard standard)
