################################################################################
# Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
# Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

add_custom_target(bigtest)
add_custom_command(
    TARGET bigtest
    POST_BUILD
    COMMENT "All tests were successful!" VERBATIM
)

function(SACA_REGISTER header type)
    string(REGEX REPLACE
        "[^a-zA-Z0-9\\.\\-]"
        "_"
        safe_type
        "${type}")

    set(SACA_HEADER "${header}")
    set(SACA_TYPE "${type}")
    configure_file(register.cpp.in "${CMAKE_CURRENT_BINARY_DIR}/${safe_type}.cpp")

    target_sources(sacabench PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/${safe_type}.cpp"
    )
endfunction(SACA_REGISTER)

include(../sacabench/saca/register.cmake)
