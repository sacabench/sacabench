################################################################################
#Copyright(C) 2018 Marvin Böcker < marvin.boecker @tu - dortmund.de >
#Copyright(C) 2018 Marvin Löbel < loebel.marvin @gmail.com >
#All rights reserved.Published under the BSD - 3 license in the LICENSE file.
    ################################################################################

    add_executable(bigtest_testrunner)

        target_sources(bigtest_testrunner PUBLIC bigtest_driver.cpp)

            target_link_libraries(bigtest_testrunner saca_bench_lib gtest)

                target_compile_options(
                    bigtest_testrunner PUBLIC ${SACA_BENCH_FLAGS} $<
                        $<CONFIG : Debug> : ${SACA_BENCH_DEBUG_FLAGS}>
                        $<$<CONFIG : Release> : ${SACA_BENCH_RELEASE_FLAGS}>)

                    function(SACA_REGISTER header type) string(
                        REGEX REPLACE "[^a-zA-Z0-9\\.\\-]"
                                      "_" safe_type "${type}")

                        set(SACA_HEADER "${header}") set(SACA_TYPE "${type}")
                            configure_file(bigtest.cpp.in
                                           "${CMAKE_CURRENT_BINARY_DIR}/"
                                           "bigtest_${safe_type}.cpp")

                                target_sources(bigtest_testrunner PUBLIC
                                               "${CMAKE_CURRENT_BINARY_DIR}/"
                                               "bigtest_${safe_type}.cpp")
                                    endfunction(SACA_REGISTER)

                                        add_custom_target(
                                            bigtest COMMAND bigtest_testrunner
                                                DEPENDS bigtest_testrunner)

                                            include(../ sacabench / saca /
                                                    register.cmake)
