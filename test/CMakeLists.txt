################################################################################
# test/CMakeLists.txt
#
# Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

include(test/testsuite.cmake)

run_test(merge_sa_dc_tests DEPS saca_bench_lib)
