################################################################################
# Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

include(test/testsuite.cmake)

run_test(example_tests DEPS saca_bench_lib)
run_test(test_uint40)
run_test(span_tests DEPS saca_bench_lib)
run_test(container_tests DEPS saca_bench_lib)
run_test(alphabet_tests DEPS saca_bench_lib)
run_test(bucketsort_tests DEPS saca_bench_lib)
run_test(read_text_tests DEPS saca_bench_lib)
run_test(test_ternary_quicksort DEPS saca_bench_lib)
run_test(sa_check_tests DEPS saca_bench_lib)
run_test(test_multikey_quicksort DEPS saca_bench_lib)
run_test(test_is_sorted DEPS saca_bench_lib)
run_test(compare_tests DEPS saca_bench_lib)
