################################################################################
# Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
################################################################################

include(test/testsuite.cmake)

run_test(example_tests DEPS saca_bench_lib)
run_test(test_uint40 DEPS saca_bench_lib)
run_test(span_tests DEPS saca_bench_lib)
run_test(container_tests DEPS saca_bench_lib)
run_test(bucket_size_tests DEPS saca_bench_lib)
run_test(alphabet_tests DEPS saca_bench_lib)
run_test(isa2sa_tests DEPS saca_bench_lib)
run_test(bucketsort_tests DEPS saca_bench_lib)
run_test(read_text_tests DEPS saca_bench_lib)
run_test(test_ternary_quicksort DEPS saca_bench_lib)
run_test(sa_check_tests DEPS saca_bench_lib)
run_test(test_multikey_quicksort DEPS saca_bench_lib)
run_test(test_is_sorted DEPS saca_bench_lib)
run_test(compare_tests DEPS saca_bench_lib)
run_test(insertsort_tests DEPS saca_bench_lib)
run_test(heapsort_tests DEPS saca_bench_lib)
run_test(introsort_tests DEPS saca_bench_lib)
run_test(merge_sa_dc_tests DEPS saca_bench_lib)
run_test(radixsort_tests DEPS saca_bench_lib)
run_test(sa_induce_dc_tests DEPS saca_bench_lib)
run_test(bucket_pointer_refinement_tests DEPS saca_bench_lib)
run_test(test_signed_size_type DEPS saca_bench_lib)
run_test(bits_tests DEPS saca_bench_lib)
run_test(gsaca_tests DEPS saca_bench_lib)
run_test(test_saca_deep_shallow DEPS saca_bench_lib)
run_test(prefix_doubling_tests DEPS saca_bench_lib)
run_test(ringbuffer DEPS saca_bench_lib)
