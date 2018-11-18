/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/radixsort_parallel.hpp>
#include <util/container.hpp>

using namespace sacabench::util;

TEST(radixsort_parallel_tests, no_conflicting_positions) {
    container<int> test_input = {456, 123, 912, 691};
    container<int> test_output = make_container<int>(4);
    container<int> test_control = {123, 456, 691, 912};
    sort::radixsort_parallel(test_input, test_output);
    ASSERT_EQ(test_output, test_control);
}

TEST(radixsort_parallel_tests, conflicting_positions) {
    container<int> test_input = {123, 223, 912, 691};
    container<int> test_output = make_container<int>(4);
    container<int> test_control = {123, 223, 691, 912};
    sort::radixsort_parallel(test_input, test_output);
    ASSERT_EQ(test_output, test_control);
}

TEST(radixsort_parallel_tests, element_number_regardless_of_thred_count) {
    container<int> test_input = {123, 223, 912, 691, 735};
    container<int> test_output = make_container<int>(5);
    container<int> test_control = {123, 223, 691, 735, 912};
    sort::radixsort_parallel(test_input, test_output);
    ASSERT_EQ(test_output, test_control);
}

TEST(radixsort_parallel_tests, big_random_test) {
    container<int> test_input = {123, 223, 912, 691, 735};
    container<int> test_output = make_container<int>(5);
    container<int> test_control = {123, 223, 691, 735, 912};
    sort::radixsort_parallel(test_input, test_output);
    ASSERT_EQ(test_output, test_control);
}