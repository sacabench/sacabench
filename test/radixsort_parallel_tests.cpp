/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/radixsort_parallel.hpp>
#include <util/container.hpp>
#include <random>

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

    // https://stackoverflow.com/a/32887614
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(100, 999);

    container<int> test_input(1000000);
    std::generate(test_input.begin(), test_input.end(), [&] { return distribution(generator); });

    container<int> test_output = make_container<int>(1000000);
    
    sort::radixsort_parallel(test_input, test_output);
    
    for (int index = 0; index < test_input.size() - 1; index++) {
        ASSERT_LE(test_output[index], test_output[index + 1]);
    }
}