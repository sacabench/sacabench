/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/radixsort_parallel.hpp>
#include <util/container.hpp>
#include <random>
#include <tuple>

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
    
    for (size_t index = 0; index < test_input.size() - 1; index++) {
        ASSERT_LE(test_output[index], test_output[index + 1]);
    }
}

TEST(radixsort_parallel_tests, triple_sorting_integer) {
    std::tuple<int, int, int> tuple_1(1, 2, 3);
    std::tuple<int, int, int> tuple_2(2, 2, 3);
    std::tuple<int, int, int> tuple_3(9, 1, 2);
    std::tuple<int, int, int> tuple_4(6, 9, 1);
    std::tuple<int, int, int> tuple_5(7, 3, 5);
    std::vector<std::tuple<int, int, int>> test_input = {tuple_1, tuple_2, tuple_3, tuple_4, tuple_5};

    std::vector<std::tuple<int, int, int>> test_output = std::vector<std::tuple<int, int, int>>(5);

    std::vector<std::tuple<int, int, int>> test_control = {tuple_1, tuple_2, tuple_4, tuple_5, tuple_3};

    sort::radixsort_parallel(test_input, test_output);
    ASSERT_EQ(test_output, test_control);
}

TEST(radixsort_parallel_tests, triple_sorting_chars) {
    std::tuple<char, char, char> tuple_1('a', 'b', 'c');
    std::tuple<char, char, char> tuple_2('b', 'b', 'x');
    std::tuple<char, char, char> tuple_3('z', 'a', 'd');
    std::tuple<char, char, char> tuple_4('z', 'b', 'd');
    std::tuple<char, char, char> tuple_5('b', 'b', 'c');
    std::vector<std::tuple<char, char, char>> test_input = {tuple_1, tuple_2, tuple_3, tuple_4, tuple_5};

    // construct alphabet
    sacabench::util::string input_chars = "abcdxz"_s;
    sacabench::util::alphabet alphabet = sacabench::util::alphabet(input_chars);

    std::vector<std::tuple<char, char, char>> test_output = std::vector<std::tuple<char, char, char>>(5);

    std::vector<std::tuple<char, char, char>> test_control = {tuple_1, tuple_5, tuple_2, tuple_3, tuple_4};

    sort::radixsort_parallel(test_input, test_output, alphabet);
    ASSERT_EQ(test_output, test_control);
}