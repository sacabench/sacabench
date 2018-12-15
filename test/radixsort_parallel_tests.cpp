/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/radixsort_parallel.hpp>
#include <util/container.hpp>
//#include <random>
#include <tuple>

using namespace sacabench::util;
/*
TEST(radixsort_parallel_tests, string_sorting_without_start_index) {
    string string_1 = make_string("abcy"_s);
    string string_2 = make_string("bbyx"_s);
    string string_3 = make_string("zyad"_s);
    string string_4 = make_string("zybd"_s);
    string string_5 = make_string("bbyc"_s);

    container<string> test_input = {string_1, string_2, string_3, string_4, string_5};

    // construct alphabet
    sacabench::util::string input_chars = "abcdxyz"_s;
    sacabench::util::alphabet alphabet = sacabench::util::alphabet(input_chars);

    container<string> test_output = make_container<string>(5);
    container<string> test_control = {string_1, string_5, string_2, string_3, string_4};

    sort::radixsort_parallel<string>(test_input, test_output, alphabet);
    ASSERT_EQ(test_output, test_control);
}

TEST(radixsort_parallel_tests, string_sorting_with_start_index) {
    string string_1 = make_string("abc"_s);
    string string_2 = make_string("bbx"_s);
    string string_3 = make_string("zad"_s);
    string string_4 = make_string("zbd"_s);
    string string_5 = make_string("bbc"_s);

    container<string> test_input = {string_1, string_2, string_3, string_4, string_5};

    // construct alphabet
    sacabench::util::string input_chars = "abcdxz"_s;
    sacabench::util::alphabet alphabet = sacabench::util::alphabet(input_chars);

    container<string> test_output = make_container<string>(5);
    container<string> test_control = {string_1, string_5, string_2, string_3, string_4};

    sort::radixsort_parallel<string>(test_input, test_output, alphabet, 2);
    ASSERT_EQ(test_output, test_control);
}

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
}*/

TEST(radixsort_parallel_tests, hopefully_final_triple_sorting) {

    std::tuple<char, int, int> tuple_0('a', 1, 7);
    std::tuple<char, int, int> tuple_1('a', 0, 2);
    std::tuple<char, int, int> tuple_2('b', 3, 1);
    std::tuple<char, int, int> tuple_3('b', 2, 0);
    std::tuple<char, int, int> tuple_4('b', 4, 9);

    auto test_input = make_container<std::tuple<char, int, int>>(5);
    test_input[0] = tuple_0;
    test_input[1] = tuple_1;
    test_input[2] = tuple_2;
    test_input[3] = tuple_3;
    test_input[4] = tuple_4;

    // construct alphabet
    sacabench::util::string input_chars = "ab"_s;
    sacabench::util::alphabet alphabet = sacabench::util::alphabet(input_chars);

    auto test_output = make_container<std::tuple<char, int, int>>(5);

    auto test_control = make_container<std::tuple<char, int, int>>(5);
    test_control[0] = tuple_1;
    test_control[1] = tuple_0;
    test_control[2] = tuple_3;
    test_control[3] = tuple_2;
    test_control[4] = tuple_4;

    sort::radixsort_parallel(test_input, test_output, alphabet);

    // Check position in third item of tripple
    ASSERT_EQ(std::get<2>(test_output[0]), 0);
    ASSERT_EQ(std::get<2>(test_output[1]), 1);
    ASSERT_EQ(std::get<2>(test_output[2]), 2);
    ASSERT_EQ(std::get<2>(test_output[3]), 3);
    ASSERT_EQ(std::get<2>(test_output[4]), 4);
}
