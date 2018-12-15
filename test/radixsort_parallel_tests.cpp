/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/sort/radixsort_parallel.hpp>
#include <util/container.hpp>
#include <tuple>

using namespace sacabench::util;

TEST(radixsort_parallel_tests, hopefully_final_triple_sorting) {

    std::tuple<char, size_t, size_t> tuple_0('a', 1, 7);
    std::tuple<char, size_t, size_t> tuple_1('a', 0, 2);
    std::tuple<char, size_t, size_t> tuple_2('b', 3, 1);
    std::tuple<char, size_t, size_t> tuple_3('b', 2, 0);
    std::tuple<char, size_t, size_t> tuple_4('b', 4, 9);

    auto test_input = make_container<std::tuple<char, size_t, size_t>>(5);
    test_input[0] = tuple_0;
    test_input[1] = tuple_1;
    test_input[2] = tuple_2;
    test_input[3] = tuple_3;
    test_input[4] = tuple_4;

    // construct alphabet
    sacabench::util::string input_chars = "ab"_s;
    sacabench::util::alphabet alphabet = sacabench::util::alphabet(input_chars);

    auto test_control = make_container<std::tuple<char, size_t, size_t>>(5);
    test_control[0] = tuple_1;
    test_control[1] = tuple_0;
    test_control[2] = tuple_3;
    test_control[3] = tuple_2;
    test_control[4] = tuple_4;

    sort::radixsort_parallel<char, size_t>(test_input, alphabet);

    // Check position in third item of tripple
    ASSERT_EQ(std::get<2>(test_input[0]), 0);
    ASSERT_EQ(std::get<2>(test_input[1]), 1);
    ASSERT_EQ(std::get<2>(test_input[2]), 2);
    ASSERT_EQ(std::get<2>(test_input[3]), 3);
    ASSERT_EQ(std::get<2>(test_input[4]), 4);
}
