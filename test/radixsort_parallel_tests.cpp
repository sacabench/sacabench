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

TEST(radixsort_parallel_tests, triple_sorting) {
    
    std::tuple<size_t, size_t, size_t> tuple_0(3, 6, 0);
    std::tuple<size_t, size_t, size_t> tuple_1(5, 2, 3);
    std::tuple<size_t, size_t, size_t> tuple_2(8, 4, 6);
    std::tuple<size_t, size_t, size_t> tuple_3(1, 1, 9);

    auto test_input = make_container<std::tuple<size_t, size_t, size_t>>(4);
    test_input[0] = tuple_0;
    test_input[1] = tuple_1;
    test_input[2] = tuple_2;
    test_input[3] = tuple_3;

    auto test_control = make_container<std::tuple<size_t, size_t, size_t>>(4);
    test_control[0] = tuple_3;
    test_control[1] = tuple_0;
    test_control[2] = tuple_1;
    test_control[3] = tuple_2;

    sort::radixsort_parallel<size_t, size_t>(test_input, 9);

    ASSERT_EQ(test_input[0], test_control[0]);
    ASSERT_EQ(test_input[1], test_control[1]);
    ASSERT_EQ(test_input[2], test_control[2]);
    ASSERT_EQ(test_input[3], test_control[3]);
}
