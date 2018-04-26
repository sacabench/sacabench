/*******************************************************************************
 * test/sa_check_tests.cpp
 *
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/span.hpp>
#include <util/container.hpp>
#include <util/sa_check.hpp>

using namespace sacabench::util;

TEST(SaCheck, example_string_1_ok) {
    auto text = "caabaccaabacaa"_s;
    auto sa = container<size_t> {
        13, 12, 7, 1, 8, 2, 10, 4, 9, 3, 11, 6, 0, 5,
    };
    ASSERT_EQ(sa_check(sa, text), sa_check_result::ok);
}

TEST(SaCheck, example_string_1_not_suffix_sorted) {
    auto text = "caabaccaabacaa"_s;
    auto sa = container<size_t> {
        13, 12, 7, 1, 8, 2, 4, 10, 9, 3, 11, 6, 0, 5,
        //                  ^ swapped 4 and 10
    };
    ASSERT_EQ(sa_check(sa, text), sa_check_result::not_suffix_sorted);
}

TEST(SaCheck, example_string_1_not_a_permutation) {
    auto text = "caabaccaabacaa"_s;
    auto sa = container<size_t> {
        13, 12, 7, 1, 8, 2, 4, 4, 9, 3, 11, 6, 0, 5,
        //                  ^ overwrote 10 with 4
    };
    ASSERT_EQ(sa_check(sa, text), sa_check_result::not_a_permutation);
}

TEST(SaCheck, example_string_1_wrong_length) {
    auto text = "caabaccaabacaa"_s;
    auto sa = container<size_t> {
        13, 12, 7, 1, 8, 2, 4,
        //                  ^ cut off after 4
    };
    ASSERT_EQ(sa_check(sa, text), sa_check_result::wrong_length);
}
