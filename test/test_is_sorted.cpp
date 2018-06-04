/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <functional>
#include <gtest/gtest.h>
#include <util/is_sorted.hpp>
#include <util/span.hpp>
#include <vector>

using namespace sacabench::util;

constexpr auto cmp = std::less<>();

TEST(is_sorted, correctly_sorted_empty) {
    std::vector<size_t> test_case;
    ASSERT_TRUE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, correctly_sorted_one) {
    std::vector<size_t> test_case = {5};
    ASSERT_TRUE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, correctly_sorted_two) {
    std::vector<size_t> test_case = {4, 5};
    ASSERT_TRUE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, correctly_sorted_five) {
    std::vector<size_t> test_case = {1, 2, 3, 4, 5};
    ASSERT_TRUE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, wrongly_sorted_two) {
    std::vector<size_t> test_case = {1, 0};
    ASSERT_FALSE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, wrongly_sorted_five) {
    std::vector<size_t> test_case = {5, 4, 1, 2, 3};
    ASSERT_FALSE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, correctly_sorted_five_default_cmp) {
    std::vector<size_t> test_case = {1, 2, 3, 4, 5};
    ASSERT_TRUE(is_sorted(span(test_case)));
}

TEST(is_sorted, wrongly_sorted_two_default_cmp) {
    std::vector<size_t> test_case = {1, 0};
    ASSERT_FALSE(is_sorted(span(test_case)));
}
