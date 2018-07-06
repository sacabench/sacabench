/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <util/sa_check.hpp>
#include <util/is_sorted.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/string.hpp>

using namespace sacabench::util;

constexpr auto test_strlen = [](size_t strl) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('a', 'd');

    // Test with 1000 different random arrays.
    for (size_t k = 0; k < 1000; ++k) {
        std::vector<character> input;
        std::vector<size_t> array;

        // Insert 1000 random numbers.
        for (size_t i = 0; i < strl; ++i) {
            input.push_back(dist(gen));
            array.push_back(i);
        }
        input.push_back(SENTINEL);
        array.push_back(strl);

        sort::multikey_quicksort::multikey_quicksort(span(array), span(input));
        ASSERT_TRUE(sa_check(span(array), span(input)));
    }
};

TEST(multikey_quicksort, test_equal_partitions) {
    const auto input = "acbaacbaacba\0"_s;

    std::vector<size_t> array;
    for (size_t i = 0; i < input.size(); ++i) {
        array.push_back(0);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), span(input));

    ASSERT_TRUE(is_partially_suffix_sorted<size_t>(array, input));
}

TEST(multikey_quicksort, test_compare_function) {
    const auto input = "acba"_s;
    const sort::multikey_quicksort::compare_one_character_at_depth<size_t> less(
        input);

    // acba < cba
    ASSERT_TRUE(less(0, 1));
    ASSERT_FALSE(less(1, 0));

    // cba < ba
    ASSERT_FALSE(less(1, 2));
    ASSERT_TRUE(less(2, 1));

    // ba < cba
    ASSERT_FALSE(less(2, 3));
    ASSERT_TRUE(less(3, 2));

    // a == acba
    ASSERT_FALSE(less(3, 0));
    ASSERT_FALSE(less(0, 3));
}

TEST(multikey_quicksort, random_string_3) { test_strlen(3); }

TEST(multikey_quicksort, random_string_4) { test_strlen(4); }

TEST(multikey_quicksort, random_string_5) { test_strlen(5); }

TEST(multikey_quicksort, random_string_6) { test_strlen(6); }

TEST(multikey_quicksort, random_string_7) { test_strlen(7); }

TEST(multikey_quicksort, random_string_8) { test_strlen(8); }

TEST(multikey_quicksort, random_string_1000) { test_strlen(1000); }

TEST(multikey_quicksort, deep_sort_gfedcbaaaa) {
    const string_span input = "gfedcbaaaa\0"_s;

    std::vector<size_t> array;
    for (size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    bool used_deep_sort = false;

    auto deep_sort = [&](span<size_t>) { used_deep_sort = true; };

    sort::multikey_quicksort::multikey_quicksort<2>(span(array), input,
                                                 deep_sort);
    ASSERT_TRUE(used_deep_sort);
}
