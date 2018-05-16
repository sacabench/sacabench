/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <random>
#include <iostream>
#include <gtest/gtest.h>
#include <util/is_sorted.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/string.hpp>
#include <util/sa_check.hpp>

using namespace sacabench::util;

TEST(multikey_quicksort, test_compare_function) {
    const auto input = "acba"_s;
    const sort::multikey_quicksort::compare_one_character_at_depth<size_t> less(input);

    // acba < cba
    ASSERT_TRUE(less(0,1));
    ASSERT_FALSE(less(1,0));

    // cba < ba
    ASSERT_FALSE(less(1,2));
    ASSERT_TRUE(less(2,1));

    // ba < cba
    ASSERT_FALSE(less(2,3));
    ASSERT_TRUE(less(3,2));

    // a == acba
    ASSERT_FALSE(less(3,0));
    ASSERT_FALSE(less(0,3));

    // $ < a
    ASSERT_TRUE(less(4,3));
    ASSERT_FALSE(less(3,4));

    // $$ < acba
    ASSERT_TRUE(less(5,0));
    ASSERT_FALSE(less(0,5));
}

TEST(multikey_quicksort, abc) {
    const string_span input = "abc"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    ASSERT_EQ(array.size(), input.size());

    sort::multikey_quicksort::multikey_quicksort(span(array), input);
    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, ba) {
    const string_span input = "ba"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);
    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, baca) {
    const string_span input = "baca"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);
    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, aaaabcas) {
    const string_span input = "aaaabcas"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);
    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, caabaccaabacaa) {
    const string_span input = "caabaccaabacaa"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);
    ASSERT_TRUE(sa_check(span(array), input));
}

constexpr auto test_strlen = [](size_t strl) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('a', 'd');

    // Test with 1000 different random arrays.
    for (size_t k = 0; k < 1000; ++k) {
        string input;
        std::vector<size_t> array;

        // Insert 1000 random numbers.
        for (int i = 0; i < strl; ++i) {
            input.push_back(dist(gen));
            array.push_back(i);
        }

        // for(size_t aa = 0; aa < input.size(); ++aa) {
        //     std::cout << input[aa];
        // }
        // std::cout << std::endl;

        sort::multikey_quicksort::multikey_quicksort(span(array), span(input));
        ASSERT_TRUE(sa_check(span(array), span(input)));
    }
};

TEST(multikey_quicksort, random_string_3) {
    test_strlen(3);
}

TEST(multikey_quicksort, random_string_4) {
    test_strlen(4);
}

TEST(multikey_quicksort, random_string_5) {
    test_strlen(5);
}

TEST(multikey_quicksort, random_string_6) {
    test_strlen(6);
}

TEST(multikey_quicksort, random_string_7) {
    test_strlen(7);
}

TEST(multikey_quicksort, random_string_8) {
    test_strlen(8);
}

TEST(multikey_quicksort, random_string_1000) {
    test_strlen(1000);
}
