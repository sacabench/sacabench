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

TEST(multikey_quicksort, caabaccaabacaa) {
    const string_span input = "caabaccaabacaa"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);
    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, random_string) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('a', 'd');

    // Test with 1000 different random arrays.
    for (size_t k = 0; k < 1000; ++k) {
        string input;
        std::vector<size_t> array;

        // Insert 1000 random numbers.
        for (int i = 0; i < 1000; ++i) {
            input.push_back(dist(gen));
            array.push_back(i);
        }

        sort::multikey_quicksort::multikey_quicksort(span(array), span(input));
        ASSERT_TRUE(sa_check(span(array), span(input)));
    }
}
