/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <functional>
#include <gtest/gtest.h>
#include <random>
#include <util/is_sorted.hpp>
#include <util/sort/ternary_quicksort.hpp>
#include <util/span.hpp>

using namespace sacabench::util::sort::ternary_quicksort;

inline int cmp(size_t a, size_t b) { return a - b; }

TEST(ternary_quicksort, empty_set) {
    auto test_set = std::vector<size_t>();
    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set), cmp));
}

TEST(ternary_quicksort, one_element) {
    auto test_set = std::vector<size_t>{5};
    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set), cmp));
}

TEST(ternary_quicksort, two_elements_unsorted) {
    auto test_set = std::vector<size_t>{5, 4};
    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set), cmp));
}

TEST(ternary_quicksort, two_elements_sorted) {
    auto test_set = std::vector<size_t>{4, 5};
    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set), cmp));
}

TEST(ternary_quicksort, example_array_1) {
    auto test_set =
        std::vector<size_t>{10, 5, 7, 2, 8, 10, 756, 1, 0, 65, 4, 42};

    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set), cmp));
}

TEST(ternary_quicksort, example_array_2) {
    auto test_set = std::vector<size_t>{1, 1, 1, 1, 1, 1, 5, 1, 1, 11, 1, 1};
    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set), cmp));
}

TEST(ternary_quicksort, random_array) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 10);

    // Test with 1000 different random arrays.
    for (size_t k = 0; k < 1000; ++k) {
        std::vector<size_t> test_set;

        // Insert 1000 random numbers.
        for (int i = 0; i < 1000; ++i) {
            test_set.push_back(dist(gen));
        }

        ternary_quicksort(span(test_set), cmp);
        ASSERT_TRUE(is_sorted(span(test_set), cmp));
    }
}

TEST(ternary_quicksort, big_random_array) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 10);

    // Test with 1000 different random arrays.
    for (size_t k = 0; k < 10; ++k) {
        std::vector<size_t> test_set;

        // Insert 1000000 random numbers.
        for (int i = 0; i < 1000000; ++i) {
            test_set.push_back(dist(gen));
        }

        ternary_quicksort(span(test_set), cmp);
        ASSERT_TRUE(is_sorted(span(test_set), cmp));
    }
}
