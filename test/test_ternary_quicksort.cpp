/*******************************************************************************
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

using sacabench::util::is_sorted;
using namespace sacabench::util::sort::ternary_quicksort;
using sacabench::util::span;

constexpr auto cmp = std::less<size_t>();

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

TEST(ternary_quicksort, three_elements_equal_mod_three) {
    auto test_set = std::vector<size_t>{1, 4, 7};

    constexpr auto cmp2 = [](size_t a, size_t b) {
        return (a%3) < (b%3);
    };

    ternary_quicksort(span(test_set), cmp2);
    ASSERT_TRUE(is_sorted(span(test_set), cmp2));
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

TEST(ternary_quicksort, array_sizes) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 10);

    // Test with 1000 different array-sizes
    for (size_t k = 0; k < 1000; ++k) {
        // Try 100 times
        for (size_t j = 0; j < 10; ++j) {
            std::vector<size_t> test_set;

            // Insert k random numbers.
            for (size_t i = 0; i < k; ++i) {
                test_set.push_back(dist(gen));
            }

            ternary_quicksort(span(test_set), cmp);
            ASSERT_TRUE(is_sorted(span(test_set), cmp));
        }
    }
}

TEST(ternary_quicksort, not_size_t) {

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 10);

    // Test with 1000 different array-sizes
    for (size_t k = 0; k < 1000; ++k) {
        auto cmp2 = [](uint8_t a, uint8_t b) -> bool {
            return a < b;
        };

        // Try 100 times
        for (size_t j = 0; j < 10; ++j) {
            std::vector<uint8_t> test_set;

            // Insert k random numbers.
            for (size_t i = 0; i < k; ++i) {
                test_set.push_back(dist(gen));
            }

            ternary_quicksort(span(test_set), cmp2);
            ASSERT_TRUE(is_sorted(span(test_set), cmp2));
        }
    }
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
