/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "../sacabench/util/sort/ternary_quicksort.hpp"

TEST(ternary_quicksort, empty_set) {
    using util::sort::ternary_quicksort;

    auto test_set = std::vector();

    struct test_key_func {
        int compare(const size_t a, const size_t b) {
            return a < b;
        }
        int max(const size_t a, const size_t b) {
            return compare(a,b) < 0 ? b : a;
        }
        int min(const size_t a, const size_t b) {
            return compare(a,b) < 0 ? a : b;
        }
    };

    ternary_quicksort(test_set, test_key_func);

    ASSERT_EQ(true, is_sorted(ewigjewiogjwgioew))
}
