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

TEST(is_sorted, correctly_sorted) {
    std::vector<size_t> test_case = {1, 2, 3, 4, 5};
    std::function<int(size_t, size_t)> cmp = [](size_t a, size_t b) {
        return (a - b);
    };
    ASSERT_TRUE(is_sorted(span(test_case), cmp));
}

TEST(is_sorted, wrongly_sorted) {
    std::vector<size_t> test_case = {5, 4, 1, 2, 3};
    std::function<int(size_t, size_t)> cmp = [](size_t a, size_t b) {
        return (a - b);
    };
    ASSERT_FALSE(is_sorted(span(test_case), cmp));
}
