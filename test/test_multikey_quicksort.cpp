/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/is_sorted.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/string.hpp>

TEST(multikey_quicksort, simple_test) {
    using namespace sacabench::util;

    string_span input = "hallo"_s;
    std::vector<size_t> array = {0, 1, 2, 3, 4};

    sort::multikey_quicksort::multikey_quicksort(span(array), input);

    auto cmp = sort::multikey_quicksort::generate_multikey_key_function(input);

    ASSERT_TRUE(is_sorted(array, cmp));
}
