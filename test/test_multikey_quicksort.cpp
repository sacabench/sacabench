/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <util/assertions.hpp>
#include <iostream>
#include <gtest/gtest.h>
#include <util/is_sorted.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/string.hpp>
#include <util/sa_check.hpp>

TEST(multikey_quicksort, abc) {
    using namespace sacabench::util;

    const string_span input = "abc"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    ASSERT_EQ(array.size(), input.size());

    sort::multikey_quicksort::multikey_quicksort(span(array), input);

    for(size_t i = 0; i < array.size(); ++i) {
        std::cout << array[i] << std::endl;
    }

    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, ba) {
    using namespace sacabench::util;

    const string_span input = "ba"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);

    for(size_t i = 0; i < array.size(); ++i) {
        std::cout << array[i] << std::endl;
    }

    ASSERT_TRUE(sa_check(span(array), input));
}

TEST(multikey_quicksort, caabaccaabacaa) {
    using namespace sacabench::util;

    const string_span input = "caabaccaabacaa"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);

    for(size_t i = 0; i < array.size(); ++i) {
        std::cout << array[i] << std::endl;
    }

    ASSERT_TRUE(sa_check(span(array), input));
}
