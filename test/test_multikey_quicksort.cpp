/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <iostream>
#include <gtest/gtest.h>
#include <util/is_sorted.hpp>
#include <util/sort/multikey_quicksort.hpp>
#include <util/string.hpp>
//#include <util/sa_check.hpp>

TEST(multikey_quicksort, simple_test) {
    using namespace sacabench::util;

    const string_span input = "caabaccaabacaa"_s;

    std::vector<size_t> array;
    for(size_t i = 0; i < input.size(); ++i) {
        array.push_back(i);
    }

    sort::multikey_quicksort::multikey_quicksort(span(array), input);

    auto cmp = sort::multikey_quicksort::generate_multikey_key_function<size_t>(input);

    for(size_t i = 0; i < array.size(); ++i) {
        std::cout << array[i] << std::endl;
    }

    ASSERT_TRUE(is_sorted(span(array), cmp));
    //ASSERT_TRUE(sa_check(span(array), input));
}
