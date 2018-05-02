/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "../sacabench/util/span.hpp"
#include "../sacabench/util/sort/ternary_quicksort.hpp"
#include "../sacabench/util/is_sorted.hpp"

TEST(ternary_quicksort, empty_set) {
    using namespace util::sort::ternary_quicksort;

    auto test_set = std::vector<size_t>();
    auto cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
}
TEST(ternary_quicksort, exampleArray1) {
    using namespace util::sort::ternary_quicksort;

    auto test_set = std::vector<size_t>{10,5,7,2,8,10,756,1,0,65,4,42};
    auto cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
    
    //ASSERT_TRUE(is_sorted(test_set,cmp));
}
TEST(ternary_quicksort, exampleArray2) {
    using namespace util::sort::ternary_quicksort;

    auto test_set = std::vector<size_t>{1,1,1,1,1,1,5,1,1,11,1,1};
    auto cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
    
    for(size_t element : test_set) {
        std::cout<<element<<", ";
    }
    std::cout<<std::endl;
}
TEST(ternary_quicksort, random_array) {
    using namespace util::sort::ternary_quicksort;

    //auto test_set = std::vector<size_t>{1,1,1,1,1,1,5,1,1,11,1,1};
    std::vector<size_t> test_set;
    for(int i=0;i<1000;++i) {
        test_set.push_back(std::rand());
    }
    auto cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
    
    for(size_t element : test_set) {
        std::cout<<element<<", ";
    }
    std::cout<<std::endl;
}
