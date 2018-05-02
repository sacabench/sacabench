/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "../sacabench/util/span.hpp"
#include "../sacabench/util/sort/ternary_quicksort.hpp"
#include "../sacabench/util/is_sorted.hpp"
#include <functional>

TEST(ternary_quicksort, empty_set) {
    using namespace util::sort::ternary_quicksort;

    auto test_set = std::vector<size_t>();
    std::function<int(size_t,size_t)> cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
}
TEST(ternary_quicksort, example_array_1) {
    using namespace util::sort::ternary_quicksort;

    auto test_set = std::vector<size_t>{10,5,7,2,8,10,756,1,0,65,4,42};

    std::function<int(size_t,size_t)> cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);

    ASSERT_TRUE(is_sorted(span(test_set),cmp));
}

TEST(ternary_quicksort, example_array_2) {
    using namespace util::sort::ternary_quicksort;

    auto test_set = std::vector<size_t>{1,1,1,1,1,1,5,1,1,11,1,1};
    std::function<int(size_t,size_t)> cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
    ASSERT_TRUE(is_sorted(span(test_set),cmp));
    /*for(size_t elem:test_set) {
        std::cout<<elem<<", ";
    }
    std::cout<<std::endl;
*/
    
}

TEST(ternary_quicksort, random_array) {
    using namespace util::sort::ternary_quicksort;

    std::vector<size_t> test_set;
    for(int i=0;i<1000;++i) {
        test_set.push_back(std::rand());
    }
    std::function<int(size_t,size_t)> cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
     ASSERT_TRUE(is_sorted(span(test_set),cmp));
   
}
TEST(ternary_quicksort, big_random_array) {
    using namespace util::sort::ternary_quicksort;

    std::vector<size_t> test_set;
    for(int i=0;i<1000000;++i) {
        test_set.push_back(std::rand());
    }
    std::function<int(size_t,size_t)> cmp = [](size_t a, size_t b){ return a - b; };
    ternary_quicksort(span(test_set), cmp);
     ASSERT_TRUE(is_sorted(span(test_set),cmp));
   
}
