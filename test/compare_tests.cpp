/*******************************************************************************
 * test/compare_tests.cpp
 *
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/compare.hpp>
#include <map>

using namespace sacabench;

template<typename compare_type = std::less<int>>
void example_sort(compare_type func = compare_type()) {
    ASSERT_FALSE(func(0, 0));
    ASSERT_TRUE(func(0, 1));
    ASSERT_FALSE(func(1, 0));
}

template<typename compare_type = std::less<int>>
void example_map(compare_type func = compare_type()) {
    auto less_map = std::map<int, int, compare_type>(func);
    less_map[0] = 5;
    less_map[7] = 10;
    less_map[2] = 20;
    auto less_is = std::vector<std::pair<int, int>>();
    for (auto kv : less_map) {
        less_is.push_back(kv);
    }
    auto less_should = std::vector<std::pair<int, int>> {
        {0, 5},
        {2, 20},
        {7, 10},
    };
    ASSERT_EQ(less_is, less_should);

    auto greater_map = std::map<int, int, util::as_greater<compare_type>>(func);
    greater_map[0] = 5;
    greater_map[7] = 10;
    greater_map[2] = 20;
    auto greater_is = std::vector<std::pair<int, int>>();
    for (auto kv : greater_map) {
        greater_is.push_back(kv);
    }
    auto greater_should = std::vector<std::pair<int, int>> {
        {7, 10},
        {2, 20},
        {0, 5},
    };
    ASSERT_EQ(greater_is, greater_should);
}

TEST(Compare, standalone) {
    example_sort();
    example_sort(std::less<int>());
    example_sort([](int a, int b){ return a < b; });

    example_map();
    example_map(std::less<int>());
    example_map([](int a, int b){ return a < b; });
}

TEST(Compare, standalone_greater) {
    auto x = util::as_greater(std::less<int>());
    ASSERT_FALSE(x(0, 0));
    ASSERT_FALSE(x(0, 1));
    ASSERT_TRUE(x(1, 0));
}

TEST(Compare, standalone_equal) {
    auto x = util::as_equal(std::less<int>());
    ASSERT_TRUE(x(0, 0));
    ASSERT_FALSE(x(0, 1));
    ASSERT_FALSE(x(1, 0));
}

TEST(Compare, standalone_less_equal) {
    auto x = util::as_less_equal(std::less<int>());
    ASSERT_TRUE(x(0, 0));
    ASSERT_TRUE(x(0, 1));
    ASSERT_FALSE(x(1, 0));
}

TEST(Compare, standalone_greater_equal) {
    auto x = util::as_greater_equal(std::less<int>());
    ASSERT_TRUE(x(0, 0));
    ASSERT_FALSE(x(0, 1));
    ASSERT_TRUE(x(1, 0));
}

TEST(Compare, standalone_not_equal) {
    auto x = util::as_not_equal(std::less<int>());
    ASSERT_FALSE(x(0, 0));
    ASSERT_TRUE(x(1, 0));
    ASSERT_TRUE(x(0, 1));
}
