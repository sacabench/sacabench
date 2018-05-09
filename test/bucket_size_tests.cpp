/*******************************************************************************
 * test/bucket_size_tests.cpp
 *
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/container.hpp>
#include <util/bucket_size.hpp>

using namespace sacabench::util;

TEST(Bucket_Size, get) {
    string input = {'b','l','a','b','l','a','b','l','u','b'};
    container<int> bucket_sizes = get_bucket_sizes(input);
    ASSERT_EQ(bucket_sizes.size(), 5);
}

TEST(Bucket_Size, sizes){
        string input = {'b','l','a','b','l','a','b','l','u','b'};
        container<int> bucket_sizes = get_bucket_sizes(input);
        ASSERT_EQ(bucket_sizes.at(1), 2);
        ASSERT_EQ(bucket_sizes.at(2), 4);
        ASSERT_EQ(bucket_sizes.at(3), 3);
        ASSERT_EQ(bucket_sizes.at(4), 1);
}

