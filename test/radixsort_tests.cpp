/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/radixsort.hpp>

TEST(Radixsort, sort) {
    std::vector<std::string> strings = { "abbs", "hjgu", "abhg", "achg", "abas", "chgz", "lgju", "aghq"};
    sacabench::util::radixsort(&strings);
    ASSERT_EQ(strings[0], "abas");
    ASSERT_EQ(strings[1], "abbs");
    ASSERT_EQ(strings[2], "abhg");
    ASSERT_EQ(strings[3], "achg");
    ASSERT_EQ(strings[4], "aghq");
    ASSERT_EQ(strings[5], "chgz");
    ASSERT_EQ(strings[6], "hjgu");
    ASSERT_EQ(strings[7], "lgju"); 
}
