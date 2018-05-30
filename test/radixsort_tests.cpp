/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/radixsort.hpp>

TEST(Radixsort, sort) {
    using namespace sacabench::util;

    container<string> strings = {
        "abbs"_s, "hjgu"_s, "abhg"_s, "achg"_s,
        "abas"_s, "chgz"_s, "lgju"_s, "aghq"_s,
    };
    radixsort(&strings);
    ASSERT_EQ(strings[0], "abas"_s);
    ASSERT_EQ(strings[1], "abbs"_s);
    ASSERT_EQ(strings[2], "abhg"_s);
    ASSERT_EQ(strings[3], "achg"_s);
    ASSERT_EQ(strings[4], "aghq"_s);
    ASSERT_EQ(strings[5], "chgz"_s);
    ASSERT_EQ(strings[6], "hjgu"_s);
    ASSERT_EQ(strings[7], "lgju"_s);
}
