/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>

#include <util/tagged_number.hpp>
#include <util/container.hpp>

using namespace sacabench::util;

TEST(tagged_number, simple) {
    tagged_number<size_t, 0> a = 0;
    ++a;

    tagged_number<size_t, 0> b = 1;
    tagged_number<size_t, 0> c = 2;
    tagged_number<size_t, 0> d = 3;

    ASSERT_EQ(a, b);

    --a;
    ASSERT_NE(a, b);

    ASSERT_EQ(b + c, d);
    ASSERT_EQ(b * c, c);

    // ASSERT_EQ(a, 0);
    // ASSERT_EQ(b, 1);
    // ASSERT_EQ(c, 2);
    // ASSERT_EQ(d, 3);
}

TEST(tagged_number, bits) {
    tagged_number<size_t, 1> a = 20;

    ASSERT_EQ(a.number(), 20u);
    a.set<0>(true);
    ASSERT_EQ(a.number(), 20u);
    a.set<0>(false);
    ASSERT_EQ(a.number(), 20u);

    a.set<1>(true);
    ASSERT_NE(a.number(), 20u);
}

TEST(tagged_number, cast) {
    auto array = make_container<size_t>(10);
    for(size_t i = 0; i < 10; ++i) {
        array[i] = i;
    }

    auto s = cast_to_tagged_numbers<size_t, 1>(array);

    for(size_t i = 0; i < 10; ++i) {
        ASSERT_EQ(s[i].number(), i);
        ASSERT_EQ(s[i].get<0>(), false);
        s[i].set<0>(true);
    }

    for(size_t i = 0; i < 10; ++i) {
        ASSERT_EQ(s[i].number(), i);
        ASSERT_EQ(s[i].get<0>(), true);
    }
}
