/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>

#include <util/tagged_number.hpp>

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
