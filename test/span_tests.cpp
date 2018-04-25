/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/span.hpp>

// VERY quick and dirty tests that the code for span compiles

template < typename T >
void test_const_abstract_span(T& m)
{
    auto f = [&](auto&& m) {
        m.begin();
        m.end();
        m.size();
        m.empty();
        m[0];
        m.at(0);
        m.front();
        m.back();
        m.data();
    };

    f(m);
    f(static_cast< T const& >(m));
}

template < typename T >
void test_abstract_span(T& m)
{
    test_const_abstract_span(m);
    m.fill();
}

void test_string_span()
{
    string_span_t s = "hello"_s;
    test_const_abstract_span(s);
}

TEST(Span, test1) {
    /* implement your quick tests here! */
    ASSERT_TRUE(true);
}
