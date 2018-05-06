/*******************************************************************************
 * test/span_tests.cpp
 *
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/span.hpp>
#include <util/string.hpp>

using namespace sacabench::util;

// VERY quick and dirty tests that the code for span compiles

template < typename T >
void test_const_abstract_span(T const& m) {
    m.begin();
    m.end();
    m.size();
    m.empty();
    m[0];
    m.at(0);
    m.front();
    m.back();
    m.data();
}

template < typename T >
void test_abstract_span(T const& m) {
    test_const_abstract_span(m);
    m.fill();
}

void test_string_span() {
}

TEST(Span, test1) {
    string_span s = "hello"_s;
    test_const_abstract_span(s);

    std::vector<uint8_t> data = { 0, 1, 2 };
    span<uint8_t> v;
    test_const_abstract_span(v);
    test_abstract_span(v);

    /* implement your quick tests here! */
    ASSERT_TRUE(true);
}
