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

TEST(Span, string) {
    string_span s = "hello"_s;
    test_const_abstract_span(s);
}

TEST(Span, vector) {
    std::vector<uint8_t> data = { 0, 1, 2 };
    span<uint8_t> v { data };
    test_const_abstract_span(v);
    test_abstract_span(v);
}

TEST(Span, error_messages) {
    // NB: Disabled because they cause
    // errors not checkable in a gtest

    //auto s = "foo"_s;
    //s[4];
    //s.at(4);
    //s.slice(4, 5);
}
