/*******************************************************************************
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

TEST(Span, const_vector) {
    std::vector<uint8_t> const data = { 0, 1, 2 };
    span<uint8_t const> v { data };
    test_const_abstract_span(v);
}

TEST(Span, array) {
    std::array<uint8_t, 3> data = { 0, 1, 2 };
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

TEST(Compare, less) {
    string_span a = "hello0"_s;
    string_span b = "hello1"_s;
    ASSERT_LT(a, b);
}
TEST(Compare, greater) {
    string_span a = "hello1"_s;
    string_span b = "hello0"_s;
    ASSERT_GT(a, b);
}
TEST(Compare, equal) {
    string_span a = "hello0"_s;
    string_span b = "hello0"_s;
    ASSERT_EQ(a, b);
}
TEST(Compare, not_equal) {
    string_span a = "hello0"_s;
    string_span b = "hello1"_s;
    ASSERT_NE(a, b);
}
TEST(Compare, less_equal) {
    string_span a = "hello0"_s;
    string_span b = "hello1"_s;
    string_span c = "hello1"_s;
    ASSERT_LE(a, b);
    ASSERT_LE(b, c);
}
TEST(Compare, greater_equal) {
    string_span a = "hello1"_s;
    string_span b = "hello0"_s;
    string_span c = "hello0"_s;
    ASSERT_GE(a, b);
    ASSERT_GE(b, c);
}
