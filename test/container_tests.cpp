/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

using namespace sacabench::util;

TEST(Container, construct_empty) { container<uint8_t> c; }

TEST(Container, make_container) {
    container<uint8_t> c = make_container<uint8_t>(10);
    ASSERT_EQ(c.size(), 10u);
}

TEST(Container, make_container_span) {
    std::array<uint8_t, 3> arr{1, 2, 3};

    container<uint8_t> c = make_container<uint8_t>(span(arr));
    ASSERT_EQ(c.size(), 3u);
    ASSERT_EQ(c[1], 2);
}

TEST(String, make_string) {
    string c = make_string("hello"_s);
    ASSERT_EQ(string_span(c), "hello"_s);
}

TEST(String, make_cstring) {
    string c = make_string("hello");
    ASSERT_EQ(string_span(c), "hello"_s);
}

TEST(Container, string_container) {
    container<string> v = {"foo"_s, "bar"_s};

    ASSERT_EQ(v[1], "bar"_s);
    ASSERT_EQ(v[0], "foo"_s);
}
