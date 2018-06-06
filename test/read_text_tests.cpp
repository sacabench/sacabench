/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/string.hpp>
#include "util/read_text.hpp"

TEST(read_text, read_text_without_trailing_newline) {

    using namespace sacabench::util;

    // The txt file is contained in the test directory.
    // So when running this test with make check in the build directory,
    // we have to go to the prevoius directory (sacabench) and then into the test directory.
    std::string path = "../test/read_text_tests.txt";
    auto context = read_text_context(path);
    auto container = make_container<character>(context.size);
    context.read_text(container);

    ASSERT_EQ(container.size(), size_t(27));
    ASSERT_EQ(container, "Hello!\nread_text test.\nBye!"_s);
}

TEST(read_text, read_text_with_trailing_newline) {

    using namespace sacabench::util;

    std::string path = "../test/read_text_tests_newline_at_end.txt";
    auto context = read_text_context(path);
    auto container = make_container<character>(context.size);
    context.read_text(container);

    ASSERT_EQ(container.size(), size_t(28));
    ASSERT_EQ(container, "Hello!\nread_text test.\nBye!\n"_s);
}