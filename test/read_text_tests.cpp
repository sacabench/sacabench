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
    string input = read_text(path);

    ASSERT_EQ(input.size(), 27);
    ASSERT_EQ(input[0], 'H');
    ASSERT_EQ(input[1], 'e');
    ASSERT_EQ(input[2], 'l');
    ASSERT_EQ(input[3], 'l');
    ASSERT_EQ(input[4], 'o');
    ASSERT_EQ(input[5], '!');
    ASSERT_EQ(input[6], '\n');
    ASSERT_EQ(input[7], 'r');
    ASSERT_EQ(input[8], 'e');
    ASSERT_EQ(input[9], 'a');
    ASSERT_EQ(input[10], 'd');
    ASSERT_EQ(input[11], '_');
    ASSERT_EQ(input[12], 't');
    ASSERT_EQ(input[13], 'e');
    ASSERT_EQ(input[14], 'x');
    ASSERT_EQ(input[15], 't');
    ASSERT_EQ(input[16], ' ');
    ASSERT_EQ(input[17], 't');
    ASSERT_EQ(input[18], 'e');
    ASSERT_EQ(input[19], 's');
    ASSERT_EQ(input[20], 't');
    ASSERT_EQ(input[21], '.');
    ASSERT_EQ(input[22], '\n');
    ASSERT_EQ(input[23], 'B');
    ASSERT_EQ(input[24], 'y');
    ASSERT_EQ(input[25], 'e');
    ASSERT_EQ(input[26], '!');
}

TEST(read_text, read_text_with_trailing_newline) {

    using namespace sacabench::util;

    std::string path = "../test/read_text_tests_newline_at_end.txt";
    string input = read_text(path);

    ASSERT_EQ(input.size(), 28);
    ASSERT_EQ(input[0], 'H');
    ASSERT_EQ(input[1], 'e');
    ASSERT_EQ(input[2], 'l');
    ASSERT_EQ(input[3], 'l');
    ASSERT_EQ(input[4], 'o');
    ASSERT_EQ(input[5], '!');
    ASSERT_EQ(input[6], '\n');
    ASSERT_EQ(input[7], 'r');
    ASSERT_EQ(input[8], 'e');
    ASSERT_EQ(input[9], 'a');
    ASSERT_EQ(input[10], 'd');
    ASSERT_EQ(input[11], '_');
    ASSERT_EQ(input[12], 't');
    ASSERT_EQ(input[13], 'e');
    ASSERT_EQ(input[14], 'x');
    ASSERT_EQ(input[15], 't');
    ASSERT_EQ(input[16], ' ');
    ASSERT_EQ(input[17], 't');
    ASSERT_EQ(input[18], 'e');
    ASSERT_EQ(input[19], 's');
    ASSERT_EQ(input[20], 't');
    ASSERT_EQ(input[21], '.');
    ASSERT_EQ(input[22], '\n');
    ASSERT_EQ(input[23], 'B');
    ASSERT_EQ(input[24], 'y');
    ASSERT_EQ(input[25], 'e');
    ASSERT_EQ(input[26], '!');
    ASSERT_EQ(input[22], '\n');
}