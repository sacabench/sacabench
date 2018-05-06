/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/string.hpp>
#include "util/read_text.hpp"

TEST(read_text, example_test_ok) {

    using namespace sacabench::util;

    // The txt file is contained in the test directory.
    // So when running this test with make check in the build directory,
    // we have to go to the prevoius directory (sacabench) and then into the test directory.
    std::string path = "../test/read_text_tests.txt";
    string input = read_text(path);

    ASSERT_EQ(input.size(), 25);
    ASSERT_EQ(input[0], 'H');
    ASSERT_EQ(input[1], 'e');
    ASSERT_EQ(input[2], 'l');
    ASSERT_EQ(input[3], 'l');
    ASSERT_EQ(input[4], 'o');
    ASSERT_EQ(input[5], '!');
    ASSERT_EQ(input[6], 'r');
    ASSERT_EQ(input[7], 'e');
    ASSERT_EQ(input[8], 'a');
    ASSERT_EQ(input[9], 'd');
    ASSERT_EQ(input[10], '_');
    ASSERT_EQ(input[11], 't');
    ASSERT_EQ(input[12], 'e');
    ASSERT_EQ(input[13], 'x');
    ASSERT_EQ(input[14], 't');
    ASSERT_EQ(input[15], ' ');
    ASSERT_EQ(input[16], 't');
    ASSERT_EQ(input[17], 'e');
    ASSERT_EQ(input[18], 's');
    ASSERT_EQ(input[19], 't');
    ASSERT_EQ(input[20], '.');
    ASSERT_EQ(input[21], 'B');
    ASSERT_EQ(input[22], 'y');
    ASSERT_EQ(input[23], 'e');
    ASSERT_EQ(input[24], '!');
}