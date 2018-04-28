/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>

TEST(Alphabet, construct) {
    container<uint8_t> c = make_container<uint8_t>(10);
    string input = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
    alphabet a = alphabet(input);
    ASSERT_EQ(a.size, 3);
}

TEST(Alphabet, convert) {
    container<uint8_t> c = make_container<uint8_t>(10);
    string input = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
    alphabet a = alphabet(input);

    apply_effective_alphabet(input, a);
    ASSERT_EQ(input[0], 3);
    ASSERT_EQ(input[1], 1);
    ASSERT_EQ(input[2], 1);
    ASSERT_EQ(input[3], 2);
    // ...
}
