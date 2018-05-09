/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>

TEST(Alphabet, construct) {
    sacabench::util::string input = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
    sacabench::util::alphabet a = sacabench::util::alphabet(input);
    ASSERT_EQ(a.size, std::size_t(3));
}

TEST(Alphabet, convert) {
    sacabench::util::string input = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};
    sacabench::util::alphabet a = sacabench::util::alphabet(input);

    sacabench::util::apply_effective_alphabet(input, a);
    ASSERT_EQ(input[0], 3);
    ASSERT_EQ(input[1], 1);
    ASSERT_EQ(input[2], 1);
    ASSERT_EQ(input[3], 2);
    // ...
}
