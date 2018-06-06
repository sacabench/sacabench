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
    ASSERT_EQ(a.size_without_sentinel(), std::size_t(3));
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

TEST(Alphabet, test_helper_function) {
    sacabench::util::string input1 = {'c', 'a', 'a', 'b', 'a', 'c', 'c',
            'a', 'a', 'b', 'a', 'c', 'a', 'a'};

    sacabench::util::string input2 = input1;

    sacabench::util::alphabet a = sacabench::util::alphabet(input1);
    sacabench::util::apply_effective_alphabet(input1, a);
    sacabench::util::apply_effective_alphabet(input2);

    for(size_t i = 0; i < input1.size(); ++i) {
        ASSERT_EQ(input1[i], input2[i]);
    }
}

IF_DEBUG(TEST(Alphabet, test_null) {
    sacabench::util::string input1 = "hello\0world"_s;
    ASSERT_ANY_THROW(sacabench::util::alphabet{input1});
})
