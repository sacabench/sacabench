/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/deep_shallow/saca.hpp>
#include "test/saca.hpp"

using namespace sacabench;
using ds = sacabench::deep_shallow::saca;

TEST(deep_shallow, simple) {
    util::string input = util::make_string("hello");
    auto sa = util::make_container<size_t>(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        sa[i] = i;
    }

    auto alphabet = util::apply_effective_alphabet(input);

    ds::construct_sa<size_t>(input, alphabet.size, sa);
    ASSERT_TRUE(true);
}

TEST(deep_shallow, corner_cases) {
    test::saca_corner_cases<ds>();
}
