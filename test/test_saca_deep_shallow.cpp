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
#include <util/sa_check.hpp>

using namespace sacabench;
using u_char = sacabench::util::character;
using ds = sacabench::deep_shallow::saca;

// TEST(deep_shallow, blind_sort) {
//     util::string input = util::make_string("banabanabas");
//     util::apply_effective_alphabet(input);
//
//     auto sa = util::make_container<size_t>(input.size());
//     for(size_t i = 0; i < sa.size(); ++i) {
//         sa[i] = i;
//     }
//
//     sacabench::deep_shallow::blind_trie<size_t>(util::span(input), util::span(sa));
// }

TEST(deep_shallow, simple) {
    util::string input = util::make_string("hello");
    auto alphabet = util::apply_effective_alphabet(input);

    auto sa = util::make_container<size_t>(input.size());

    ds::construct_sa<size_t>(input, alphabet, sa);
    ASSERT_TRUE(true);
}

TEST(deep_shallow, with_nullbytes) {
    util::string input = util::make_string("abc\0abc"_s);
    auto alphabet = util::apply_effective_alphabet(input);

    auto sa = util::make_container<size_t>(input.size());

    ds::construct_sa<size_t>(input, alphabet.size, sa);

    ASSERT_TRUE(sa_check(util::span<size_t>(sa), util::span<util::character>(input)));
}

TEST(deep_shallow, corner_cases) {
    test::saca_corner_cases<ds>();
}
