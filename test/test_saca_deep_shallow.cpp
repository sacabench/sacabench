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

// TEST(deep_shallow, print_blind_trie) {
//     using blind_trie = sacabench::deep_shallow::blind_trie::trie<size_t>;
//
//     util::string input = util::make_string("nbanana");
//
//     blind_trie my_trie(input, 6);
//     my_trie.insert(5);
//     my_trie.insert(4);
//     my_trie.insert(3);
//     my_trie.insert(2);
//     my_trie.insert(1);
//     my_trie.insert(0);
//
//     my_trie.print();
// }

TEST(deep_shallow, blind_trie_traverse) {
    using blind_trie = sacabench::deep_shallow::blind_trie::trie<size_t>;

    util::string input = util::make_string("nbanana");

    blind_trie my_trie(input, 6);
    my_trie.insert(5);
    my_trie.insert(4);
    my_trie.insert(3);
    my_trie.insert(2);
    my_trie.insert(1);
    my_trie.insert(0);

    my_trie.print();

    auto bucket = util::make_container<size_t>(7);
    my_trie.traverse(bucket);

    std::cout << bucket << std::endl;
}

TEST(deep_shallow, simple) {
    util::string input = util::make_string("hello");
    auto alphabet = util::apply_effective_alphabet(input);

    auto sa = util::make_container<size_t>(input.size());

    ds::construct_sa<size_t>(input, alphabet.size, sa);
    ASSERT_TRUE(true);
}

TEST(deep_shallow, corner_cases) {
    test::saca_corner_cases<ds>();
}
