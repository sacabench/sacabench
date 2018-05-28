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

TEST(blind_trie, display_trie_banana_2) {
    using namespace sacabench::deep_shallow;
    using blind_node = node<size_t>;
    using blind_trie = blind_trie<size_t>;

    const auto input = "banana"_s;

    // Create two leafs for "banana" and "na".
    leaf<size_t> banana(6, 0);
    leaf<size_t> na(2, 4);

    // root has LCP 0
    inner_node<size_t> root(0);

    blind_node node_banana(banana);
    blind_node node_na(na);

    edge<size_t> edge_banana(u_char('b'), &node_banana);
    edge<size_t> edge_na(u_char('n'), &node_na);

    root.insert_child(edge_banana);
    root.insert_child(edge_na);

    blind_node node_root(root);
    blind_trie bt(input, &node_root);

    bt.print_trie();
}

TEST(blind_trie, display_trie_banana_3) {
    using namespace sacabench::deep_shallow;
    using blind_node = node<size_t>;
    using blind_trie = blind_trie<size_t>;

    const auto input = "banana"_s;

    leaf<size_t> banana(6, 0);
    inner_node<size_t> inner_ana(3);
    leaf<size_t> anana(5, 1);
    leaf<size_t> ana(3, 3);

    blind_node node_banana(banana);
    blind_node node_anana(anana);
    blind_node node_ana(ana);

    inner_ana.insert_child(edge<size_t>((u_char)'a', &node_anana));
    inner_ana.insert_child(edge<size_t>(util::SENTINEL, &node_ana));

    blind_node node_inner_ana(inner_ana);

    inner_node<size_t> root(0);

    edge<size_t> edge_banana(u_char('b'), &node_banana);
    edge<size_t> edge_root_ana(u_char('n'), &node_inner_ana);

    root.insert_child(edge_root_ana);
    root.insert_child(edge_banana);

    blind_node node_root(root);
    blind_trie bt(input, &node_root);

    bt.print_trie();
}

TEST(blind_trie, follow_edges) {
    using namespace sacabench::deep_shallow;
    using blind_node = node<size_t>;
    using blind_trie = blind_trie<size_t>;

    const auto input = "banana"_s;

    // Create two leafs for "banana" and "na".
    leaf<size_t> banana(6, 0);
    leaf<size_t> na(2, 4);

    // root has LCP 0
    inner_node<size_t> root(0);

    blind_node node_banana(banana);
    blind_node node_na(na);

    edge<size_t> edge_banana(u_char('b'), &node_banana);
    edge<size_t> edge_na(u_char('n'), &node_na);

    root.insert_child(edge_banana);
    root.insert_child(edge_na);

    blind_node node_root(root);
    blind_trie bt(input, &node_root);

    bt.print_trie();

    {
        // Follow the edges for "nana".
        auto result = bt.follow_edges(2);
        print_result(result);
    }

    {
        // Follow the edges for "anana".
        auto result = bt.follow_edges(1);
        print_result(result);
    }
}

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

    ds::construct_sa<size_t>(input, alphabet.size, sa);
    ASSERT_TRUE(true);
}

TEST(deep_shallow, corner_cases) {
    test::saca_corner_cases<ds>();
}
