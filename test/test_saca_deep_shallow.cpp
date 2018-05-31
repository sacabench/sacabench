/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "test/saca.hpp"
#include <gtest/gtest.h>
#include <random>
#include <saca/deep_shallow/blind/sort.hpp>
#include <saca/deep_shallow/blind/trie.hpp>
#include <saca/deep_shallow/saca.hpp>
#include <util/alphabet.hpp>
#include <util/container.hpp>
#include <util/sa_check.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

using namespace sacabench;
using u_char = sacabench::util::character;
using ds = sacabench::deep_shallow::saca;

TEST(blind_trie, traverse) {
    using blind_trie = sacabench::deep_shallow::blind::trie<size_t>;

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

struct blind_saca {
    template <typename sa_index_type>
    inline static void construct_sa(util::string_span text,
                                    size_t /*alphabet_size*/,
                                    util::span<sa_index_type> sa) {
        using blind_trie = sacabench::deep_shallow::blind::trie<size_t>;

        if (text.size() < 2) {
            return;
        }

        blind_trie my_trie(text, text.size() - 1);
        for (size_t i = 0; i < text.size() - 1; ++i) {
            size_t j = text.size() - 2 - i;
            my_trie.insert(j);
        }
        // my_trie.print();
        my_trie.traverse(sa);
    }
};

TEST(blind_trie, as_saca) { test::saca_corner_cases<blind_saca>(); }

constexpr auto test_blind_trie = [](const util::string_span text) {
    using blind_trie = sacabench::deep_shallow::blind::trie<size_t>;

    if (text.size() < 2) {
        return true;
    }

    auto space = util::make_container<size_t>(text.size());
    util::span<size_t> sa = space;

    // std::cout << "#########################################################"
    // << std::endl;

    blind_trie my_trie(text, text.size() - 1);
    for (size_t i = 0; i < text.size() - 1; ++i) {
        size_t j = text.size() - 2 - i;
        my_trie.insert(j);
        // std::cout <<
        // "---------------------------------------------------------" <<
        // std::endl; my_trie.print(); std::cout <<
        // "---------------------------------------------------------" <<
        // std::endl;
    }

    my_trie.traverse(sa);

    const auto result = util::sa_check(sa, text);

    if (!bool(result)) {
        std::cout << "tree failed on " << text << ": " << result << std::endl;
        return false;
    } else {
        return true;
    }
};

TEST(blind_trie, bbabaabaaa) {
    const auto text = "bbabaabaaa"_s;
    test_blind_trie(text);
}

TEST(blind_trie, sort) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('a', 'b');

    // Try with 100 text lengths.
    for (size_t j = 0; j < 100; ++j) {
        std::vector<util::character> input;
        auto space = util::make_container<size_t>(j);

        // Insert j random numbers.
        for (size_t i = 0; i < j; ++i) {
            input.push_back(dist(gen));
            space[i] = i;
        }

        auto input_span = util::span<util::character>(input);
        auto space_span = util::span<size_t>(space);

        sacabench::deep_shallow::blind::sort(input_span, space_span);

        ASSERT_TRUE(bool(sa_check(space_span, input_span)));
    }
}

TEST(blind_trie, random) {
    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('a', 'd');

    // Try with 100 text lengths.
    for (size_t j = 0; j < 100; ++j) {
        std::vector<util::character> input;

        // Insert j random numbers.
        for (size_t i = 0; i < j; ++i) {
            input.push_back(dist(gen));
        }

        ASSERT_TRUE(test_blind_trie(util::span(input)));
    }

    // Try with 100 texts.
    for (size_t j = 0; j < 100; ++j) {
        std::vector<util::character> input;

        // Insert 100 random numbers.
        for (size_t i = 0; i < 100; ++i) {
            input.push_back(dist(gen));
        }

        ASSERT_TRUE(test_blind_trie(util::span(input)));
    }
}

TEST(deep_shallow, simple) {
    util::string input = util::make_string("hello");
    auto alphabet = util::apply_effective_alphabet(input);

    auto sa = util::make_container<size_t>(input.size());

    ds::construct_sa<size_t>(input, alphabet.size, sa);
    ASSERT_TRUE(true);
}

TEST(deep_shallow, corner_cases) { test::saca_corner_cases<ds>(); }
