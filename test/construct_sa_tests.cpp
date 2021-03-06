/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "test/saca.hpp"
#include <gtest/gtest.h>

#include <saca/naive.hpp>
#include <saca/prefix_doubling.hpp>

using namespace sacabench;

template <typename algo, size_t Sentinels>
struct Adapter {
    static constexpr size_t EXTRA_SENTINELS = algo::EXTRA_SENTINELS + Sentinels;

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet_size,
                             util::span<sa_index> out_sa) {
        ASSERT_EQ(out_sa.size(), text.size());

        for (size_t i = 0; i < Sentinels; i++) {
            out_sa[i] = (out_sa.size() - 1) - i;
        }

        out_sa = out_sa.slice(Sentinels);

        text = text.slice(0, text.size() - Sentinels);
        algo::construct_sa(text, alphabet_size, out_sa);
    }
};

TEST(CornerCases, naive) { test::saca_corner_cases<naive::naive>(); }

TEST(CornerCases, adapter_naive) {
    test::saca_corner_cases<Adapter<naive::naive, 3>>();
}

TEST(CornerCases, adapter_doubling) {
    test::saca_corner_cases<Adapter<prefix_doubling::prefix_doubling, 3>>();
}

TEST(CornerCases, adapter_discarding) {
    test::saca_corner_cases<
        Adapter<prefix_doubling::prefix_discarding_2, 3>>();
}
