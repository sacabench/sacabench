/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "test/saca.hpp"
#include <gtest/gtest.h>

#include <saca/prefix_doubling.hpp>

using namespace sacabench::prefix_doubling;

TEST(prefix_doubling, naive) { test::saca_corner_cases<prefix_doubling>(); }

TEST(prefix_doubling, discarding_2) {
    test::saca_corner_cases<prefix_discarding_2>();
}

TEST(prefix_doubling, discarding_4) {
    test::saca_corner_cases<prefix_discarding_4>();
}

struct prefix_doubling_4 {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Doubling4";
    static constexpr char const* DESCRIPTION = "";

    template <typename sa_index>
    static void construct_sa(sacabench::util::string_span text,
                             sacabench::util::alphabet const&,
                             sacabench::util::span<sa_index> out_sa) {
        prefix_doubling_impl<sa_index, 4, ips4o_sorter>::doubling(text, out_sa);
    }
};

TEST(prefix_doubling, naive4) { test::saca_corner_cases<prefix_doubling_4>(); }
