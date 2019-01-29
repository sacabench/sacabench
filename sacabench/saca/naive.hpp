/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include <util/alphabet.hpp>
#include <util/bits.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/sort/std_sort.hpp>
#include <util/sort/ips4o.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::naive {

struct naive {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Naiv";
    static constexpr char const* DESCRIPTION =
        "Naiver Algorithmus. Sortiert Suffixe durch paarweise "
        "Stringvergleiche.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        DCHECK_EQ(text.size(), out_sa.size());

        // Fill SA with all index positions
        for (size_t i = 0; i < out_sa.size(); i++) {
            out_sa[i] = i;
        }

        // Construct a SA by sorting according
        // to the suffix starting at that index.
        util::sort::std_sort(
            out_sa, util::compare_key([&](size_t i) { return text.slice(i); }));
    }
}; // struct prefix_doubling_discarding

struct naive_ips4o {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "NaivIps4o";
    static constexpr char const* DESCRIPTION =
        "Naiver Algorithmus. Sortiert Suffixe durch paarweise "
        "Stringvergleiche. Nutzt Ips4o als Sortieralgorithmus.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        DCHECK_EQ(text.size(), out_sa.size());

        // Fill SA with all index positions
        for (size_t i = 0; i < out_sa.size(); i++) {
            out_sa[i] = i;
        }

        // Construct a SA by sorting according
        // to the suffix starting at that index.
        util::sort::ips4o_sort(
            out_sa, util::compare_key([&](size_t i) { return text.slice(i); }));
    }
}; // struct prefix_doubling_discarding

struct naive_ips4o_parallel {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "NaivIps4oParallel";
    static constexpr char const* DESCRIPTION =
        "Naiver Algorithmus. Sortiert Suffixe durch paarweise "
        "Stringvergleiche. Nutzt parallel Ips4o als Sortieralgorithmus.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        DCHECK_EQ(text.size(), out_sa.size());

        // Fill SA with all index positions
        for (size_t i = 0; i < out_sa.size(); i++) {
            out_sa[i] = i;
        }

        // Construct a SA by sorting according
        // to the suffix starting at that index.
        util::sort::ips4o_sort_parallel(
            out_sa, util::compare_key([&](size_t i) { return text.slice(i); }));
    }
}; // struct prefix_doubling_discarding

struct naive_parallel {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "NaivParallel";
    static constexpr char const* DESCRIPTION =
        "Naiver Algorithmus. Sortiert Suffixe durch paarweise "
        "Stringvergleiche. Nutzt paralell std::sort als Sortieralgorithmus.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        DCHECK_EQ(text.size(), out_sa.size());

        // Fill SA with all index positions
        for (size_t i = 0; i < out_sa.size(); i++) {
            out_sa[i] = i;
        }

        // Construct a SA by sorting according
        // to the suffix starting at that index.
        util::sort::std_par_stable_sort(
            out_sa, util::compare_key([&](size_t i) { return text.slice(i); }));
    }
}; // struct prefix_doubling_discarding

} // namespace sacabench::naive
