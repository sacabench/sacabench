/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <tuple>
#include <array>

#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/bits.hpp>
#include <util/sort/std_sort.hpp>
#include <util/compare.hpp>

namespace sacabench::prefix_doubling {
    struct std_sorting_algorithm {
        template <typename T, typename Compare = std::less<T>>
        static void sort(util::span<T> data, Compare comp = Compare()) {
            util::sort::std_sort(data, comp);
        }
    };

    using name_type = size_t;
    using atuple = std::array<name_type, 2>;
    using S_tuple = std::pair<atuple, size_t>;
    using P_tuple = std::pair<name_type, size_t>;

    /// Create a unique name for each S tuple,
    /// returning true if all names are unique.
    ///
    /// Precondition: The tuple in S are lexicographical sorted.
    inline static bool name(util::span<S_tuple> S, util::span<P_tuple> P) {
        // A name is determined as follows:
        // `last_pair` contains the last `S` tuple looked at, or
        // ($, $) initially.
        //
        // We iterate over all tuples in `S`, and set `name` to the current
        // iteration index + 1 each time we see a different tuple.
        //
        // Since the tuple in S are sorted, `name`s are therefore always
        // integer that sort to the same positions as their tuple would.
        //
        // We skip `name=0`, because we want to reserve
        // it for the sentinel value.
        //
        // In `only_unique` we keep track of whether we've seen a tuple more
        // than once.
        //
        // Example:
        //
        // Text: ababab$
        // |      S            P   |
        // | ((a, b), 0) => (1, 0) |
        // | ((a, b), 2) => (1, 2) |
        // | ((a, b), 4) => (1, 4) |
        // | ((b, a), 1) => (4, 1) |
        // | ((b, a), 3) => (4, 3) |
        // | ((b, $), 5) => (6, 5) |
        // only_unique = false

        name_type name = 0;
        auto last_pair = atuple { util::SENTINEL, util::SENTINEL };
        bool only_unique = true;

        for (size_t i = 0; i < S.size(); i++) {
            auto const& pair = S[i].first;
            if (pair != last_pair) {
                name = i + 1;
                last_pair = pair;
            } else {
                only_unique = false;
            }
            P[i] = P_tuple { name, S[i].second };
        }

        return only_unique;
    }

    inline void print_S(util::span<S_tuple> S) {
        std::cout << "S [\n";
        for (auto& e: S) {
            std::cout << "  ("
                << int(e.first[0]) << ", "
                << int(e.first[1]) << "), "
                << e.second << "\n";
        }
        std::cout << "]\n";
    }

    inline void print_P(util::span<P_tuple> S) {
        std::cout << "S [\n";
        for (auto& e: S) {
            std::cout << "  "
                << int(e.first) << ", "
                << e.second << "\n";
        }
        std::cout << "]\n";
    }

    /// Naive variant, roughly identical to description in Paper
    template<typename sorting_algorithm = std_sorting_algorithm>
    class prefix_doubling {
        public:
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     size_t /*alphabet_size*/,
                                     util::span<sa_index> out_sa) {
                size_t const N = text.size();

                // To simplify some of the logic below,
                // we don't even try to process texts of size less than 2.
                if (N == 0) {
                    return;
                } else if (N == 1) {
                    out_sa[0] = 0;
                    return;
                }

                // Allocate the S and P arrays.
                // TODO: They are never needed concurrently, so could
                // combine into a single array and interpret either
                // as S or P
                auto S = util::make_container<S_tuple>(N);
                auto P = util::make_container<P_tuple>(N);

                // Create the initial S array of character tuples + text
                // position
                for(size_t i = 0; i < N - 1; ++i) {
                    S[i] = std::make_pair(
                        atuple { text[i], text[i + 1] }, i);
                }
                S[N - 1] = std::make_pair(
                    atuple { text[N - 1], util::SENTINEL }, N - 1);

                // We iterate up to ceil(log2(N)) times - but because we
                // always have a break condition in the loop body,
                // we don't need to check for it explicitly
                for(size_t k = 1;;k++) {
                    DCHECK_LE(k, util::ceil_log2(N));

                    size_t const k_length = 1ull << k;

                    // Sort the S tuples lexicographical
                    sorting_algorithm::sort(util::span(S));

                    // Rename the S tuples into P
                    bool only_unique = name(S, P);

                    // The algorithm terminates if P only contains unique values
                    if (only_unique) {
                        for (size_t i = 0; i < N; i++) {
                            out_sa[i] = P[i].second;
                        }
                        return;
                    }

                    // Sort P by its i position mapped to the tuple
                    // (i % (2**k), i / (2**k), implemented as a single
                    // integer value
                    sorting_algorithm::sort(util::span(P), util::compare_key(
                        [k](auto value) {
                            size_t const i = value.second;
                            auto const anti_k = util::bits_of<size_t> - k;
                            return (i << anti_k) | (i >> k);
                        }
                    ));

                    for (size_t j = 0; j < N; j++) {
                        size_t const i1 = P[j].second;

                        name_type const c1 = P[j].first;
                        name_type c2;

                        if ((j + 1) >= N) {
                            c2 = util::SENTINEL;
                        } else {
                            c2 = P[j + 1].first;

                            size_t const i2 = P[j + 1].second;

                            if (i2 != (i1 + k_length)) {
                                c2 = util::SENTINEL;
                            }
                        }
                        S[j] = std::make_pair(atuple { c1, c2 }, i1);
                    }
                }
            }
    }; // class prefix_doubling

} // namespace sacabench::prefix_doubling
