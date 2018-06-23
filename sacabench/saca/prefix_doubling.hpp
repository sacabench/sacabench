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
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::prefix_doubling {

struct std_sorting_algorithm {
    template <typename T, typename Compare = std::less<T>>
    static void sort(util::span<T> data, Compare comp = Compare()) {
        util::sort::std_sort(data, comp);
    }
};

template <typename sa_index, typename sorting_algorithm = std_sorting_algorithm>
struct prefix_doubling_impl {
    /// The type to use for lexicographical sorted "names".
    /// For discarding, is required to have a width of at least 1 bit more than
    /// needed.
    using name_type = sa_index;
    using atuple = std::array<name_type, 2>;
    using names_tuple = std::pair<atuple, sa_index>;
    using name_tuple = std::pair<name_type, sa_index>;

    /// Create a unique name for each S tuple,
    /// returning true if all names are unique.
    ///
    /// Precondition: The tuple in S are lexicographical sorted.
    inline static bool name(util::span<names_tuple> S,
                            util::span<name_tuple> P) {
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
        auto last_pair = atuple{util::SENTINEL, util::SENTINEL};
        bool only_unique = true;

        for (size_t i = 0; i < S.size(); i++) {
            auto const& pair = S[i].first;
            if (pair != last_pair) {
                name = i + 1;
                last_pair = pair;
            } else {
                only_unique = false;
            }
            P[i] = name_tuple{name, S[i].second};
        }

        return only_unique;
    }

    inline static name_type unique_mask() {
        auto mask = std::numeric_limits<name_type>::max() >> 1;
        mask += 1;
        return mask;
    }

    /// Check if the extra bit in `name_type` is set.
    inline static bool is_marked(name_type v) {
        return (v & unique_mask()) != 0;
    }

    /// Sets the extra bit in `v`, and returns the changed value.
    inline static name_type marked(name_type v) {
        v = v | unique_mask();
        DCHECK(is_marked(v));
        return v;
    }

    /// Unsets the extra bit in `v`, and returns the changed value.
    inline static name_type unmarked(name_type v) {
        v = v & ~unique_mask();
        DCHECK(!is_marked(v));
        return v;
    }

    /// Debug output, prints a span of `names_tuple`
    inline static void print_names(util::span<names_tuple> S) {
        std::cout << "[\n";
        for (auto& e : S) {
            std::cout << "  (" << (e.first[0]) << ", " << (e.first[1]) << "), "
                      << e.second << "\n";
        }
        std::cout << "]\n";
    }

    /// Debug output, prints a span of `name_tuple`, while isolating the extra
    /// bit
    inline static void print_name(util::span<name_tuple> S) {
        std::cout << "[\n";
        for (auto& e : S) {
            auto name = unmarked(e.first);
            if (is_marked(e.first)) {
                std::cout << "  * " << name << ", " << e.second << "\n";
            } else {
                std::cout << "    " << name << ", " << e.second << "\n";
            }
        }
        std::cout << "]\n";
    }

    /// Naive variant, roughly identical to description in Paper
    static void doubling(util::string_span text, util::span<sa_index> out_sa) {
        tdc::StatPhase phase("Initialization");
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
        auto S = util::make_container<names_tuple>(N);
        auto P = util::make_container<name_tuple>(N);

        // Create the initial S array of character tuples + text
        // position
        for (size_t i = 0; i < N - 1; ++i) {
            S[i] = names_tuple{atuple{text[i], text[i + 1]}, i};
        }
        S[N - 1] = names_tuple{atuple{text[N - 1], util::SENTINEL}, N - 1};

        // We iterate up to ceil(log2(N)) times - but because we
        // always have a break condition in the loop body,
        // we don't need to check for it explicitly
        for (size_t k = 1;; k++) {
            phase.split("Iteration");

            DCHECK_LE(k, util::ceil_log2(N));
            size_t const k_length = 1ull << k;

            phase.log("current_iteration", k);
            phase.log("prefix_size", k_length);

            // tdc::StatPhase loop_phase("Sort S");

            // Sort the S tuples lexicographical
            sorting_algorithm::sort(S.slice());

            // loop_phase.split("Rename S tuples");

            // Rename the S tuples into P
            bool only_unique = name(S, P);

            // loop_phase.split("Check unique");

            // The algorithm terminates if P only contains unique values
            if (only_unique) {
                phase.split("Write out result");
                for (size_t i = 0; i < N; i++) {
                    out_sa[i] = P[i].second;
                }
                return;
            }

            // loop_phase.split("Sort P tuples");

            // Sort P by its i position mapped to the tuple
            // (i % (2**k), i / (2**k), implemented as a single
            // integer value
            sorting_algorithm::sort(
                P.slice(), util::compare_key([k](auto value) {
                    size_t const i = value.second;
                    auto const anti_k = util::bits_of<size_t> - k;
                    return (i << anti_k) | (i >> k);
                }));

            // loop_phase.split("Pair names");

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
                S[j] = names_tuple{atuple{c1, c2}, i1};
            }
        }
    }

    /// Marks elements as not unique by setting the highest extra bit
    inline static void mark_not_unique(util::span<name_tuple> U) {
        for (size_t i = 1; i < U.size(); i++) {
            auto& a = U[i - 1].first;
            auto& b = U[i].first;

            if (unmarked(a) == unmarked(b)) {
                a = marked(a);
                b = marked(b);
            }
        }
    }

    /// Helper class to contain the S,U,P and F arrays.
    ///
    /// This type exists so that further optimizations could reduce the
    /// memory footprint by merging some of the arrays.
    class supf_containers {
        util::container<names_tuple> m_s;
        util::container<name_tuple> m_u;
        util::container<name_tuple> m_p;
        util::container<name_tuple> m_f;

        // TODO: Might not be needed
        util::span<names_tuple> m_s_span;
        util::span<name_tuple> m_u_span;
        util::span<name_tuple> m_p_span;
        util::span<name_tuple> m_f_span;

    public:
        inline supf_containers(size_t N) {
            m_s = util::make_container<names_tuple>(N);
            m_u = util::make_container<name_tuple>(N);
            m_p = util::make_container<name_tuple>(N);
            m_f = util::make_container<name_tuple>(N);

            m_s_span = m_s;
            m_u_span = m_u;
            m_p_span = util::span<name_tuple>();
            m_f_span = util::span<name_tuple>();
        }

        inline auto S() { return m_s_span; }
        inline auto U() { return m_u_span; }
        inline auto P() { return m_p_span; }
        inline auto F() { return m_f_span; }

        inline auto extend_u_by(size_t size) {
            auto r = m_u.slice(m_u_span.size(), m_u_span.size() + size);
            m_u_span = m_u.slice(0, m_u_span.size() + size);
            return r;
        }

        inline void reset_p() { m_p_span = util::span<name_tuple>(); }
        inline void reset_s() { m_s_span = util::span<names_tuple>(); }

        inline void resize_u(size_t size) { m_u_span = m_u.slice(0, size); }

        inline void append_f(name_tuple v) {
            m_f_span = m_f.slice(0, m_f_span.size() + 1);
            m_f_span.back() = v;
        }

        inline void append_p(name_tuple v) {
            m_p_span = m_p.slice(0, m_p_span.size() + 1);
            m_p_span.back() = v;
        }

        inline void append_s(names_tuple v) {
            m_s_span = m_s.slice(0, m_s_span.size() + 1);
            m_s_span.back() = v;
        }
    };

    /// Merges the (name,index) tuples from P into U,
    /// and ensures the combined U is sorted according to `k`.
    inline static void
    sort_U_by_index_and_merge_P_into_it(supf_containers& supf, size_t k) {

        auto P = supf.P();

        auto U_P_extra = supf.extend_u_by(P.size());

        for (size_t i = 0; i < P.size(); i++) {
            U_P_extra[i] = P[i];
        }
        supf.reset_p();

        auto U_merged = supf.U();

        // TODO: Change to just merge U_P_extra later

        // Sort <U?> by its i position mapped to the tuple
        // (i % (2**k), i / (2**k), implemented as a single
        // integer value
        sorting_algorithm::sort(U_merged, util::compare_key([k](auto value) {
                                    size_t const i = value.second;
                                    auto const anti_k =
                                        util::bits_of<size_t> - k;
                                    return (i << anti_k) | (i >> k);
                                }));
    }

    /// Get a clean (== not extra bit set) `names_tuple`
    /// of the names at `U[j]` and `U[j+1]`.
    inline static names_tuple get_next(util::span<name_tuple> U, size_t j,
                                       size_t k) {
        size_t const k_length = 1ull << k;

        name_type c1 = unmarked(U[j].first);
        name_type c2 = util::SENTINEL;

        auto i1 = U[j].second;

        if (j + 1 < U.size()) {
            auto i2 = U[j + 1].second;
            if (i2 == i1 + static_cast<sa_index>(k_length)) {
                c2 = unmarked(U[j + 1].first);
            }
        }

        return names_tuple{atuple{c1, c2}, i1};
    }

    /// Create a unique name for each S tuple.
    ///
    /// Precondition: The tuple in S are lexicographical sorted.
    inline static void name2(util::span<names_tuple> S,
                             util::span<name_tuple> U) {
        /// Names are formed by taking the name of the first tuple element,
        /// and adding a offset that increments if the second element changes.
        ///
        // Example:
        //
        // |      S                U       |
        // | ((1, 2), 0) => (1 + 0 = 1, 0) |
        // | ((1, 2), 2) => (1 + 0 = 1, 2) |
        // | ((2, 1), 1) => (2 + 0 = 2, 1) |
        // | ((2, 0), 5) => (2 + 1 = 3, 5) |

        name_type name_counter = 0;
        name_type name_offset = 0;

        auto last_pair = atuple{util::SENTINEL, util::SENTINEL};

        DCHECK_EQ(S.size(), U.size());

        for (size_t i = 0; i < S.size(); i++) {
            auto const& pair = S[i].first;

            if (pair[0] != last_pair[0]) {
                name_counter = 0;
                name_offset = 0;
                last_pair = pair;
            } else if (pair[1] != last_pair[1]) {
                name_offset = name_counter;
                last_pair[1] = pair[1];
            }
            U[i] = name_tuple{pair[0] + name_offset, S[i].second};
            name_counter += 1;
        }
    }

    /// Doubling with discarding
    static void doubling_discarding(util::string_span text,
                                    util::span<sa_index> out_sa) {
        tdc::StatPhase phase("Initialization");
        size_t const N = text.size();

        // To simplify some of the logic below,
        // we don't even try to process texts of size less than 2.
        if (N == 0) {
            return;
        } else if (N == 1) {
            out_sa[0] = 0;
            return;
        }

        // Allocate all arrays
        auto supf = supf_containers(N);

        // Create the initial S array of character tuples + text
        // position
        for (size_t i = 0; i < N - 1; ++i) {
            supf.S()[i] = names_tuple{atuple{text[i], text[i + 1]}, i};
        }
        supf.S()[N - 1] =
            names_tuple{atuple{text[N - 1], util::SENTINEL}, N - 1};

        // Sort the S tuples lexicographical
        sorting_algorithm::sort(supf.S());

        // Rename the S tuples into U
        name(supf.S(), supf.U());

        // We iterate up to ceil(log2(N)) times - but because we
        // always have a break condition in the loop body,
        // we don't need to check for it explicitly
        for (size_t k = 1;; k++) {
            phase.split("Iteration");

            DCHECK_LE(k, util::ceil_log2(N));
            size_t const k_length = 1ull << k;

            phase.log("current_iteration", k);
            phase.log("prefix_size", k_length);
            phase.log("fully_discarded", supf.F().size());
            phase.log("partially_discarded", supf.P().size());
            phase.log("remaining", supf.U().size());

            // Mark all not unique names in U by setting their extra bit.
            mark_not_unique(supf.U());

            // Merge the previous unique names from P
            // into U and sort them by index.
            sort_U_by_index_and_merge_P_into_it(supf, k);

            // Reset P, because we just emptied it.
            supf.reset_p();

            // Reset S, because we want to fill it anew
            supf.reset_s();

            // Iterate through U, either appending tuples to S
            // or to F or P.
            size_t count = 0;
            for (size_t j = 0; j < supf.U().size(); j++) {
                // Get the name at U[j], and check if its unique,
                // while also removing the extra bit in any case.
                name_type c = unmarked(supf.U()[j].first);
                bool const is_uniq = !is_marked(supf.U()[j].first);

                size_t const i = supf.U()[j].second;

                if (is_uniq) {
                    if (count < 2) {
                        // fully discard tuple
                        supf.append_f(name_tuple{c, i});
                    } else {
                        // partially discard tuple
                        supf.append_p(name_tuple{c, i});
                    }
                    count = 0;
                } else {
                    // Get neighboring names from U[j], U[j+1]
                    auto c1c2i1 = get_next(supf.U(), j, k);

                    IF_DEBUG({
                        auto c1 = c1c2i1.first[0];
                        auto i1 = c1c2i1.second;
                        DCHECK_EQ(static_cast<size_t>(c1),
                                  static_cast<size_t>(c));
                        DCHECK_EQ(static_cast<size_t>(i1),
                                  static_cast<size_t>(i));
                    })

                    supf.append_s(c1c2i1);
                    count += 1;
                }
            }

            // If S is empty, the algorithm terminates, and
            // F contains names for unique prefixes.
            if (supf.S().empty()) {
                phase.split("Write out result");
                auto F = supf.F();

                // Sort the F tuples lexicographical
                sorting_algorithm::sort(F);

                for (size_t i = 0; i < N; i++) {
                    out_sa[i] = F[i].second;
                }
                return;
            }

            // Sort the S tuples lexicographical
            sorting_algorithm::sort(supf.S());

            // Make U the same length as S
            // TODO: Make this nicer
            supf.resize_u(supf.S().size());

            // Rename S tuple into U tuple
            name2(supf.S(), supf.U());
        }
    }
};

struct prefix_doubling {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Doubling";
    static constexpr char const* DESCRIPTION =
        "In-Memory variant of Naive Prefix Doubling by R. Dementiev, J. "
        "Kärkkäinen, J. Mehnert, and P. Sanders";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        prefix_doubling_impl<sa_index>::doubling(text, out_sa);
    }
}; // struct prefix_doubling

struct prefix_doubling_discarding {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Discarding";
    static constexpr char const* DESCRIPTION =
        "In-Memory variant of Naive Doubling with Discarding by R. "
        "Dementiev, J. Kärkkäinen, J. Mehnert, and P. Sanders";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        prefix_doubling_impl<sa_index>::doubling_discarding(text, out_sa);
    }
}; // struct prefix_doubling_discarding

} // namespace sacabench::prefix_doubling
