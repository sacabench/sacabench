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

using name_type = size_t;
using atuple = std::array<name_type, 2>;
using names_tuple = std::pair<atuple, size_t>;
using name_tuple = std::pair<name_type, size_t>;

/// Create a unique name for each S tuple,
/// returning true if all names are unique.
///
/// Precondition: The tuple in S are lexicographical sorted.
inline static bool name(util::span<names_tuple> S, util::span<name_tuple> P) {
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

inline static bool is_marked(name_type v) { return (v & unique_mask()) != 0; }

inline static name_type set_marked(name_type v) {
    v |= unique_mask();
    DCHECK(is_marked(v));
    return v;
}

inline static name_type unset_marked(name_type v) {
    v &= ~unique_mask();
    DCHECK(!is_marked(v));
    return v;
}

inline void print_names(util::span<names_tuple> S) {
    std::cout << "[\n";
    for (auto& e : S) {
        std::cout << "  (" << (e.first[0]) << ", " << (e.first[1]) << "), "
                  << e.second << "\n";
    }
    std::cout << "]\n";
}

inline void print_name(util::span<name_tuple> S) {
    std::cout << "[\n";
    for (auto& e : S) {
        auto name = unset_marked(e.first);
        if (is_marked(e.first)) {
            std::cout << "  * " << name << ", " << e.second << "\n";
        } else {
            std::cout << "    " << name << ", " << e.second << "\n";
        }
    }
    std::cout << "]\n";
}

/// Naive variant, roughly identical to description in Paper
template <typename sorting_algorithm = std_sorting_algorithm>
class prefix_doubling {
public:
    template <typename sa_index>
    static void construct_sa(util::string_span text, size_t /*alphabet_size*/,
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
        auto S = util::make_container<names_tuple>(N);
        auto P = util::make_container<name_tuple>(N);

        // Create the initial S array of character tuples + text
        // position
        for (size_t i = 0; i < N - 1; ++i) {
            S[i] = std::make_pair(atuple{text[i], text[i + 1]}, i);
        }
        S[N - 1] = std::make_pair(atuple{text[N - 1], util::SENTINEL}, N - 1);

        // We iterate up to ceil(log2(N)) times - but because we
        // always have a break condition in the loop body,
        // we don't need to check for it explicitly
        for (size_t k = 1;; k++) {
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
            sorting_algorithm::sort(
                util::span(P), util::compare_key([k](auto value) {
                    size_t const i = value.second;
                    auto const anti_k = util::bits_of<size_t> - k;
                    return (i << anti_k) | (i >> k);
                }));

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
                S[j] = std::make_pair(atuple{c1, c2}, i1);
            }
        }
    }
}; // class prefix_doubling

/// Marks elements as not unique by setting the highest extra bit
inline static void mark_not_unique(util::span<name_tuple> U) {
    for (size_t i = 1; i < U.size(); i++) {
        auto& a = U[i - 1].first;
        auto& b = U[i].first;

        if (unset_marked(a) == unset_marked(b)) {
            a = set_marked(a);
            b = set_marked(b);
        }
    }
}

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
        auto r = util::span(m_u).slice(m_u_span.size(), m_u_span.size() + size);
        m_u_span = util::span(m_u).slice(0, m_u_span.size() + size);
        return r;
    }

    inline void reset_p() { m_p_span = util::span<name_tuple>(); }

    inline void reset_s() { m_s_span = util::span<names_tuple>(); }

    inline void reset_u() { m_u_span = util::span<name_tuple>(); }

    inline void append_f(name_tuple v) {
        m_f_span = util::span(m_f).slice(0, m_f_span.size() + 1);
        m_f_span.back() = v;
    }

    inline void append_p(name_tuple v) {
        m_p_span = util::span(m_p).slice(0, m_p_span.size() + 1);
        m_p_span.back() = v;
    }

    inline void append_s(names_tuple v) {
        m_s_span = util::span(m_s).slice(0, m_s_span.size() + 1);
        m_s_span.back() = v;
    }
};

template <typename sorting_algorithm>
inline static void sort_U_by_index_and_merge_P_into_it(supf_containers& supf,
                                                       size_t k) {

    auto P = supf.P();
    auto U_P_extra = supf.extend_u_by(P.size());

    for (size_t i = 0; i < P.size(); i++) {
        U_P_extra[i] = P[i];
    }
    supf.reset_p();

    auto U_merged = supf.U();

    /*
    // TODO: Check which one need actual unsetting
    for(size_t i = 0; i < U_merged.size(); i++) {
        U_merged[i].first = unset_marked(U_merged[i].first);
    }
    */

    // TODO: Change to just merge U_P_extra later

    // Sort <U?> by its i position mapped to the tuple
    // (i % (2**k), i / (2**k), implemented as a single
    // integer value
    sorting_algorithm::sort(U_merged, util::compare_key([k](auto value) {
                                size_t const i = value.second;
                                auto const anti_k = util::bits_of<size_t> - k;
                                return (i << anti_k) | (i >> k);
                            }));
}

inline static names_tuple get_next(util::span<name_tuple> U, size_t j,
                                   size_t k_length) {
    name_type c1 = unset_marked(U[j].first);
    name_type c2 = util::SENTINEL;

    auto i1 = U[j].second;

    if (j + 1 < U.size()) {
        auto i2 = U[j + 1].second;
        if (i2 == i1 + k_length) {
            c2 = unset_marked(U[j + 1].first);
        }
    }

    return names_tuple{atuple{c1, c2}, i1};
}

inline static void name2(util::span<names_tuple> S, util::span<name_tuple> U) {
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
template <typename sorting_algorithm = std_sorting_algorithm>
class prefix_doubling_discarding {
public:
    template <typename sa_index>
    static void construct_sa(util::string_span text, size_t /*alphabet_size*/,
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

        // Allocate all arrays
        auto supf = supf_containers(N);

        // Create the initial S array of character tuples + text
        // position
        for (size_t i = 0; i < N - 1; ++i) {
            supf.S()[i] = std::make_pair(atuple{text[i], text[i + 1]}, i);
        }
        supf.S()[N - 1] =
            std::make_pair(atuple{text[N - 1], util::SENTINEL}, N - 1);

        // Sort the S tuples lexicographical
        sorting_algorithm::sort(supf.S());

        // Rename the S tuples into U
        name(supf.S(), supf.U());

        // We iterate up to ceil(log2(N)) times - but because we
        // always have a break condition in the loop body,
        // we don't need to check for it explicitly
        for (size_t k = 1;; k++) {
            DCHECK_LE(k, util::ceil_log2(N));

            size_t const k_length = 1ull << k;

            mark_not_unique(supf.U());

            sort_U_by_index_and_merge_P_into_it<sorting_algorithm>(supf, k);

            supf.reset_s();
            supf.reset_p();

            size_t count = 0;
            for (size_t j = 0; j < supf.U().size(); j++) {
                name_type c = supf.U()[j].first;
                bool const is_uniq = !is_marked(c);
                c = unset_marked(c);

                size_t const i = supf.U()[j].second;

                if (is_uniq) {
                    auto ci_tuple = std::make_pair(c, i);

                    if (count < 2) {
                        supf.append_f(ci_tuple);
                    } else {
                        supf.append_p(ci_tuple);
                    }

                    count = 0;
                } else {
                    auto c1c2i1 = get_next(supf.U(), j, k_length);

                    auto c1 = c1c2i1.first[0];
                    auto i1 = c1c2i1.second;
                    DCHECK_EQ(c1, c);
                    DCHECK_EQ(i1, i);

                    supf.append_s(c1c2i1);
                    count += 1;
                }
            }
            if (supf.S().empty()) {
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

            supf.reset_u();
            supf.extend_u_by(supf.S().size());
            name2(supf.S(), supf.U());
        }
    }
}; // class prefix_doubling_discarding

} // namespace sacabench::prefix_doubling
