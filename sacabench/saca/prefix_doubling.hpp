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
#include <util/macros.hpp>
#include <util/sort/std_sort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::prefix_doubling {

struct std_sorting_algorithm {
    template <typename T, typename Compare>
    SB_NO_INLINE static void sort(util::span<T> data, Compare comp) {
        util::sort::std_sort(data, comp);
    }
};

template <typename sa_index, size_t a_size>
struct a_size_helper_type {
    inline SB_FORCE_INLINE static uint64_t pow_a_k(uint64_t k) {
        return util::powi(a_size, k);
    }
    /*
    inline SB_FORCE_INLINE static bool idx_compare(uint64_t const k, sa_index a,
                                                   sa_index b) {
        uint64_t const k_length = pow_a_k(k);

        sa_index const high_a = a % k_length;
        sa_index const high_b = b % k_length;
        sa_index const low_a = a / k_length;
        sa_index const low_b = b / k_length;

        bool const high_diff = high_a < high_b;
        bool const high_eq = high_a == high_b;
        bool const low_diff = low_a < low_b;

        return high_diff | (high_eq & low_diff);
    }
    */
};

template <typename sa_index>
struct a_size_helper_type<sa_index, 2> {
    inline SB_FORCE_INLINE static uint64_t pow_a_k(uint64_t k) {
        return 1ull << k;
    }
    inline SB_FORCE_INLINE static sa_index rotate(uint64_t const k,
                                                  sa_index v) {
        auto const anti_k = util::bits_of<sa_index> - k;
        return (v << anti_k) | (v >> k);
    }
    inline SB_FORCE_INLINE static sa_index unrotate(uint64_t const k,
                                                    sa_index v) {
        auto const anti_k = util::bits_of<sa_index> - k;
        return (v >> anti_k) | (v << k);
    }
    inline SB_FORCE_INLINE static size_t index_phase(uint64_t const k,
                                                     sa_index idx) {
        sa_index k_mask = pow_a_k(k) - 1;
        return (idx & k_mask) >> (k - 1);
    }
};

template <typename sa_index>
struct a_size_helper_type<sa_index, 4> {
    inline SB_FORCE_INLINE static uint64_t pow_a_k(uint64_t k) {
        return 1ull << (k << 1);
    }
    inline SB_FORCE_INLINE static sa_index rotate(uint64_t const k,
                                                  sa_index v) {
        return a_size_helper_type<sa_index, 2>::rotate(k << 1, v);
    }
    inline SB_FORCE_INLINE static sa_index unrotate(uint64_t const k,
                                                    sa_index v) {
        return a_size_helper_type<sa_index, 2>::unrotate(k << 1, v);
    }
    inline SB_FORCE_INLINE static size_t index_phase(uint64_t const k,
                                                     sa_index idx) {
        sa_index k_mask = pow_a_k(k) - 1;
        return (idx & k_mask) >> ((k - 1) << 1);
    }
};

template <typename sa_index, size_t a_size,
          typename sorting_algorithm = std_sorting_algorithm>
struct prefix_doubling_impl {
    using a_size_helper = a_size_helper_type<sa_index, a_size>;
    static constexpr bool USE_WORDPACKING = true;
    static constexpr size_t WP_SIZE = 4;
    static constexpr bool HAS_SCRATCH_2 = a_size > 2;

    /// The type to use for lexicographical sorted "names".
    /// For discarding, is required to have a width of at least 1 bit more than
    /// needed.
    using name_type = sa_index;
    using atuple = std::array<name_type, a_size>;

    inline static atuple make_sentinel_tuple() {
        auto a = atuple();
        for (auto& e : a) {
            e = util::SENTINEL;
        }
        return a;
    }

    inline static sa_index rotate(size_t k, sa_index v) {
        return a_size_helper::rotate(k, v);
    }

    inline static sa_index unrotate(size_t k, sa_index v) {
        return a_size_helper::unrotate(k, v);
    }

    class hybrid_tuple {
        std::array<sa_index, a_size + 1> m_data;

    public:
        inline hybrid_tuple() = default;

        inline sa_index& idx() { return m_data[a_size]; }
        inline name_type& name() { return m_data[0]; }
        inline name_type& scratch1() { return m_data[1]; }
        inline name_type& scratch2() {
            DCHECK_GT(a_size, 2);
            return m_data[2];
        }
        inline util::span<name_type> names() {
            return util::span<name_type>(m_data).slice(0, a_size);
        }

        inline sa_index const& idx() const { return m_data[a_size]; }
        inline name_type const& name() const { return m_data[0]; }
        inline name_type const& scratch1() const { return m_data[1]; }
        inline name_type const& scratch2() const {
            DCHECK_GT(a_size, 2);
            return m_data[2];
        }
        inline util::span<name_type const> names() const {
            return util::span<name_type const>(m_data).slice(0, a_size);
        }
    };

    /// Hybrid array. Each entry stores either a `(atuple, idx)` or a `(name,
    /// idx)` tuple.
    class HArray {
        util::container<hybrid_tuple> m_hybrids;

    public:
        inline HArray(size_t N) {
            m_hybrids = util::make_container<hybrid_tuple>(N);
        }
        inline size_t size() const { return m_hybrids.size(); }
        inline util::span<hybrid_tuple> hybrids() { return m_hybrids; }

        inline util::span<name_type> s_names(size_t i) {
            return m_hybrids[i].names();
        }
        inline sa_index& s_idx(size_t i) { return m_hybrids[i].idx(); }

        inline name_type& p_name(size_t i) { return m_hybrids[i].name(); }
        inline sa_index& p_idx(size_t i) { return m_hybrids[i].idx(); }
    };

    /// Create a unique name for each S tuple,
    /// returning true if all names are unique.
    ///
    /// Precondition: The tuple in S are lexicographical sorted.
    inline static bool rename_inplace_and_rotate_idx(util::span<hybrid_tuple> H,
                                                     size_t k) {
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
        auto last_pair_arr = make_sentinel_tuple();
        auto last_pair = util::span(last_pair_arr);
        bool only_unique = true;

        for (size_t i = 0; i < H.size(); i++) {
            auto pair = H[i].names();
            if (pair != last_pair) {
                name = i + 1;
                last_pair.copy_from(pair);
            } else {
                only_unique = false;
            }
            H[i].name() = name;
            H[i].idx() = rotate(k, unrotate(k - 1, H[i].idx()));
        }

        return only_unique;
    }

    inline SB_FORCE_INLINE static bool
    names_less(util::span<name_type const> a, util::span<name_type const> b) {
        DCHECK_EQ(a.size(), a_size);
        DCHECK_EQ(b.size(), a_size);
        return a.slice(0, a_size) < b.slice(0, a_size);
    }

    /// Naive variant, roughly identical to description in Paper
    static void doubling(util::string_span text, util::span<sa_index> out_sa) {
        tdc::StatPhase::log("A_Tuple", a_size);
        tdc::StatPhase phase("Initialization");
        size_t const N = text.size();

        // To simplify some of the logic below,
        // we don't even try to process texts of size less than 1.
        if (N == 0) {
            return;
        }

        // Allocate the hybrid array.
        auto h = HArray(N);

        // How many bytes to initially pack into a word
        // TODO: Only works with power-of-two sizes
        size_t wp_initial_loop_offset = 0;
        if constexpr (USE_WORDPACKING) {
            wp_initial_loop_offset = WP_SIZE / a_size;
        }

        // Create the initial S array of character tuples + text
        // position
        for (size_t i = 0; i < N; ++i) {
            auto p = make_sentinel_tuple();

            if constexpr (USE_WORDPACKING) {
                for (size_t j = 0; j < a_size * WP_SIZE; j++) {
                    uint64_t v = p[j / WP_SIZE];
                    v <<= 8;
                    if ((i + j) < N) {
                        v |= text[i + j];
                    }
                    p[j / WP_SIZE] = v;
                }
            } else {
                for (size_t j = 0; j < a_size; j++) {
                    if ((i + j) < N) {
                        p[j] = text[i + j];
                    }
                }
            }

            h.s_names(i).copy_from(p);
            h.s_idx(i) = rotate(1 + wp_initial_loop_offset - 1, i);
        }

        // We iterate up to ceil(log_a(N)) times - but because we
        // always have a break condition in the loop body,
        // we don't need to check for it explicitly
        for (size_t k = 1 + wp_initial_loop_offset;; k++) {
            phase.split("Iteration");

            // TODO: check new condition
            // DCHECK_LE(k, util::ceil_log_base(N, a_size));
            size_t const k_length = a_size_helper::pow_a_k(k);

            phase.log("current_iteration", k);
            phase.log("prefix_size", k_length);

            // tdc::StatPhase loop_phase("Sort S");

            // Sort the S tuples lexicographical
            sorting_algorithm::sort(
                h.hybrids(), [k](auto const& a, auto const& b) SB_FORCE_INLINE {
                    return names_less(a.names(), b.names());
                });

            // loop_phase.split("Rename S tuples");

            // Rename the S tuples into P
            bool only_unique = rename_inplace_and_rotate_idx(h.hybrids(), k);

            // loop_phase.split("Check unique");

            // The algorithm terminates if P only contains unique values
            if (only_unique) {
                phase.split("Write out result");
                phase.log("current_iteration", k);
                phase.log("prefix_size", k_length);

                for (size_t i = 0; i < N; i++) {
                    out_sa[i] = unrotate(k, h.p_idx(i));
                }
                return;
            }

            // loop_phase.split("Sort P tuples");

            // Sort P by its i position mapped to the tuple
            // (i % (2**k), i / (2**k), implemented as a single
            // integer value
            sorting_algorithm::sort(
                h.hybrids(), [k](auto const& a, auto const& b)
                                 SB_FORCE_INLINE { return a.idx() < b.idx(); });

            // loop_phase.split("Pair names");

            for (size_t j = 0; j < N; j++) {
                auto t = make_sentinel_tuple();
                t[0] = h.p_name(j);

                auto last_idx = h.p_idx(j);

                size_t trunc_size = std::min(a_size, N - j);
                for (size_t l = 1; l < trunc_size; l++) {
                    auto const idx = h.p_idx(j + l);

                    if (unrotate(k, idx) ==
                        uint64_t(unrotate(k, last_idx)) + k_length) {
                        t[l] = h.p_name(j + l);
                        last_idx = idx;
                    } else {
                        break;
                    }
                }

                h.s_names(j).copy_from(t);
            }
        }
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

    /// Helper class to contain the logical S,U,P and F arrays.
    ///
    /// It provides different views to the underlying allocation, depending
    /// on which phase in the loop it is in.
    ///
    /// phase 0: initialization
    /// [                       H(S)                                           ]
    ///
    /// phase 1: merging P and H(U) from last iteration
    /// [     P          |      H(U)                               |     F     ]
    /// [                 H(U)                                     |     F     ]
    ///
    /// phase 2: iterate over H(U) and split into P, H(S) and F
    /// [next P | next S | next F |         H(U)                   |     F     ]
    ///         >        >        >
    ///
    /// phase 3: process H(S) to H(U), or end iteration with F
    /// [     P          |      H(S)                               |     F     ]
    /// [     P          |      H(U)                               |     F     ]
    /// or
    /// [                            F                                         ]
    class DiscardingHArray {
        util::container<hybrid_tuple> m_disc_h;

    public:
        inline DiscardingHArray(size_t N) {
            m_disc_h = util::make_container<hybrid_tuple>(N);
        }

        inline auto phase_0_H() { return m_disc_h.slice(); }
        struct phase_1_PU_type {
            util::span<hybrid_tuple> m_u;
            util::span<hybrid_tuple> m_p;
            util::span<hybrid_tuple> m_pu;
            inline auto U() { return m_u; }
            inline auto P() { return m_p; }
            inline auto PU() { return m_pu; }
        };
        inline auto phase_1_PU(size_t P_size, size_t F_size) {
            auto PU = m_disc_h.slice(0, m_disc_h.size() - F_size);
            auto P = PU.slice(0, P_size);
            auto U = PU.slice(P_size);

            return phase_1_PU_type{U, P, PU};
        }
        class phase_2_U2PSF_type {
            util::span<hybrid_tuple> m_disc_h;

            size_t m_U_start = 0;
            size_t m_U_end = 0;

            size_t m_P_end = 0;
            size_t m_S_end = 0;
            size_t m_F_end = 0;

            inline auto p_span() { return m_disc_h.slice(0, m_P_end); }
            inline auto s_span() { return m_disc_h.slice(m_P_end, m_S_end); }
            inline auto f_span() { return m_disc_h.slice(m_S_end, m_F_end); }
            inline auto u_span() { return m_disc_h.slice(m_U_start, m_U_end); }

        public:
            inline phase_2_U2PSF_type(util::span<hybrid_tuple> disc_h,
                                      size_t F_size) {
                m_disc_h = disc_h;
                m_U_end = disc_h.size() - F_size;
            }

            inline bool has_next_u_elem() { return m_U_start != m_U_end; }
            inline hybrid_tuple const& get_next_u_elem() {
                return m_disc_h[m_U_start];
            }
            inline util::span<hybrid_tuple> get_u_elems_after_next(size_t n) {
                size_t start = std::min(m_U_start + 1, m_U_end);
                size_t end = std::min(m_U_start + 1 + n, m_U_end);
                return m_disc_h.slice(start, end);
            }
            inline void drop_u_elem() { m_U_start++; }
            inline void append_f(sa_index name, sa_index idx) {
                auto& new_f_elem = m_disc_h[m_F_end];
                new_f_elem.name() = name;
                new_f_elem.idx() = idx;
                m_F_end++;
            }
            inline void append_s(util::span<sa_index> names, sa_index idx) {
                m_disc_h[m_F_end] = m_disc_h[m_S_end];
                m_F_end++;

                auto& new_s_elem = m_disc_h[m_S_end];
                new_s_elem.names().copy_from(names);
                new_s_elem.idx() = idx;
                m_S_end++;
            }
            inline void append_p(sa_index name, sa_index idx) {
                m_disc_h[m_F_end] = m_disc_h[m_S_end];
                m_F_end++;
                m_disc_h[m_S_end] = m_disc_h[m_P_end];
                m_S_end++;

                auto& new_p_elem = m_disc_h[m_P_end];
                new_p_elem.name() = name;
                new_p_elem.idx() = idx;
                m_P_end++;
            }
            inline size_t additional_f_size() { return m_F_end - m_S_end; }
            inline size_t p_size() { return m_P_end; }
        };
        inline auto phase_2_U2PSF(size_t F_size) {
            return phase_2_U2PSF_type(m_disc_h, F_size);
        }
        struct phase_3_PSF_type {
            util::span<hybrid_tuple> m_p;
            util::span<hybrid_tuple> m_s;
            util::span<hybrid_tuple> m_f;
            inline auto P() { return m_p; }
            inline auto S() { return m_s; }
            inline auto F() { return m_f; }
        };
        inline auto phase_3_PSF(size_t P_size, size_t F_size) {
            auto P = m_disc_h.slice(0, P_size);
            auto S = m_disc_h.slice(P_size, m_disc_h.size() - F_size);
            auto F = m_disc_h.slice(m_disc_h.size() - F_size);

            return phase_3_PSF_type{P, S, F};
        }
    };

    class p_bucket_sizes {
        std::array<size_t, a_size> m_bucket_sizes;
        bool m_finalized = false;

    public:
        inline p_bucket_sizes() { reset(); }
        inline void reset() {
            for (auto& e : m_bucket_sizes) {
                e = 0;
            }
            m_finalized = false;
        }
        inline void count(size_t idx) {
            DCHECK_EQ(m_finalized, false);
            DCHECK_LT(idx, a_size);
            m_bucket_sizes[idx]++;
        }
        inline void finalize() {
            DCHECK_EQ(m_finalized, false);
            for (size_t i = 1; i < a_size; i++) {
                m_bucket_sizes[i] += m_bucket_sizes[i - 1];
            }

            for (size_t i = a_size - 1; i > 0; i--) {
                m_bucket_sizes[i] = m_bucket_sizes[i - 1];
            }
            m_bucket_sizes[0] = 0;
            m_finalized = true;
        }
        inline size_t insert(size_t idx) {
            DCHECK_EQ(m_finalized, true);
            return m_bucket_sizes[idx]++;
        }
    };

    /// Merges the (name,index) tuples from P into U,
    /// and ensures the combined U is sorted according to `k`.
    template <typename pu_type>
    inline static void
    sort_U_by_index_and_merge_P_into_it(pu_type& pu, size_t k,
                                        p_bucket_sizes const& Pbs) {
        // Bucket-sort the entries in P for the k+1 order
        {
            auto P = pu.P();

            auto merge = [=](auto assign) {
                auto bs = Pbs;
                for (size_t i = 0; i < P.size(); i++) {
                    auto idx = a_size_helper::index_phase(k, P[i].idx());
                    assign(P[bs.insert(idx)], P[i]);
                }
            };

            if constexpr (HAS_SCRATCH_2) {
                merge([](auto& dst, auto& src) {
                    dst.scratch1() = src.name();
                    dst.scratch2() = src.idx();
                });

                for (size_t i = 0; i < P.size(); i++) {
                    P[i].idx() = rotate(k, P[i].scratch2());
                    P[i].name() = P[i].scratch1();
                }
            } else {
                merge(
                    [](auto& dst, auto& src) { dst.scratch1() = src.name(); });
                merge([](auto& dst, auto& src) { dst.name() = src.idx(); });

                for (size_t i = 0; i < P.size(); i++) {
                    P[i].idx() = rotate(k, P[i].name());
                    P[i].name() = P[i].scratch1();
                }
            }
        }

        // Sort U
        sorting_algorithm::sort(pu.U(), [](auto const& a, auto const& b) {
            return a.idx() < b.idx();
        });

        // Merge the adjacent P and U arrays in place
        {
            auto PU = pu.PU();
            size_t const p_end = pu.P().size();
            size_t const u_end = PU.size();

            auto merge = [PU, p_end, u_end](auto assign) {
                size_t pi = 0;
                size_t ui = p_end;

                size_t i = 0;
                while ((pi < p_end) && (ui < u_end)) {
                    if (PU[pi].idx() < PU[ui].idx()) {
                        assign(PU[i++], PU[pi++]);
                    } else {
                        assign(PU[i++], PU[ui++]);
                    }
                }
                while (pi < p_end) {
                    assign(PU[i++], PU[pi++]);
                }
                while (ui < u_end) {
                    assign(PU[i++], PU[ui++]);
                }
            };

            // For a_size = 2, we only have a single field as scratch space,
            // so we need to run the merge step twice.
            // for a_size > 2, we have two free fields, and can do it in one
            // iteration.
            if constexpr (HAS_SCRATCH_2) {
                merge([](auto& dst, auto& src) {
                    dst.scratch1() = src.name();
                    dst.scratch2() = src.idx();
                });

                for (size_t i = 0; i < PU.size(); i++) {
                    PU[i].idx() = unrotate(k, PU[i].scratch2());
                    PU[i].name() = PU[i].scratch1();
                }
            } else {
                merge(
                    [](auto& dst, auto& src) { dst.scratch1() = src.name(); });
                merge([](auto& dst, auto& src) { dst.name() = src.idx(); });

                for (size_t i = 0; i < PU.size(); i++) {
                    PU[i].idx() = unrotate(k, PU[i].name());
                    PU[i].name() = PU[i].scratch1();
                }
            }
        }
    }

    inline static void
    rename_inplace_and_mark_non_unique(util::span<hybrid_tuple> H) {
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

        name_type name = 0;
        auto last_pair_arr = make_sentinel_tuple();
        auto last_pair = util::span(last_pair_arr);

        for (size_t i = 0; i < H.size(); i++) {
            bool duplicate = false;
            auto pair = H[i].names();
            if (pair != last_pair) {
                name = i + 1;
                last_pair.copy_from(pair);
            } else {
                duplicate = true;
            }
            H[i].name() = name;
            if (duplicate) {
                auto& a = H[i - 1].name();
                auto& b = H[i].name();

                a = marked(a);
                b = marked(b);
            }
        }
    }

    /// Create a unique name for each S tuple.
    ///
    /// Precondition: The tuple in S are lexicographical sorted.
    inline static void
    rename2_inplace_and_mark_non_unique(util::span<hybrid_tuple> H) {
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

        auto last_pair_arr = make_sentinel_tuple();
        auto last_pair = util::span(last_pair_arr);

        for (size_t i = 0; i < H.size(); i++) {
            bool duplicate = false;
            auto pair = H[i].names();

            if (pair[0] != last_pair[0]) {
                name_counter = 0;
                name_offset = 0;
                last_pair.copy_from(pair);
            } else if (pair.slice(1) != last_pair.slice(1)) {
                name_offset = name_counter;
                last_pair.copy_from(pair);
            } else {
                duplicate = true;
            }
            H[i].name() = pair[0] + name_offset;
            name_counter += 1;
            if (duplicate) {
                auto& a = H[i - 1].name();
                auto& b = H[i].name();

                a = marked(a);
                b = marked(b);
            }
        }
    }

    /// Doubling with discarding
    static void doubling_discarding(util::string_span text,
                                    util::span<sa_index> out_sa) {
        tdc::StatPhase::log("A_Tuple", a_size);
        tdc::StatPhase phase("Initialization");
        size_t const N = text.size();

        // To simplify some of the logic below,
        // we don't even try to process texts of size less than 2.
        if (N == 0) {
            return;
        }

        // Allocate all logical arrays
        auto disc_h = DiscardingHArray(N);

        // How many bytes to initially pack into a word
        // TODO: Only works with power-of-two sizes
        size_t wp_initial_loop_offset = 0;
        if constexpr (USE_WORDPACKING) {
            wp_initial_loop_offset = WP_SIZE / a_size;
        }

        {
            auto H = disc_h.phase_0_H();

            // Create the initial S array of character tuples + text
            // position
            for (size_t i = 0; i < N; ++i) {
                auto p = make_sentinel_tuple();

                if constexpr (USE_WORDPACKING) {
                    for (size_t j = 0; j < a_size * WP_SIZE; j++) {
                        uint64_t v = p[j / WP_SIZE];
                        v <<= 8;
                        if ((i + j) < N) {
                            v |= text[i + j];
                        }
                        p[j / WP_SIZE] = v;
                    }
                } else {
                    for (size_t j = 0; j < a_size; j++) {
                        if ((i + j) < N) {
                            p[j] = text[i + j];
                        }
                    }
                }

                H[i].names().copy_from(p);
                H[i].idx() = rotate(1 + wp_initial_loop_offset, i);
            }

            // Sort the S tuples lexicographical
            sorting_algorithm::sort(
                H, [&](auto const& a, auto const& b) SB_FORCE_INLINE {
                    return names_less(a.names(), b.names());
                });

            // Rename the S tuples into U
            rename_inplace_and_mark_non_unique(H);
        }

        size_t P_size = 0;
        size_t F_size = 0;
        auto Pbs = p_bucket_sizes();

        // We iterate up to ceil(log_a(N)) times - but because we
        // always have a break condition in the loop body,
        // we don't need to check for it explicitly
        for (size_t k = 1 + wp_initial_loop_offset;; k++) {
            phase.split("Iteration");

            // TODO: check new condition
            // DCHECK_LE(k, util::ceil_log_base(N, a_size));
            size_t const k_length = a_size_helper::pow_a_k(k);

            phase.log("current_iteration", k);
            phase.log("prefix_size", k_length);
            phase.log("fully_discarded", F_size);
            phase.log("partially_discarded", P_size);
            phase.log("remaining", N - (F_size + P_size));

            {
                auto UP = disc_h.phase_1_PU(P_size, F_size);

                // Mark all not unique names in U by setting their extra bit.
                // NB: Happens automatically during renaming now.
                // mark_not_unique(UP.U());

                // Merge the previous unique names from P
                // into U and sort them by index.
                sort_U_by_index_and_merge_P_into_it(UP, k, Pbs);
                Pbs.reset();
            }

            auto U2PSF = disc_h.phase_2_U2PSF(F_size);

            // Iterate through U, either appending tuples to S
            // or to F or P.
            size_t count = 0;
            while (U2PSF.has_next_u_elem()) {
                // Get the name at U[j], and check if its unique,
                // while also removing the extra bit in any case.
                auto const& next_u_elem = U2PSF.get_next_u_elem();

                name_type c = unmarked(next_u_elem.name());
                bool const is_uniq = !is_marked(next_u_elem.name());
                auto const i = next_u_elem.idx();

                if (is_uniq) {
                    U2PSF.drop_u_elem();

                    if (count < 2) {
                        // fully discard tuple
                        U2PSF.append_f(c, i);
                    } else {
                        // partially discard tuple
                        U2PSF.append_p(c, i);

                        Pbs.count(a_size_helper::index_phase(k + 1, i));
                    }
                    count = 0;
                } else {
                    // Get neighboring names U[j], U[j+1], ...

                    auto tuple = make_sentinel_tuple();
                    tuple[0] = c;

                    {
                        auto neighbor_names =
                            U2PSF.get_u_elems_after_next(tuple.size() - 1);

                        auto last_i = i;
                        for (size_t l = 0; l < neighbor_names.size(); l++) {
                            auto const& n = neighbor_names[l];

                            if (n.idx() == uint64_t(last_i) + k_length) {
                                tuple[l + 1] = unmarked(n.name());
                                last_i = n.idx();
                            } else {
                                break;
                            }
                        }
                    }

                    U2PSF.drop_u_elem();
                    U2PSF.append_s(tuple, rotate(k + 1, i));
                    count += 1;
                }
            }

            F_size += U2PSF.additional_f_size();
            P_size = U2PSF.p_size();
            Pbs.finalize();

            auto PSF = disc_h.phase_3_PSF(P_size, F_size);

            // If S is empty, the algorithm terminates, and
            // F contains names for unique prefixes.
            if (PSF.S().empty()) {
                phase.split("Write out result");
                phase.log("current_iteration", k);
                phase.log("prefix_size", k_length);
                phase.log("fully_discarded", F_size);
                phase.log("partially_discarded", P_size);
                phase.log("remaining", N - (F_size + P_size));
                auto F = PSF.F();

                // Sort the F tuples lexicographical
                sorting_algorithm::sort(
                    F, [](auto const& a, auto const& b)
                           SB_FORCE_INLINE { return a.name() < b.name(); });

                for (size_t i = 0; i < N; i++) {
                    out_sa[i] = F[i].idx();
                }
                return;
            }

            // Sort the S tuples lexicographical
            sorting_algorithm::sort(
                PSF.S(), [](auto const& a, auto const& b) SB_FORCE_INLINE {
                    return names_less(a.names(), b.names());
                });

            // Rename S tuple into U tuple
            rename2_inplace_and_mark_non_unique(PSF.S());
        }
    }
};

struct prefix_doubling {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Doubling";
    static constexpr char const* DESCRIPTION =
        "In-Memory variant of Prefix Doubling by R. Dementiev, J. "
        "Kärkkäinen, J. Mehnert, and P. Sanders";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        prefix_doubling_impl<sa_index, 2>::doubling(text, out_sa);
    }
};

struct prefix_discarding_2 {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Discarding2";
    static constexpr char const* DESCRIPTION =
        "In-Memory variant of 2-Tupling with Discarding by R. "
        "Dementiev, J. Kärkkäinen, J. Mehnert, and P. Sanders";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        prefix_doubling_impl<sa_index, 2>::doubling_discarding(text, out_sa);
    }
};

struct prefix_discarding_4 {
    static constexpr size_t EXTRA_SENTINELS = 0;
    static constexpr char const* NAME = "Discarding4";
    static constexpr char const* DESCRIPTION =
        "In-Memory variant of 4-Tupling with Discarding by R. "
        "Dementiev, J. Kärkkäinen, J. Mehnert, and P. Sanders";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& /*alphabet_size*/,
                             util::span<sa_index> out_sa) {
        prefix_doubling_impl<sa_index, 4>::doubling_discarding(text, out_sa);
    }
};

} // namespace sacabench::prefix_doubling
