/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::sads {
using namespace sacabench::util;
class sads {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "SADS";
    static constexpr char const* DESCRIPTION =
        "Suffix Array D-Critical Sorting by Nong, Zhang and Chan";

    static const size_t L_Type = 0;
    static const size_t S_Type = 1;

    template <typename T>
    static void compute_types(T s, span<bool> t) {
        t[s.size() - 1] = S_Type;

        for (ssize i = s.size() - 2; i >= 0; i--) {
            if (s[i + 1] < s[i]) {
                t[i] = L_Type;
            } else if (s[i + 1] > s[i]) {
                t[i] = S_Type;
            } else {
                t[i] = t[i + 1];
            }
        }
    }

    template <typename T, typename sa_index>
    static size_t compute_dc(T s, span<bool> t, span<sa_index> p_1, size_t d) {
        ssize i = -1;
        size_t j = 0;
        while (i < static_cast<ssize>(s.size() - 1)) {
            size_t h, isLMS = 0;
            for (h = 2; h <= d + 1; h++) {
                if (t[i + h - 1] == L_Type && t[i + h] == S_Type) {
                    isLMS = 1;
                    break;
                }
            }

            if (j == 0 && !isLMS) {
                i += d;
                continue;
            }
            // next dc
            i = (isLMS) ? i + h : i + d;
            p_1[j++] = i;
        }

        return j;
    }

    template <typename T>
    static void generate_buckets(T s, span<size_t> buckets, size_t K,
                                 bool end) {
        size_t sum = 0;

        for (size_t i = 0; i <= K; i++) {
            buckets[i] = 0;
        }

        // bucket size for each char
        for (size_t i = 0; i < s.size(); i++) {
            buckets[s.at(i)]++;
        }

        // sum up to bucket ends
        for (size_t i = 0; i <= K; i++) {
            sum += buckets[i];
            buckets[i] = (end ? sum : sum - buckets[i]);
        }
    }

    template <typename T, typename sa_index>
    static void induce_L_Types(T s, span<size_t> buckets, size_t K, bool end,
                               span<sa_index> SA, span<bool> t) {
        generate_buckets(s, buckets, K, end);
        for (size_t i = 0; i < s.size(); i++) {
            ssize pre_index =
                SA[i] - (sa_index)1; // pre index of the ith suffix array position

            if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 &&
                t[pre_index] == L_Type) { // pre index is type L
                SA[buckets[s.at(pre_index)]++] =
                    pre_index; // "sort" index in the bucket
            }
        }
    }

    template <typename T, typename sa_index>
    static void induce_S_Types(T s, span<size_t> buckets, size_t K, bool end,
                               span<sa_index> SA, span<bool> t) {
        generate_buckets(s, buckets, K, end);
        for (ssize i = s.size() - 1; i >= 0; i--) {
            ssize pre_index = SA[i] - (sa_index)1;

            if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 && t[pre_index] == S_Type) {
                SA[--buckets[s.at(pre_index)]] = pre_index;
            }
        }
    }

    template <typename T>
    static ssize get_omega_weight(T s, span<bool> t, size_t pos) {
        return ((size_t)(s[pos] * 2 + t[pos]));
    }

    template <typename T, typename sa_index>
    static void bucket_sort_LS(span<sa_index> source, span<sa_index> dest, T s,
                               span<bool> t, size_t n1, size_t h) {
        size_t c[] = {0, n1 - 1};
        size_t j = 0;

        for (size_t i = 0; i < n1; i++) {
            j = source[i] + (sa_index)h;
            if (j > s.size() - 1) {
                j = s.size() - 1;
            }

            if (t[j] == S_Type) {
                dest[c[1]--] = source[i];
            } else {
                dest[c[0]++] = source[i];
            }
        }
    }

    template <typename T, typename sa_index>
    static void bucket_sort(span<sa_index> src, span<sa_index> dst, T s, size_t n1,
                            size_t K, span<size_t> buckets, size_t d) {
        size_t sum = 0;
        ssize j = 0;

        for (size_t i = 0; i < (K + 1); i++) {
            buckets[i] = 0;
        }
        for (size_t i = 0; i < n1; i++) {
            if ((j = src[i] + (sa_index)d) > static_cast<ssize>(s.size() - 1)) {
                j = s.size() - 1;
            }
            buckets[s[j]]++;
        }
        for (size_t i = 0; i < (K + 1); i++) {
            size_t len = buckets[i];
            buckets[i] = sum;
            sum += len;
        }
        for (size_t i = 0; i < n1; i++) {
            if ((j = src[i] + (sa_index)d) > static_cast<ssize>(s.size() - 1)) {
                j = s.size() - 1;
            }

            dst[buckets[s[j]]++] = src[i];
        }
    }

    template <typename T, typename sa_index>
    static void run_saca(span<T> s, span<sa_index> SA, size_t K) {

        container<bool> t = make_container<bool>(s.size());
        container<size_t> buckets = make_container<size_t>(K + 1);

        // fixed d = 3 ... will change when optimizing algorithm
        compute_types(s, t);

        size_t n1 = compute_dc<span<T>, sa_index>(s, t, SA, 3);


        span<sa_index> s1 = SA.slice(s.size() - n1, s.size());

        size_t j = 0;

        // pre sort based on character types
        bucket_sort_LS<span<T>, sa_index>(SA, s1, s, t, n1, 4);

        for (ssize i = 4; i >= 0; i--) {
            bucket_sort<span<T>, sa_index>(i % 2 == 0 ? s1 : SA, i % 2 == 0 ? SA : s1, s, n1, K,
                        buckets, i);
        }

        for (ssize i = n1 - 1; i >= 0; i--) {
            j = 2 * i;
            SA[j] = SA[i];
            SA[j + 1] = -1;
        }

        for (size_t i = 2 * (n1 - 1) + 3; i < s.size(); i += 2) {
            SA[i] = -1;
        }

        size_t name = 0;
        ssize c[] = {-1, -1, -1, -1, -1};

        for (size_t i = 0; i < n1; i++) {
            size_t h;
            size_t pos = SA[2 * i];
            size_t diff = 0;

            for (h = 0; h < 4; h++) {
                if ((ssize)s[pos + h] != c[h]) {
                    diff = true;
                    break;
                }
            }
            if ((pos + 4 >= s.size()) ||
                get_omega_weight(s, t, pos + 4) != c[4]) {
                diff = true;
            }

            if (diff) {
                name++;
                for (h = 0; h < 4; h++) {
                    c[h] = (pos + h < s.size()) ? (ssize)s[pos + h] : -1;
                }

                c[h] =
                    (pos + h < s.size()) ? get_omega_weight(s, t, pos + h) : -1;
            }

            if (pos % 2 == 0) {
                pos--;
            }
            SA[pos] = name - 1;
        }

        for (ssize i = (((s.size() / 2) * 2) - 1), j = s.size() - 1;
             i >= 0 && j >= 0; i -= 2) {
            if (SA[i] != (sa_index)-1) {
                SA[j--] = SA[i];
            }
        }

        if (name < n1) {
            run_saca<sa_index const, sa_index>(s1, SA, name - 1);
        }
        else {
            for (size_t i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        compute_dc(s, t, s1, 3);

        generate_buckets(s, buckets, K, true);

        for (size_t i = 0; i < n1; i++) {
            if (SA[i] != (sa_index)-1 && SA[i] < s1.size())
            SA[i] = s1[SA[i]];
        }

        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = -1;
        }

        for (ssize i = n1 - 1; i >= 0; i--) {
            j = SA[i];
            SA[i] = -1;

            if (j != (sa_index)-1 && t[j] == S_Type && t[j - 1] == L_Type) {
                SA[--buckets[s[j]]] = j;
            }
        }

        induce_L_Types<span<T>, sa_index>(s, buckets, K, false, SA, t);
        induce_S_Types(s, buckets, K, true, SA, t);
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {

        tdc::StatPhase sais("Main Phase");
        if (text.size() > 1) {
            run_saca<character const, sa_index>(text, out_sa, alphabet.max_character_value());
        }
    }
};
} // namespace sacabench::sads
