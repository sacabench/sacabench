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

namespace sacabench::sais {
using namespace sacabench::util;
class sais {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static const size_t L_Type = 0;
    static const size_t S_Type = 1;

    template <typename T>
    static void compute_types(span<size_t> t, span<size_t> p_1,
                              T s) {
        t[s.size() - 1] = S_Type;

        for (ssize i = s.size() - 2; i >= 0; i--) {
            if (s.at(i + 1) < s.at(i)) {
                t[i] = L_Type;
                if (t[i + 1] == S_Type) {
                    p_1[i + 1] = 1;
                }
            } else if (s.at(i + 1) > s.at(i)) {
                t[i] = S_Type;
            } else {
                t[i] = t[i + 1];
            }
        }
    }

    static bool is_LMS(span<size_t> p_1, ssize position) {
        return (position > 0) && (p_1[position] == 1);
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

    template <typename T>
    static void induce_L_Types(T s, span<size_t> buckets, size_t K,
                               bool end, span<ssize> SA,
                               span<size_t> t) {
        generate_buckets(s, buckets, K, end);
        for (size_t i = 0; i < s.size(); i++) {
            ssize pre_index =
                SA[i] - 1; // pre index of the ith suffix array position

            if (pre_index >= 0 &&
                t[pre_index] == L_Type) { // pre index is type L
                SA[buckets[s.at(pre_index)]++] =
                    pre_index; // "sort" index in the bucket
            }
        }
    }

    template <typename T>
    static void induce_S_Types(T s, span<size_t> buckets, size_t K,
                               bool end, span<ssize> SA,
                               span<size_t> t) {
        generate_buckets(s, buckets, K, end);
        for (ssize i = s.size() - 1; i >= 0; i--) {
            ssize pre_index = SA[i] - 1;

            if (pre_index >= 0 && t[pre_index] == S_Type) {
                SA[--buckets[s.at(pre_index)]] = pre_index;
            }
        }
    }

    template <typename T>
    static void run_saca(T s, span<ssize> SA, size_t K) {
        container<size_t> t = make_container<size_t>(s.size());
        container<size_t> p_1 = make_container<size_t>(s.size());
        container<size_t> buckets = make_container<size_t>(K + 1);

        compute_types(t, p_1, s);

        generate_buckets(s, buckets, K, true);
        // Initialize each entry in SA with -1
        for (size_t i = 0; i < s.size(); i++) {
            SA[i] = -1;
        }
        // iterate from left to right (starting by 1 cause 0 can never be LMS)
        // and put LMS to end of the bucket and move bucket's tail backwards
        for (size_t i = 1; i < s.size(); i++) {
            if (is_LMS(p_1, i)) {
                SA[--buckets[s.at(i)]] = i;
            }
        }

        // sort LMS substrings
        induce_L_Types(s, buckets, K, false, SA, t);
        induce_S_Types(s, buckets, K, true, SA, t);

        // because we have at most n/2 LMS, we can store the sorted indices in
        // the first half of the SA
        ssize n1 = 0;
        for (size_t i = 0; i < s.size(); i++) {
            if (is_LMS(p_1, SA[i]) == 1) {
                SA[n1++] = SA[i];
            }
        }

        // All LMS are now stored in the (at most) first half of the SA, so the
        // rest half of the suffix array can be used
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = -1;
        }

        // The given names correspond to the buckets the LMS are sorted ssizeo.
        // To find the names, the strings have to be compared by its type and
        // lexicographical value per char
        ssize name = 0;
        ssize previous_LMS = -1;
        for (ssize i = 0; i < n1; i++) { // max n/2 iterations
            bool diff = false;
            // compare types and chars
            ssize current_LMS = SA[i];
            for (size_t j = 0; j < s.size(); j++) {
                if (previous_LMS == -1 ||
                    s.at(current_LMS + j) != s.at(previous_LMS + j) ||
                    t[current_LMS + j] != t[previous_LMS + j]) {
                    diff = true;
                    break;
                } else if (j > 0 &&
                           (is_LMS(p_1, current_LMS + j) ||
                            is_LMS(p_1,
                                   previous_LMS +
                                       j))) { // check if next LMS is reached
                    break;                    // no diff was found
                }
            }

            if (diff) { // if diff was found, adjust the name and continue with
                        // current LMS as previous
                name++;
                previous_LMS = current_LMS;
            }

            current_LMS = (current_LMS % 2 == 0) ? current_LMS / 2
                                                 : (current_LMS - 1) / 2;
            SA[n1 + current_LMS] = name - 1;
        }

        // not needed
        for (ssize i = s.size() - 1, j = s.size() - 1; i >= n1; i--) {
            if (SA[i] >= 0) {
                SA[j--] = SA[i];
            }
        }

        // if there are more names than LMS, we have duplicates and have to
        // recurse

        container<size_t> s1 = make_container<size_t>(n1);
        for (size_t i = s.size() - n1; i < s.size(); i++) {
            s1[i - (s.size() - n1)] = SA[i];
        }

        if (name < n1) {
            run_saca<span<size_t const>>(s1, SA, name - 1);
        } else {
            for (ssize i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        // induce the final SA
        generate_buckets(s, buckets, K, true);
        size_t j;
        for (size_t i = 1, j = 0; i < s.size(); i++) {
            if (is_LMS(p_1, i)) {
                s1[j++] = i;
            }
        }
        for (ssize i = 0; i < n1; i++) {
            SA[i] = s1[SA[i]];
        }
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = -1;
        }
        for (ssize i = n1 - 1; i >= 0; i--) {
            j = SA[i];
            SA[i] = -1;
            SA[--buckets[s.at(j)]] = j;
        }

        induce_L_Types(s, buckets, K, false, SA, t);
        induce_S_Types(s, buckets, K, true, SA, t);
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        container<ssize> SA = make_container<ssize>(text.size());
        run_saca<string_span>(text, SA, alphabet.max_character_value());
        for (size_t i = 0; i < text.size(); i++) {
            DCHECK_LE(SA[i], std::numeric_limits<sa_index>::max());
            out_sa[i] = static_cast<sa_index>(SA[i]);
        }
    }
};
} // namespace sacabench::sais
