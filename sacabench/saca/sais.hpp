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
#include <util/type_extraction.hpp>
#include <vector>
#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::sais {
using namespace sacabench::util;
class sais {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "SAIS";
    static constexpr char const* DESCRIPTION =
        "Suffix Array Induced Sorting by Nong, Zhang and Chan";

    static const size_t L_Type = 0;
    static const size_t S_Type = 1;

    static bool is_LMS(std::vector<bool>& t, ssize position) {
        return (position > 0) && (t[position] == S_Type && t[position - 1] == L_Type);
    }

    template <typename T>
    static void compute_types(std::vector<bool>& t, T s) {
       t[s.size() - 1] = S_Type;

       for (ssize i = s.size() - 2; i >= 0; i--) {
           if (s.at(i + 1) < s.at(i)) {
               t[i] = L_Type;
           } else if (s.at(i + 1) > s.at(i)) {
               t[i] = S_Type;
           } else {
               t[i] = t[i + 1];
           }
       }
   }

    template <typename T, typename sa_index>
    static void generate_buckets(T s, span<sa_index> buckets, size_t K,
                                 bool end) {
        size_t sum = 0;

        for (size_t i = 0; i < K; i++) {
            buckets[i] = 0;
        }

        // bucket size for each char
        for (size_t i = 0; i < s.size(); i++) {
            buckets[s.at(i)]++;
        }

        // sum up to bucket ends
        for (size_t i = 0; i < K; i++) {
            sum += buckets[i];
            buckets[i] = (end ? sum : sum - buckets[i]);
        }
    }

    template <typename T, typename sa_index>
    static void induce_L_Types(T s, span<sa_index> buckets, std::vector<bool>& t, size_t K,
                               bool end, span<sa_index> SA) {
        generate_buckets<T, sa_index>(s, buckets, K, end);
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
    static void induce_S_Types(T s, span<sa_index> buckets, std::vector<bool>& t, size_t K,
                               bool end, span<sa_index> SA) {
        generate_buckets<T, sa_index>(s, buckets, K, end);
        for (ssize i = s.size() - 1; i >= 0; i--) {
            ssize pre_index = SA[i] - (sa_index)1;

            if (SA[i] != (sa_index)-1 && SA[i] != (sa_index)0 && t[pre_index] == S_Type) {
                SA[--buckets[s.at(pre_index)]] = pre_index;
            }
        }
    }

    template <typename T, typename sa_index>
    static void run_saca(T s, span<sa_index> SA, size_t K) {

        container<sa_index> buckets = make_container<sa_index>(K);
        // bit vector for now and on the fly computation later
        std::vector<bool> t(s.size());

        compute_types(t, s);

        generate_buckets<T, sa_index>(s, buckets, K, true);
        // Initialize each entry in SA with -1
        for (size_t i = 0; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }
        // iterate from left to right (starting by 1 cause 0 can never be LMS)
        // and put LMS to end of the bucket and move bucket's tail backwards
        for (size_t i = 1; i < s.size(); i++) {
            if (is_LMS(t, i)) {
                SA[--buckets[s.at(i)]] = i;
            }
        }

        // sort LMS substrings
        induce_L_Types<T, sa_index>(s, buckets, t, K, false, SA);
        induce_S_Types<T, sa_index>(s, buckets, t, K, true, SA);

        // because we have at most n/2 LMS, we can store the sorted indices in
        // the first half of the SA
        ssize n1 = 0;
        for (size_t i = 0; i < s.size(); i++) {
            if (is_LMS(t, SA[i]) == 1 || s.size() == 1) {
                SA[n1++] = SA[i];
            }
        }

        // All LMS are now stored in the (at most) first half of the SA, so the
        // rest half of the suffix array can be used
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }

        // The given names correspond to the buckets the LMS are sorted into.
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
                           (is_LMS(t, current_LMS + j) ||
                            is_LMS(t,
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
            if (SA[i] >= (sa_index)0 && SA[i] != ((sa_index)-1)) {
                SA[j--] = SA[i];
            }
        }


        span<sa_index> s1 = SA.slice(s.size() - n1, s.size());

        if (name < n1) {
            run_saca<span<sa_index const>, sa_index>(s1, SA, name);
        } else {
            for (ssize i = 0; i < n1; i++) {
                SA[s1[i]] = i;
            }
        }

        // induce the final SA
        generate_buckets<T, sa_index>(s, buckets, K, true);
        size_t j;
        for (size_t i = 1, j = 0; i < s.size(); i++) {
            if (is_LMS(t, i)) {
                s1[j++] = i;
            }
        }
        for (ssize i = 0; i < n1; i++) {
            SA[i] = s1[SA[i]];
        }
        for (size_t i = n1; i < s.size(); i++) {
            SA[i] = (sa_index)-1;
        }
        for (ssize i = n1 - 1; i >= 0; i--) {
            j = SA[i];
            SA[i] = (sa_index)-1;
            SA[--buckets[s.at(j)]] = j;
        }

        induce_L_Types<T, sa_index>(s, buckets, t, K, false, SA);
        induce_S_Types<T, sa_index>(s, buckets, t, K, true, SA);
    }

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {

        tdc::StatPhase sais("Main Phase");
        if (text.size() > 1) {
            run_saca<string_span, sa_index>(text, out_sa, alphabet.size_with_sentinel());
        }
    }
};
} // namespace sacabench::sais

