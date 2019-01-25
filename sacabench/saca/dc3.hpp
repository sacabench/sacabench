/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <tuple>
#include <util/alphabet.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/induce_sa_dc.hpp>
#include <util/macros.hpp>
#include <util/merge_sa_dc.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/sort/radixsort_parallel.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::dc3 {

class dc3 {
public:
    static constexpr size_t EXTRA_SENTINELS = 3;
    static constexpr char const* NAME = "DC3";
    static constexpr char const* DESCRIPTION =
        "Difference Cover Modulo 3 as described by Juha Kärkkäinen and Peter "
        "Sanders";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        if (text.size() != 3) {
            construct_sa_dc3<sa_index, false, sacabench::util::character>(
                text, out_sa.slice(3, out_sa.size()),
                alphabet.size_with_sentinel());
        }
    }

private:
    template <typename C, typename sa_index, typename T, typename S>
    static void determine_triplets(const T& INPUT_STRING, S& triplets_12,
                                   size_t alphabet_size) {

        DCHECK_MSG(
            triplets_12.size() == 2 * (INPUT_STRING.size() - 2) / 3,
            "triplets_12 must have the length (2*INPUT_STRING.size()/3)");

        /*sacabench::util::container<sa_index> result(triplets_12.size());

        radixsort_triple<sa_index>(INPUT_STRING, result, triplets_12,
        alphabet_size, 2, true); radixsort_triple<sa_index>(INPUT_STRING,
        triplets_12, result, alphabet_size, 1, false);
        radixsort_triple<sa_index>(INPUT_STRING, result, triplets_12,
        alphabet_size, 0, false);*/

        // Container to store all tuples with the same length as triplets_12
        // Tuples contains three chararcters (triplet) and the start position i
        // mod 3 != 0

        (void)alphabet_size;
        sa_index counter = 0;
        sa_index one = 1;
        sa_index two = 2;

        auto triplets_12_to_be_sorted =
            sacabench::util::make_container<std::tuple<C, C, C, sa_index>>(
                2 * (INPUT_STRING.size() - 2) / 3);

        for (sa_index i = 1; i < INPUT_STRING.size() - 2; ++i) {
            if (i % 3 != 0) {
                triplets_12_to_be_sorted[counter++] =
                    std::tuple<C, C, C, sa_index>(INPUT_STRING[i],
                                                  INPUT_STRING[i + one],
                                                  INPUT_STRING[i + two], i);
            }
        }

        std::sort(triplets_12_to_be_sorted.begin(),
                  triplets_12_to_be_sorted.end());

        for (sa_index i = 0; i < triplets_12_to_be_sorted.size(); ++i) {
            triplets_12[i] = std::get<3>(triplets_12_to_be_sorted[i]);
        }
    }

    template <typename C, typename sa_index, typename T>
    static util::span<const C> retrieve_triplets(T& text, sa_index pos,
                                                 sa_index count) {
        if ((pos + count) < text.size()) {
            return util::span<const C>(&text[pos], count);
        } else {
            return util::span<const C>(&text[pos], text.size() - pos);
        }
    }

    template <typename sa_index, typename T, typename S, typename L>
    static void determine_leq(const T& INPUT_STRING, const S& triplets_12,
                              L& t_12, bool& recursion, size_t alphabet_size) {

        (void)alphabet_size;
        DCHECK_MSG(triplets_12.size() == t_12.size(),
                   "triplets_12 must have the same length as t_12");

        sa_index leq_name = 1;

        for (size_t i = 0; i < triplets_12.size(); ++i) {
            // set the lexicographical names at correct positions:
            //[----names at positions i mod 3 = 1----||----names at positions i
            // mod 3 = 2----]
            if (triplets_12[i] % 3 == 1) {
                t_12[triplets_12[i] / 3] = leq_name;
            } else {
                if (t_12.size() % 2 == 0) {
                    t_12[t_12.size() / 2 + triplets_12[i] / 3] = leq_name;
                } else {
                    t_12[t_12.size() / 2 + 1 + triplets_12[i] / 3] = leq_name;
                }
            }

            if (i + 1 < triplets_12.size()) {
                if (sacabench::util::span(&INPUT_STRING[triplets_12[i]], 3) !=
                    sacabench::util::span(&INPUT_STRING[triplets_12[i + 1]],
                                          3)) {
                    ++leq_name;
                } else { // if lexicographical names are not uniqe set recursion
                         // = true
                    recursion = true;
                    //--alphabet_size;
                }
            }
        }
    }

    template <typename S, typename I>
    static void determine_sa(const S& t_12, I& isa_12) {

        DCHECK_MSG(isa_12.size() == t_12.size(),
                   "isa_12 must have the same length as t_12");

        for (size_t i = 0; i < t_12.size(); ++i) {
            isa_12[t_12[i]] = i + 1;
        }
    }

    template <typename sa_index, bool rec, typename C, typename S>
    static void construct_sa_dc3(S& text, util::span<sa_index> out_sa,
                                 size_t alphabet_size) {
        (void)alphabet_size;

        //-----------------------Phase 1------------------------------//
        tdc::StatPhase dc3("Phase 1");
        // empty container which will contain indices of triplet
        // at positions i mod 3 != 0
        auto triplets_12 = sacabench::util::make_container<sa_index>(
            2 * (text.size() - 2) / 3);

        // determine positions and calculate the sorted order
        determine_triplets<C, sa_index>(text, triplets_12, alphabet_size);

        // empty SA which should be filled correctly with lexicographical
        // names of triplets (+3 because of dummy triplet)
        auto t_12 = sacabench::util::make_container<sa_index>(
            2 * (text.size() - 2) / 3 + 3);

        // bool which will be set true in determine_leq if the names are not
        // unique
        bool recursion = false;

        // fill t_12 with lexicographical names
        auto span_t_12 = util::span(&t_12[0], t_12.size() - 3);
        //alphabet_size = span_t_12.size();
        determine_leq<sa_index>(text, triplets_12, span_t_12, recursion,
                                alphabet_size);

        util::span<sa_index> sa_12 = util::span(&out_sa[0], t_12.size() - 3);

        // run the algorithm recursivly if the names are not unique
        if (recursion) {
            dc3.split("Rekursion");
            // run algorithm recursive
            construct_sa_dc3<sa_index, true, sa_index>(t_12, sa_12,
                                                       alphabet_size);
        }

        //-----------------------Phase2------------------------------//
        dc3.split("Phase 2");

        // if in recursion use temporary sa. Otherwise t_12
        if (recursion) {
            determine_sa(sa_12, span_t_12);

            // index of the first value which represents the positions i mod 3 =
            // 2
            size_t end_of_mod_eq_1 =
                triplets_12.size() / 2 + ((triplets_12.size() % 2) != 0);

            sa_index one = 1;
            // correct the order of sa_12 with result of recursion
            for (size_t i = 0; i < end_of_mod_eq_1; ++i) {
                triplets_12[span_t_12[i] - one] = 3 * i + 1;
            }
            for (size_t i = end_of_mod_eq_1; i < triplets_12.size(); ++i) {
                triplets_12[span_t_12[i] - one] = 3 * (i - end_of_mod_eq_1) + 2;
            }
        }

        // empty sa_0 which should be filled correctly with method induce_sa_dc
        auto sa_0 = sacabench::util::make_container<sa_index>(
            (text.size() - 2) / 3 + ((text.size() - 2) % 3 != 0));

        // fill sa_0 by inducing with characters at i mod 3 = 0 and ranks of
        // triplets beginning in positions i mod 3 != 0
        induce_sa_dc<C, sa_index>(text, span_t_12, sa_0);

        //-----------------------Phase 3------------------------------//
        dc3.split("Phase 3");

        // merging the SA's of triplets in i mod 3 != 0 and ranks of i mod 3 = 0
        if constexpr (rec) {

            merge_sa_dc<const sa_index, sa_index>(
                sacabench::util::span(&text[0], text.size()), sa_0, triplets_12,
                span_t_12, out_sa,
                std::less<sacabench::util::span<const sa_index>>(),
                [](auto a, auto b, auto c) {
                    return get_substring_recursion<sa_index>(a, b, c);
                });

        } else {
            merge_sa_dc<const sacabench::util::character, sa_index>(
                sacabench::util::span(&text[0], text.size()), sa_0, triplets_12,
                span_t_12, out_sa, std::less<sacabench::util::string_span>(),
                [](auto a, auto b, auto c) { return get_substring(a, b, c); });
        }
    }

    // implementation of get_substring method with type of character not in
    // recursion
    static const sacabench::util::span<const sacabench::util::character>
    get_substring(const sacabench::util::string_span t,
                  const sacabench::util::character* ptr, size_t n) {
        // Suppress unused variable warnings:
        (void)t;
        return sacabench::util::span(ptr, n);
    }

    // implementation of get_substring method with type size_t in recursion
    template <typename sa_index>
    static const sacabench::util::span<const sa_index>
    get_substring_recursion(const sacabench::util::span<sa_index> t,
                            const sa_index* ptr, size_t n) {
        // Suppress unused variable warnings:
        (void)t;
        return sacabench::util::span(ptr, n);
    }

    template <typename C, typename sa_index, typename T, typename I, typename S>
    /**\brief Identify order of chars starting in position i mod 3 = 0 with
     * difference cover
     * \tparam T Type of input string
     * \tparam C Type of input characters
     * \tparam I Type of inverse Suffix Array
     * \tparam S Type of Suffix Array
     * \param t_0 input text t_0 with chars beginning in i mod 3 = 0 of input
     * text \param isa_12 ISA for triplets beginning in i mod 3 != 0 \param sa_0
     * memory block for resulting SA for positions beginning in i mod 3 = 0
     *
     * This method identifies the order of the characters in input_string in
     * positions i mod 3 = 0 with information of ranks of triplets starting in
     * position i mod 3 != 0 of input string. This method works correct because
     * of the difference cover idea.
     */
    static void induce_sa_dc(const T& text, const I& isa_12, S& sa_0) {
        /*
        //The commented part looks better and needs less memory. But it's much
        slower. for (sa_index i = 0; i < sa_0.size(); ++i) { sa_0[i] = 3 * i;
        }

        // index of first rank for triplets beginning in i mod 3 = 2
        size_t start_pos_mod_2 = isa_12.size() / 2 + ((isa_12.size() % 2) != 0);

        auto comp = [&](size_t i, size_t j) {
            if (text[i] < text[j])
                return true;
            else if (text[i] > text[j])
                return false;
            else {

                if (start_pos_mod_2 <= i / 3)
                    return true;
                else if (start_pos_mod_2 <= j / 3)
                    return false;
                if (isa_12[i / 3] < isa_12[j / 3])
                    return true;
                else
                    return false;
            }
        };

        std::sort(sa_0.begin(), sa_0.end(), comp);*/

        // Container to store all tuples with the same length as sa_0
        // Tuples contains a char, a rank and the position
        auto sa_0_to_be_sorted =
            util::make_container<std::tuple<C, sa_index, sa_index>>(
                sa_0.size());

        // index of first rank for triplets beginning in i mod 3 = 2
        sa_index start_pos_mod_2 =
            isa_12.size() / 2 + ((isa_12.size() % 2) != 0);

        sa_index counter = 0;
        size_t i = 0;
        for (i = 0; i < 3 * start_pos_mod_2; i += 3) {
                sa_0_to_be_sorted[counter++] =
                    (std::tuple<C, sa_index, sa_index>(text[i], isa_12[i / 3],
                                                       i));  
        }
        for (i = 3 * start_pos_mod_2; i < text.size() - 2; i += 3) {
                sa_0_to_be_sorted[counter++] =
                    (std::tuple<C, sa_index, sa_index>(text[i], 0, i));
        }
        
        std::sort(sa_0_to_be_sorted.begin(), sa_0_to_be_sorted.end());
        //util::sort::radixsort_parallel<C, sa_index>(sa_0_to_be_sorted, text.size());

        for (sa_index i = 0; i < sa_0_to_be_sorted.size(); ++i) {
            sa_0[i] = std::get<2>(sa_0_to_be_sorted[i]);
        }
    }

    /**\brief Merge two suffix array with the difference cover idea.
     * \tparam T input string
     * \tparam C input characters
     * \tparam I ISA
     * \tparam S SA
     * \param t input text
     * \param sa_0 calculated SA for triplets beginning in i mod 3 = 0
     * \param sa_12 calculated SA for triplets beginning in i mod 3 != 0
     * \param isa_12 calculated ISA for triplets beginning in i mod 3 != 0
     * \param sa memory block for merged SA
     * \param comp function which compares for strings a and b if a is
     *        lexicographically smaller than b
     * \param get_substring function which expects a string t, an index i and an
     *        integer n and returns a substring of t beginning in i where n
     *        equally calculated substrings are concatenated
     *
     * This method merges the suffix arrays s_0, which contains the
     * lexicographical ranks of positions i mod 3 = 0, and s_12, which
     * contains the lexicographical ranks of positions i mod 3 != 0.
     * This method works correct because of the difference cover idea.
     */
    template <typename C, typename sa_index, typename T, typename I, typename S,
              typename X, typename Compare, typename Substring,
              typename nocheintemplate>
    static void merge_sa_dc(const T& t, const S& sa_0,
                            const nocheintemplate& sa_12, const I& isa_12,
                            X& sa, const Compare comp,
                            const Substring get_substring) {

        // DCHECK_MSG(sa.size() == t.size(),
        //           "sa must be initialised and must have the same length as
        //           t.");
        // DCHECK_MSG(
        //    sa.size() == (sa_0.size() + sa_12.size()),
        //    "the length of sa must be the sum of the length of sa_0 and
        //    sa_12");
        // DCHECK_MSG(sa_12.size() == isa_12.size(),
        //           "the length of sa_12 must be equal to isa_12");

        // index of first rank for triplets beginning in i mod 3 = 2
        size_t start_pos_mod_2 = isa_12.size() / 2 + ((isa_12.size() % 2) != 0);
        size_t position_i_isa = 0;
        size_t position_j_isa = 0;
        sa_index i = 0;
        sa_index j = 0;

        sa_index one = 1;
        sa_index two = 2;

        sa_index size_isa_12 = isa_12.size();

        if (t.size() % 3 == 0) {
            ++i;
        } else {
            ++j;
        }

        for (size_t index = 0; index < sa.size(); ++index) {
            if (i < sa_0.size() && j < sa_12.size()) {
                sacabench::util::span<C> t_0;
                sacabench::util::span<C> t_12;
                if (sa_12[j] % 3 == 1) {
                    t_0 = get_substring(t, &t[sa_0[i]], 1);
                    t_12 = get_substring(t, &t[sa_12[j]], 1);
                } else {
                    t_0 = get_substring(t, &t[sa_0[i]], 2);
                    t_12 = get_substring(t, &t[sa_12[j]], 2);
                }

                sa_index size_t_0 = t_0.size();
                sa_index size_t_12 = t_12.size();

                if (sa_12[j] % 3 == 1) {
                    position_j_isa = start_pos_mod_2 + (sa_12[j] + one) / 3;
                    position_i_isa = (sa_0[i] + one) / 3;

                    if (position_i_isa >= size_isa_12 ||
                        position_j_isa >= size_isa_12) {
                        position_i_isa = 0;
                        position_j_isa = 0;
                    }
                } else {
                    position_j_isa = (sa_12[j] + two) / 3;
                    position_i_isa = start_pos_mod_2 + (sa_0[i] + two) / 3;
                    if (position_i_isa >= size_isa_12 ||
                        position_j_isa >= size_isa_12) {
                        position_i_isa = 0;
                        position_j_isa = 0;
                    }
                }

                const bool less_than = comp(t_0, t_12);
                const bool eq = sacabench::util::as_equal(comp)(t_0, t_12);
                // NB: This is a closure so that we can evaluate it later,
                // because evaluating it if `eq` is not true causes
                // out-of-bounds errors.
                auto lesser_suf = [&]() {
                    return !((2 * (sa_12[j] + size_t_12)) / 3 >=
                             size_isa_12) && // if index to compare for t_12
                                             // is out of bounds of isa then
                                             // sa_0[i] is never
                                             // lexicographically smaller than
                                             // sa_12[j]
                           ((2 * (sa_0[i] + size_t_0)) / 3 >=
                                size_isa_12 || // if index to compare for t_0
                                               // is out of bounds of isa then
                                               // sa_0[i] is lexicographically
                                               // smaller
                            isa_12[position_i_isa] < isa_12[position_j_isa]);
                };

                if (less_than || (eq && lesser_suf())) {

                    sa[index] = sa_0[i++];
                } else {

                    sa[index] = sa_12[j++];
                }
            }

            else if (i >= sa_0.size()) {

                sa[index] = sa_12[j++];
            }

            else {

                sa[index] = sa_0[i++];
            }
        }
    }

    template <typename sa_index, typename T, typename S, typename N>
    static void radixsort_triple(const T& text, const S& tuples, S& result,
                                 const N& alphabet_size,
                                 const sa_index& position, const bool& first) {
        auto buckets = util::container<sa_index>(alphabet_size);
        if (first) {
            for (sa_index i = 1; i < text.size() - 2; ++i) {
                if (i % 3 != 0) {
                    ++buckets[text[i + position]];
                }
            }

            sa_index sum = 0;
            for (size_t i = 0; i < buckets.size(); ++i) {
                sum += buckets[i];
                buckets[i] = sum - buckets[i];
            }

            for (sa_index i = 1; i < text.size() - 2; ++i) {
                if (i % 3 != 0) {
                    result[buckets[text[i + position]]++] = i;
                }
            }
        } else {
            for (sa_index i = 0; i < tuples.size(); ++i) {
                ++buckets[text[tuples[i] + position]];
            }

            sa_index sum = 0;
            for (sa_index i = 0; i < buckets.size(); ++i) {
                sum += buckets[i];
                buckets[i] = sum - buckets[i];
            }

            for (sa_index i = 0; i < tuples.size(); ++i) {
                result[buckets[text[tuples[i] + position]]++] = tuples[i];
            }
        }
    }
}; // class dc3

} // namespace sacabench::dc3
