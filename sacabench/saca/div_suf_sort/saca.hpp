/*******************************************************************************
 * Copyright (C) 2018 Oliver Magiera <magiera.o@googlemail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "comparison_fct.hpp"
#include "induce.hpp"
#include "init.hpp"
#include "rms_sorting.hpp"
#include "utils.hpp"
#include <iostream>
#include <util/alphabet.hpp>
#include <util/assertions.hpp>
#include <util/compare.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::div_suf_sort {

class div_suf_sort {
public:
    static const size_t EXTRA_SENTINELS = 1;

    static constexpr char const* NAME = "DivSufSort";
    static constexpr char const* DESCRIPTION =
        "DivSufSort (Y. Mori) without repetition detection";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    util::alphabet const& alphabet,
                                    util::span<sa_index> out_sa) {
        // Check if enough bits free for negation.
        DCHECK(util::assert_text_length<sa_index>(text.size(), 1u));

        // Contains sentinel only
        if (text.size() == 1) {
            out_sa[0] = 0;
        } else {
            DCHECK_EQ(text.size(), out_sa.size());
            // Create container for l/s types
            tdc::StatPhase dss("Initialize DivSufSort");
            // Initialize buckets: alphabet_size slots for l-buckets,
            // alphabet_size² for s-buckets

            auto s_bkt = util::make_container<sa_index>(
                pow(alphabet.max_character_value() + 1, 2));
            auto l_bkt = util::make_container<sa_index>(
                alphabet.max_character_value() + 1);

            buckets<sa_index> bkts = {
                /*.alphabet_size=*/alphabet.max_character_value() + 1,
                /*.l_buckets=*/l_bkt, /*.s_buckets=*/s_bkt};

            auto rms_count = extract_rms_suffixes_and_count_buckets<sa_index>(
                text, out_sa, bkts);
            prefix_sum(alphabet.max_character_value(), bkts);
            /*check_buckets(text, alphabet.max_character_value() + 1,
                          sa_type_container, bkts);*/

            // Initialize struct rms_suffixes with text, relative positions
            // (first rms_count positions in out_sa) and absolute positions
            // (last rms_count positions in out_sa) for rms-suffixes
            // partial_isa contains isa for relative_indices; in out_sa from
            // rms_count to 2*rms_count (safe, because at most half of suffixes
            // rms-type)
            rms_suffixes<sa_index> rms_suf = {
              /*.text=*/text, /*.relative_indices=*/
              out_sa.slice(0, rms_count),
              /*.absolute_indices=*/
              out_sa.slice(out_sa.size() - rms_count, out_sa.size()),
              /*.partial_isa=*/out_sa.slice(rms_count, 2 * rms_count)};

            if (rms_count > 0) {
                insert_into_buckets<sa_index>(rms_suf, bkts);
                dss.split("Sort RMS-Substrings");
                // auto substrings = extract_rms_substrings(rms_suf);
                sort_rms_substrings<sa_index>(
                    rms_suf, alphabet.max_character_value(), bkts);

                // Compute ISA
                dss.split("Sort RMS-Suffixes");
                compute_initial_isa<sa_index>(rms_suf.relative_indices,
                                              rms_suf.partial_isa);

                sort_rms_suffixes<sa_index>(rms_suf);
                sort_rms_indices_to_order<sa_index>(rms_suf, rms_count, out_sa);

                // Check if order correct
                // check_sorted_rms(text, rms_suf.relative_indices);
                insert_rms_into_correct_pos<sa_index>(
                    rms_count, bkts, alphabet.max_character_value(), out_sa);
            }
            dss.split("Inducing L/S-Suffixes");
            induce_s_suffixes<sa_index>(text, bkts, out_sa,
                                        alphabet.max_character_value());

            induce_l_suffixes<sa_index>(text, bkts, out_sa);
        }
    }

    template <typename sa_index>
    static void
    check_substrings(util::span<sa_index> rel_ind,
                     util::container<std::pair<sa_index, sa_index>> substrings,
                     util::string_span text) {
        // Check, wether substrings sorted correctly
        size_t a_size;
        size_t b_size;
        size_t max_pos;
        sa_index a, b, a_start, b_start;
        for (size_t i = 0; i < rel_ind.size() - 1; ++i) {
            a = (rel_ind[i] | utils<sa_index>::NEGATIVE_MASK) ^
                utils<sa_index>::NEGATIVE_MASK;
            b = (rel_ind[i + 1] | utils<sa_index>::NEGATIVE_MASK) ^
                utils<sa_index>::NEGATIVE_MASK;

            if (((rel_ind[i] | utils<sa_index>::NEGATIVE_MASK) ^
                 utils<sa_index>::NEGATIVE_MASK) == 109106 ||
                ((rel_ind[i + 1] | utils<sa_index>::NEGATIVE_MASK) ^
                 utils<sa_index>::NEGATIVE_MASK) == 109106) {
                std::cout << "Index 109106 found !" << std::endl;
            }
            if (((rel_ind[i] | utils<sa_index>::NEGATIVE_MASK) ^
                 utils<sa_index>::NEGATIVE_MASK) == 109114 ||
                ((rel_ind[i + 1] | utils<sa_index>::NEGATIVE_MASK) ^
                 utils<sa_index>::NEGATIVE_MASK) == 109114) {
                std::cout << "Index 109114 found !" << std::endl;
            }

            std::cout << "Comparing substring <" << std::get<0>(substrings[a])
                      << "," << std::get<1>(substrings[a]) << "> with <"
                      << std::get<0>(substrings[b]) << ","
                      << std::get<1>(substrings[b]) << ">" << std::endl;
            a_size = std::get<1>(substrings[a]) - std::get<0>(substrings[a]) +
                     sa_index(1);
            b_size = std::get<1>(substrings[b]) - std::get<0>(substrings[b]) +
                     sa_index(1);
            a_start = std::get<0>(substrings[a]);
            b_start = std::get<0>(substrings[b]);
            if (a_start == sa_index(76651)) {
                std::cout << "Index 76651 found!" << std::endl;
            } else if (a_start == sa_index(76659)) {
                std::cout << "Index 76659 found! rel_ind " << a << std::endl;
            }
            max_pos = std::min(a_size, b_size) + 1;
            for (sa_index j = 0; j < max_pos; ++j) {
                // Check characterwise, until smaller
                std::cout << "Comparing " << size_t(text[a_start + j])
                          << ", index" << a_start+j <<
                             ", to "
                          << size_t(text[b_start + j]) << ", index "
                          << b_start + j << std::endl;
                DCHECK_LE(text[a_start + j], text[b_start + j]);
                if (text[a_start + j] < text[b_start + j]) {
                    break;
                }
            }
        }
    }

    template <typename sa_index>
    static void check_rel_ind(util::span<sa_index> rel_ind,
                              util::span<sa_index> abs) {
        auto check = util::make_container<sa_index>(rel_ind.size());
        for (size_t i = 0; i < rel_ind.size(); ++i) {
            DCHECK_LT(check[rel_ind[i]], sa_index(1));
            ++check[rel_ind[i]];
            if (rel_ind[i] == sa_index(24755)) {
                std::cout << "Index " << abs[rel_ind[i]]
                          << " found, ref: " << rel_ind[i] << std::endl;
            }
        }
    }

    template <typename sa_index>
    static void check_rms_buckets(util::string_span text,
                                  util::span<sa_index> rel,
                                  util::span<sa_index> abs,
                                  buckets<sa_index>& bkts, size_t max_char) {
        // Buckets contain left border of themselves
        size_t interval_begin, interval_end = rel.size(), bkt_index;
        for (size_t first_letter = max_char - 1; 0 < first_letter;
             --first_letter) {

            for (size_t second_letter = max_char; first_letter < second_letter;
                 --second_letter) {
                bkt_index =
                    bkts.get_rms_bucket_index(first_letter, second_letter);
                interval_begin = bkts.s_buckets[bkt_index];
                for (size_t i = interval_begin; i < interval_end; ++i) {
                    DCHECK_EQ(first_letter, text[abs[rel[i]]]);
                    DCHECK_EQ(second_letter, text[abs[rel[i]] + sa_index(1)]);
                }
                // Old interval_begin is right border for next bucket
                interval_end = interval_begin;
            }
        }
    }

    template <typename sa_index>
    static void check_sorted_rms(util::string_span text,
                                 util::span<sa_index> rms_ind) {
        // Check for each rms-suffix if it is smaller than its successor
        size_t elem, succ, j;

        for (size_t i = 0; i < rms_ind.size() - 1; ++i) {
            j = 0;
            elem = (rms_ind[i] | utils<sa_index>::NEGATIVE_MASK) ^
                   utils<sa_index>::NEGATIVE_MASK;
            succ = (rms_ind[i + 1] | utils<sa_index>::NEGATIVE_MASK) ^
                   utils<sa_index>::NEGATIVE_MASK;
            DCHECK_NE(elem, succ);
            std::cout << "elem " << elem << " | succ " << succ << std::endl;
            while (text[succ + j] <= text[elem + j]) {
                std::cout << "Comparing " << size_t(text[elem + j]) << " to "
                          << size_t(text[succ + j]) << std::endl;
                // (Possibly) runs through whole suffix
                DCHECK_LE(text[elem + j], text[succ + j]);
                ++j;
            }
            std::cout << "Stopped because comparing " << size_t(text[elem + j])
                      << " to " << size_t(text[succ + j]) << std::endl;
        }
    }

    template <typename sa_index>
    static void check_buckets(util::string_span text, size_t max,
                              util::container<bool>& types,
                              buckets<sa_index> bkts) {
        std::cout << "Checking buckets" << std::endl;
        auto bkt_cmp_s = util::make_container<sa_index>(pow(max, 2));
        for (size_t i = 0; i < text.size() - 1; ++i) {
            if (sa_types::is_rms_type(i, types)) {
                ++bkt_cmp_s[text[i] * max + text[i + 1]];
            }
        }
        size_t current_count = 0;
        size_t index;
        for (size_t first_letter = 0; first_letter < max; ++first_letter) {

            for (size_t second_letter = first_letter + 1; second_letter < max;
                 ++second_letter) {
                 std::cout << "<" << first_letter << "," << second_letter
                 << ">" << std::endl;
                index = first_letter * max + second_letter;
                current_count += bkt_cmp_s[index];
                bkt_cmp_s[index] = current_count;
                DCHECK_EQ(bkt_cmp_s[index],
                          bkts.s_buckets[bkts.get_rms_bucket_index(
                              first_letter, second_letter)]);
            }
        }
        std::cout << "Finished checking buckets" << std::endl;
    }
};

} // namespace sacabench::div_suf_sort
