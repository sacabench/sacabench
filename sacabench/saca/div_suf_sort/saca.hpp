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
#include <util/compare.hpp>
#include <util/assertions.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::div_suf_sort {

class div_suf_sort {
public:
    static const size_t EXTRA_SENTINELS = 1;

    static constexpr char const* NAME = "DSS";
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
            auto sa_type_container = util::make_container<bool>(text.size());
            // auto sa_types = util::span<bool>(sa_type_container);

            // Compute l/s types for given text; TODO: Replace with version from
            // 'extract_types.hpp' after RTL-Insertion was merged.
            get_types_tmp<sa_index>(text, sa_type_container);
            auto rms_count =
                extract_rms_suffixes<sa_index>(text, sa_type_container, out_sa);
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

            // Initialize buckets: alphabet_size slots for l-buckets,
            // alphabet_size² for s-buckets

            auto s_bkt = util::make_container<sa_index>(
                pow(alphabet.max_character_value() + 1, 2));
            auto l_bkt = util::make_container<sa_index>(
                alphabet.max_character_value() + 1);

            buckets<sa_index> bkts = {
                /*.alphabet_size=*/alphabet.max_character_value() + 1,
                /*.l_buckets=*/l_bkt, /*.s_buckets=*/s_bkt};

            compute_buckets<sa_index>(text, alphabet.max_character_value(),
                                      sa_type_container, bkts);
            if (rms_count > 0) {
                insert_into_buckets<sa_index>(rms_suf, bkts);
                auto substrings = extract_rms_substrings(rms_suf);

                sort_rms_substrings<sa_index>(
                    rms_suf, alphabet.max_character_value(), bkts);

                // Check, wether substrings sorted correctly
                util::span<sa_index> rel_ind = rms_suf.relative_indices;
                size_t a_size;
                size_t b_size;
                size_t max_pos;
                for(size_t i = 0; i < rel_ind.size()-1; ++i) {
                    if((rel_ind[i] & utils<sa_index>::NEGATIVE_MASK) > 0 ||
                        (rel_ind[i+1] & utils<sa_index>::NEGATIVE_MASK) > 0) {
                        std::cout << "Skipping because one index negated (i.e. same)" <<std::endl;
                        continue;
                    }
                    std::cout << "Comparing substring <" << std::get<0>(substrings[rel_ind[i]])
                     << "," << std::get<1>(substrings[rel_ind[i]]) << "> with <" <<
                     std::get<0>(substrings[rel_ind[i+1]]) << "," << std::get<1>(substrings[rel_ind[i+1]]) << ">"<< std::endl;
                    a_size = std::get<1>(substrings[rel_ind[i]]) - std::get<0>(substrings[rel_ind[i]]) + sa_index(1);
                    b_size = std::get<1>(substrings[rel_ind[i+1]]) - std::get<0>(substrings[rel_ind[i+1]]) + sa_index(1);
                    sa_index a_start = std::get<0>(substrings[rel_ind[i]]);
                    sa_index b_start = std::get<0>(substrings[rel_ind[i+1]]);
                    max_pos = std::min(a_size, b_size)+1;
                    for(sa_index j=0; j < max_pos; ++j) {
                        // Check characterwise, until smaller
                        std::cout << "Comparing " << size_t(a_start+j) << ", index "<< rel_ind[i]+j << "  to "
                        << size_t(b_start +j) << ", index " << rel_ind[i+1]+j << std::endl;
                        DCHECK_LE(text[a_start + j], text[b_start+j]);
                        if(text[a_start + j] < text[b_start+j]) {
                            break;
                        }
                    }
                }

                // Compute ISA

                compute_initial_isa<sa_index>(rms_suf.relative_indices,
                                              rms_suf.partial_isa);

                sort_rms_suffixes<sa_index>(rms_suf);
                sort_rms_indices_to_order<sa_index>(rms_suf, rms_count,
                                                    sa_type_container, out_sa);

                insert_rms_into_correct_pos<sa_index>(
                    rms_count, bkts, alphabet.max_character_value(), out_sa);
            }
            induce_s_suffixes<sa_index>(text, sa_type_container, bkts, out_sa,
                                        alphabet.max_character_value());

            induce_l_suffixes<sa_index>(text, bkts, out_sa);
        }
    }
};

} // namespace sacabench::div_suf_sort
