/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>
#include <functional>

#include <util/string.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/signed_size_type.hpp>
#include <util/alphabet.hpp>
#include <util/induce_sa_dc.hpp>
#include <util/compare.hpp>
#include <util/sort/tuple_sort.hpp>

#include <saca/dc3_lite.hpp>
#include <tudocomp_stat/StatPhase.hpp>


namespace sacabench::nzsufsort_par {

    class nzsufsort_par {
        public:    
            static constexpr size_t EXTRA_SENTINELS = 3;
            static constexpr char const* NAME = "nzSufSort-Par";
            static constexpr char const* DESCRIPTION =
                "Optimal lightweight SACA by G. Nong and S. Zhang parallelized";


            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     util::alphabet const& alphabet,
                                     util::span<sa_index> out_sa) {
                tdc::StatPhase phase("Count s-type-positions");
                
                // count number of s-type-positions in text
                size_t count_s_type_pos = 1;
                bool s_type = true;
                for (size_t i = text.size()-1; i > 0; --i) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { ++count_s_type_pos; }
                }
                phase.log("count_s_type_pos", count_s_type_pos);
                phase.log("less_s_type_pos_than_l_type_pos", 
                    count_s_type_pos <= text.size()/2);
                
                /*if there are more s-type-positions than l-type-positions
                  revert the characters in t*/
                bool is_text_reverted = false;  
                util::container<util::character> tmp_text;
                if (count_s_type_pos > text.size()/2) {
                    phase.split("Transform input text");
                    //TODO: Text kann nicht überschrieben werden
                    tmp_text = util::make_container<util::character>(text.size());
                    //for (size_t i = 0; i < text.size(); ++i) { tmp_text[i] = text[i]; }
                    #pragma omp parallel for
                    for (size_t i = 0; i < text.size(); ++i) { tmp_text[i] = alphabet.max_character_value()-text[i]+1; }
                    text = tmp_text;
                    
                    count_s_type_pos = text.size()-count_s_type_pos;
                    is_text_reverted = true;
                }
                
                // calculate position array of s-type-positions with i mod 3 != 0
                size_t mod_1 = count_s_type_pos/3 + (count_s_type_pos % 3 > 1);
                size_t mod_2 = count_s_type_pos/3;
                util::span<sa_index> p_12 = out_sa.slice(0, mod_1+mod_2);
                phase.split("Calculate position arrays");
                calculate_p_12(text, p_12, count_s_type_pos);
                
                // calculate length of Z_3-Strings
                phase.split("Calculate array of length of Z_3-Strings");
                util::span<sa_index> h_12 = out_sa.slice(mod_1+mod_2, 2*(mod_1+mod_2));
                size_t first_mod_2 = h_12.size()/2 + (h_12.size() % 2 != 0);
                size_t max_length_12 = 0;
                #pragma omp parallel for reduction(max:max_length_12)
                for (size_t i = 0; i < h_12.size(); ++i) {
                    /* last positions in mod_1 and mod_2 group are excluded
                       because these are the last positions */
                    if ((0 <= i && i < first_mod_2-1) || (first_mod_2 <= i && i < h_12.size()-1)) {
                        h_12[i] = p_12[i+1]-p_12[i]+(sa_index)1;
                        if (h_12[i] > max_length_12) { max_length_12 = h_12[i]; }
                    }
                }
                if (first_mod_2 < h_12.size()) {
                    h_12[first_mod_2-1] = text.size()-p_12[first_mod_2-1];
                }
                if (h_12.size() > 0) {
                    h_12[h_12.size()-1] = text.size()-p_12[h_12.size()-1];
                }
                
                //sort p_12 with radix sort
                phase.split("Sort p_12 with RadixSort");
                phase.log("max_length_12", max_length_12);
                util::span<sa_index> to_be_sorted_12 = out_sa.slice(2*(mod_1+mod_2), 3*(mod_1+mod_2));
                #pragma omp parallel for
                for (size_t i = 0; i < to_be_sorted_12.size(); ++i) {
                    to_be_sorted_12[i] = i;
                }
                auto key_function_12 = [&](size_t i, size_t p) {
                    if (p < h_12[i]) {
                        return text[p_12[i]+(sa_index)p];
                    }
                    return (util::character)(alphabet.max_character_value()+1);
                };
                auto comp_function_12 = [&](size_t i, size_t j, size_t index, size_t length) {
                    // suppress warnings
                    (void) length;
                    
                    size_t pos_1 = p_12[i]+(sa_index)index+(sa_index)1;
                    size_t pos_2 = p_12[j]+(sa_index)index+(sa_index)1;
                    util::span<const util::character> t_1;
                    util::span<const util::character> t_2;
                    if (pos_1 <= p_12[i]+h_12[i]) {
                        t_1 = text.slice(pos_1, p_12[i]+h_12[i]);
                    }
                    else {
                        t_1 = util::span<const util::character>(); 
                    }
                    if (pos_2 <= p_12[j]+h_12[j]) {
                        t_2 = text.slice(pos_2, p_12[j]+h_12[j]);
                    }
                    else {
                        t_2 = util::span<const util::character>(); 
                    }
                    return comp_z_strings(t_1, t_2);
                };
                util::msd_radixsort_inplace_with_key_and_no_recursion_on_specific_buckets(to_be_sorted_12, 
                    alphabet.size_with_sentinel()+1, 0, max_length_12, alphabet.size_with_sentinel(), key_function_12,
                    comp_function_12);
                //update to_be_sorted_12 with positions in p_12
                #pragma omp parallel for
                for (size_t i = 0; i < to_be_sorted_12.size(); ++i) {
                    to_be_sorted_12[i] = p_12[to_be_sorted_12[i]];
                }
                std::copy(to_be_sorted_12.begin(), to_be_sorted_12.end(), p_12.begin());
                
                //calculate t_12 in the begin of out_sa
                phase.split("Calculate reduced text t_12");
                bool recursion = false;
                size_t rank = 1;
                calculate_t_12<util::character>(text, out_sa, 0, mod_1+mod_2, count_s_type_pos, recursion, rank);
                util::span<sa_index> t_12 = p_12;
                
                //calculate SA(t_12) by calling the lightweight variant of DC3
                phase.split("Call DC3-Lite");
                auto u = out_sa.slice(t_12.size(), 2*t_12.size());
                auto v = out_sa.slice(2*t_12.size(), 3*t_12.size());
                if (recursion) {
                    dc3_lite::dc3_lite::lightweight_dc3<sa_index, sa_index>(t_12, t_12, u, v, rank);
                    #pragma omp parallel for
                    for (size_t i = 0; i < t_12.size(); ++i) {
                        t_12[i] = v[i];
                    }
                }
                else {
                    std::copy(t_12.begin(), t_12.end(), v.begin());
                    #pragma omp parallel for
                    for (size_t i = 0; i < t_12.size(); ++i) {
                        t_12[v[i]-(sa_index)1] = i;
                    }
                }
                util::span<sa_index> sa_12 = t_12;
                
                //calculate position array p_0
                phase.split("calculate length of Z_3-Strings");
                size_t mod_0 = count_s_type_pos/3 + (count_s_type_pos % 3 > 0);
                util::span<sa_index> p_0 = out_sa.slice(t_12.size(), t_12.size()+mod_0);
                calculate_p_0(text, p_0, count_s_type_pos);
                
                //calculate length of Z_3-Strings
                util::span<sa_index> h_0 = out_sa.slice(t_12.size()+2*mod_0, t_12.size()+3*mod_0);
                size_t max_length_0 = 0;
                
                #pragma omp parallel for reduction(max:max_length_0)
                for (size_t i = 0; h_0.size() > 0 && i < h_0.size()-1; ++i) {
                    h_0[i] = p_0[i+1]-p_0[i]+(sa_index)1;
                    if (h_0[i] > max_length_0) { max_length_0 = h_0[i]; }
                }
                if (h_0.size() > 0) { h_0[h_0.size()-1] = text.size()-p_0[p_0.size()-1]; }
                
                // sort p_0 with radixsort
                phase.split("sort p_0 with radixsort");
                phase.log("max_length_0", max_length_0);
                util::span<sa_index> to_be_sorted_0 = out_sa.slice(t_12.size()+mod_0, t_12.size()+2*mod_0);
                #pragma omp parallel for
                for (size_t i = 0; i < to_be_sorted_0.size(); ++i) {
                    to_be_sorted_0[i] = i;
                }
                auto key_function_0 = [&](size_t i, size_t p) {
                    if (p < h_0[i]) {
                        return text[p_0[i]+(sa_index)p];
                    }
                    return (util::character)(alphabet.max_character_value()+1);
                };
                auto comp_function_0 = [&](size_t i, size_t j, size_t index,size_t length) {
                    // suppress warnings
                    (void) length;
                    
                    size_t pos_1 = p_0[i]+(sa_index)index+(sa_index)1;
                    size_t pos_2 = p_0[j]+(sa_index)index+(sa_index)1;
                    util::span<const util::character> t_1;
                    util::span<const util::character> t_2;
                    if (pos_1 <= p_0[i]+h_0[i]) {
                        t_1 = text.slice(pos_1, p_0[i]+h_0[i]);
                    }
                    else {
                        t_1 = util::span<const util::character>(); 
                    }
                    if (pos_2 <= p_0[j]+h_0[j]) {
                        t_2 = text.slice(pos_2, p_0[j]+h_0[j]);
                    }
                    else {
                        t_2 = util::span<const util::character>(); 
                    }
                    return comp_z_strings(t_1, t_2);
                };
                util::msd_radixsort_inplace_with_key_and_no_recursion_on_specific_buckets(to_be_sorted_0, 
                    alphabet.size_with_sentinel()+1, 0, max_length_0, alphabet.size_with_sentinel(), key_function_0,
                    comp_function_0);
                //update to_be_sorted_0 with positions in p_0
                #pragma omp parallel for
                for (size_t i = 0; i < to_be_sorted_0.size(); ++i) {
                    to_be_sorted_0[i] = p_0[to_be_sorted_0[i]];
                }
                std::copy(to_be_sorted_0.begin(), to_be_sorted_0.end(), p_0.begin());
                
                //calculate_t_0
                phase.split("Calculate t_0");
                calculate_t_0<util::character>(text, out_sa, 0, sa_12.size(), sa_12.size(), 
                    p_0.size(), count_s_type_pos);
                util::span<sa_index> t_0 = p_0;
                
                //induce SA_0
                phase.split("Induce SA_0");
                util::span<sa_index> isa_12 = out_sa.slice(count_s_type_pos+mod_0, 2*count_s_type_pos);
                util::span<sa_index> sa_0 = out_sa.slice(count_s_type_pos, count_s_type_pos+mod_0);
                #pragma omp parallel for
                for (size_t i = 0; i < sa_12.size(); ++i) {
                    isa_12[sa_12[i]] = i;
                }
                util::induce_sa_dc<size_t>(t_0, isa_12, sa_0);
                std::copy(sa_0.begin(), sa_0.end(), t_0.begin());
                sa_0 = t_0;
                
                /* positions in sa_0 are multiplied by 3 so divide by 3 */
                #pragma omp parallel for
                for (size_t i = 0; i < sa_0.size(); ++i) { sa_0[i] = sa_0[i]/3; }
                
                //update SA(t_0) and SA(t_12) with position arrays
                phase.split("Calculate Position arrays again");
                p_0 = out_sa.slice(count_s_type_pos, count_s_type_pos+mod_0);
                p_12 = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1+mod_2);
                
                calculate_p_0(text, p_0, count_s_type_pos);
                calculate_p_12(text, p_12, count_s_type_pos);
                
                phase.split("Update indices of SA_0 and SA_12 with position arrays");
                #pragma omp parallel for
                for (size_t i = 0; i < sa_0.size(); ++i) {
                    sa_0[i] = p_0[sa_0[i]];
                }
                
                #pragma omp parallel for
                for (size_t i = 0; i < sa_12.size(); ++i) {
                    sa_12[i] = p_12[sa_12[i]];
                }
                
                // revert sa_0 and sa_12 so we can traverse them easier later
                revert(sa_0);
                revert(sa_12);
                
                // Swap sa_0 and sa_12
                auto sa_0_tmp = out_sa.slice(count_s_type_pos, count_s_type_pos+sa_0.size());
                auto sa_12_tmp = out_sa.slice(count_s_type_pos+sa_0.size(), 2*count_s_type_pos);
                std::copy(sa_0.begin(), sa_0.end(), sa_0_tmp.begin());
                std::copy(sa_12.begin(), sa_12.end(), sa_12_tmp.begin());
                sa_0 = out_sa.slice(0, sa_0.size());
                sa_12 = out_sa.slice(sa_0.size(), count_s_type_pos);
                std::copy(sa_0_tmp.begin(), sa_0_tmp.end(), sa_0.begin());
                std::copy(sa_12_tmp.begin(), sa_12_tmp.end(), sa_12.begin());
                
                // copy sa_0 and sa_12 into l-type-positions of out_sa
                phase.split("Copy SA_0 and SA_12 into l-type-positions of out_sa");
                size_t h = 0;
                size_t curr_pos_sa_0 = sa_0.size();
                size_t curr_pos_sa_12 = sa_12.size();
                util::character last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) { 
                        if (curr_pos_sa_12 > 0) { out_sa[i-1] = sa_12[--curr_pos_sa_12]; }
                        else if (curr_pos_sa_0 > 0) { 
                            if (curr_pos_sa_0 == sa_0.size()) { h = i-1; }
                            out_sa[i-1] = sa_0[--curr_pos_sa_0]; 
                        }
                        else { break; }
                    }
                    last_char = text[i-1];
                }
                
                // calculate isa_0 and isa_12 into s-type-positions of out_sa
                phase.split("Calculate ISA_0 and ISA_12 into s-type-positions of out_sa");
                size_t count = 0; // only l-type-positions with sa elements
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (count == sa_0.size()+sa_12.size()) { break; }
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) { 
                        out_sa[out_sa[i-1]] = i-1; 
                        ++count;
                    }
                    last_char = text[i-1];
                }
                
                // merge sa_0 and sa_12 by calculating positions in merged sa
                phase.split("Merge SA_0 and SA_12");
                curr_pos_sa_0 = h+1;
                util::character last_char_sa_12 = util::SENTINEL;
                curr_pos_sa_12 = out_sa.size()+1;
                size_t count_sa_0 = 0;
                size_t count_sa_12 = 0;
                size_t pos_in_merged_sa = 0;
                bool s_type_sa_0 = false;
                bool s_type_sa_12 = true;
                // traverse l-type-positions
                while (count_sa_0 < sa_0.size() && count_sa_12 < sa_12.size()) {
                    // get next index for sa_0
                    while (s_type_sa_0) { 
                        --curr_pos_sa_0;
                        if (curr_pos_sa_0 == 0) { break; }
                        if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                        else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                    }
                    if (curr_pos_sa_0 == 0) { break; }
                    // get next index for sa_12
                    while (s_type_sa_12) { 
                        --curr_pos_sa_12;
                        if (curr_pos_sa_12-1 == h) { break; }
                        if (text[curr_pos_sa_12-1] > last_char_sa_12) { s_type_sa_12 = false; }
                        else if (text[curr_pos_sa_12-1] < last_char_sa_12) { s_type_sa_12 = true; }
                        last_char_sa_12 = text[curr_pos_sa_12-1];
                    }
                    if (curr_pos_sa_12-1 == h) { break; }
                    
                    util::string_span t_0;
                    util::string_span t_12;
                    
                    bool t_12_is_residue_1 = false;
                    util::string_span check_residue_t_12 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_12-1], 1);
                    size_t end_pos = ((size_t)out_sa[curr_pos_sa_12-1])+check_residue_t_12.size()-1;
                    if (out_sa[end_pos] > h) { t_12_is_residue_1 = true; }
                    
                    if (t_12_is_residue_1) {
                        t_0 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_0-1], 1);
                        t_12 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_12-1], 1);
                    }
                    else {
                        t_0 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_0-1], 2);
                        t_12 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_12-1], 2);
                    }
                    
                    const bool less_than = comp_z_strings(t_0, t_12);
                    const bool eq = util::as_equal(comp_z_strings)(t_0, t_12);
                    const bool lesser_suf = out_sa[((size_t)out_sa[curr_pos_sa_0-1])+t_0.size()-1] >   // greater because sa_0 and sa_12 are stored from right to left
                            out_sa[((size_t)out_sa[curr_pos_sa_12-1])+t_12.size()-1];
                    
                    if (less_than || (eq && lesser_suf)) {
                        out_sa[(curr_pos_sa_0--)-1] = pos_in_merged_sa++; 
                        ++count_sa_0;
                        if (curr_pos_sa_0 == 0) { break; }
                        if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                        else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                    }
                    else { 
                        out_sa[(curr_pos_sa_12--)-1] = pos_in_merged_sa++;
                        ++count_sa_12;
                        if (curr_pos_sa_12-1 == h) { break; }
                        if (text[curr_pos_sa_12-1] > last_char_sa_12) { s_type_sa_12 = false; }
                        else if (text[curr_pos_sa_12-1] < last_char_sa_12) { s_type_sa_12 = true; }
                        last_char_sa_12 = text[curr_pos_sa_12-1];
                    }
                }
                // There are positions in sa_0 left
                while (count_sa_0 < sa_0.size()) {
                    // get next index for sa_0
                    while (s_type_sa_0) { 
                        --curr_pos_sa_0;
                        if (curr_pos_sa_0 == 0) { break; }
                        if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                        else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                    }
                    if (curr_pos_sa_0 == 0) { break; }
                    out_sa[(curr_pos_sa_0--)-1] = pos_in_merged_sa++;
                    ++count_sa_0;
                    if (curr_pos_sa_0 == 0) { break; }
                    if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                    else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                }
                // There are positions in sa_12 left
                while (count_sa_12 < sa_12.size()) {
                    // get next index for sa_12
                    while (s_type_sa_12) { 
                        --curr_pos_sa_12;
                        if (curr_pos_sa_12-1 == h) { break; }
                        if (text[curr_pos_sa_12-1] > last_char_sa_12) { s_type_sa_12 = false; }
                        else if (text[curr_pos_sa_12-1] < last_char_sa_12) { s_type_sa_12 = true; }
                        last_char_sa_12 = text[curr_pos_sa_12-1];
                    }
                    if (curr_pos_sa_12-1 == h) { break; }
                    out_sa[(curr_pos_sa_12--)-1] = pos_in_merged_sa++;
                    ++count_sa_12;
                    if (curr_pos_sa_12-1 == h) { break; }
                    if (text[curr_pos_sa_12-1] > last_char_sa_12) { s_type_sa_12 = false; }
                    else if (text[curr_pos_sa_12-1] < last_char_sa_12) { s_type_sa_12 = true; }
                    last_char_sa_12 = text[curr_pos_sa_12-1];
                }
                
                /* update isa_0 and isa_12 with positions in merged arrays to calculate isa_012 */
                phase.split("Update ISA_0 and ISA_12 with positions in merged arrays");
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { out_sa[i-1] = out_sa[out_sa[i-1]]; }
                    last_char = text[i-1];
                }
                
                /* move isa_012 to the end of out_sa */
                phase.split("Move ISA_012 to the end of out_sa");
                last_char = util::SENTINEL;
                s_type = true;
                size_t counter = text.size()-1;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { out_sa[counter--] = out_sa[i-1]; }
                    last_char = text[i-1];
                }
                
                /* calculate sa_012 by traversing isa_012 */
                phase.split("Calculate SA_012");
                util::span<sa_index> sa_012 = out_sa.slice(0, count_s_type_pos);
                util::span<sa_index> isa_012 = out_sa.slice(out_sa.size()-count_s_type_pos, out_sa.size());
                #pragma omp parallel for
                for (size_t i = 0; i < sa_012.size(); ++i) {
                    sa_012[isa_012[i]] = i;
                }
                
                /* calculate position array of s-type-positions in right half */
                phase.split("Calculate position array of s-type-positions");
                util::span<sa_index> p_012 = out_sa.slice(out_sa.size()-count_s_type_pos, out_sa.size());
                s_type = true;
                last_char = util::SENTINEL;
                counter = p_012.size()-1;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { p_012[counter--] = i-1; }
                    last_char = text[i-1];
                }
                
                /* update sa_012 with positions in p_012 */ 
                phase.split("Update SA_012 with s-type-positions");
                #pragma omp parallel for
                for (size_t i = 0; i < sa_012.size(); ++i) {
                    sa_012[i] = p_012[sa_012[i]];
                }
                
                /* induction scan to calculate correct sa */
                phase.split("Induce SA");
                left_induction_scan(text, out_sa, alphabet.size_with_sentinel()+1, count_s_type_pos);
                
                /* if text was reverted at the beginning out_sa must be reverted */
                if (is_text_reverted) { 
                    phase.split("Revert SA");
                    revert(out_sa); 
                }
            }
          
        private:
            template<typename T, typename sa_index>
            static void left_induction_scan(const T& text, util::span<sa_index> out_sa, const size_t alphabet_size, const size_t count_s_type_pos) {
                const size_t UNDEFINED = text.size();
                
                /* calculate bucket sizes */ 
                auto buckets = util::make_container<size_t>(alphabet_size);
                for (size_t i = 0; i < text.size(); ++i) { buckets[text[i]]++; }
                
                /* calculate last index of buckets */
                size_t sum = 0;
                for (size_t i = 0; i < buckets.size(); ++i) {
                    sum += buckets[i];
                    buckets[i] = sum-1;
                }
                
                /* move sa_s to the end of their buckets */
                for (size_t i = count_s_type_pos; i < out_sa.size(); ++i) { out_sa[i] = UNDEFINED; }
                for (size_t i = count_s_type_pos; i > 0; --i) {
                    auto elem = text[out_sa[i-1]];
                    if (buckets[elem] != i-1) {
                        out_sa[buckets[elem]] = out_sa[i-1];
                        out_sa[i-1] = UNDEFINED;
                    }
                    if (buckets[elem] > 0) { --buckets[elem]; }
                }
                
                /* reset buckets */
                for (size_t i = 0; i < buckets.size(); ++i) { buckets[i] = 0; }
                
                /* calculate bucket sizes */ 
                for (size_t i = 0; i < text.size(); ++i) { buckets[text.at(i)]++; }
                
                /* calculate front index of buckets */
                sum = 0;
                for (size_t i = 0; i < buckets.size(); ++i) {
                    size_t old = buckets[i];
                    buckets[i] = sum;
                    sum += old;
                }
                
                /* induction scan */
                for (size_t i = 0; i < out_sa.size(); ++i) {
                    if (out_sa[i] != UNDEFINED && out_sa[i] != 0u) {
                        size_t curr_pos = out_sa[i];
                        size_t pre_pos = ((size_t)out_sa[i])-1u;
                        bool curr_pos_is_s_type = (i < buckets[text[curr_pos]]); // bucket pointer points at l-type-positions 
                        if (text[pre_pos] > text[curr_pos] || (text[pre_pos] == text[curr_pos] && curr_pos_is_s_type)) {
                            auto elem = text[pre_pos];
                            out_sa[buckets[elem]] = pre_pos;
                            ++buckets[elem];
                        }
                    }
                }
            }
            
            template<typename T, typename sa_index>
            static void calculate_p_0(const T& text, 
                    const util::span<sa_index> p_0, size_t count_s_type_pos) {
                size_t size_mod_0 = count_s_type_pos/3 + (count_s_type_pos % 3 > 0);
                auto mod_0 = p_0.slice(0, size_mod_0);
                // field which indicates where next position must be saved 
                size_t mod = (count_s_type_pos+3-1) % 3;  
                /* fields which indicate how many position of mod_i are 
                   currently saved */
                size_t count_mod_0 = 0;
                size_t count_mod_1 = 0;
                size_t count_mod_2 = 0;  

                // save s-type-positions in the correct arrays
                util::character last_char = util::SENTINEL;
                bool s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) {
                        if (mod == 0) { mod_0[count_mod_0++] = i-1; }
                        else if (mod == 1) { ++count_mod_1; }
                        else { ++count_mod_2; }
                        
                        mod = (mod+3-1) % 3;
                    }
                    last_char = text[i-1];
                }
                
                // arrays must be reverted, because we started at the last index
                // TODO: since we know the memory sizes, we can save the positions directly at the right index
                revert(mod_0);
            }    
            
            template<typename T, typename sa_index>
            static void calculate_p_12(const T& text, 
                const util::span<sa_index> p_12, size_t count_s_type_pos) {
                size_t size_mod_1 = count_s_type_pos/3 + (count_s_type_pos % 3 > 1);
                size_t size_mod_2 = count_s_type_pos/3;
                auto mod_1 = p_12.slice(0, size_mod_1);
                auto mod_2 = p_12.slice(size_mod_1, size_mod_1+size_mod_2);
                // field which indicates where next position must be saved 
                size_t mod = (count_s_type_pos+3-1) % 3;  
                /* fields which indicate how many position of mod_i are 
                   currently saved */
                size_t count_mod_0 = 0;
                size_t count_mod_1 = 0;
                size_t count_mod_2 = 0;  

                // save s-type-positions in the correct arrays
                util::character last_char = util::SENTINEL;
                bool s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) {
                        if (mod == 0) { ++count_mod_0; }
                        else if (mod == 1) { mod_1[count_mod_1++] = i-1; }
                        else { mod_2[count_mod_2++] = i-1; }
                        
                        mod = (mod+3-1) % 3;
                    }
                    last_char = text[i-1];
                }
                
                // arrays must be reverted, because we started at the last index
                // TODO: since we know the memory sizes, we can save the positions directly at the right index
                revert(mod_1);
                revert(mod_2);
            }
            
            template<typename C, typename T, typename sa_index>
            static void calculate_t_12(const T& text, util::span<sa_index> out_sa, size_t start_p_12, 
                    size_t length_p_12, size_t count_s_type_pos, bool& recursion, size_t& rank) {
                // revert p_0 and p_12 so we can access them later from left to right
                util::span<sa_index> p_12 = out_sa.slice(start_p_12, start_p_12+length_p_12);
                revert(p_12);
                
                // Copy p_12 in L-type positions in out_sa
                size_t curr_pos_p_12 = start_p_12+length_p_12;
                size_t end_pos_p_12 = -1;
                
                util::character last_char = util::SENTINEL;
                bool s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) {
                        if (curr_pos_p_12 == 0) {
                            end_pos_p_12 = i-1;
                            break; 
                        }
                        out_sa[i-1] = out_sa[--curr_pos_p_12];
                    }
                    last_char = text[i-1];
                }
                
                /* Determine lexicographical ranks of Positions p_12 and save
                   them in correct positions in out_sa */
                util::span<const C> last_t;
                size_t last_i = -1;
                
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) { 
                        if (i-1 == end_pos_p_12) { break; }
                        auto curr_t = retrieve_s_string<C>(text, out_sa[i-1], 3);
                        if (!last_t.empty()) {
                            out_sa[out_sa[last_i-1]] = rank;
                            if (comp_z_strings(last_t, curr_t)) { ++rank; }
                            else { recursion = true; }
                        }
                        last_t = curr_t;
                        last_i = i;
                    }
                    last_char = text[i-1];
                }
                if (length_p_12 > 0) { out_sa[out_sa[last_i-1]] = rank; }
                
                /* Determine t_12 by looking up the lexicographical ranks 
                   in out_sa and save them in l-type-positions of out_sa in reverted order*/
                size_t mod = (count_s_type_pos+3-1) % 3;  
                last_char = util::SENTINEL;
                s_type = true;
                size_t last_l_type = -1;
                
                /* calculate first l-type-position */
                util::character last_char_in_l_loop = util::SENTINEL;
                bool s_type_in_l_loop = true;
                for (size_t j = text.size(); j > 0; --j) {
                    if (text[j-1] > last_char_in_l_loop) { s_type_in_l_loop = false; }
                    else if (text[j-1] < last_char_in_l_loop) { s_type_in_l_loop = true; }
                    
                    if (!s_type_in_l_loop) { 
                        
                        last_l_type = j-1;
                        break;
                    }
                    last_char_in_l_loop = text[j-1];
                } 
                
                mod = (count_s_type_pos+3-1) % 3;  
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { 
                        if (mod == 1) {
                            out_sa[last_l_type] = out_sa[i-1];
                            s_type_in_l_loop = false;
                            for (size_t j = last_l_type; j > 0; --j) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    last_l_type = j-1;
                                    break;
                                }
                            } 
                        }
                        mod = (mod+3-1) % 3; 
                    }
                    last_char = text[i-1];
                }
                
                mod = (count_s_type_pos+3-1) % 3;  
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { 
                        if (mod == 2) {
                            out_sa[last_l_type] = out_sa[i-1];
                            bool s_type_in_l_loop = false;
                            for (size_t j = last_l_type; j > 0; --j) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    last_l_type = j-1;
                                    break;
                                }
                            } 
                        }
                        mod = (mod+3-1) % 3; 
                    }
                    last_char = text[i-1];
                }
                
                /* move l-type-positions to the end of out_sa */
                size_t counter = text.size()-1;
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) { out_sa[counter--] = out_sa[i-1]; }
                    last_char = text[i-1];
                }
                
                /* sizes of t_1 and t_2 */
                size_t size_t_1 = count_s_type_pos/3 + (count_s_type_pos % 3 > 1);
                size_t size_t_2 = count_s_type_pos/3;
                
                /* move t_1 and t_2 to the begin of out_sa */
                #pragma omp parallel for
                for (size_t i = text.size()-1; i > text.size()-size_t_1-size_t_2-1; --i) {
                    out_sa[text.size()-(i+1)] = out_sa[i];
                }
                
                /* revert t_1 and t_2 */
                util::span<sa_index> t_1 = out_sa.slice(0, size_t_1);
                util::span<sa_index> t_2 = out_sa.slice(size_t_1, size_t_1+size_t_2);
                revert(t_1);
                revert(t_2);
            }
            
            template<typename C, typename T, typename sa_index>
            static void calculate_t_0(const T& text, util::span<sa_index> out_sa, 
                    size_t start_sa_12, size_t length_sa_12, size_t start_p_0, 
                    size_t length_p_0, size_t count_s_type_pos) {
                util::span<sa_index> sa_12 = out_sa.slice(start_sa_12, start_sa_12+length_sa_12);
                util::span<sa_index> p_0 = out_sa.slice(start_p_0, start_p_0+length_p_0);
                // revert p_0 so we can access it later from left to right
                revert(p_0);
                
                // Copy p_0 and sa_12 in L-type positions in out_sa
                size_t curr_pos_p_0 = start_p_0+length_p_0;
                size_t curr_pos_sa_12 = start_sa_12+length_sa_12;
                size_t start_pos_sa_12 = -1;
                
                util::character last_char = util::SENTINEL;
                bool s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) {
                        if (curr_pos_p_0 > start_p_0) { out_sa[i-1] = out_sa[--curr_pos_p_0]; }
                        else {
                            if (curr_pos_sa_12 == start_sa_12+length_sa_12) { start_pos_sa_12 = i-1; }
                            if (curr_pos_sa_12 == 0) { break; }
                            out_sa[i-1] = out_sa[--curr_pos_sa_12]; 
                        }
                    }
                    last_char = text[i-1];
                    if (start_sa_12+length_sa_12 == 0) { start_pos_sa_12 = i-2; }
                    if (curr_pos_sa_12 == start_sa_12) { break; }
                }
                
                /* Determine lexicographical ranks of Positions p_0 and save
                   them in correct positions in out_sa */
                size_t rank = 1;
                util::span<const C> last_t;
                size_t last_i = -1;
                
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (i-1 == start_pos_sa_12) { break; }
                    
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) {
                        auto curr_t = retrieve_s_string<C>(text, out_sa[i-1], 3);
                        if (!last_t.empty()) {
                            out_sa[out_sa[last_i-1]] = rank;
                            if (comp_z_strings(last_t, curr_t)) { ++rank; }
                        }
                        last_t = curr_t;
                        last_i = i;
                    }
                    last_char = text[i-1];
                }
                if (length_p_0 > 0) { out_sa[out_sa[last_i-1]] = rank; }
                
                /* Determine t_0 by looking up the lexicographical ranks 
                   in out_sa and save them in l-type-positions of out_sa in reverted order*/
                size_t mod = (count_s_type_pos+3-1) % 3;  
                last_char = util::SENTINEL;
                s_type = true;
                size_t last_l_type = -1;
                
                /* calculate first l-type-position */
                util::character last_char_in_l_loop = util::SENTINEL;
                bool s_type_in_l_loop = true;
                for (size_t j = text.size(); j > 0; --j) {
                    if (text[j-1] > last_char_in_l_loop) { s_type_in_l_loop = false; }
                    else if (text[j-1] < last_char_in_l_loop) { s_type_in_l_loop = true; }
                    
                    if (!s_type_in_l_loop) { 
                        
                        last_l_type = j-1;
                        break;
                    }
                    last_char_in_l_loop = text[j-1];
                } 
                            
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { 
                        if (mod == 0) {
                            out_sa[last_l_type] = out_sa[i-1];
                            s_type_in_l_loop = false;
                            for (size_t j = last_l_type; j > 0; --j) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    last_l_type = j-1;
                                    break;
                                }
                            } 
                        }
                        mod = (mod+3-1) % 3; 
                    }
                    last_char = text[i-1];
                }

                /* move l-type-positions to the end of out_sa */
                size_t counter = text.size()-1;
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) { out_sa[counter--] = out_sa[i-1]; }
                    last_char = text[i-1];
                }
                
                /* move sa_12 and t_0 to the begin of out_sa */
                size_t p_0_first = text.size()-length_p_0;
                #pragma omp parallel for
                for (size_t i = length_p_0; i > 0; --i) {
                    p_0[i-1] = out_sa[p_0_first+i-1];
                }
                size_t sa_12_first = text.size()-length_p_0-length_sa_12;
                #pragma omp parallel for
                for (size_t i = length_sa_12; i > 0; --i) {
                    sa_12[i-1] = out_sa[sa_12_first+i-1];
                }
            }
            
            template<typename C, typename T>
            static util::span<const C> retrieve_s_string(T& text, size_t s_pos, size_t count) {
                size_t curr_s_pos = s_pos;
                for (size_t c = 1; c <= count; ++c) {
                    if (curr_s_pos == text.size()-1) { break; }
                    if (text[curr_s_pos] == text[curr_s_pos+1]) { ++curr_s_pos; }
                    else {
                        size_t k = curr_s_pos+2;
                        while (k < text.size() && text[k-1] >= text[k]) { ++k; }
                        size_t j = k-1;
                        while (j > curr_s_pos+1 && text[j-1] <= text[j]) { --j; }
                        curr_s_pos = j;
                    }
                }
                return util::span<const C>(&text[s_pos], curr_s_pos-s_pos+1); 
            }
            
            template<typename C, typename T>
            static util::span<const C> retrieve_triplets(T& text, size_t pos, size_t count) {
                if((pos+count) < text.size()){
                    return util::span<const C>(&text[pos], count); 
                }else{
                    return util::span<const C>(&text[pos], text.size()-pos); 
                }
            }
            
            template<typename A>
            static void revert(const A& a) {
                #pragma omp parallel for
                for (size_t i = 0; i < a.size()/2; ++i) {
                    auto tmp = a[i];
                    a[i] = a[a.size()-1-i];
                    a[a.size()-1-i] = tmp;
                }
            }
            
            //shorter strings have higher priority
            static bool comp_z_strings (util::string_span t_0, util::string_span t_12) {
                size_t min_length;
                if (t_0.size() <= t_12.size()) { min_length = t_0.size(); }
                else { min_length = t_12.size(); }
                
                util::string_span t_0_slice = t_0.slice(0, min_length);
                util::string_span t_12_slice = t_12.slice(0, min_length);
                
                return t_0_slice < t_12_slice || (t_0_slice == t_12_slice && t_0.size() > t_12.size());
            }
    }; // class nzsufsort_par
} // namespace sacabench::nzsufsort_par