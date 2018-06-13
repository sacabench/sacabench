/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
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


namespace sacabench::nzSufSort {

    class nzSufSort {
        public:    
            static constexpr size_t EXTRA_SENTINELS = 1;

            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     util::alphabet const& alphabet,
                                     util::span<sa_index> out_sa) {
                                         
                std::cout << "Running nzSufSort" << std::endl;
                
                // count number of s-type-positions in text
                size_t count_s_type_pos = 1;
                bool s_type = true;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { count_s_type_pos++; }
                }
                
                //TODO: Annahme dass mehr S-Typ-Positionen existieren
                DCHECK_MSG(count_s_type_pos <= text.size()/2, 
                    "There are more S-Type-Positions than L-Type-Positions");
                
                // calculate position arrays of s-type-positions
                size_t mod_0 = count_s_type_pos/3 + (count_s_type_pos % 3 > 0);
                size_t mod_1 = count_s_type_pos/3 + (count_s_type_pos % 3 > 1);
                size_t mod_2 = count_s_type_pos/3;
                util::span<sa_index> p_0 = out_sa.slice(0, mod_0);
                util::span<sa_index> p_12 = out_sa.slice(mod_0, count_s_type_pos);
                util::span<sa_index> u = out_sa.slice(count_s_type_pos, count_s_type_pos+mod_0);
                util::span<sa_index> v = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1);
                util::span<sa_index> w = out_sa.slice(count_s_type_pos+mod_0+mod_1, count_s_type_pos+mod_0+mod_1+mod_2);
                std::cout << "Running nzSufSort" << std::endl;
                calculate_position_arrays(text, p_0, p_12, u, v, w, count_s_type_pos);
                
                //check for correct position arrays
                std::cout << "p_0: " << p_0 << std::endl;
                std::cout << "p_12: " << p_12 << std::endl;
                
                //TODO sort p_0 and p_12 with radix sort
                auto comp = [&](size_t i, size_t j) {
                    util::string_span t_1 = retrieve_s_string<util::character>(text, i, 3);
                    util::string_span t_2 = retrieve_s_string<util::character>(text, j, 3);
                    return t_1 < t_2;
                };
                std::sort(p_0.begin(), p_0.end(), comp);
                std::sort(p_12.begin(), p_12.end(), comp);
                
                //check for correct sorted position arrays
                std::cout << "p_0: " << p_0 << std::endl;
                std::cout << "p_12: " << p_12 << std::endl;
                
                //calculate t_0 and t_12 in the begin of out_sa
                determine_leq<util::character>(text, out_sa, 0, mod_0, mod_0, count_s_type_pos-mod_0, count_s_type_pos);
                util::span<sa_index> t_0 = out_sa.slice(0, mod_0);
                util::span<sa_index> t_12 = out_sa.slice(mod_0, count_s_type_pos);
                std::cout << "t_0: " << t_0 << std::endl;
                std::cout << "t_12: " << t_12 << std::endl;
                
                //calculate SA(t_12) by calling the lightweight variant of DC3
                //TODO
                util::span<sa_index> tmp_out_sa = out_sa.slice(count_s_type_pos+mod_0, 2*count_s_type_pos);
                naive_sa(t_12, tmp_out_sa);
                for (size_t i = 0; i < t_12.size(); i++) {
                    t_12[i] = tmp_out_sa[i];
                }
                util::span<sa_index> sa_12 = t_12;
                
                std::cout << "sa_12: " << sa_12 << std::endl;
                
                //induce SA(t_0)
                util::span<sa_index> isa_12 = tmp_out_sa;
                for (size_t i = 0; i < sa_12.size(); i++) {
                    isa_12[sa_12[i]] = i;
                }
                util::induce_sa_dc<size_t>(t_0, isa_12, t_0);
                util::span<sa_index> sa_0 = t_0;
                
                /* positions in sa_0 are multiplied by 3 so divide by 3 */
                for (size_t i = 0; i < sa_0.size(); i++) { sa_0[i] = sa_0[i]/3; }
                
                std::cout << "sa_0: " << sa_0 << std::endl;
                
                //update SA(t_0) and SA(t_12) with position arrays
                p_0 = out_sa.slice(count_s_type_pos, count_s_type_pos+mod_0);
                p_12 = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1+mod_2);
                util::span<sa_index> p_1 = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1);
                util::span<sa_index> p_2 = out_sa.slice(count_s_type_pos+mod_0+mod_1, count_s_type_pos+mod_0+mod_1+mod_2);
                
                calculate_position_arrays(text, p_0, p_12, p_0, p_1, p_2, count_s_type_pos);
                
                for (size_t i = 0; i < sa_0.size(); i++) {
                    sa_0[i] = p_0[sa_0[i]];
                }
                
                for (size_t i = 0; i < sa_12.size(); i++) {
                    sa_12[i] = p_12[sa_12[i]];
                }
                
                std::cout << "sa_0: " << sa_0 << std::endl;
                std::cout << "sa_12: " << sa_12 << std::endl;
                
                // revert sa_0 and sa_12 so we can traverse them easier later
                revert(sa_0);
                revert(sa_12);
                
                // copy sa_0 and sa_12 into l-type-positions of out_sa
                std::cout << "out_sa: " << out_sa << std::endl;
                size_t h = 0;
                size_t curr_pos_sa_0 = sa_0.size();
                size_t curr_pos_sa_12 = sa_12.size();
                s_type = true;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (!s_type) { 
                        if (curr_pos_sa_12 > 0) { out_sa[i-1] = sa_12[--curr_pos_sa_12]; }
                        else if (curr_pos_sa_0 > 0) { 
                            if (curr_pos_sa_0 == sa_0.size()) { h = i-1; }
                            out_sa[i-1] = sa_0[--curr_pos_sa_0]; 
                        }
                        else { break; }
                    }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                
                // calculate isa_0 and isa_12 into s-type-positions of out_sa
                size_t count = 0; // only l-type-positions with sa elements
                s_type = true;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (count == sa_0.size()+sa_12.size()) { break; }
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (!s_type) { 
                        out_sa[out_sa[i-1]] = i-1; 
                        ++count;
                    }
                }
                std::cout << "after calculating isa_0 and isa_12: " << std::endl;
                std::cout << "out_sa: " << out_sa << std::endl;
                
                // merge sa_0 and sa_12 by calculating positions in merged sa
                curr_pos_sa_0 = h+1;
                curr_pos_sa_12 = out_sa.size()-1;
                size_t count_sa_0 = 0;
                size_t count_sa_12 = 0;
                size_t pos_in_merged_sa = 0;
                bool s_type_sa_0 = false;
                bool s_type_sa_12 = false;
                // traverse l-type-positions
                while (count_sa_0 < sa_0.size() && count_sa_12 < sa_12.size()) {
                    // get next index for sa_0
                    while (s_type_sa_0) { 
                        curr_pos_sa_0--;
                        if (curr_pos_sa_0 == 0) { break; }
                        if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                        else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                    }
                    if (curr_pos_sa_0 == 0) { break; }
                    // get next index for sa_12
                    while (s_type_sa_12) { 
                        curr_pos_sa_12--;
                        if (curr_pos_sa_12-1 == h) { break; }
                        if (text[curr_pos_sa_12-1] > text[curr_pos_sa_12]) { s_type_sa_12 = false; }
                        else if (text[curr_pos_sa_12-1] < text[curr_pos_sa_12]) { s_type_sa_12 = true; }
                    }
                    if (curr_pos_sa_12-1 == h) { break; }
                    
                    util::string_span t_0;
                    util::string_span t_12;
                    if (out_sa[curr_pos_sa_12-1] % 3 == 1) {
                        t_0 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_0-1], 1);
                        t_12 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_12-1], 1);
                    }
                    else {
                        t_0 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_0-1], 2);
                        t_12 = retrieve_s_string<util::character>(text, out_sa[curr_pos_sa_12-1], 2);
                    }
                    
                    //TODO: shorter strings have higher priority
                    const bool less_than = t_0 < t_12;
                    const bool eq = t_0 == t_12;
                    // NB: This is a closure so that we can evaluate it later, because
                    // evaluating it if `eq` is not true causes out-of-bounds errors.
                    auto lesser_suf = [&]() {
                        return out_sa[out_sa[curr_pos_sa_0-1]+t_0.size()-1] >   // greater because sa_0 and sa_12 are stored from right to left
                            out_sa[out_sa[curr_pos_sa_12-1]+t_12.size()-1];
                    };
                    if (less_than || (eq && lesser_suf())) {
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
                        if (text[curr_pos_sa_12-1] > text[curr_pos_sa_12]) { s_type_sa_12 = false; }
                        else if (text[curr_pos_sa_12-1] < text[curr_pos_sa_12]) { s_type_sa_12 = true; }
                    }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                // There are positions in sa_0 left
                while (count_sa_0 < sa_0.size()) {
                    // get next index for sa_0
                    while (s_type_sa_0) { 
                        curr_pos_sa_0--;
                        if (curr_pos_sa_0 == 0) { break; }
                        if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                        else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                    }
                    if (curr_pos_sa_0 == 0) { break; }
                    out_sa[(curr_pos_sa_0--)-1] = pos_in_merged_sa++;
                    ++count_sa_0;
                    if (curr_pos_sa_12 == 0) { break; }
                    if (text[curr_pos_sa_0-1] > text[curr_pos_sa_0]) { s_type_sa_0 = false; }
                    else if (text[curr_pos_sa_0-1] < text[curr_pos_sa_0]) { s_type_sa_0 = true; }
                }
                // There are positions in sa_12 left
                while (count_sa_12 < sa_12.size()) {
                    // get next index for sa_12
                    while (s_type_sa_12) { 
                        curr_pos_sa_12--;
                        if (curr_pos_sa_12-1 == h) { break; }
                        if (text[curr_pos_sa_12-1] > text[curr_pos_sa_12]) { s_type_sa_12 = false; }
                        else if (text[curr_pos_sa_12-1] < text[curr_pos_sa_12]) { s_type_sa_12 = true; }
                    }
                    if (curr_pos_sa_12-1 == h) { break; }
                    out_sa[(curr_pos_sa_12--)-1] = pos_in_merged_sa++;
                    ++count_sa_12;
                    if (curr_pos_sa_12-1 == h) { break; }
                    if (text[curr_pos_sa_12-1] > text[curr_pos_sa_12]) { s_type_sa_12 = false; }
                    else if (text[curr_pos_sa_12-1] < text[curr_pos_sa_12]) { s_type_sa_12 = true; }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                
                /* update isa_0 and isa_12 with positions in merged arrays to calculate isa_012 */
                s_type = true;
                out_sa[text.size()-1] = out_sa[out_sa[text.size()-1]];
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { out_sa[i-1] = out_sa[out_sa[i-1]]; }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                
                /* move isa_012 to the end of out_sa */
                s_type = true;
                size_t counter = text.size()-2;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { out_sa[counter--] = out_sa[i-1]; }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                
                /* calculate sa_012 by traversing isa_012 */
                util::span<sa_index> sa_012 = out_sa.slice(0, count_s_type_pos);
                util::span<sa_index> isa_012 = out_sa.slice(out_sa.size()-count_s_type_pos, out_sa.size()); 
                std::cout << "sa_012: " << sa_012 << std::endl;
                std::cout << "isa_012: " << isa_012 << std::endl;
                for (size_t i = 0; i < sa_012.size(); i++) {
                    sa_012[isa_012[i]] = i;
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                
                /* calculate position array of s-type-positions in right half */
                util::span<sa_index> p_012 = out_sa.slice(out_sa.size()-count_s_type_pos, out_sa.size());
                s_type = true;
                counter = p_012.size()-1;
                p_012[counter--] = text.size()-1;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { p_012[counter--] = i-1; }
                }
                std::cout << "p_012: " << p_012 << std::endl;
                
                /* update sa_012 with positions in p_012 */ 
                for (size_t i = 0; i < sa_012.size(); i++) {
                    sa_012[i] = p_012[sa_012[i]];
                }
                std::cout << "sa_012: " << sa_012 << std::endl;
                
                /* induction scan to calculate correct sa */
                std::cout << "start of induction scan" << std::endl;
                left_induction_scan(text, out_sa, alphabet.size_with_sentinel(), count_s_type_pos);
            }
          
        private:
            //TODO: Naive SACA for testing purposes until the lightweight DC3 is finished
            template<typename sa_index>
            static void naive_sa(util::span<sa_index> text,
                             util::span<sa_index> out_sa) {

                // Fill SA with all index positions
                for (size_t i = 0; i < out_sa.size(); i++) {
                    out_sa[i] = i;
                }

                // Construct a SA by sorting according
                // to the suffix starting at that index.
                util::sort::std_sort(
                    out_sa, util::compare_key([&](size_t i) { return text.slice(i); }));
            }
            
            template<typename T, typename sa_index>
            static void left_induction_scan(const T& text, util::span<sa_index> out_sa, const size_t alphabet_size, const size_t count_s_type_pos) {
                const size_t UNDEFINED = text.size();
                
                /* calculate bucket sizes */ 
                auto buckets = util::make_container<size_t>(alphabet_size);
                for (size_t i = 0; i < text.size(); i++) { buckets[text.at(i)]++; }
                std::cout << "buckets: " << buckets << std::endl;
                
                /* calculate last index of buckets */
                size_t sum = 0;
                for (size_t i = 0; i < buckets.size(); i++) {
                    sum += buckets[i];
                    buckets[i] = sum-1;
                }
                std::cout << "buckets: " << buckets << std::endl;
                
                /* move sa_s to the end of their buckets */
                //TODO: Here is a bug
                for (size_t i = count_s_type_pos; i < out_sa.size(); i++) { out_sa[i] = UNDEFINED; }
                std::cout << "out_sa: " << out_sa << std::endl;
                for (size_t i = count_s_type_pos; i > 0; i--) {
                    auto elem = text[out_sa[i-1]];
                    std::cout << "out_sa[i-1]: " << out_sa[i-1] << std::endl;
                    if (buckets[elem] != i-1) {
                        out_sa[buckets[elem]] = out_sa[i-1];
                        out_sa[i-1] = UNDEFINED;
                    }
                    if (buckets[elem] > 0) { buckets[elem]--; }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
                
                /* reset buckets */
                for (size_t i = 0; i < buckets.size(); i++) { buckets[i] = 0; }
                std::cout << "buckets: " << buckets << std::endl;
                
                /* calculate bucket sizes */ 
                for (size_t i = 0; i < text.size(); i++) { buckets[text.at(i)]++; }
                std::cout << "buckets: " << buckets << std::endl;
                
                /* calculate front index of buckets */
                sum = 0;
                for (size_t i = 0; i < buckets.size(); i++) {
                    size_t old = buckets[i];
                    buckets[i] = sum;
                    sum += old;
                }
                std::cout << "buckets: " << buckets << std::endl;
                
                /* induction scan */
                for (size_t i = 0; i < out_sa.size(); i++) {
                    if (out_sa[i] != UNDEFINED && out_sa[i] != 0) {
                        size_t curr_pos = out_sa[i];
                        size_t pre_pos = out_sa[i]-1;
                        bool curr_pos_is_s_type = (i < buckets[text[curr_pos]]); // bucket pointer points at l-type-positions 
                        if (text[pre_pos] > text[curr_pos] || (text[pre_pos] == text[curr_pos] && curr_pos_is_s_type)) {
                            auto elem = text[pre_pos];
                            out_sa[buckets[elem]] = pre_pos;
                            buckets[elem]++;
                        }
                    }
                }
                std::cout << "out_sa: " << out_sa << std::endl;
            }
            
            //TODO: we dont need the arrays mod_0, mod_1 and mod_2 since we can use p_0, p_1 and p_2
            template<typename T, typename sa_index>
            static void calculate_position_arrays(const T& text, 
                    const util::span<sa_index> p_0, const util::span<sa_index> p_12, 
                    const util::span<sa_index> mod_0, const util::span<sa_index> mod_1, 
                    const util::span<sa_index> mod_2, size_t count_s_type_pos) {
                // 
                size_t mod = (count_s_type_pos+3-1) % 3;  
                size_t count_mod_0 = 0;
                size_t count_mod_1 = 0;
                size_t count_mod_2 = 0;  

                // execute logic for sentinel
                if (mod == 0) { mod_0[count_mod_0++] = text.size()-1; }
                else if (mod == 1) { mod_1[count_mod_1++] = text.size()-1; }
                else { mod_2[count_mod_2++] = text.size()-1; }   
                mod = (mod+3-1) % 3;
                bool s_type = true;
                
                // save s-type-positions in the correct arrays
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) {
                        if (mod == 0) { mod_0[count_mod_0++] = i-1; }
                        else if (mod == 1) { mod_1[count_mod_1++] = i-1; }
                        else { mod_2[count_mod_2++] = i-1; }
                        
                        mod = (mod+3-1) % 3;
                    }
                }
                std::cout << "mod_0: " << mod_0 << std::endl;
                std::cout << "mod_1: " << mod_1 << std::endl;
                std::cout << "mod_2: " << mod_2 << std::endl;
                
                // arrays must be reverted, because we started at the last index
                // TODO: since we know the memory sizes, we can save the positions directly at the right index
                revert(mod_0);
                revert(mod_1);
                revert(mod_2);
                
                //copy positions in p_0 and p_12
                for (size_t i = 0; i < p_0.size(); i++) {
                    p_0[i] = mod_0[i];
                }
                size_t count_p_12 = 0;
                for (const auto elem : mod_1) {
                    p_12[count_p_12++] = elem;
                }
                for(const auto elem : mod_2) {
                    p_12[count_p_12++] = elem;
                }

            }
            
            template<typename C, typename T, typename sa_index>
            static void determine_leq(const T& text, util::span<sa_index> out_sa, size_t start_p_0, size_t length_p_0, size_t start_p_12, size_t length_p_12, size_t count_s_type_pos) {
                std::cout << "start determine_leq" << std::endl;
                std::cout << "start_p_12: " << start_p_12 << std::endl;
                
                // Copy p_0 in L-type positions in out_sa
                std::cout << out_sa << std::endl;
                
                size_t curr_pos_p_0 = start_p_0;
                size_t end_p_0 = 0;
                bool s_type = true;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (!s_type) { out_sa[i-1] = out_sa[curr_pos_p_0++]; }
                    if (curr_pos_p_0 >= start_p_0+length_p_0) { 
                        end_p_0 = i-1;
                        break; 
                    }
                }
                std::cout << out_sa << std::endl;
                
                /* Determine lexicographical ranks of Positions in p_12 and save
                   them in correct positions in out_sa */
                size_t rank = 1;
                for (size_t i = start_p_12+1; i < start_p_12+length_p_12; i++) {
                    //TODO: Länge < 2?
                    
                    auto t_1 = retrieve_s_string<C>(text, out_sa[i-1], 3);
                    auto t_2 = retrieve_s_string<C>(text, out_sa[i], 3);
                    out_sa[out_sa[i-1]] = rank;
                    if (t_1 < t_2) { rank++; }
                }
                out_sa[out_sa[start_p_12+length_p_12-1]] = rank;
                
                std::cout << out_sa << std::endl;
                
                /* Determine lexicographical ranks of Positions in p_0 and save
                   them in correct positions in out_sa */
                rank = 1;
                util::span<const C> last_t;
                size_t last_i;
                
                s_type = true;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (i-1 < end_p_0) { break; }
                    
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (!s_type) { 
                    
                        auto curr_t = retrieve_s_string<C>(text, out_sa[i-1], 3);
                        if (!last_t.empty()) {
                            out_sa[out_sa[last_i-1]] = rank;
                            if (last_t < curr_t) { rank++; }
                        }
                        last_t = curr_t;
                        last_i = i;
                    }
                }
                out_sa[out_sa[last_i-1]] = rank;
                
                std::cout << out_sa << std::endl;
                
                /* Determine t_0 and t_12 by looking up the lexicographical ranks 
                   in out_sa and save them in l-type-positions of out_sa in reverted order*/
                std::cout << "Determine t_0 and t_12" << std::endl;
                size_t mod = (count_s_type_pos+3-1) % 3;  
                s_type = true;
                size_t last_l_type = text.size()-1;
                if (mod == 0) {
                    bool s_type_in_l_loop = true;
                    for (size_t j = last_l_type; j > 0; j--) {
                        if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                        else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                        
                        if (!s_type_in_l_loop) { 
                            
                            last_l_type = j-1;
                            out_sa[j-1] = out_sa[text.size()-1];
                            break;
                        }
                    } 
                }
                mod = (mod+3-1) % 3; 
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { 
                        if (mod == 0) {
                            bool s_type_in_l_loop = false;
                            for (size_t j = last_l_type; j > 0; j--) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    
                                    last_l_type = j-1;
                                    out_sa[j-1] = out_sa[i-1];
                                    break;
                                }
                            } 
                        }
                        mod = (mod+3-1) % 3; 
                    }
                } 
                std::cout << "out_sa: " << out_sa << std::endl;
                
                mod = (count_s_type_pos+3-1) % 3;  
                s_type = true;
                if (mod == 1) {
                    bool s_type_in_l_loop = false;
                    for (size_t j = last_l_type; j > 0; j--) {
                        if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                        else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                        
                        if (!s_type_in_l_loop) { 
                            
                            last_l_type = j-1;
                            out_sa[j-1] = out_sa[text.size()-1];
                            break;
                        }
                    } 
                }
                mod = (mod+3-1) % 3; 
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { 
                        if (mod == 1) {
                            bool s_type_in_l_loop = false;
                            for (size_t j = last_l_type; j > 0; j--) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    last_l_type = j-1;
                                    out_sa[j-1] = out_sa[i-1];
                                    break;
                                }
                            } 
                        }
                        mod = (mod+3-1) % 3; 
                    }
                }
                
                mod = (count_s_type_pos+3-1) % 3;  
                s_type = true;
                if (mod == 2) {
                    bool s_type_in_l_loop = false;
                    for (size_t j = last_l_type; j > 0; j--) {
                        if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                        else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                        
                        if (!s_type_in_l_loop) { 
                            
                            last_l_type = j-1;
                            out_sa[j-1] = out_sa[text.size()-1];
                            break;
                        }
                    } 
                }
                mod = (mod+3-1) % 3; 
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { 
                        if (mod == 2) {
                            bool s_type_in_l_loop = false;
                            for (size_t j = last_l_type; j > 0; j--) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    last_l_type = j-1;
                                    out_sa[j-1] = out_sa[i-1];
                                    break;
                                }
                            } 
                        }
                        mod = (mod+3-1) % 3; 
                    }
                }    

                std::cout << out_sa << std::endl;   

                /* move l-type-positions to the end of out_sa */
                size_t counter = text.size()-1;
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (!s_type) { out_sa[counter--] = out_sa[i-1]; }
                }
                std::cout << out_sa << std::endl;   
                
                /* move t_0, t_1 and t_2 to the begin of out_sa */
                for (size_t i = text.size()-1; i > text.size()-count_s_type_pos-1; i--) {
                    out_sa[text.size()-(i+1)] = out_sa[i];
                }
                std::cout << out_sa << std::endl; 
                
                /* sizes of t_0, t_1 and t_2 */
                size_t size_t_0 = count_s_type_pos/3 + (count_s_type_pos % 3 > 0);
                size_t size_t_1 = count_s_type_pos/3 + (count_s_type_pos % 3 > 1);
                size_t size_t_2 = count_s_type_pos/3;
                
                /* revert t_0, t_1 and t_2 */
                util::span<sa_index> t_0 = out_sa.slice(0, size_t_0);
                util::span<sa_index> t_1 = out_sa.slice(size_t_0, size_t_0+size_t_1);
                util::span<sa_index> t_2 = out_sa.slice(size_t_0+size_t_1, size_t_0+size_t_1+size_t_2);
                revert(t_0);
                revert(t_1);
                revert(t_2);
                
                std::cout << out_sa << std::endl; 
            }
            
            template<typename C, typename T>
            static util::span<const C> retrieve_s_string(T& text, size_t s_pos, size_t count) {
                size_t curr_s_pos = s_pos;
                for (size_t c = 1; c <= count; c++) {
                    if (curr_s_pos == text.size()-1) { break; }
                    if (text[curr_s_pos] == text[curr_s_pos+1]) { ++curr_s_pos; }
                    else {
                        size_t k = curr_s_pos+2;
                        while (k < text.size() && text[k-1] >= text[k]) { k++; }
                        size_t j = k-1;
                        while (j > curr_s_pos+1 && text[j-1] <= text[j]) { j--; }
                        curr_s_pos = j;
                    }
                }
                return util::span<const C>(&text[s_pos], curr_s_pos-s_pos+1); 
            }
            
            template<typename A>
            static void revert(const A& a) {
                for (size_t i = 0; i < a.size()/2; i++) {
                    auto tmp = a[i];
                    a[i] = a[a.size()-1-i];
                    a[a.size()-1-i] = tmp;
                }
            }
    }; // class nzSufSort

} // namespace sacabench::nzSufSort