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


namespace sacabench::nzsufsort {

    class nzsufsort {
        public:    
            static constexpr size_t EXTRA_SENTINELS = 1;
            static constexpr char const* NAME = "nzSufSort";
            static constexpr char const* DESCRIPTION =
                "Optimal lightweight SACA by G. Nong and S. Zhang";


            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     util::alphabet const& alphabet,
                                     util::span<sa_index> out_sa) {
                // count number of s-type-positions in text
                size_t count_s_type_pos = 1;
                bool s_type = true;
                for (size_t i = text.size()-1; i > 0; --i) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { ++count_s_type_pos; }
                }
                
                /*if there are more s-type-positions than l-type-positions
                  revert the characters in t*/
                bool is_text_reverted = false;  
                util::container<util::character> tmp_text;
                if (count_s_type_pos > text.size()/2) {
                    //TODO: Text kann nicht überschrieben werden
                    tmp_text = util::make_container<util::character>(text.size());
                    for (size_t i = 0; i < text.size(); ++i) { tmp_text[i] = text[i]; }
                    for (size_t i = 0; i < text.size(); ++i) { tmp_text[i] = alphabet.max_character_value()-text[i]+1; }
                    text = tmp_text;
                    
                    count_s_type_pos = text.size()-count_s_type_pos;
                    is_text_reverted = true;
                }
                
                // calculate position arrays of s-type-positions
                size_t mod_0 = count_s_type_pos/3 + (count_s_type_pos % 3 > 0);
                size_t mod_1 = count_s_type_pos/3 + (count_s_type_pos % 3 > 1);
                size_t mod_2 = count_s_type_pos/3;
                util::span<sa_index> p_0 = out_sa.slice(0, mod_0);
                util::span<sa_index> p_12 = out_sa.slice(mod_0, count_s_type_pos);
                util::span<sa_index> u = out_sa.slice(count_s_type_pos, count_s_type_pos+mod_0);
                util::span<sa_index> v = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1);
                util::span<sa_index> w = out_sa.slice(count_s_type_pos+mod_0+mod_1, count_s_type_pos+mod_0+mod_1+mod_2);
                calculate_position_arrays(text, p_0, p_12, u, v, w, count_s_type_pos);
                
                //TODO sort p_0 and p_12 with radix sort
                auto comp = [&](size_t i, size_t j) {
                    util::string_span t_1 = retrieve_s_string<util::character>(text, i, 3);
                    util::string_span t_2 = retrieve_s_string<util::character>(text, j, 3);
                    return comp_z_strings(t_1, t_2);
                };
                std::sort(p_0.begin(), p_0.end(), comp);
                std::sort(p_12.begin(), p_12.end(), comp);
                
                //calculate t_0 and t_12 in the begin of out_sa
                determine_leq<util::character>(text, out_sa, 0, mod_0, mod_0, count_s_type_pos-mod_0, count_s_type_pos);
                util::span<sa_index> t_0 = out_sa.slice(0, mod_0);
                util::span<sa_index> t_12 = out_sa.slice(mod_0, count_s_type_pos);
                
                //calculate SA(t_12) by calling the lightweight variant of DC3
                /*TODO: calculate t_12 first, then call lightweight_dc3 with 3 spans 
                  each of size n/3. Then calculate t_0. */
                auto u_1 = util::make_container<sa_index>(t_12.size());
                auto v_1 = util::make_container<sa_index>(t_12.size());
                lightweight_dc3<sa_index>(t_12, u_1, v_1);
                for (size_t i = 0; i < t_12.size(); ++i) {
                    t_12[i] = v_1[i];
                }
                util::span<sa_index> sa_12 = t_12;
                
                //induce SA(t_0)
                util::span<sa_index> isa_12 = out_sa.slice(count_s_type_pos+mod_0, 2*count_s_type_pos);
                for (size_t i = 0; i < sa_12.size(); ++i) {
                    isa_12[sa_12[i]] = i;
                }
                util::induce_sa_dc<size_t>(t_0, isa_12, t_0);
                util::span<sa_index> sa_0 = t_0;
                
                /* positions in sa_0 are multiplied by 3 so divide by 3 */
                for (size_t i = 0; i < sa_0.size(); ++i) { sa_0[i] = sa_0[i]/3; }
                
                //update SA(t_0) and SA(t_12) with position arrays
                p_0 = out_sa.slice(count_s_type_pos, count_s_type_pos+mod_0);
                p_12 = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1+mod_2);
                util::span<sa_index> p_1 = out_sa.slice(count_s_type_pos+mod_0, count_s_type_pos+mod_0+mod_1);
                util::span<sa_index> p_2 = out_sa.slice(count_s_type_pos+mod_0+mod_1, count_s_type_pos+mod_0+mod_1+mod_2);
                
                calculate_position_arrays(text, p_0, p_12, p_0, p_1, p_2, count_s_type_pos);
                
                for (size_t i = 0; i < sa_0.size(); ++i) {
                    sa_0[i] = p_0[sa_0[i]];
                }
                
                for (size_t i = 0; i < sa_12.size(); ++i) {
                    sa_12[i] = p_12[sa_12[i]];
                }
                
                // revert sa_0 and sa_12 so we can traverse them easier later
                revert(sa_0);
                revert(sa_12);
                
                // copy sa_0 and sa_12 into l-type-positions of out_sa
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
                    size_t end_pos = out_sa[curr_pos_sa_12-1]+check_residue_t_12.size()-1;
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
                    // NB: This is a closure so that we can evaluate it later, because
                    // evaluating it if `eq` is not true causes out-of-bounds errors.
                    const bool lesser_suf = out_sa[out_sa[curr_pos_sa_0-1]+t_0.size()-1] >   // greater because sa_0 and sa_12 are stored from right to left
                            out_sa[out_sa[curr_pos_sa_12-1]+t_12.size()-1];
                    
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
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) { out_sa[i-1] = out_sa[out_sa[i-1]]; }
                    last_char = text[i-1];
                }
                
                /* move isa_012 to the end of out_sa */
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
                util::span<sa_index> sa_012 = out_sa.slice(0, count_s_type_pos);
                util::span<sa_index> isa_012 = out_sa.slice(out_sa.size()-count_s_type_pos, out_sa.size());
                for (size_t i = 0; i < sa_012.size(); ++i) {
                    sa_012[isa_012[i]] = i;
                } 
                
                /* calculate position array of s-type-positions in right half */
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
                for (size_t i = 0; i < sa_012.size(); ++i) {
                    sa_012[i] = p_012[sa_012[i]];
                }
                
                /* induction scan to calculate correct sa */
                left_induction_scan(text, out_sa, alphabet.size_with_sentinel()+1, count_s_type_pos);
                
                /* if text was reverted at the beginning out_sa must be reverted */
                if (is_text_reverted) { revert(out_sa); }
            }
          
        private:
            template<typename sa_index, typename T, typename H>
            static void lightweight_dc3(T& text, H& u, H& v) {
                //end of recursion if text.size() < 3 
                if (text.size() < 3) {
                    if (text.size() == 1) {
                        v[0] = 0;
                    }
                    else if (text.size() == 2) {
                        auto t_1 = text.slice(0,2);
                        auto t_2 = text.slice(1,2);
                        if (t_1 < t_2) {
                            v[0] = 0;
                            v[1] = 1;
                        }
                        else {
                            v[0] = 1;
                            v[1] = 0;
                        }
                    }
                    return;
                }
                
                //save indices of text in u
                for (size_t i = 0; i < text.size(); ++i) { u[i] = i; }
                
                //TODO sort triplets_12 with radix sort
                auto comp = [&](size_t i, size_t j) {
                    util::span<const sa_index> t_1 = retrieve_triplets<const sa_index>(text, i, 3);
                    util::span<const sa_index> t_2 = retrieve_triplets<const sa_index>(text, j, 3);
                    return t_1 < t_2;
                };
                
                //sort all triplets
                std::sort(u.begin(), u.end(), comp);
                
                //Determine lexicographical names of all triplets
                //if triplets are the same, they will get the same rank
                size_t rank = 1;
                for(size_t i = 0; i < u.size();++i){
                    v[u[i]] = rank; // save ranks in correct positions
                    if((i+1)<u.size()){
                        size_t index_1 = u[i]; //Position 1
                        size_t index_2 = u[i+1]; //Position2
                        
                        if(index_1 < text.size()-3){
                            if(index_2 < text.size()-3){
                                if(text[index_1]!=text[index_2] || text[index_1+1]!=text[index_2+1] || text[index_1+2]!=text[index_2+2]){
                                    ++rank;
                                } //tripletes are the same
                            }else ++rank; //if one of the triplets would be out of bounce, they can't be the same
                        }else ++rank; //if one of the triplets would be out of bounce, they can't be the same
                    }else ++rank; //last element 
                }
                
                //position of first index i mod 3 = 0;
                size_t end_pos_of_0 = u.size()/3 + (u.size() % 3 > 0);
                
                //Store lexicographical names in correct positions of text as:
                //[---i%3=0---||---i%3=1---||---i%3=2---]
                size_t counter_0 = 0;
                size_t counter_1 = end_pos_of_0;
                size_t counter_2 = 2*u.size()/3 + (u.size() % 3 > 0); 
                for(size_t i = 0; i < v.size(); ++i){
                    if(i % 3 == 0){
                        text[counter_0++] = v[i];
                    }else if(i % 3 == 1){
                        text[counter_1++] = v[i];
                    }else{
                        text[counter_2++] = v[i];
                    }
                }
                
                //unfortunately it's not working, if I pass the spans directly
                auto u_1 = util::span<sa_index>(u).slice(end_pos_of_0, u.size());
                auto v_1 = util::span<sa_index>(v).slice(end_pos_of_0, v.size());
                auto text_1 = text.slice(end_pos_of_0,text.size());
                
                /*save text_1 temporally in unused space of u and v so we can 
                  copy it back after the recursion */
                const size_t start_pos_mod_2 = text_1.size()/2 + (text_1.size() % 2 == 1);
                auto tmp_1 = util::span<sa_index>(u).slice(0, start_pos_mod_2);
                auto tmp_2 = util::span<sa_index>(v).slice(0, text_1.size()/2);
                for (size_t i = 0; i < tmp_1.size(); ++i) { tmp_1[i] = text_1[i]; }
                for (size_t i = 0; i < tmp_2.size(); ++i) { tmp_2[i] = text_1[start_pos_mod_2+i]; }
                
                //Rekursion
                lightweight_dc3<sa_index>(text_1, u_1, v_1);
                
                //copy old value of text_1 back
                for (size_t i = 0; i < tmp_1.size(); ++i) { text_1[i] = tmp_1[i]; }
                for (size_t i = 0; i < tmp_2.size(); ++i) { text_1[start_pos_mod_2+i] = tmp_2[i]; }
                
                //Next step: Induce SA_0 with SA_12
                for (size_t i = 0; i < u_1.size(); ++i) { u_1[v_1[i]] = i; }
                
                auto text_0 = text.slice(0, end_pos_of_0);
                auto v_0 = util::span<sa_index>(v).slice(0, end_pos_of_0);
                util::induce_sa_dc<sa_index>(text_0, u_1, v_0);
                
                /* positions in sa_0 are multiplied by 3 so divide by 3 */
                for (size_t i = 0; i < v_0.size(); ++i) { v_0[i] = v_0[i]/3; }
                
                /* calculate isa_0 into u_0 */
                auto u_0 = util::span<sa_index>(u).slice(0, end_pos_of_0);
                for (size_t i = 0; i < u_0.size(); ++i) { u_0[v_0[i]] = i; }
                
                /* merge sa_0 and sa_12 by calculating positions in merged sa */
                size_t count_sa_0 = 0;
                size_t count_sa_12 = 0;
                size_t position = 0;
                while (count_sa_0 < v_0.size() && count_sa_12 < v_1.size()) {
                    auto pos_in_text_0 = v_0[count_sa_0];
                    auto pos_in_text_1 = v_1[count_sa_12];
                    auto char_text_0 = text_0[pos_in_text_0];
                    auto char_text_1 = text_1[pos_in_text_1];
                    
                    const bool less_than = char_text_0 < char_text_1;
                    const bool eq = char_text_0 == char_text_1;
                    
                    bool lesser_suf = false;
                    if (pos_in_text_1 < start_pos_mod_2) {
                        if (pos_in_text_0 < u_1.size() && start_pos_mod_2+pos_in_text_1 < u_1.size()) {
                            lesser_suf = u_1[pos_in_text_0] < u_1[start_pos_mod_2+pos_in_text_1];
                        }
                        else if (pos_in_text_0 == u_1.size()) {
                            lesser_suf = true;
                        }
                    }
                    else {
                        if (start_pos_mod_2+pos_in_text_0 < u_1.size() && pos_in_text_1-start_pos_mod_2+1<start_pos_mod_2) {
                            lesser_suf = u_1[start_pos_mod_2+pos_in_text_0] < u_1[pos_in_text_1-start_pos_mod_2+1];
                        }
                        else if (start_pos_mod_2+pos_in_text_0 == u_1.size()) {
                            lesser_suf = true;
                        }
                    }
                        
                    if (less_than || (eq && lesser_suf)) {
                        v_0[count_sa_0++] = position++;
                    } 
                    else {
                        v_1[count_sa_12++] = position++;
                    }
                }
                while (count_sa_0 < v_0.size()) {
                    v_0[count_sa_0++] = position++;
                }
                while (count_sa_12 < v_1.size()) {
                    v_1[count_sa_12++] = position++;
                }
                
                /* update isa_0 and isa_12 with positions in sa_0 and sa_12 to calculate isa_012 */
                for (size_t i = 0; i < u_0.size(); ++i) { u_0[i] = v_0[u_0[i]]; }
                for (size_t i = 0; i < u_1.size(); ++i) { u_1[i] = v_1[u_1[i]]; }
                
                /* compute sa_012 by traversing isa_012 */
                for (size_t i = 0; i < v.size(); ++i) { v[u[i]] = i; }
                
                /* compute sa by equation */
                size_t m_0 = text_0.size();
                size_t m_1 = text_0.size()+text_1.size()/2+(text_1.size() % 2 != 0);
                for (size_t i = 0; i < v.size(); ++i) {
                    if (0 <= v[i] && v[i] < m_0) { v[i] = 3*v[i]; }
                    else if (m_0 <= v[i] && v[i] < m_1) {
                        v[i] = 3*(v[i]-m_0)+1;
                    }
                    else { v[i] = 3*(v[i]-m_1)+2; }
                }
            }
            
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
                    if (out_sa[i] != UNDEFINED && out_sa[i] != 0) {
                        size_t curr_pos = out_sa[i];
                        size_t pre_pos = out_sa[i]-1;
                        bool curr_pos_is_s_type = (i < buckets[text[curr_pos]]); // bucket pointer points at l-type-positions 
                        if (text[pre_pos] > text[curr_pos] || (text[pre_pos] == text[curr_pos] && curr_pos_is_s_type)) {
                            auto elem = text[pre_pos];
                            out_sa[buckets[elem]] = pre_pos;
                            ++buckets[elem];
                        }
                    }
                }
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

                util::character last_char = util::SENTINEL;
                bool s_type = true;
                
                // save s-type-positions in the correct arrays
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (s_type) {
                        if (mod == 0) { mod_0[count_mod_0++] = i-1; }
                        else if (mod == 1) { mod_1[count_mod_1++] = i-1; }
                        else { mod_2[count_mod_2++] = i-1; }
                        
                        mod = (mod+3-1) % 3;
                    }
                    last_char = text[i-1];
                }
                
                // arrays must be reverted, because we started at the last index
                // TODO: since we know the memory sizes, we can save the positions directly at the right index
                revert(mod_0);
                revert(mod_1);
                revert(mod_2);
                
                //copy positions in p_0 and p_12
                for (size_t i = 0; i < p_0.size(); ++i) {
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
                // revert p_0 and p_12 so we can access them later from left to right
                util::span<sa_index> p_0 = out_sa.slice(start_p_0, start_p_0+length_p_0);
                util::span<sa_index> p_12 = out_sa.slice(start_p_12, start_p_12+length_p_12);
                revert(p_0);
                revert(p_12);
                
                // Copy p_0 and p_12 in L-type positions in out_sa
                size_t curr_pos_p_0 = start_p_0+length_p_0;
                size_t curr_pos_p_12 = start_p_12+length_p_12;
                size_t start_pos_p_0;
                
                util::character last_char = util::SENTINEL;
                bool s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (text[i-1] > last_char) { s_type = false; }
                    else if (text[i-1] < last_char) { s_type = true; }
                    
                    if (!s_type) {
                        if (curr_pos_p_12 > start_p_12) { out_sa[i-1] = out_sa[--curr_pos_p_12]; }
                        else {
                            if (curr_pos_p_0 == start_p_0+length_p_0) { start_pos_p_0 = i-1; }
                            if (curr_pos_p_0 == 0) { break; }
                            out_sa[i-1] = out_sa[--curr_pos_p_0]; 
                        }
                    }
                    last_char = text[i-1];
                    if (curr_pos_p_0 == start_p_0) { break; }
                }
                
                /* Determine lexicographical ranks of Positions p_12 and save
                   them in correct positions in out_sa */
                size_t rank = 1;
                util::span<const C> last_t;
                size_t last_i;
                
                last_char = util::SENTINEL;
                s_type = true;
                for (size_t i = text.size(); i > 0; --i) {
                    if (i-1 == start_pos_p_0) { break; }
                    
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
                if (length_p_12 > 0) { out_sa[out_sa[last_i-1]] = rank; }
                
                /* Determine lexicographical ranks of Positions p_0 and save
                   them in correct positions in out_sa */
                rank = 1;
                size_t count = 1;
                
                s_type = false;
                last_t = retrieve_s_string<C>(text, out_sa[start_pos_p_0], 3);
                last_i = start_pos_p_0+1;
                for (size_t i = start_pos_p_0+1; i > 0; --i) {
                    if (length_p_0 <= 1) { break; }
                    if (count > length_p_0) { break; }
                    
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (!s_type) { 
                        auto curr_t = retrieve_s_string<C>(text, out_sa[i-1], 3);
                        
                        out_sa[out_sa[last_i-1]] = rank;
                        if (comp_z_strings(last_t, curr_t)) { ++rank; }
                        
                        last_t = curr_t;
                        last_i = i;
                        
                        ++count;
                    }
                }
                out_sa[out_sa[last_i-1]] = rank;
                
                /* Determine t_0 and t_12 by looking up the lexicographical ranks 
                   in out_sa and save them in l-type-positions of out_sa in reverted order*/
                size_t mod = (count_s_type_pos+3-1) % 3;  
                last_char = util::SENTINEL;
                s_type = true;
                size_t last_l_type;
                
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
                
                /* move t_0, t_1 and t_2 to the begin of out_sa */
                for (size_t i = text.size()-1; i > text.size()-count_s_type_pos-1; --i) {
                    out_sa[text.size()-(i+1)] = out_sa[i];
                }
                
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
    }; // class nzsufsort

} // namespace sacabench::nzsufsort