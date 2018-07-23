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

#include <tudocomp_stat/StatPhase.hpp>


namespace sacabench::dc3_lite {

    class dc3_lite {
        public:    
            static constexpr size_t EXTRA_SENTINELS = 0;
            static constexpr char const* NAME = "DC3-Lite";
            static constexpr char const* DESCRIPTION =
                "Lightweight variant of DC3 by G. Nong and S. Zhang";


            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     util::alphabet const& alphabet,
                                     util::span<sa_index> out_sa) {
                                         
                auto new_text_cont = util::make_container<sa_index>(text.size());
                util::span<sa_index> new_text = new_text_cont;
                auto u = util::make_container<sa_index>(text.size());
                lightweight_dc3<const util::character, sa_index>(text, new_text, u, out_sa,
                    alphabet.size_with_sentinel());
            }
            
            template<typename C, typename sa_index>
            static void lightweight_dc3(util::span<C> text, util::span<sa_index> new_text,
                                        util::span<sa_index> u, util::span<sa_index> out_sa,
                                        const size_t alphabet_size) {
                //end of recursion if text.size() < 3 
                if (text.size() < 3) {
                    if (text.size() == 1) {
                        out_sa[0] = 0;
                    }
                    else if (text.size() == 2) {
                        auto t_1 = text.slice(0,2);
                        auto t_2 = text.slice(1,2);
                        if (t_1 < t_2) {
                            out_sa[0] = 0;
                            out_sa[1] = 1;
                        }
                        else {
                            out_sa[0] = 1;
                            out_sa[1] = 0;
                        }
                    }
                    return;
                }
                
                tdc::StatPhase phase("Sort triplets");
                
                /* save indices of text in out_sa in reverted order
                   so that the lastc complete triplet gets higher priority */
                for (size_t i = 0; i < text.size(); ++i) { out_sa[i] = text.size()-i-1; }
                
                auto key_function = [&](size_t i, size_t p) {
                    if (i+p < text.size()) {
                        return text[i+p];
                    }
                    else { return (C)0; }
                };
                
                radixsort_with_key(out_sa, u, alphabet_size, 2, key_function);
                
                //Determine lexicographical names of all triplets
                //if triplets are the same, they will get the same rank
                phase.split("Calculate t_0 and t_12");
                size_t rank = 1;
                for(size_t i = 0; i < u.size(); ++i){
                    out_sa[u[i]] = rank; // save ranks in correct positions
                    if ((i+1) < u.size()) {
                        util::span<C> t_1 = retrieve_triplets<C>(text, u[i], 3);
                        util::span<C> t_2 = retrieve_triplets<C>(text, u[i+1], 3);
                        if (t_1 < t_2 || (t_1 == t_2 && ((size_t)u[i])+3u == text.size())) { ++rank; }
                    }
                }
                
                //position of first index i mod 3 = 0;
                size_t end_pos_of_0 = u.size()/3 + (u.size() % 3 > 0);
                
                //Store lexicographical names in correct positions of text as:
                //[---i%3=0---||---i%3=1---||---i%3=2---]
                size_t counter_0 = 0;
                size_t counter_1 = end_pos_of_0;
                size_t counter_2 = 2*u.size()/3 + (u.size() % 3 > 0); 
                for(size_t i = 0; i < out_sa.size(); ++i){
                    if(i % 3 == 0){
                        new_text[counter_0++] = out_sa[i];
                    }else if(i % 3 == 1){
                        new_text[counter_1++] = out_sa[i];
                    }else{
                        new_text[counter_2++] = out_sa[i];
                    }
                }
                
                //unfortunately it's not working, if I pass the spans directly
                auto u_1 = util::span<sa_index>(u).slice(end_pos_of_0, u.size());
                auto v_1 = util::span<sa_index>(out_sa).slice(end_pos_of_0, out_sa.size());
                auto text_1 = new_text.slice(end_pos_of_0,new_text.size());
                
                /*save text_1 temporally in unused space of u and out_sa so we can 
                  copy it back after the recursion */
                phase.split("Copy text_1 temporally");
                const size_t start_pos_mod_2 = text_1.size()/2 + (text_1.size() % 2 == 1);
                auto tmp_1 = util::span<sa_index>(u).slice(0, start_pos_mod_2);
                auto tmp_2 = util::span<sa_index>(out_sa).slice(0, text_1.size()/2);
                for (size_t i = 0; i < tmp_1.size(); ++i) { tmp_1[i] = text_1[i]; }
                for (size_t i = 0; i < tmp_2.size(); ++i) { tmp_2[i] = text_1[start_pos_mod_2+i]; }
                
                //Rekursion
                lightweight_dc3<sa_index, sa_index>(text_1, text_1, u_1, v_1, rank+1);
                
                //copy old value of text_1 back
                phase.split("Copy text_1 back");
                for (size_t i = 0; i < tmp_1.size(); ++i) { text_1[i] = tmp_1[i]; }
                for (size_t i = 0; i < tmp_2.size(); ++i) { text_1[start_pos_mod_2+i] = tmp_2[i]; }
                
                //Calculate ISA_12
                for (size_t i = 0; i < u_1.size(); ++i) { u_1[v_1[i]] = i; }
                
                //Induce SA_0 with SA_12
                phase.split("Induce SA_0");
                auto text_0 = new_text.slice(0, end_pos_of_0);
                auto v_0 = util::span<sa_index>(out_sa).slice(0, end_pos_of_0);
                util::induce_sa_dc<sa_index>(text_0, u_1, v_0);
                
                /* positions in sa_0 are multiplied by 3 so divide by 3 */
                for (size_t i = 0; i < v_0.size(); ++i) { v_0[i] = v_0[i]/3; }
                
                /* calculate isa_0 into u_0 */
                phase.split("Calculate ISA_0");
                auto u_0 = util::span<sa_index>(u).slice(0, end_pos_of_0);
                for (size_t i = 0; i < u_0.size(); ++i) { u_0[v_0[i]] = i; }
                
                /* merge sa_0 and sa_12 by calculating positions in merged sa */
                phase.split("Merge SA_0 and SA_12");
                size_t count_sa_0 = 0;
                size_t count_sa_12 = 0;
                size_t position = 0;
                while (count_sa_0 < v_0.size() && count_sa_12 < v_1.size()) {
                    size_t pos_in_text_0 = v_0[count_sa_0];
                    size_t pos_in_text_1 = v_1[count_sa_12];
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
                for (size_t i = 0; i < out_sa.size(); ++i) { out_sa[u[i]] = i; }
                
                /* compute sa by equation */
                phase.split("Calculate SA by equation");
                size_t m_0 = text_0.size();
                size_t m_1 = text_0.size()+text_1.size()/2+(text_1.size() % 2 != 0);
                for (size_t i = 0; i < out_sa.size(); ++i) {
                    if (0u <= out_sa[i] && out_sa[i] < m_0) { out_sa[i] = 3*out_sa[i]; }
                    else if (m_0 <= out_sa[i] && out_sa[i] < m_1) {
                        out_sa[i] = 3*(((size_t)out_sa[i])-m_0)+1;
                    }
                    else { out_sa[i] = 3*(((size_t)out_sa[i])-m_1)+2; }
                }
            }
            
        private:    
            template<typename C>
            static util::span<C> retrieve_triplets(util::span<C> text, size_t pos, size_t count) {
                if((pos+count) < text.size()){
                    return util::span<C>(&text[pos], count); 
                }else{
                    return util::span<C>(&text[pos], text.size()-pos); 
                }
            }
    }; // class dc3_lite
} // namespace sacabench::dc3_lite