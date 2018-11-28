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


namespace sacabench::dc3_lite_par {

    class dc3_lite_par {
        public:    
            static constexpr size_t EXTRA_SENTINELS = 0;
            static constexpr char const* NAME = "DC3-Lite-Parallel";
            static constexpr char const* DESCRIPTION =
                "Parallelized lightweight variant of DC3 by G. Nong and S. Zhang";


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
                //std::cout << "text: " << text << std::endl;
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
                    //std::cout << "out_sa: " << out_sa << std::endl;
                    return;
                }
                tdc::StatPhase phase("Sort triplets");
                
                // positions of text to be sorted
                #pragma omp parallel for
                for (size_t i = 0; i < text.size(); ++i) { u[i] = text.size()-i-1; }
                
                auto key_function = [&](size_t i, size_t p) {
                    if (p < 2) {
                        if (i+p < text.size()) {
                            return text[i+p];
                        }
                        else { return (C)0; }
                    }
                    else {
                        auto last_char = text[text.size()-1];
                        if (i+p < text.size()) {
                            if (text[i+p] < last_char || (i+p == text.size()-1)) {
                                return text[i+p];
                            }
                            else {
                                return (C)(text[i+p]+(C)1);
                            }
                        }
                        else { return (C)0; }
                    }
                };
                auto compare_function = [&](size_t i, size_t j, size_t index, size_t length) {
                    size_t pos_1 = i+index+1;
                    size_t pos_2 = j+index+1;
                    util::span<C> t_1 = retrieve_triplets<C>(text, pos_1, length);
                    util::span<C> t_2 = retrieve_triplets<C>(text, pos_2, length);
                    return t_1 < t_2 
                        || (t_1 == t_2 && pos_1+length == text.size()); 
                                /* If equal the last tuple must have lower
                                  priority to simulate dummy tuple */
                };
                
                #pragma omp parallel for
                for (size_t i = 0; i < out_sa.size(); ++i) {
                    out_sa[i] = 0;
                }
                
                radixsort_for_big_alphabet(u, out_sa, alphabet_size+1, 0, 2, key_function, compare_function);
                
                //position of last index i mod 3 = 0;
                size_t start_pos_of_0 = u.size()/3 + (u.size() % 3 > 0);
                //position of last index i mod 3 = 1;
                size_t start_pos_of_1 = 2*u.size()/3 + (u.size() % 3 > 0);
                
                // spans for groups with same i mod 3
                
                auto new_text_0 = new_text.slice(0, start_pos_of_0);
                auto new_text_1 = new_text.slice(start_pos_of_0, start_pos_of_1);
                auto new_text_2 = new_text.slice(start_pos_of_1, new_text.size());
                
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
                
                //Store lexicographical names in correct positions of text as:
                //[---i%3=0---||---i%3=1---||---i%3=2---]
                size_t counter_0 = 0;
                size_t counter_1 = 0;
                size_t counter_2 = 0; 
                for(size_t i = 0; i < out_sa.size(); ++i){
                    if(i % 3 == 0){
                        new_text_0[counter_0++] = out_sa[i];
                    }else if(i % 3 == 1){
                        new_text_1[counter_1++] = out_sa[i];
                    }else{
                        new_text_2[counter_2++] = out_sa[i];
                    }
                }
                
                // temporary space for correct ranks of new_text
                size_t text_1_size = new_text.size()-start_pos_of_0;
                auto t_1 = util::span<sa_index>(out_sa).slice(0, text_1_size/2 + (text_1_size%2>0));
                auto t_2 = util::span<sa_index>(out_sa).slice(text_1_size/2 + (text_1_size%2>0), text_1_size);
                
                // calculate t_12
                bool recursion = false;
                size_t rank_12 = 1;
                for (size_t i = 0; i < u.size(); ++i) { //TODO: Dummy-Triplett?
                    // Write ranks at correct position
                    size_t pos = u[i]/3;
                    if (u[i] % 3 == 1) {
                        t_1[pos] = rank_12;
                    }
                    else if (u[i] % 3 == 2) {
                        t_2[pos] = rank_12;
                    }
                    
                    // Calculate new ranks
                    if (u[i] % 3 != 0) {
                        size_t next_mod_12_pos = 0;
                        for (size_t j = i+1; j < u.size(); ++j) {
                            if (u[j] % 3 != 0) {
                                next_mod_12_pos = j;
                                break;
                            }
                        }
                        
                        if (next_mod_12_pos != 0) {
                            auto pos_i = u[i]/3;
                            auto pos_next_mod_12_pos = u[next_mod_12_pos]/3;
                            auto char_i = new_text[0]; // Dummy initializer
                            auto char_next_mod_12_pos = new_text[0]; // Dummy initializer
                            if (u[i] % 3 == 1) { char_i = new_text_1[pos_i]; }
                            else { char_i = new_text_2[pos_i]; }
                            if (u[next_mod_12_pos] % 3 == 1) { char_next_mod_12_pos = new_text_1[pos_next_mod_12_pos]; }
                            else { char_next_mod_12_pos = new_text_2[pos_next_mod_12_pos]; }
                            
                            if (char_i < char_next_mod_12_pos) { ++rank_12; }
                            else { recursion = true; }
                        }
                    }
                }
                
                std::cout << "t_1: " << t_1 << std::endl;
                std::cout << "t_2: " << t_2 << std::endl;
                
                auto u_1 = u.slice(start_pos_of_0, u.size());
                auto v_1 = out_sa.slice(start_pos_of_0, out_sa.size());
                auto text_1 = new_text.slice(start_pos_of_0,new_text.size());
                
                const size_t start_pos_mod_2 = text_1.size()/2 + (text_1.size() % 2 == 1);
                
                auto tmp_2 = out_sa.slice(0, text_1_size/2);
                auto tmp_1 = u.slice(0, start_pos_mod_2);
                    
                if (recursion) {    
                    /*save text_1 temporally in unused space of u and out_sa so we can 
                      copy it back after the recursion */
                    phase.split("Copy text_1 temporally");
                    // copy new_text_1 into unused space of u
                    std::copy(new_text_1.begin(), new_text_1.end(), tmp_1.begin());
                    // copy t_1 to new_text_1
                    std::copy(t_1.begin(), t_1.end(), new_text_1.begin());
                    // copy new_text_2 into unused space of out_sa
                    std::copy(new_text_2.begin(), new_text_2.end(), tmp_2.begin());
                    // copy t_2 to new_text_2
                    std::copy(t_2.begin(), t_2.end(), new_text_2.begin());
                    
                    //Rekursion
                    lightweight_dc3<sa_index, sa_index>(text_1, text_1, u_1, v_1, rank_12+1);
                    
                    //copy old value of text_1 back
                    phase.split("Copy text_1 back");
                    std::copy(tmp_1.begin(), tmp_1.end(), new_text_1.begin());
                    std::copy(tmp_2.begin(), tmp_2.end(), new_text_2.begin());
                }
                else {
                    auto t_12 = out_sa.slice(0, text_1_size);
                    auto sa_12 = u.slice(0, text_1_size);
                    
                    // calculate sa out of isa
                    #pragma omp parallel for
                    for (size_t i = 0; i < sa_12.size(); ++i) {
                        sa_12[t_12[i]-(sa_index)1] = i;
                    }
                    
                    std::copy(sa_12.begin(), sa_12.end(), v_1.begin());
                }
                
                //Calculate ISA_12
                #pragma omp parallel for
                for (size_t i = 0; i < u_1.size(); ++i) { u_1[v_1[i]] = i; }
                
                //Induce SA_0 with SA_12
                phase.split("Induce SA_0");
                auto text_0 = new_text.slice(0, start_pos_of_0);
                auto v_0 = util::span<sa_index>(out_sa).slice(0, start_pos_of_0);
                util::induce_sa_dc<sa_index>(text_0, u_1, v_0);
                
                /* positions in sa_0 are multiplied by 3 so divide by 3 */
                #pragma omp parallel for
                for (size_t i = 0; i < v_0.size(); ++i) { v_0[i] = v_0[i]/3; }
                
                /* calculate isa_0 into u_0 */
                phase.split("Calculate ISA_0");
                auto u_0 = util::span<sa_index>(u).slice(0, start_pos_of_0);
                #pragma omp parallel for
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
                #pragma omp parallel for
                for (size_t i = 0; i < u_0.size(); ++i) { u_0[i] = v_0[u_0[i]]; }
                #pragma omp parallel for
                for (size_t i = 0; i < u_1.size(); ++i) { u_1[i] = v_1[u_1[i]]; }
                
                /* compute sa_012 by traversing isa_012 */
                #pragma omp parallel for
                for (size_t i = 0; i < out_sa.size(); ++i) { out_sa[u[i]] = i; }
                
                /* compute sa by equation */
                phase.split("Calculate SA by equation");
                size_t m_0 = text_0.size();
                size_t m_1 = text_0.size()+text_1.size()/2+(text_1.size() % 2 != 0);
                
                #pragma omp parallel for
                for (size_t i = 0; i < out_sa.size(); ++i) {
                    if (0u <= out_sa[i] && out_sa[i] < m_0) { out_sa[i] = 3*out_sa[i]; }
                    else if (m_0 <= out_sa[i] && out_sa[i] < m_1) {
                        out_sa[i] = 3*(((size_t)out_sa[i])-m_0)+1;
                    }
                    else { out_sa[i] = 3*(((size_t)out_sa[i])-m_1)+2; }
                }
                //std::cout << "out_sa: " << out_sa << std::endl;
            }
            
        private:    
            template<typename C>
            static util::span<C> retrieve_triplets(util::span<C> text, size_t pos, size_t count) {
                if (pos >= text.size()) {
                    return util::span<C>(); 
                }
                else if((pos+count) < text.size()){
                    return util::span<C>(&text[pos], count); 
                }else{
                    return util::span<C>(&text[pos], text.size()-pos); 
                }
            }
    }; // class dc3_lite_par
} // namespace sacabench::dc3_lite_par