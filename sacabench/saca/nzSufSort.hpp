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

namespace sacabench::nzSufSort {

    class nzSufSort {
        public:    
            static constexpr size_t EXTRA_SENTINELS = 1;

            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     util::alphabet const& alphabet,
                                     util::span<sa_index> out_sa) {
                // Suppress Warnings
                (void) alphabet; 
                                         
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
                
                determine_leq<util::character>(text, out_sa, 0, mod_0, mod_0, count_s_type_pos-mod_0, count_s_type_pos);
            }
          
        private:
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
                mod = (mod-1) % 3;
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
                for (size_t i = start_p_12; i < start_p_12+length_p_12; i++) {
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
                            bool s_type_in_l_loop = true;
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
                if (mod == 1) {
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
                        if (mod == 1) {
                            bool s_type_in_l_loop = true;
                            for (size_t j = last_l_type; j > 0; j--) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    //std::cout << j << std::endl;
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
                        if (mod == 2) {
                            
                                    std::cout << i-1 << std::endl;
                            bool s_type_in_l_loop = true;
                            for (size_t j = last_l_type; j > 0; j--) {
                                if (text[j-1] > text[j]) { s_type_in_l_loop = false; }
                                else if (text[j-1] < text[j]) { s_type_in_l_loop = true; }
                                
                                if (!s_type_in_l_loop) { 
                                    //std::cout << j << std::endl;
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
