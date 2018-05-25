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

namespace sacabench::nzSufSort {

    class nzSufSort {
        public:
            template<typename sa_index>
            static void construct_sa(util::string_span text,
                                     size_t alphabet_size,
                                     util::span<sa_index> out_sa) {
                // Suppress Warnings
                (void) alphabet_size; 
                
                // TODO: Sentinel am Ende einfügen (lassen)
                const util::character SENTINEL = 0;
                auto cont_text = util::make_container<util::character>(text.size()+1);
                for (size_t i = 0; i < text.size(); i++) {
                    cont_text[i] = text[i];
                }
                cont_text[text.size()] = SENTINEL;
                text = cont_text;
                // empty string
                if (text.size() == 0) { return; }
                                         
                std::cout << "Running nzSufSort" << std::endl;
                
                //TODO: Annahme dass mehr S-Typ-Positionen existieren
                
                // count number of s-type-positions in text
                size_t count_s_type_pos = 1;
                bool s_type = true;
                for (size_t i = text.size()-1; i > 0; i--) {
                    std::cout << i << std::endl;
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) { count_s_type_pos++; }
                }
                
                //TODO
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
                std::cout << "p_0" << std::endl;
                for (size_t i = 0; i < p_0.size(); i++) {
                    std::cout << i << ": " << p_0[i] << std::endl;
                }
                std::cout << "p_12" << std::endl;
                for (size_t i = 0; i < p_12.size(); i++) {
                    std::cout << i << ": " << p_12[i] << std::endl;
                }
                
                //TODO sort p_0 and p_12 with radix sort
                /*struct Comp {
                    Comp(util::string_span& text) { this->text = text; }
                    bool comp(size_t i, size_t j) { 
                        util::string_span t_1 = retrieve_s_string<util::character>(text, i, 3);
                        util::string_span t_2 = retrieve_s_string<util::character>(text, j, 3);
                        return t_1 < t_2;
                    }

                    util::string_span text;
                };*/
                auto comp_bind = std::bind(comp, text, std::placeholders::_1, std::placeholders::_2);
                std::sort(p_0.begin(), p_0.end(), comp_bind);
                std::sort(p_12.begin(), p_12.end(), comp_bind);
            }
          
        private:
            template<typename T, typename sa_index>
            static void calculate_position_arrays(const T& text, 
                    const util::span<sa_index>& p_0, const util::span<sa_index>& p_12, 
                    const util::span<sa_index>& mod_0, const util::span<sa_index>& mod_1, 
                    const util::span<sa_index>& mod_2, size_t count_s_type_pos) {
                // 
                size_t mod = (count_s_type_pos-1) % 3;  
                size_t count_mod_0 = 0;
                size_t count_mod_1 = 0;
                size_t count_mod_2 = 0;  

                // execute logic for sentinel
                if (mod == 0) { mod_0[count_mod_0++] = text.size()-1; }
                else if (mod == 1) { mod_1[count_mod_1++] = text.size()-1; }
                else { mod_2[count_mod_2++] = text.size()-1; }   
                mod = (mod+1) % 3;
                bool s_type = true;
                
                // save s-type-positions in the correct arrays
                for (size_t i = text.size()-1; i > 0; i--) {
                    if (text[i-1] > text[i]) { s_type = false; }
                    else if (text[i-1] < text[i]) { s_type = true; }
                    
                    if (s_type) {
                        if (mod == 0) { mod_0[count_mod_0++] = i-1; }
                        else if (mod == 1) { mod_1[count_mod_1++] = i-1; }
                        else { mod_2[count_mod_2++] = i-1; }
                        mod = (mod+1) % 3;
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
                for (size_t i = 0; i < mod_1.size(); i++) {
                    p_12[count_p_12++] = mod_1[i];
                }
                for (size_t i = 0; i < mod_2.size(); i++) {
                    p_12[count_p_12++] = mod_2[i];
                }
            }
            
            template<typename C, typename T>
            static util::span<const C> retrieve_s_string(T& text, size_t s_pos, size_t count) {
                //TODO: Check for "out of bounds"
                //TODO: Concatenate count s_stringss
                if (text[s_pos] == text[s_pos+1]) {
                    return util::span<const C>(&text[s_pos], 2);
                }
                else {
                    //TODO
                }
            }
            
            template<typename T>
            static bool comp(T& text, size_t i, size_t j) {
                util::string_span t_1 = retrieve_s_string<util::character>(text, i, 3);
                util::string_span t_2 = retrieve_s_string<util::character>(text, j, 3);
                return t_1 < t_2;
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
