/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>

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
                
                std::cout << "Running nzSufSort" << std::endl;
                //TODO
                DCHECK_MSG(count_s_type_pos <= text.size()/2, 
                    "There are more S-Type-Positions than L-Type-Positions");
                
                // calculate position arrays of s-type-positions
                size_t q = count_s_type_pos/3 + (count_s_type_pos % 3 != 0);
                util::span<sa_index> p_0 = out_sa.slice(0, q);
                util::span<sa_index> p_12 = out_sa.slice(q+1, count_s_type_pos);
                util::span<sa_index> u = out_sa.slice(count_s_type_pos+1, count_s_type_pos+q);
                util::span<sa_index> v = out_sa.slice(count_s_type_pos+q+1, count_s_type_pos+2*q);
                util::span<sa_index> w = out_sa.slice(count_s_type_pos+2*q+1, count_s_type_pos+3*q);
                std::cout << "Running nzSufSort" << std::endl;
                calculate_position_arrays(text, p_0, p_12, u, v, w);
                
                std::cout << "p_0" << std::endl;
                for (size_t i = 0; i < p_0.size(); i++) {
                    std::cout << i << ": " << p_0[i] << std::endl;
                }
                
                std::cout << "p_12" << std::endl;
                for (size_t i = 0; i < p_12.size(); i++) {
                    std::cout << i << ": " << p_12[i] << std::endl;
                }
            }
          
        private:
            template<typename T, typename sa_index>
            static void calculate_position_arrays(const T& text, 
                    const util::span<sa_index>& p_0, const util::span<sa_index>& p_12, 
                    const util::span<sa_index>& u, const util::span<sa_index>& v, 
                    const util::span<sa_index>& w) {
                u[0] = text.size()-1;     
                     
                size_t count_u = 1;
                size_t count_v = 0;
                size_t count_w = 0;
                size_t mod = 1;
                bool s_type = true;
                for (size_t i = text.size()-2; i >= 0; i--) {
                    if (text[i] > text[i+1]) { s_type = false; }
                    else if (text[i] < text[i+1]) { s_type = true; }
                    
                    if (s_type) {
                        if (mod == 0) { u[count_u++] = i; }
                        else if (mod == 1) { v[count_v++] = i; }
                        else { w[count_w++] = i; }
                        mod = (mod+1) % 3;
                    }
                }
                
                util::span<sa_index> mod_0;
                util::span<sa_index> mod_1;
                util::span<sa_index> mod_2;
                // when last position was added it was mod == 2
                if (mod == 0) {
                    mod_0 = w.slice(0, count_w-1);
                    mod_1 = u.slice(0, count_u-1);
                    mod_2 = v.slice(0, count_v-1);
                }
                else if (mod == 1) {
                    mod_0 = u.slice(0, count_u-1);
                    mod_1 = v.slice(0, count_v-1);
                    mod_2 = w.slice(0, count_w-1);
                }
                else {
                    mod_0 = v.slice(0, count_v-1);
                    mod_1 = w.slice(0, count_w-1);
                    mod_2 = u.slice(0, count_u-1);
                }
                
                revert(mod_0);
                revert(mod_1);
                revert(mod_2);
                
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
            
            template<typename A>
            static void revert(const A& a) {
                for (size_t i = 0; i <= a.size()/2; i++) {
                    auto tmp = a[i];
                    a[i] = a[a.size()-1-i];
                    a[a.size()-1-i] = tmp;
                }
            }
    }; // class nzSufSort

} // namespace sacabench::nzSufSort
