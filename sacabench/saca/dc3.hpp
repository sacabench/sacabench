/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <iostream>
#include <util/string.hpp>
#include <vector>
#include <util/container.hpp>
#include <tuple>

namespace sacabench::saca {
    
    /**\brief Identify order of chars starting in position i mod 3 = 0 with difference cover
    * \param t_0 input text t_0 with chars beginning in i mod 3 = 0 of input text
    * \param isa_12 calculated ISA for triplets beginning in i mod 3 != 0
    * \param sa_0 memory block for resulting SA for positions beginning in i mod 3 = 0
    *
    * This method identifies the order of the characters in input_string in positions i mod 3 = 0
    * with information of ranks of triplets starting in position i mod 3 != 0 of input string.
    * This method works correct because of the difference cover idea.
    */
    
    template<typename C, typename T, typename S>
    void determine_triplets(const T& INPUT_STRING, S& sa_12) {
        
        //const sacabench::util::string SMALLEST_CHAR = {' '};
        const unsigned char SMALLEST_CHAR = ' ';
        //Container to store all tuples with the same length as sa_0
        //Tuples contains a char and a rank 
        auto sa_12_to_be_sorted = sacabench::util::make_container<std::tuple<C, C, C, size_t>>(2*INPUT_STRING.size()/3);
        
        size_t counter = 0;
        for(size_t i = 1; i < INPUT_STRING.size(); i++) {
            if(i % 3 == 1){
                if((i+2) >= INPUT_STRING.size()){
                    if((i+1) >= INPUT_STRING.size()){
                        sa_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                    }else{
                        sa_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);
                    }
                }else{
                    sa_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                }
                
            }
            else if(i % 3 == 2){
                if((i+2) >= INPUT_STRING.size()){
                    if((i+1) >= INPUT_STRING.size()){
                        sa_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], SMALLEST_CHAR, SMALLEST_CHAR, i);
                    }else{
                        sa_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], SMALLEST_CHAR, i);

                    }
                }else{
                    sa_12_to_be_sorted[counter++] = std::tuple<C, C, C, size_t>(INPUT_STRING[i], INPUT_STRING[i+1], INPUT_STRING[i+2], i);
                }    
            }
        }  
        
        //TODO: sort Tupels with radix_sort
        //radix_sort(sa0_to_be_sorted, sa0);        
        std::sort(sa_12_to_be_sorted.begin(),sa_12_to_be_sorted.end());
        
        for(size_t i = 0; i<sa_12_to_be_sorted.size();i++){   
            sa_12[i] = std::get<3>(sa_12_to_be_sorted[i]);
        }
                
    }
}  // namespace sacabench::util

