/*******************************************************************************
 * util/induce_sa_dc.hpp
 *
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <tuple>
#include <vector>
#include <util/assertions.hpp>
#include <algorithm>

#pragma once
namespace sacabench::util {
    template<typename C, typename T, typename I, typename S>
    //template C for input characters
    //template T for input string
    //template I for ISA
    //template S for SA

    /**\brief Identify order of chars starting in position i mod 3 = 0 with difference cover
    * \param t_0 input text t_0 with chars beginning in i mod 3 = 0 of input text
    * \param isa_12 calculated ISA for triplets beginning in i mod 3 != 0
    * \param sa_0 memory block for resulting SA for positions beginning in i mod 3 = 0
    *
    * This method identifies the order of the characters in input_string in positions i mod 3 = 0
    * with information of ranks of triplets starting in position i mod 3 != 0 of input string.
    * This method works correct because of the difference cover idea.
    */
    void induce_sa_dc(T& t_0, I& isa_12, S& sa_0) {
        
        DCHECK_MSG(sa_0.size() == t_0.size(), "sa_0 must have the same length as t_0");

        std::vector<std::tuple<C, int, int>> sa_0_to_be_sorted;

        for (size_t i = 0; i < sa_0.size(); i++) {
            sa_0_to_be_sorted.push_back(std::tuple<C, int, int>(t_0[i], isa_12[i], i));
        }  
    
        //TODO: sort Tupels with radix_sort
        //radix_sort(sa0_to_be_sorted, sa0);
        
        std::sort(sa_0_to_be_sorted.begin(),sa_0_to_be_sorted.end());
        for(size_t i = 0; i<sa_0_to_be_sorted.size();i++){
            sa_0[i] = std::get<2>(sa_0_to_be_sorted[i]);
        }
    }
}  // namespace sacabench::util