/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <algorithm>
#include <tuple>
#include <util/assertions.hpp>
#include <util/container.hpp>

#pragma once
namespace sacabench::util {
template<typename C, typename T, typename I, typename S>
    /**\brief Identify order of chars starting in position i mod 3 = 0 with difference cover
    * \tparam T Type of input string
    * \tparam C Type of input characters
    * \tparam I Type of inverse Suffix Array
    * \tparam S Type of Suffix Array
    * \param t_0 input text t_0 with chars beginning in i mod 3 = 0 of input text
    * \param isa_12 ISA for triplets beginning in i mod 3 != 0
    * \param sa_0 memory block for resulting SA for positions beginning in i mod 3 = 0
    *
    * This method identifies the order of the characters in input_string in positions i mod 3 = 0
    * with information of ranks of triplets starting in position i mod 3 != 0 of input string.
    * This method works correct because of the difference cover idea.
    */
    void induce_sa_dc(const T& t_0, const I& isa_12, S& sa_0) {

    DCHECK_MSG(sa_0.size() == t_0.size(),
               "sa_0 must have the same length as t_0");
    DCHECK_MSG(!(isa_12.size() >= 1 && t_0.size() >= 1) ||
                   isa_12.size() >= t_0.size(),
               "isa_12 must have at least the same length as t_0 if "
               "isa_12 and t_1 have at least a length of 1");

    for(size_t i = 0; i < sa_0.size(); ++i){
            sa_0[i] = 3*i;
    }

    // index of first rank for triplets beginning in i mod 3 = 2
    size_t start_pos_mod_2 = isa_12.size() / 2 + ((isa_12.size() % 2) != 0);
    
    // TODO sort with radix sort
    auto comp = [&](size_t i, size_t j) {
        if(t_0[i/3] < t_0[j/3]) return true;
        else if(t_0[i/3] > t_0[j/3]) return false;
        else{
            
            if(start_pos_mod_2 <= i/3) return true;
            else if (start_pos_mod_2 <= j/3) return false;
            if(isa_12[i/3] < isa_12[j/3]) return true;
            else return false;
        }
    };

    std::sort(sa_0.begin(), sa_0.end(), comp);
    /*
    for (size_t i = 0; i < sa_0.size(); i++) {
        if (start_pos_mod_2 > i) {
            sa_0_to_be_sorted[i] =
                (std::tuple<C, size_t, size_t>(t_0[i], isa_12[i], 3 * i));
        } else {
            sa_0_to_be_sorted[i] =
                (std::tuple<C, size_t, size_t>(t_0[i], 0, 3 * i));
        }
    }

    // TODO: sort Tupels with radix_sort
    // radix_sort(sa0_to_be_sorted, sa0);

    std::sort(sa_0_to_be_sorted.begin(), sa_0_to_be_sorted.end());
    for (size_t i = 0; i < sa_0_to_be_sorted.size(); i++) {
        sa_0[i] = std::get<2>(sa_0_to_be_sorted[i]);
    }*/
}
} // namespace sacabench::util