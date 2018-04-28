/*******************************************************************************
 * util/induce_sa_dc.hpp
 *
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 * Copyright (C) 2018 Johannes Bahne <johannes.bahne@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <tuple>
#include <array>

#pragma once

namespace sacabench::util {
  template<typename T, typename C, typename I, typename S> 
  //template T for input string
  //template C for input characters
  //template I for ISA
  //template C for SA

  /**\brief Identify order of chars starting in position i mod 3 = 0 with difference cover
  * \param input input text
  * \param isa12 calculated ISA for triplets beginning in i mod 3 != 0
  * \param sa0 memory block for resulting SA for positions beginning in i mod 3 = 0
  * 
  * This method identifies the order of the characters in input_string in positions i mod 3 = 0 
  * with information of ranks of triplets starting in position i mod 3 != 0 of input string.
  * This method works correct because of the difference cover idea.
  */
  void induce_sa_dc(T& input, I& isa_12, S& sa_0) {
    
    static_assert(sa_0.size() > 0, "Suffix Array has to be initialised.");
    
    std::array<std::tuple<C,int>, sa_0.size()> sa0_to_be_sorted;
    
    for(int i = 0; i < sa0.size(); i++) {
      sa0_to_be_sorted[i] = std::tuple(input[3*i], isa_12[i]);
    }  
    
    //TODO: sort Tupels with radix_sort
    //radix_sort(sa0_to_be_sorted, sa0);
  }
}  // namespace sacabench::util

/******************************************************************************/
