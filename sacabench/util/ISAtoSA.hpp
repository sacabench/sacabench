/*******************************************************************************
 * bench/ISAtoSA.hpp
 *
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 * 
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

namespace sacabench::util {
template<typename T>
/**\brief Transforms inverse Suffix Array to Suffix Array
 * \param isa calculated ISA
 * \param sa memory block for resulting SA
 * 
 * This method transforms the ISA to SA using a simple scan from right to left
 */
void isa2sa_simple_scan(T& isa, T& sa) {
  for(int i = 0; i < isa.size(); i++) {
    sa[isa[i]] = i;
  }  
}
}
/******************************************************************************/
