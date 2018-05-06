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
void isa2sa_simple_scan(const T& isa, T& sa) {
  for(size_t i = 0; i < isa.size(); i++) {
    sa[isa[i]] = i;
  }
}

template<typename T>
/**\brief Transforms inverse Suffix Array to Suffix Array, using no extra space
 * \param isa calculated ISA
 *
 * This method transforms the ISA to SA by
 * cyclically replacing ranks with suffix-indices
 * Note: ranks are always < 0!
 */
void isa2sa_inplace(T& isa) {
    for(size_t i = 0; i < isa.size(); i++) {
        if(isa[i] < 0) {
            ssize_t start = i;
            ssize_t suffix = i;
            auto rank = -isa[i]-1;
            do {
                auto tmp_rank = -isa[rank]-1;
                isa[rank] = suffix;
                suffix = rank;
                rank = tmp_rank;

            } while(rank != start);
            isa[rank] = suffix;
        }
    }
}

}
/******************************************************************************/
