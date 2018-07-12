/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <util/alphabet.hpp>

namespace sacabench::util {
template <typename sa_index>
sa_index own_pow(sa_index base, sa_index exp)
{
    if(exp==sa_index(0)) return 1;
    sa_index result =1;
    for(sa_index index =0; index<exp;++index)
    {
        result=result*base;
    }
    return result;
}
    
    
template <typename T, typename sa_index>
void word_packing(T text, util::container<sa_index>& result, alphabet alpha,
                  size_t sentinels, size_t needed_tag_bits) {

    if (!alpha.is_effective()) {
        alpha = apply_effective_alphabet(string(text));
    }
    sa_index max_value = alpha.max_character_value() + 1;

    auto r =static_cast<sa_index> (static_cast<size_t>((log(std::numeric_limits<sa_index>::max() >> needed_tag_bits) / log(static_cast<size_t>(max_value)))));

    sa_index n =
        text.size() -
        sentinels; // alpha.size_without_sentinel();//text.size()-sentinels;
    for (sa_index index = 0; index < r && index < n; ++index) {
        result[0] += sa_index((text[index])) *
                     sa_index((own_pow(max_value, r - index - sa_index(1))));
    }

    for (size_t index = 1; index < n; ++index) {
        //std::cout<<(result[index - 1] % (sa_index(size_t(pow(max_value, r - 1)))))<<std::endl;
        result[index] =
            (result[index - 1] % (sa_index(size_t(own_pow(max_value, r-sa_index(1)))))) *
                max_value +
            (((index + r - 1) < n) ? sa_index(size_t(text[index + r - 1])) : sa_index(0));
    }
    std::cout<<"Packed"<<std::endl;
}
//TODO Fast & inplace
} // namespace sacabench::util
