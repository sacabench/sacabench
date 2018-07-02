/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <math.h>
#include <util/alphabet.hpp>

namespace sacabench::util
{

    template <typename T, typename sa_index>
    void word_packing(T text, util::container<sa_index>& result, alphabet alpha,size_t sentinels, size_t needed_tag_bits) {
        
        if(!alpha.is_effective()) {
            alpha= apply_effective_alphabet(string(text));
        }
        size_t max_value = alpha.max_character_value()+1;
        size_t r = sa_index(size_t(log( std::numeric_limits<sa_index>::max()>>needed_tag_bits)/log(max_value)));
        sa_index n = text.size()-sentinels;//alpha.size_without_sentinel();//text.size()-sentinels;
        for(size_t index=0;index<r&&index<n;++index) {
            result[0]+=(size_t(text[index]))* sa_index(size_t(pow(max_value, r-index-1)));
        }
        for(size_t index =1; index<n;++index) {
            result[index]=(result[index-1]%(sa_index(size_t(pow(max_value,r-1)))))*max_value + (((index+r-1)<n)? size_t(text[index+r-1]):0);
        }
    }
}
