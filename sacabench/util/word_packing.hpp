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
sa_index own_pow(sa_index base, sa_index exp) {
    if (exp == sa_index(0))
        return 1;
    sa_index result = 1;
    for (sa_index index = 0; index < exp; ++index) {
        result = result * base;
    }
    return result;
}

/**\brief Transforms a given text with word packing technique
 * \param text Given Text
 * \param result Container to save the result in
 * \param alpha Alphabet belonging to text
 * \param sentinels Number of sentinels
 * \param needed_tag_bits Number of reserved bits at the end
 * 
 * This method transforms a given text by packing as much as possible 
 * characters into one register. By doing this, comparing two indices
 * at the packed array allows you to use information about the 
 * following indices as well, without changing the original order.
 * Note that the type stored in the result array doesnt have to 
 * match to sa_index. A bigger type allows to pack more characters in
 * one register, leading to include more indices to compare at once, but
 * for the cost of higher memory usage.
 */
template <typename T, typename sa_index>
void word_packing(T text, util::container<sa_index>& result, alphabet alpha,
                  size_t sentinels, size_t needed_tag_bits) {

    if (!alpha.is_effective()) {
        alpha = apply_effective_alphabet(string(text));
    }
    sa_index max_value = alpha.max_character_value() + 1;
    // nicer solutions are welcome...
    auto r = static_cast<sa_index>(static_cast<size_t>(
        (log(std::numeric_limits<sa_index>::max() >> needed_tag_bits) /
         log(static_cast<size_t>(max_value)))));
    sa_index n = text.size() - sentinels;
    // Pack first index "by hand"
    for (sa_index index = 0; index < r && index < n; ++index) {
        result[0] += sa_index((text[index])) *
                     sa_index((own_pow(max_value, r - index - sa_index(1))));
    }
    // Pack other indices accoring to their predecessor
    for (size_t index = 1; index < n; ++index) {
        result[index] =
            (result[index - 1] %
             (sa_index(size_t(own_pow(max_value, r - sa_index(1)))))) *
                max_value +
            (((index + r - 1) < n) ? sa_index(size_t(text[index + r - 1]))
                                   : sa_index(0));
    }
}
// TODO Fast & inplace
} // namespace sacabench::util
