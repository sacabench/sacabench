/*******************************************************************************
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include "string.hpp"
#include "util/signed_size_type.hpp"
#include <array>
#include <tuple>

#pragma once

namespace sacabench::util {

/** Returns a type b (L=1, S=0) and a number n so that the next n characters
 * have type b */
template <typename string_type>
inline std::tuple<bool, size_t> get_type_ltr_dynamic(const string_type& t_0, size_t index) {
    if (static_cast<size_t>(t_0[index]) == util::SENTINEL)
        return std::make_tuple(false, 1);

    size_t same_char_amount = 0;
    character current_char = t_0[index];

    for (size_t i = index; i < t_0.size(); i++) {
        same_char_amount++;
        if (i + 1 >= t_0.size() || current_char != t_0[i + 1]) {
            break;
        }
    }

    bool const right_before_sentinel =
        (index + same_char_amount - 1 == t_0.size() - 2);
    character next_different_char = t_0[index + same_char_amount];

    return std::make_tuple(
        (right_before_sentinel || current_char > next_different_char),
        same_char_amount);
}

/**\Returns the type (L=1, S=0) of the character on the given index
 * \param t_0 input text
 * \param index index of the character
 * \param right_type type of the character on the right of the current index
 */
template <typename string_type>
inline bool get_type_rtl_dynamic(string_type t_0, size_t index,
                                 bool right_type) {
    // sentinel is never included in actual string t_0!!

    bool const right_before_sentinel = (index == t_0.size() - 1);

    return (right_before_sentinel || // Symbol is not sentinel
            (t_0[index] >
                 t_0[index + 1] || // Symbol is larger than following Symbol OR
             (t_0[index] == t_0[index + 1] &&
              right_type))); // Symbol is equal to following Symbol
}

/**\Save the L/S-Types of the input text in a given array (boolean: 1 = L, 0 =
 * S) \param t_0 input text \param alph alphabet \param ba boolean array where
 * the result is saved
 */
inline void get_types(string_span t_0, span<bool> ba) {

    DCHECK_MSG(t_0.size() == ba.size(),
               "t_0 must have the same length as the type array ba");

    character last_char = util::SENTINEL;

    // For size_t we need to shift everything by -1 (because size_t always is >=
    // 0)

    for (size_t j = t_0.size(); j > 0; j--) {

        size_t i = j - 1;

        bool const not_sentinel = (i != t_0.size() - 1);

        ba[i] =
            (not_sentinel &&        // Symbol is not sentinel
             (t_0[i] > last_char || // Symbol is larger than following Symbol OR
              (t_0[i] == last_char &&
               ba[i + 1]))); // Symbol is equal to following Symbol

        last_char = t_0[i];
    }
}

} // namespace sacabench::util

/******************************************************************************/
