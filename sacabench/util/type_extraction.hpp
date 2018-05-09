/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Janina Michaelis <janina.michaelis@tu-dortmund.de>
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#include "string.hpp"

#pragma once


namespace sacabench::util {

    /** Check if c_1 comes after c_2 in the given alphabet
    */
    size_t get_alphabet_index(char c, string_span alph) {

        for (size_t i = 0; i < alph.size(); i++) {
            if (alph[i] == c) {
                return i;
            }
        }
    }

    /**\Save the L/S-Types of the input text in a given array (boolean: 1 = L, 0 = S)
    * \param t_0 input text
    * \param alph alphabet
    * \param ba boolean array where the result is saved
    */
    void get_types(string_span t_0, string_span alph, span<bool> ba) {

        DCHECK_MSG(t_0.size() == ba.size(), "t_0 must have the same length as the type array ba");

        size_t last_alphabet_index = 0;

        // For size_t we need to shift everything by -1 (because size_t always is >= 0)

        for (size_t j = t_0.size(); j > 0; j--) {

            size_t i = j - 1;

            bool const not_sentinel = (i != t_0.size() - 1);
            size_t const current_alphabet_index = get_alphabet_index(t_0[i], alph);

            ba[i] = (not_sentinel &&		                                    // Symbol is not sentinel
                (current_alphabet_index > last_alphabet_index || 	            // Symbol is larger than following Symbol OR
                (current_alphabet_index == last_alphabet_index && ba[i + 1])));	// Symbol is equal to following Symbol

            last_alphabet_index = current_alphabet_index;
        }

    }
}

/******************************************************************************/

