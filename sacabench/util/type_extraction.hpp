/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#include "string.hpp"

#pragma once


namespace sacabench::util {

    bool from_right_global = false;
    string_span t_0_global;
    string_span alph_global;

    size_t current_string_position = 0;
    bool last_type_global = 0;
    size_t last_alphabet_index_global = 0;
    bool string_is_finished = false;


    /** Check if c_1 comes after c_2 in the given alphabet
    */
    size_t get_alphabet_index(character c, string_span alph) {

        for (size_t i = 0; i < alph.size(); i++) {
            if (alph[i] == c) {

                return i;
            }
        }
    }

    /**\Initialise the inplace L-S-TypeExtraction which you can call with "get_next_type_onfly"
    * \param t_0 input text
    * \param alph alphabet
    * \param from_right whether you want to iterate the text from right or from left
    */
    void initialize_type_extraction_onfly(string_span t_0, string_span alph, bool from_right) {

        from_right_global = from_right;
        t_0_global = t_0;
        alph_global = alph;

        current_string_position = from_right_global ? t_0.size()-1 : 0;
        last_type_global = 0;
        last_alphabet_index_global = 0;
    }

    /**\Return the L/S-Type of the next character in the initialized text, in the initialized direction (boolean: 1 = L, 0 = S) */
    bool get_next_type_onfly()
    {
        size_t iteration_direction = from_right_global ? -1 : 1;

        bool const not_sentinel = (current_string_position != t_0_global.size() - 1);
        size_t const current_alphabet_index = get_alphabet_index(t_0_global[current_string_position], alph_global);

        last_type_global = (not_sentinel &&		                                    // Symbol is not sentinel
            (current_alphabet_index > last_alphabet_index_global || 	            // Symbol is larger than following Symbol OR
            (current_alphabet_index == last_alphabet_index_global && last_type_global)));	// Symbol is equal to following Symbol

        last_alphabet_index_global = current_alphabet_index;
        string_is_finished == ((current_string_position == 0 && iteration_direction == -1) || (current_string_position == t_0_global.size() - 1 && iteration_direction == 1));
        current_string_position += iteration_direction;

        return last_type_global;
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

    size_t symbols_left()
    {
        return string_is_finished? 0 : (from_right_global ? current_string_position + 1 : t_0_global.size() - current_string_position);
    }


}

/******************************************************************************/

