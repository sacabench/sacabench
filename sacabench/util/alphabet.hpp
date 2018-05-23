/*******************************************************************************
 * Copyright (C) 2018 Florian Grieskamp <florian.grieskamp@tu-dortmund.de>
 * Copyright (C) 2018 Florian Kurpicz   <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "util/string.hpp"

namespace sacabench::util {

    struct alphabet {
        std::size_t size;
        std::array<character,
                   std::numeric_limits<character>::max() + 1> effective;

        /**\brief Generates an effective alphabet by searching an input text for
         *      occurring characters
         * \param input Input text that is going to be scanned
         *
         * This constructor takes the input text and creates an effective
         * alphabet with characters {1, ..., n} where n is the number of
         * distinct characters in the input text. If the text relies on all
         * characters (0...max_char), the resulting map will be invalid.
         */
        alphabet(const string& input)
        {
            // Find all existing characters (mark with 1)
            std::fill(effective.begin(), effective.end(), 0);
            for (const character c : input) {
                effective[c] = 1;
            }

            // Create a mapping for the new (effective) alphabet (zero not
            // included)
            //
            // (Maybe use std::inclusive_scan)
            for (std::size_t i = 1; i < effective.size(); ++i) {
                effective[i] += effective[i - 1];
            }

            // The number of effective characters equals the effective value of
            // the greatest character.
            size = effective[effective.size() - 1];
        }
    };

    /**\brief Transforms a text such that it uses the effective alphabet
     * \param input Input text that is going to be transformed
     * \param alphabet_map Alphabet to be used for transformation
     *
     * This method takes the input text and replaces it's symbols according to
     * the specified mapping in the given effective alphabet. Note that the
     * transformation relies on the correctness of the alphabet map.
     */
    void apply_effective_alphabet(string& input, const alphabet& alphabet_map) {
        // Map the characters using the new effective alphabet
        for (character& c : input) {
            c = alphabet_map.effective[c];
        }
    }

    /**\brief Transforms a text such that it uses the effective alphabet
     * \param input Input text that is going to be transformed
     *
     * This method takes the input text and replaces its symbols by consecutive
     * numbers in the `character` range, holding onto the relative ordering.
     */
    alphabet apply_effective_alphabet(string& input) {
        const alphabet input_alphabet(input);
        apply_effective_alphabet(input, input_alphabet);
        return input_alphabet;
    }
}

/******************************************************************************/
