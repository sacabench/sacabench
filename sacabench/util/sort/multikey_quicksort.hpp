/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <inttypes>

namespace util {
namespace sort {
    // Sort the suffix indices in array by comparing one character in
    // input_text.
    void multikey_quicksort(span<index_type> array, const input_type& input_text);

    // Create a function with compares at one character depth.
    struct compare_one_character_at_depth {
    public:
        // The depth at which we compare.
        index_type depth;

        // 0 if equal, < 0 if the first is smaller, > 0 if the first is larger.
        //Overwrites -> operator for quicksort
        int util::sort::compare_one_character_at_depth::operator->(const
        index_type&, const index_type&) const noexcept;
        //Overwrites ->* operator for quicksort
        int util::sort::compare_one_character_at_depth::operator->*(const
        index_type&, const index_type&) const noexcept;
    private:
        // A reference to the input text.
        input_type& input_text;
    }
}
}
