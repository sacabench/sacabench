/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <vector>
#include "util/string.hpp"
#include <unordered_map>

namespace sacabench::util {
    void radixsort(container<string>* input);
    void radixsort(container<string>* input, size_t index);

    void radixsort(container<string>* input) {
        DCHECK_NE(input->front().size(), 0);
        radixsort(input, input->front().size() - 1);
    }

    void radixsort(container<string>* input, size_t index) {
        std::unordered_map<size_t, std::vector<string>> buckets;
        size_t biggest_char = 0;

        // partitioning
        for (string s : *input) {
            buckets[(size_t)s[index]].push_back(s);
            if ((size_t)s[index] > biggest_char) { biggest_char = (size_t)s[index]; }
        }

        // collecting
        size_t input_i = 0;
        for (size_t i = 0; i <= biggest_char; i++) {
            for (string s : buckets[i]) {
                (*input)[input_i++] = s;
            }
        }

        if (index != 0) {
            radixsort(input, index - 1);
        }
    }
}

