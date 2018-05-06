/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once
 
#include <vector>
#include <string>
#include <unordered_map> 

namespace sacabench::util {
    void radixsort(std::vector<std::string>* input);
    void radixsort(std::vector<std::string>* input, int index);

    void radixsort(std::vector<std::string>* input) {
        radixsort(input, input->front().length() - 1);
    }

    void radixsort(std::vector<std::string>* input, int index) {
        std::unordered_map<int, std::vector<std::string>> buckets;
        int biggest_char = -1;

        // partitioning
        for (std::string s : *input) {
            buckets[(int)s[index]].push_back(s);
            if ((int)s[index] > biggest_char) { biggest_char = (int)s[index]; }
        }

        // collecting
        input->clear();
        for (int i = 0; i <= biggest_char; i++) {
            for (std::string s : buckets[i]) {
                input->push_back(s);
            }
        }

        if (index != 0) {
            radixsort(input, index - 1);
        }
    }
}

