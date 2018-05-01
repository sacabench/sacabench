/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once
 
#include<vector>
#include<string>
#include<map>

namespace sacabench::util {
    void radixsort(std::vector<std::string>* input);
    void radixsort(std::vector<std::string>* input, int index);

    void radixsort(std::vector<std::string>* input) {
        radixsort(input, input->front().length());
    }

    void radixsort(std::vector<std::string>* input, int index) {
        std::map<int, std::vector<std::string>> buckets;

        // partitioning
        for (std::string s : *input) {
            buckets[(int)s[index]].push_back(s);
        }

        // collecting
        input->clear();
        for (auto const&[key, val] : buckets) {
            for (std::string s : val) {
                input->push_back(s);
            }
        }

        if (index != 0) {
            radixsort(input, index - 1);
        }
    }
}
