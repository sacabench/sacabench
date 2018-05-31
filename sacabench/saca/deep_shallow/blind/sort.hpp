/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "trie.hpp"
#include <util/sort/introsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow::blind {
template <typename suffix_index_type>
inline void sort(const util::string_span text,
                 const util::span<suffix_index_type> bucket) {
    if (bucket.size() < 2) {
        return;
    }

    // Sort `bucket` such that the smallest suffixes (greatest index) are
    // inserted first.
    util::sort::introsort(bucket, std::greater());

    // std::cout << bucket << std::endl;

    // Create trie
    trie my_trie(text, bucket[0]);
    for (size_t i = 1; i < bucket.size(); ++i) {
        // my_trie.print();
        my_trie.insert(bucket[i]);
    }

    // Write correct ordering to bucket
    my_trie.traverse(bucket);
}
} // namespace sacabench::deep_shallow::blind
