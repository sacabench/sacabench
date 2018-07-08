/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "trie.hpp"
#include <util/alphabet.hpp>
#include <util/sort/introsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow::blind {
template <typename suffix_index_type>
inline void sort(const util::string_span text,
                 const util::span<suffix_index_type> bucket,
                 const size_t common_prefix_length) {
    if (bucket.size() < 2) {
        return;
    }

    // Sort `bucket` such that the smallest suffixes (greatest index) are
    // inserted first.
    util::sort::introsort(bucket, std::greater<suffix_index_type>());

    // Create trie
    trie<suffix_index_type> my_trie(text, common_prefix_length, bucket[0]);
    for (size_t i = 1; i < bucket.size(); ++i) {
        my_trie.insert(bucket[i]);
    }

    // Write correct ordering to bucket
    my_trie.traverse(bucket);
}

class saca {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "BlindSACA";
    static constexpr char const* DESCRIPTION =
        "SACA using Blind Sort (Component of Deep-Shallow)";

    /// \brief Use Deep Shallow Sorting to construct the suffix array.
    template <typename sa_index_type>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index_type> sa) {

        // Check if `sa_index_type` is suitable.
        DCHECK(util::assert_text_length<sa_index_type>(text.size(), 0));

        if (text.size() < 2) {
            return;
        }

        // Create trie
        trie<sa_index_type> my_trie(text, 0, text.size() - 1);
        for (size_t i = 1; i < text.size(); ++i) {
            my_trie.insert(text.size() - i - 1);
        }

        // Write correct ordering to bucket
        my_trie.traverse(sa);
    }
};
} // namespace sacabench::deep_shallow::blind
