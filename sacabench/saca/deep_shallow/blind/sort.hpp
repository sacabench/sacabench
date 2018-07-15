/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "../log.hpp"
#include "trie.hpp"

#include <util/alphabet.hpp>
#include <util/sort/introsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow::blind {

constexpr size_t MIN_BLINDSORT_SIZE = 2;

template <typename suffix_index_type>
inline void sort(const util::string_span text,
                 const util::span<suffix_index_type> bucket,
                 const size_t common_prefix_length) {
    DCHECK_GT(bucket.size(), 2);

    logger::get() << "####################################################\n";
    logger::get() << "#                    BLIND SORT                    #\n";

    // Sort `bucket` such that the smallest suffixes (greatest index) are
    // inserted first.
    size_t ns = duration([&]() {
        util::sort::introsort(bucket, std::greater<suffix_index_type>());
    });
    logger::get() << "Initial sorting took " << ns << "ns.\n";

    // Create trie
    trie<suffix_index_type> my_trie(text, common_prefix_length, bucket[0]);

    for (size_t i = 1; i < bucket.size(); ++i) {
        ns = duration([&]() { my_trie.insert(bucket[i]); });
        logger::get() << "Insertion of element " << i << "/" << bucket.size()
                      << " took " << ns << "ns.\n";
    }

    ns = duration([&]() {
        // Write correct ordering to bucket
        my_trie.traverse(bucket);
    });

    logger::get() << "Final tree traversal took " << ns << "ns.\n";
    logger::get() << "#                                                  #\n";
    logger::get() << "####################################################\n";
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
