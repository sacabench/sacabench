/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow {

template <typename T>
using span = util::span<T>;

template <typename suffix_index_type>
class blind_node {
public:
    blind_node(suffix_index_type _content) : content(_content) {
        children = util::make_container<suffix_index_type>(0);
    }

private:
    const suffix_index_type content;
    util::container<blind_node*> children;
};

template <typename suffix_index_type>
class blind_trie {
public:
    inline blind_trie() {}

    inline void insert(const suffix_index_type val) {
        if (root.has_value()) {
            // Insert recursively.
            root->insert(val);
        } else {
            // Insert new root node.
            blind_node node(val);
            root = node;
        }
    }

    inline suffix_index_type next(size_t index) {}

private:
    std::optional<blind_node<suffix_index_type>> root;
};

template <typename suffix_index_type>
inline void blind_sort(const span<suffix_index_type> array) {
    // Create empty blind trie
    blind_trie<suffix_index_type> bt;

    // Insert all suffixes
    for (suffix_index_type elem : array) {
        bt.insert(elem);
    }

    // Traverse the blind trie, writing the sorted elements to `array`
    for (suffix_index_type i = 0; i < array.size(); ++i) {
        array[i] = bt.next(i);
    }
}
} // namespace sacabench::deep_shallow
