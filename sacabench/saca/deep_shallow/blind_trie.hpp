/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow {

// Forward-declare the `node` class.
template <typename suffix_index_type>
struct node;

/// \brief Represents an edge in the blind trie. An edge is marked by a label
///        and points to a node.
template <typename suffix_index_type>
struct edge {
    const util::character edge_label;
    node<suffix_index_type>* child;
};

/// \brief Represents a single node in the blind trie. Contains a list of
///        children, in sorted order. Each edge is marked with a character.
template <typename suffix_index_type>
class node {
public:
    inline node(suffix_index_type _content) : content(_content) {
        children = util::make_container<suffix_index_type>(0);
    }

private:
    const suffix_index_type content;
    util::container<edge<suffix_index_type>> children;
};

template <typename suffix_index_type>
class iterator;

/// \brief Represents an entire blind trie. Contains its data in extra-space.
///        Use an iterator to traverse the trie in-order.
template <typename suffix_index_type>
class blind_trie {
public:
    inline blind_trie(const util::span<suffix_index_type> bucket) {}

    inline iterator<suffix_index_type> begin() const {
        iterator<suffix_index_type> it;
        return it;
    }

    inline iterator<suffix_index_type> end() const {
        iterator<suffix_index_type> it;
        return it;
    }

private:
    inline void insert(const suffix_index_type suffix) {}

    std::optional<node<suffix_index_type>> root;
};
} // namespace sacabench::deep_shallow
