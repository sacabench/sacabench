/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/sort/std_sort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

namespace sacabench::deep_shallow {

inline void print_spaces(size_t depth) {
    for (size_t i = 0; i < depth; ++i) {
        std::cout << " ";
    }
}

/// \brief The result of a blind_node::follow_edges(.) call.
///        The possibilities are:
///        - We reached a point in the trie, were no possible edges exist in the
///          current node. We then need to split the leaf into two leafs,
///          creating an inner node in the process. One of the two new edges has
///          the label SENTINEL, and one has the rest of the string to follow.
///        - We reached a leaf. If the content of the leaf and the string to
///          look up are identical, the string is already contained in the trie.
enum follow_edges_result_type {
    no_suitable_edge,
    possible_leaf_found,
};

template <typename suffix_index_type>
struct follow_edges_result {
    follow_edges_result_type type;
    suffix_index_type content;

    inline follow_edges_result(follow_edges_result_type t, suffix_index_type c)
        : type(t), content(c) {}
};

template <typename suffix_index_type>
inline void print_result(follow_edges_result<suffix_index_type> r) {
    std::cout << "result: ";
    if (r.type == no_suitable_edge) {
        std::cout << "no suitable edge. ";
    } else {
        std::cout << "possible leaf found. ";
    }
    std::cout << (size_t)r.content << std::endl;
}

// Forward-declare the `node` class.
template <typename suffix_index_type>
class node;

/// \brief Represents an edge in the blind trie. An edge is marked by a label
///        and points to a node.
template <typename suffix_index_type>
class edge {
public:
    inline edge(const util::character l, node<suffix_index_type>* dst)
        : edge_label(l), child(dst) {}

    util::character edge_label;
    node<suffix_index_type>* child;
};

/// \brief Represents a leaf in the blind trie. Contains an entire suffix by
///        saving its suffix index.
template <typename suffix_index_type>
class leaf {
public:
    inline leaf(const suffix_index_type l, const suffix_index_type c)
        : lcp(l), content(c) {}

    inline void print_node(size_t depth) {
        print_spaces(depth);
        std::cout << "[ LCP: " << lcp << " ]" << std::endl;
        print_spaces(depth);
        std::cout << "[ Leaf: " << content << "]" << std::endl;
    }

    suffix_index_type lcp;
    suffix_index_type content;
};

/// \brief Represents a single node in the blind trie. Contains a list of
///        children, in sorted order. Each edge is marked with a character.
template <typename suffix_index_type>
class inner_node {
public:
    inline inner_node(suffix_index_type lcp_len) : lcp(lcp_len), children() {}

    inline void print_node(size_t depth) {
        print_spaces(depth);
        std::cout << "[ LCP: " << lcp << " ]" << std::endl;

        for (const auto& edge : children) {
            print_spaces(depth);
            if (edge.edge_label != '\0') {
                std::cout << "|- " << edge.edge_label << " " << std::endl;
            } else {
                std::cout << "|- <$> " << std::endl;
            }
            edge.child->print_node(depth + 4);
        }
    }

    inline void insert_child(const edge<suffix_index_type> e) {
        children.push_back(e);
        // FIXME: use insertion sort instead.
        util::sort::std_sort(children, [](const auto& a, const auto& b) {
            return a.edge_label < b.edge_label;
        });
    }

    inline follow_edges_result<suffix_index_type>
    follow_edges(const util::string_span text, suffix_index_type request) {
        for (const auto& e : children) {
            // In the comparison, skip the next LCP characters, because every
            // child of this node has the same LCP characters.
            if (e.edge_label == text[request + lcp]) {
                return e.child->follow_edges(text, request);
            }
        }

        follow_edges_result<suffix_index_type> r(
            follow_edges_result_type::no_suitable_edge, lcp);
        return r;
    }

    inline suffix_index_type get_random_leaf() {
        DCHECK_GT(children.size(), 0);
        return children[0].child->get_random_leaf();
    }

private:
    suffix_index_type lcp;
    std::vector<edge<suffix_index_type>> children;
};

/// \brief A node in the blind trie is either an inner node or a leaf.
template <typename suffix_index_type>
class node {
public:
    // This constructs a node, which is an inner node.
    inline node(inner_node<suffix_index_type> r)
        : is_leaf(false), node_ref(r) {}

    // This constructs a node, which is a leaf.
    inline node(leaf<suffix_index_type> r) : is_leaf(true), node_ref(r) {}

    inline follow_edges_result<suffix_index_type>
    follow_edges(const util::string_span text, suffix_index_type request) {
        if (is_leaf) {
            follow_edges_result<suffix_index_type> r(
                follow_edges_result_type::possible_leaf_found,
                node_ref.as_leaf.content);
            return r;
        } else {
            return node_ref.as_inner_node.follow_edges(text, request);
        }
    }

    inline suffix_index_type get_random_leaf() {
        if(is_leaf) {
            return node_ref.as_leaf.content;
        } else {
            return node_ref.as_inner_node.get_random_leaf();
        }
    }

    inline void print_node(size_t depth) {
        if (is_leaf) {
            node_ref.as_leaf.print_node(depth);
        } else {
            node_ref.as_inner_node.print_node(depth);
        }
    }

    ~node() {}

private:
    // If this is true, the `node_ref` is a `leaf`. If not, a `inner_node`.
    bool is_leaf;

    // Store either a inner_node xor a leaf.
    union contained_node {
        inner_node<suffix_index_type> as_inner_node;
        leaf<suffix_index_type> as_leaf;

        contained_node(inner_node<suffix_index_type> i) : as_inner_node(i) {}
        contained_node(leaf<suffix_index_type> l) : as_leaf(l) {}
        ~contained_node() {}
    } node_ref;
};

/// \brief An iterator over every element in the blind trie, in-order.
template <typename suffix_index_type>
class iterator;

/// \brief Represents an entire blind trie. Contains its data in extra-space.
///        Use an iterator to traverse the trie in-order.
template <typename suffix_index_type>
class blind_trie {
public:
    inline blind_trie(const util::string_span input,
                      const util::span<suffix_index_type> bucket)
        : text(input), root(nullptr) {
        // Insert every suffix from the bucket in to this trie.
        for (const suffix_index_type si : bucket) {
            insert(si);
        }
    }

    inline blind_trie(const util::string_span input,
                      node<suffix_index_type>* rnode)
        : text(input), root(rnode) {}

    inline void print_trie() {
        if (root == nullptr) {
            std::cout << "[empty trie]" << std::endl;
        } else {
            root->print_node(0);
        }
    }

    inline iterator<suffix_index_type> begin() const {
        iterator<suffix_index_type> it;
        return it;
    }

    inline iterator<suffix_index_type> end() const {
        iterator<suffix_index_type> it;
        return it;
    }

    inline follow_edges_result<suffix_index_type>
    follow_edges(suffix_index_type request) {
        return root->follow_edges(text, request);
    }

private:
    inline void insert(const suffix_index_type suffix) {
        if (root == nullptr) {
            leaf l(text.size() - suffix, suffix);
            root = new node(l);
        } else {
            // Since we don't store entire suffixes in the inner nodes, we don't
            // know where to insert the new string until we reached a leaf.
            const auto contained_string = root->follow_edges(text, suffix);
            print_result(contained_string);

            switch(contained_string.type) {
                case follow_edges_result_type::no_suitable_edge:
                    {
                        suffix_index_type lcp = contained_string.content;
                        suffix_index_type random_leaf = root->get_random_leaf(contained_string.content);
                        // Check, if the LCP is equal

                        // if it does, insert a new leaf as a child of the found node.
                    }
                    break;
                default:
                    std::cout << "could not insert." << std::endl;
                    break;
            }
        }
    }

    util::string_span text;
    node<suffix_index_type>* root;
};
} // namespace sacabench::deep_shallow
