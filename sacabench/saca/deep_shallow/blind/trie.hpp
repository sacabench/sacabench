/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>
#include <util/string.hpp>

namespace sacabench::deep_shallow::blind {

/// \brief A helper function, which prints the specified amount of spaces.
inline void print_spaces(const size_t depth) {
    for (size_t i = 0; i < depth; ++i) {
        std::cout << ' ';
    }
}

template <typename suffix_index_type>
class trie {
private:
    struct node {
        /// \brief The character on the edge that is incident to this node.
        util::character incoming_char;

        /// \brief The amount of characters every suffix in this set shares.
        suffix_index_type lcp;

        /// \brief The suffix index this leaf corresponds to.
        suffix_index_type si;

        /// \brief Child pointers.
        std::vector<node> children;

        inline node() : incoming_char(0), lcp(0), si(0), children() {}

        inline static node new_leaf(const util::string_span input_text,
                                    const suffix_index_type _si) {
            node n;
            n.incoming_char = input_text[_si];
            n.lcp = input_text.size() - _si;
            n.si = _si;
            return n;
        }

        inline static node new_inner_node(const suffix_index_type _lcp) {
            node n;
            n.lcp = _lcp;
            return n;
        }

        inline bool is_leaf() const { return children.empty(); }

        /// \brief A dumb method, which adds the given node to the children of
        ///        this node, while keeping the
        ///        "sorted-by-edge-label"-invariant.
        inline void add_child(node&& new_child) {
            // FIXME: Use insertion sort here.
            children.push_back(new_child);
            std::sort(children.begin(), children.end(),
                      [](const node& a, const node& b) {
                          // Vergleiche: das erste zeichen nach dem LCP von
                          // diesem Knoten
                          return a.incoming_char < b.incoming_char;
                      });
        }

        inline void print(const util::string_span input_text,
                          const size_t depth) const {
            util::character print_incoming_char = incoming_char;
            if (print_incoming_char == util::SENTINEL) {
                print_incoming_char = '$';
            }
            std::cout << print_incoming_char;

            if (is_leaf()) {
                const util::string_span suffix = input_text.slice(si);
                std::cout << " [ " << lcp << " ] -> " << si << ": '" << suffix
                          << "'" << std::endl;
            } else {
                std::cout << " [ " << lcp << " ]" << std::endl;
                for (const node& child : children) {
                    print_spaces(depth);
                    std::cout << "'- ";
                    child.print(input_text, depth + 3);
                }
            }
        }

        /// \brief Returns the suffix index of a random leaf which is a child
        ///        of this node.
        inline suffix_index_type get_any_leaf_si() const {
            if (is_leaf()) {
                return si;
            } else {
                return children[0].get_any_leaf_si();
            }
        }

        /// \brief Checks, if this node can contain the new_element.
        ///        That means, if the LCP is identical to the LCP this node
        ///        represents.
        inline bool can_contain(const util::string_span input_text,
                                const suffix_index_type new_element) const {
            const auto example_suffix = get_any_leaf_si();
            const util::string_span new_prefix = input_text.slice(new_element);
            const util::string_span existing_prefix =
                input_text.slice(example_suffix);

            return new_prefix.slice(0, lcp) == existing_prefix.slice(0, lcp);
        }

        /// \brief This is the main tree construction method. It inserts the
        ///        new_element either as a child to this node or into a child
        ///        of this node.
        inline void insert(const util::string_span input_text,
                           const suffix_index_type new_element) {
            // Try to find out if a suitable edge exists
            bool does_edge_exist = false;
            node* possible_child;
            for (node& child : children) {

                // Check if an edge exists with the correct character label.
                if (child.incoming_char == input_text[new_element + lcp]) {
                    does_edge_exist = true;
                    possible_child = &child;
                    if (child.can_contain(input_text, new_element)) {
                        // Case 1: There is an edge and the LCP compatible with
                        // the new suffix.

                        child.insert(input_text, new_element);
                        return;
                    }
                    break;
                }
            }

            if (does_edge_exist) {
                // Case 2: There is an edge, but the LCP is not the same
                // anymore. We need to split the selected child node into two.

                // The common prefix the next node represents.
                const suffix_index_type existing_suffix_index =
                    possible_child->get_any_leaf_si();
                const util::string_span existing_suffix =
                    input_text.slice(existing_suffix_index);
                const util::string_span new_suffix =
                    input_text.slice(new_element);

                // Find attributes of the new nodes.
                // This is the LCP of new_element and the selected child.
                suffix_index_type lcp_of_new_node;

                // Save the edge label of the selected child.
                util::character old_edge_label;

                for (suffix_index_type i = lcp;; ++i) {
                    // Compare the prefixes of existing and new suffix.
                    const util::character existing_prefix = existing_suffix[i];
                    const util::character new_prefix = new_suffix[i];

                    if (existing_prefix != new_prefix) {
                        lcp_of_new_node = i;
                        old_edge_label = existing_prefix;
                        break;
                    }
                }

                // We now have:
                // - The lcp of the new inner node (lcp_of_new_node)
                // - The edge label to the new inner node
                // (input_text[new_element])
                // - The edge label to the old child (old_edge_label)

                // Create new node structure:
                // this -> new_inner--> new_leaf
                //                  `-> old_node--> ...
                //                              `-> ...

                node new_inner = node::new_inner_node(lcp_of_new_node);
                new_inner.incoming_char = input_text[new_element + lcp];

                node new_leaf = node::new_leaf(input_text, new_element);
                new_leaf.incoming_char =
                    input_text[new_element + lcp_of_new_node];

                node old_node = *possible_child;
                old_node.incoming_char = old_edge_label;

                // Remove the old child from children.
                children.erase(
                    std::remove_if(children.begin(), children.end(),
                                   [&](const node& a) {
                                       return a.incoming_char ==
                                              input_text[new_element + lcp];
                                   }),
                    children.end());

                new_inner.add_child(std::move(old_node));
                new_inner.add_child(std::move(new_leaf));
                add_child(std::move(new_inner));
            } else {
                if (is_leaf()) {
                    // Case 4: This node is itself a leaf. We then make this
                    // node an inner node and insert a dummy leaf with no edge
                    // label.

                    node n = node::new_leaf(input_text, si);
                    n.incoming_char = util::SENTINEL;
                    add_child(std::move(n));
                    si = 0;
                }

                // Case 3: There is no edge yet for this character. Insert a
                // leaf below this node.

                node n = node::new_leaf(input_text, new_element);
                n.incoming_char = input_text[new_element + lcp];
                add_child(std::move(n));
            }
        }

        /// \brief Traverses the trie in order and saves the indices into the
        ///        bucket.
        inline size_t traverse(util::span<suffix_index_type> bucket) const {
            if (is_leaf()) {
                bucket[0] = si;
                return 1;
            } else {
                size_t n_suffixe = 0;
                for (const node& child : children) {
                    size_t n_child = child.traverse(bucket);
                    bucket = bucket.slice(n_child);
                    n_suffixe += n_child;
                }
                return n_suffixe;
            }
        }
    };

    // This text is shifted by common_prefix_length, so that the common prefix
    // is ignored in the comparisons here.
    const util::string_span m_input_text;
    node m_root;

public:
    /// \brief Construct a blind trie, which contains the initial_element.
    inline trie(const util::string_span _input_text,
                const size_t common_prefix_length,
                const suffix_index_type initial_element)
        : m_input_text(_input_text.data() + common_prefix_length, _input_text.size() - common_prefix_length) {
        m_root = node::new_inner_node(0);
        m_root.incoming_char = util::SENTINEL;

        node n = node::new_leaf(m_input_text, initial_element);
        n.incoming_char = m_input_text[initial_element];
        m_root.add_child(std::move(n));
    }

    inline void print() const { m_root.print(m_input_text, 0); }

    /// \brief Insert an element into the correct place of the blind trie.
    inline void insert(const suffix_index_type new_element) {
        m_root.insert(m_input_text, new_element);
    }

    /// \brief Traverse the blind trie in order and copy the suffixes into
    ///        bucket.
    inline void traverse(util::span<suffix_index_type> bucket) const {
        size_t n = m_root.traverse(bucket);
        DCHECK_EQ(n, bucket.size());

        // "Use" `n` so that the compiler doesn't warn
        // about it being unused.
        (void)n;
    }
};

} // namespace sacabench::deep_shallow::blind
