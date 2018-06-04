/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>
#include <util/string.hpp>

namespace sacabench::deep_shallow::blind {

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

        inline void add_child(node&& new_child) {
            // TODO: Use insertion sort here.
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

        inline suffix_index_type get_any_leaf_si() const {
            if (is_leaf()) {
                return si;
            } else {
                return children[0].get_any_leaf_si();
            }
        }

        inline bool can_contain(const util::string_span input_text,
                                const suffix_index_type new_element) const {
            const auto example_suffix = get_any_leaf_si();
            const util::string_span new_prefix = input_text.slice(new_element);
            const util::string_span existing_prefix =
                input_text.slice(example_suffix);

            // std::cout << "lcp: " << lcp << std::endl;
            // std::cout << "new: " << new_prefix << std::endl;
            // std::cout << "exi: " << existing_prefix << std::endl;

            return new_prefix.slice(0, lcp) == existing_prefix.slice(0, lcp);
        }

        inline void insert(const util::string_span input_text,
                           const suffix_index_type new_element) {
            // Try to find out, if a suitable edge exists
            bool does_edge_exist = false;
            node* possible_child;
            for (node& child : children) {
                // std::cout << "checking edge " << child.incoming_char <<
                // std::endl;
                if (child.incoming_char == input_text[new_element + lcp]) {
                    // std::cout << "match" << std::endl;
                    does_edge_exist = true;
                    possible_child = &child;
                    if (child.can_contain(input_text, new_element)) {
                        // Case 1: There is an edge and the LCP compatible with
                        // the new suffix.
                        // std::cout << "case 1:" << std::endl;
                        child.insert(input_text, new_element);
                        return;
                    }
                    break;
                }
            }

            // std::cout << std::endl;

            if (does_edge_exist) {
                // std::cout << "case 2:" << std::endl;
                //
                // this->print(input_text, 0);
                // std::cout << std::endl;
                // possible_child->print(input_text, 0);

                // Case 2: There is an edge, but the LCP is not the same
                // anymore. We need to split the selected child node into two.

                // The common prefix the next node represents.
                const suffix_index_type existing_suffix_index =
                    possible_child->get_any_leaf_si();
                const util::string_span existing_suffix =
                    input_text.slice(existing_suffix_index);
                const util::string_span new_suffix =
                    input_text.slice(new_element);

                // std::cout << "existing suffix starting at this node: " <<
                // existing_suffix << std::endl; std::cout << "trying to insert
                // here: " << new_suffix << std::endl;

                suffix_index_type lcp_of_new_node;
                util::character old_edge_label;

                // std::cout << "starting at lcp = " << lcp << std::endl;

                for (suffix_index_type i = lcp;; ++i) {
                    // Compare the prefixes of existing and new suffix.
                    // std::cout << "comparing at position " << (i) <<
                    // std::endl;
                    const util::character existing_prefix = existing_suffix[i];
                    const util::character new_prefix = new_suffix[i];
                    // std::cout << existing_prefix << " vs " << new_prefix
                    //           << std::endl;
                    if (existing_prefix != new_prefix) {
                        lcp_of_new_node = i;
                        old_edge_label = existing_prefix;
                        break;
                    }
                }

                // std::cout << "found new lcp of inner node: " <<
                // lcp_of_new_node << std::endl;

                // We now have:
                // - The lcp of the new inner node (lcp_of_new_node)
                // - The edge label to the new inner node
                // (input_text[new_element])
                // - The edge label to the old child (old_edge_label)

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
                return;
            } else {
                if (is_leaf()) {
                    // std::cout << "case 4:" << std::endl;
                    // Case 4: This node is itself a leaf. We then make this
                    // node an inner node and insert a dummy leaf with no edge
                    // label.
                    node n = node::new_leaf(input_text, si);
                    n.incoming_char = util::SENTINEL;
                    add_child(std::move(n));
                    si = 0;
                } else {
                    // std::cout << "case 3:" << std::endl;
                }

                // Case 3: There is no edge yet for this character. Insert a
                // leaf below this node.

                node n = node::new_leaf(input_text, new_element);
                n.incoming_char = input_text[new_element + lcp];
                add_child(std::move(n));
                return;
            }
        }

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

    const util::string_span m_input_text;
    node m_root;

public:
    inline trie(const util::string_span _input_text,
                const suffix_index_type initial_element)
        : m_input_text(_input_text) {
        m_root = node::new_inner_node(0);
        m_root.incoming_char = util::SENTINEL;

        node n = node::new_leaf(m_input_text, initial_element);
        n.incoming_char = m_input_text[initial_element];
        m_root.add_child(std::move(n));
    }

    inline void print() const { m_root.print(m_input_text, 0); }

    inline void insert(const suffix_index_type new_element) {
        m_root.insert(m_input_text, new_element);
    }

    inline void traverse(util::span<suffix_index_type> bucket) const {
        size_t n = m_root.traverse(bucket);
        DCHECK_EQ(n, bucket.size());

        // "Use" `n` so that the compiler doesn't warn
        // about it being unused.
        (void)n;
    }
};

} // namespace sacabench::deep_shallow::blind
