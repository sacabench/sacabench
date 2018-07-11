/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <optional>
#include <util/is_sorted.hpp>
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
    struct node;

    class arena {
        inline arena() {}

        inline node allocate(const size_t nobj) { return new node[nobj]; }

        inline node deallocate(node* ptr, const size_t nobj) { delete[] ptr; }
    };

    struct node {
        /// \brief The character on the edge that is incident to this node.
        util::character incoming_char;

        /// \brief The amount of characters every suffix in this set shares.
        suffix_index_type lcp;

        /// \brief Child pointers.
        std::vector<node> children;

        inline node() : incoming_char(0), lcp(0), children() {}
        inline node(const node&) = delete;

        inline node(node&&) = default;
        inline node& operator=(node&&) = default;

        inline static node new_leaf(const util::string_span input_text,
                                    const size_t _si) {
            node n;
            n.incoming_char = input_text[_si];
            n.lcp = input_text.size() - _si;
            return n;
        }

        inline static node new_inner_node(const size_t _lcp) {
            node n;
            n.lcp = _lcp;
            return n;
        }

        inline SB_FORCE_INLINE size_t get_si(const size_t text_size) const {
            return text_size - lcp;
        }

        inline bool is_leaf() const { return children.empty(); }

        /// \brief A dumb method, which adds the given node to the children of
        ///        this node, while keeping the
        ///        "sorted-by-edge-label"-invariant.
        inline void add_child(node&& new_child) {
            // Use binary search to find the first element in children, which is
            // larger than the new. The correct position is right in front of
            // the found element.
            auto it = std::find_if(
                children.begin(), children.end(), [&](const node& n) {
                    return n.incoming_char >= new_child.incoming_char;
                });
            children.emplace(it, std::move(new_child));
        }

        inline void print(std::ostream& out, const util::string_span input_text,
                          const size_t depth) const {
            util::character print_incoming_char = incoming_char;
            if (print_incoming_char == util::SENTINEL) {
                print_incoming_char = '$';
            }
            out << print_incoming_char;

            if (is_leaf()) {
                const size_t si = get_si(input_text.size());
                const util::string_span suffix = input_text.slice(si);
                out << " [ " << lcp << " ] -> " << si << ": '" << suffix << "'"
                    << std::endl;
            } else {
                out << " [ " << lcp << " ]" << std::endl;
                for (const node& child : children) {
                    print_spaces(depth);
                    out << "'- ";
                    child.print(out, input_text, depth + 3);
                }
            }
        }

        /// \brief Returns the suffix index of a random leaf which is a child
        ///        of this node.
        inline size_t get_any_leaf_si(const size_t text_size) const {
            if (is_leaf()) {
                return get_si(text_size);
            } else {
                return children[0].get_any_leaf_si(text_size);
            }
        }

        /// \brief Checks, if this node can contain the new_element.
        ///        That means, if the LCP is identical to the LCP this node
        ///        represents.
        inline bool can_contain(const util::string_span input_text,
                                const size_t new_element,
                                const size_t si) const {
            const util::string_span new_prefix = input_text.slice(new_element);
            const util::string_span existing_prefix = input_text.slice(si);
            return new_prefix.slice(0, lcp) == existing_prefix.slice(0, lcp);
        }

        inline void split(const util::string_span input_text,
                          node* possible_child, const size_t new_element,
                          const size_t leaf_si) {

            // The common prefix the next node represents.
            const util::string_span existing_suffix = input_text.slice(leaf_si);
            const util::string_span new_suffix = input_text.slice(new_element);

            // Find attributes of the new nodes.
            // This is the LCP of new_element and the selected child.
            size_t lcp_of_new_node;

            // Save the edge label of the selected child.
            util::character old_edge_label;

            for (size_t i = lcp;; ++i) {
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

            node new_inner = std::move(node::new_inner_node(lcp_of_new_node));
            new_inner.incoming_char = input_text[new_element + lcp];

            node new_leaf = node::new_leaf(input_text, new_element);
            new_leaf.incoming_char = input_text[new_element + lcp_of_new_node];

            node old_node = std::move(*possible_child);
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
        }

        /// \brief This method traverses the trie, but has the meta information
        ///        from the first pass-through about the LCP of the new element
        ///        and the elements on the existing path.
        /// \param max_lcp The length of the common prefix of new_element and
        ///                the elements on the path in the trie.
        /// \param si      The content of any leaf which is on the correct path.
        inline void second_pass_insert(const util::string_span input_text,
                                       const size_t new_element,
                                       const size_t max_lcp, const size_t si) {

            // Follow the path until a node is encountered, which node label is
            // greater than lcp_len.
            for (node& child : children) {

                // Check if an edge exists with the correct character label.
                if (child.incoming_char == input_text[new_element + lcp]) {

                    // Check, if the possible child still has compatible LCPs.
                    if (child.lcp > max_lcp) {

                        // We now have the following situation:
                        // - This node is the last node with a correct LCP.
                        // - possible_child is the node with the "correct" edge
                        //   label, but too large LCP

                        split(input_text, &child, new_element, si);
                        return;

                    } else {
                        return child.second_pass_insert(input_text, new_element,
                                                        max_lcp, si);
                    }
                }
            }
        }

        /// \brief   This is the main tree construction method. It inserts the
        ///          new_element either as a child to this node or into a child
        ///          of this node.
        ///
        /// \returns 0, if the node was successfully inserted. Otherwise the
        ///          length of the LCP with the existing path.
        inline std::pair<size_t, size_t>
        insert(const util::string_span input_text, const size_t new_element) {
            // Scan the children to find the right edge.
            // TODO: Replace by binary search since the children are sorted by
            // their edge labels.
            for (node& child : children) {

                // Check if an edge exists with the correct character label.
                if (child.incoming_char == input_text[new_element + lcp]) {
                    // TODO: Replace recursive call here with iterative
                    //       approach?

                    // Case 1: There is an edge with the correct edge label. We
                    // have to assume, that the LCP matches. Once we reach a
                    // leaf we will see if that assessment was correct.
                    return child.insert(input_text, new_element);
                }
            }

            // We know, that no edge exists for the suffix to insert. Therefore,
            // we can just create a leaf and add it as a child of this node.

            const size_t si = get_any_leaf_si(input_text.size());
            if (can_contain(input_text, new_element, si)) {
                if (is_leaf()) {
                    // Case 4: This node is itself a leaf. We then make this
                    // node an inner node and insert a dummy leaf with no
                    // edge label.

                    logger::get() << "Blind: Case 4.\n";

                    node n = node::new_leaf(input_text, si);
                    n.incoming_char = util::SENTINEL;
                    add_child(std::move(n));
                }

                // Case 3: There is no edge yet for this character. Insert a
                // leaf below this node.

                node n = node::new_leaf(input_text, new_element);
                n.incoming_char = input_text[new_element + lcp];
                add_child(std::move(n));

                logger::get() << "Blind: Case 3.\n";

                return std::make_pair(0, 0);
            } else {
                // Case 5: We know, that there is an usable edge existing at
                // each level, but one of the LCPs is not identical. We can
                // now compute the LCP of this node and the
                // to-be-newly-inserted node. We then return to the tree
                // root and try again, now with the information, how long
                // the LCP will be.

                logger::get() << "Blind: Case 5.\n";

                const util::string_span this_lcp = input_text.slice(si);
                const util::string_span new_lcp = input_text.slice(new_element);
                size_t lcp_len;
                for (lcp_len = 0; this_lcp[lcp_len] == new_lcp[lcp_len];
                     ++lcp_len) {
                }

                // LCP can't be 0, because then we would have inserted a
                // new edge.
                DCHECK_GT(lcp_len, 0);
                return std::make_pair(lcp_len, si);
            }
        }

        /// \brief Traverses the trie in order and saves the indices into the
        ///        bucket.
        inline size_t traverse(const size_t text_size,
                               util::span<suffix_index_type> bucket) const {
            if (is_leaf()) {
                bucket[0] = get_si(text_size);
                return 1;
            } else {
                size_t n_suffixe = 0;
                for (const node& child : children) {
                    size_t n_child = child.traverse(text_size, bucket);
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
                const size_t common_prefix_length, const size_t initial_element)
        : m_input_text(_input_text.slice(common_prefix_length)),
          m_root(std::move(node::new_inner_node(0))) {
        m_root.incoming_char = util::SENTINEL;

        node n = std::move(node::new_leaf(m_input_text, initial_element));
        n.incoming_char = m_input_text[initial_element];
        m_root.add_child(std::move(n));
    }

    inline void print(std::ostream& out) const {
        m_root.print(out, m_input_text, 0);
    }

    /// \brief Insert an element into the correct place of the blind trie.
    inline void insert(const size_t new_element) {
        const auto leaf_data = m_root.insert(m_input_text, new_element);
        if (leaf_data.first > 0) {
            m_root.second_pass_insert(m_input_text, new_element,
                                      leaf_data.first, leaf_data.second);
        }
    }

    /// \brief Traverse the blind trie in order and copy the suffixes into
    ///        bucket.
    inline void traverse(util::span<suffix_index_type> bucket) const {
        size_t n = m_root.traverse(m_input_text.size(), bucket);
        DCHECK_EQ(n, bucket.size());
        DCHECK(is_partially_suffix_sorted(bucket, m_input_text));

        // "Use" `n` so that the compiler doesn't warn
        // about it being unused.
        (void)n;
    }
};

template <typename sa_index>
inline std::ostream& operator<<(std::ostream& out, const trie<sa_index>& t) {
    t.print(out);
    return out;
}

} // namespace sacabench::deep_shallow::blind
