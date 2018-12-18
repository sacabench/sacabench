/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 * Copyright (C) 2018 Oliver Magiera <oliver.magiera@tu-dortmund.de>
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include <iostream>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <memory>
#include <util/span.hpp>
#include <util/container.hpp>
#include <util/bits.hpp>
#include <util/assertions.hpp>

namespace sacabench::util {


// Operator for sum-operation (function call for prefix-sum)
template <typename Content>
struct sum_fct{
public:
    // elem and compare_to need to be smaller than input.size()
    inline Content operator()(const Content& in1, const Content& in2) const {
        return in1 + in2;
    }
};

// Operator for max-operation (function call for prefix-sum)
template <typename Content>
struct max_fct{
public:
    // elem and compare_to need to be smaller than input.size()
    inline Content operator()(const Content& in1, const Content& in2) const {
        return std::max(in1, in2);
    }
};


// Sequential version of the prefix sum calculation.
template <typename Content, typename add_operator>
void seq_prefix_sum(span<Content> in, span<Content> out, bool inclusive,
        add_operator add, Content identity) {
    DCHECK_EQ(in.size(), out.size());
    if(inclusive) {
        out[0] = add(identity, in[0]);
        for(size_t i = 1; i < in.size(); ++i) {
            out[i] = add(in[i], out[i - 1]);
        }
    } else {
        Content tmp2, tmp = in[0];
        out[0] = identity;
        for(size_t i=1; i < in.size(); ++i) {
            tmp2 = in[i];
            out[i] = add(tmp, out[i-1]);
            tmp = tmp2;
        }
    }
}

inline size_t next_power_of_two(size_t v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v |= v >> 32;
	v++;

	return v;
}
inline size_t prev_power_of_two(size_t v) {
    v = next_power_of_two(v);
    return v >> 1;
}

// Needed for down pass
inline size_t left_child(size_t parent) { return 2*parent; }
inline size_t right_child(size_t parent) { return 2*parent+1; }
inline size_t parent_of_child(size_t child) { return child/2; }

/*
// Alternative computation with less memory usage (and possibly less runtime)
template <typename Content, typename add_operator>
void par_prefix_sum_mem_eff(span<Content> in, span<Content> out, bool inclusive, add_operator add, Content identity) {
    bool uneven = (in.size() % 2 == 1);
    const size_t corrected_len = uneven ? in.size() + 1 :
            in.size();
    // Round down to previous number of two, if in is not a power of two (leaves not in tree)
    size_t tree_size = (in.size()&(in.size()-1)) == 0 ?
        next_power_of_two(in.size()) : next_power_of_two(in.size()) >> 1;

    // Memory needed: tree_size + corrected_len/2 (for each two leaves above
    // previous power of 2 one parent is needed)
    container<Content> tree = make_container<Content>(tree_size + corrected_len/2);
    // bottom layer in tree (not containing leaves)
    for(size_t i=0, parent=tree_size; i < in.size(); i+=2, ++parent) {
        tree[parent] = add(in[i], in[i+1]);
        std::cout << "Setting value for node " << parent << " to " <<
            tree[parent] << std::endl;
    }
    // Uneven array -> assign last value to
    if(uneven) {tree[corrected_len/2] = in[in.size()-1];}

    // Up-Pass (first level done separately)
    for(size_t offset = tree_size/2; offset != 1; offset /= 2) {

        //#pragma omp parallel for
        for(size_t i = 0; i < offset; i += 2) {
            const size_t j = offset + i;
            const size_t k = offset + i + 1;
            tree[parent_of_child(j)] = add(tree[j], tree[k]);
        }
    }

    tree[1] = identity;

    // Downpass (skip last layer as it isn't part of tree)
    for(size_t offset = 1; offset != tree_size/2; offset *= 2) {

        const size_t layer_size = offset;

        //#pragma omp parallel for
        for(size_t i = 0; i < layer_size; i++) {
            const size_t j = layer_size - i - 1;

            const Content& from_left = tree[offset + j];
            const Content& left_sum = tree[left_child(offset + j)];
            tree[right_child(offset + j)] = add(from_left, left_sum);
            tree[left_child(offset + j)] = from_left;
        }
    }
}
*/

template <typename Content, typename add_operator>
void par_prefix_sum(span<Content> in, span<Content> out, bool inclusive,
        add_operator add, Content identity) {
    const size_t corrected_len = (in.size() % 2 == 1) ? in.size() + 1 : in.size();
    const size_t tree_size = next_power_of_two(in.size());
    container<Content> tree = make_container<Content>(corrected_len + tree_size);
    //std::cout << "tree size: " << tree.size() << std::endl;

    // copy values of in into last level of tree
    //#pragma omp parallel for schedule(dynamic, 2048)
    #pragma omp simd
    for(size_t i=0; i < in.size(); ++i) {
        tree[tree_size + i] = in[i];
    }

    // Up-Pass

    for(size_t offset = tree_size; offset != 1; offset /= 2) {
        #pragma omp parallel for schedule(dynamic, 2048) shared(tree)
        for(size_t i = 0; i < std::min(offset, corrected_len); i += 2) {
            const size_t j = offset + i;
            const size_t k = offset + i + 1;
            tree[parent_of_child(j)] = add(tree[j], tree[k]);
        }
    }

    // First element needs to be set to identity
    tree[1] = identity;
    // Downpass
    for(size_t offset = 1; offset != tree_size; offset *= 2) {

        const size_t layer_size = offset == tree_size / 2 ? corrected_len / 2 : offset;

        #pragma omp parallel for schedule(dynamic, 2048) shared(tree)
        for(size_t i = 0; i < layer_size; i++) {
            const size_t j = layer_size - i - 1;

            const Content& from_left = tree[offset + j];
            const Content& left_sum = tree[left_child(offset + j)];
            tree[right_child(offset + j)] = add(from_left, left_sum);
            tree[left_child(offset + j)] = from_left;
        }
    }

    // ########################################################
    // FINAL PASS (Copy to out)
    // ########################################################

    // Prefer branch outside of simd-loop
    if(inclusive) {
        #pragma omp simd
        for(size_t i = 0; i < in.size(); ++i) {
            out[i] = add(in[i], tree[tree_size + i]);
        }
    } else {
        #pragma omp simd
        for(size_t i = 0; i < in.size(); ++i) {
            out[i] = tree[tree_size + i];
        }
    }
}
// End Namespace
}
