/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/assertions.hpp>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/bits.hpp>
#include <util/sort/introsort.hpp>

//#include<iostream>
#include <vector>
#include <stack>
#include <utility>

namespace sacabench::m_suf_sort {

template <typename sa_index> static constexpr sa_index END = ((sa_index)(-1)) >> 1;

template <typename sa_index>
struct special_bits {
private:
    util::span<sa_index> isa;
    static constexpr sa_index negative_bit = ((sa_index)1) << (util::bits_of<sa_index> - 1);

public:

    void set_END(sa_index index) { isa[index] = END<sa_index>; }
    bool is_END(sa_index index) { return (isa[index] == END<sa_index>); }
    bool is_rank(sa_index index) {
        return (isa[index] & negative_bit) > 0;
    }
    void set_rank(sa_index index, sa_index rank) {
        isa[index] = rank | negative_bit;
    }
    bool is_link(sa_index index) {
        return !is_rank(index);
    }
    void set_link(sa_index index, sa_index link) {
        isa[index] = link;
    }
    sa_index get(sa_index index) {
        return isa[index];
    }
    util::span<sa_index> get_span(){return(isa);}
    sa_index size() {
        return isa.size();
    }

    special_bits(util::span<sa_index> isa_to_be) : isa(isa_to_be) {}
};

struct m_suf_sort {
public:

    template <typename sa_index>
    static void construct_sa(util::string_span text, size_t alphabet_size,
                             util::span<sa_index> out_sa) {
        // TODO: Check if sa_index type fits for text.size() and extra bits

        // make a special_bits struct out of out_sa for isa
        special_bits isa(out_sa);

        // Initialize rank
        sa_index rank = alphabet_size + 1;

        // initialize chain_stack containing pairs of
        // sa_index head of uChain and sa_index length of uChain elements (offset)
        std::stack<std::pair<sa_index, sa_index>> chain_stack;
        form_initial_chains(text, alphabet_size, isa, chain_stack);
/*
        // begin main loop:
        while(chain_stack.size() > 0){
            std::pair<sa_index, size_t> current_chain = chain_stack.pop();
            sa_index chain_index = current_chain.first;
            // if u-Chain is singleton rank it!
            if(isa.is_END(chain_index)) {
                assign_rank(chain_index, rank, isa, "uChain");
            }
        }
    */
    }
};

// private:
template <typename sa_index>
void assign_rank(sa_index index, sa_index& rank, special_bits<sa_index>& isa, std::string state) {
    if(state == "uChain") {
        isa.set_rank(index, rank);
        rank--;
    }
}
// compare function for introsort that sorts first after text symbols at given indices
// and if both text symbols are the same compares (unique) indices.
template <typename sa_index>
struct compare_uChain_elements {
public:
    compare_uChain_elements(sa_index l, const util::string_span text)
        : length(l), input_text(text) {}

    const sa_index length;

    // This returns true, if a < b.
    bool operator()(const sa_index& a, const sa_index& b) const {

        // All of the following should never occur as input_text contains 0 at the end.
        /*
        const bool a_is_too_short = length + a >= input_text.size();
        const bool b_is_too_short = length + b >= input_text.size();

        if (a_is_too_short) {
            if (b_is_too_short) {
                // but if both are over the edge, one cannot be smaller.
                return false;
            }

            // b should be larger
            return true;
        }

        if (b_is_too_short) {
            // a should be larger
            return false;
        }
        */

        DCHECK_LT(length + a, input_text.size());
        DCHECK_LT(length + b, input_text.size());

        const util::character at_a = this->input_text[a + length];
        const util::character at_b = this->input_text[b + length];
        bool is_a_smaller = at_a > at_b;
        if(at_a == at_b) {
            is_a_smaller = a < b;
        }
        return is_a_smaller;
    }

private:
    const util::string_span input_text;
};

template <typename sa_index>
void test_fun(util::string_span text, util::span<sa_index> new_chain_IDs, sa_index length){
    compare_uChain_elements comparator(length, text);
    util::sort::introsort(new_chain_IDs, comparator);
}

// function that sorts new_chain_IDs and then re-links new uChains in isa and pushes new uChain tuples on stack
template <typename sa_index>
void refine_uChain(util::string_span text, special_bits<sa_index>& isa,
     std::stack<std::pair<sa_index, sa_index>>& cstack, util::span<sa_index> new_chain_IDs, sa_index length){
         compare_uChain_elements comparator(length, text);
         util::sort::introsort(new_chain_IDs, comparator);

         // last index that is to be linked
         sa_index last_ID = END<sa_index>;

         for(sa_index i = 1; i < new_chain_IDs.size(); i++) {
             auto current_element = text[new_chain_IDs[i-1] + length];
             auto next_element = text[new_chain_IDs[i] + length];
             sa_index current_ID = new_chain_IDs[i-1];

             // no matter what, we first link the last element:
             isa.set_link(current_ID, last_ID);

             // does this chain continue?
             if(current_element != next_element) {

                 // this element marks the end of a chain, push the new chain on stack
                 std::pair<sa_index, sa_index> new_chain (current_ID, length + 1);
                 cstack.push(new_chain);

                 // next link should be set to END to mark the beginning of a new chain
                 last_ID = END<sa_index>;
             }
             else {
                 last_ID = current_ID;
             }
         }
         //last element is automatically beginning of the (lexikographically) smallest chain
         std::pair<sa_index, sa_index> new_chain (new_chain_IDs[new_chain_IDs.size() - 1], length + 1);
         cstack.push(new_chain);
}
} // namespace sacabench::m_suf_sort
