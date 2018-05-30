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

//#include<iostream>
#include <vector>
#include <stack>
#include <utility>

namespace sacabench::m_suf_sort {
template <typename sa_index>
struct special_bits {
private:
    util::span<sa_index> isa;
    static constexpr sa_index negative_bit = ((sa_index)1) << (util::bits_of<sa_index> - 1);

public:
    static constexpr sa_index THIS_IS_MAGIC = ((sa_index)(-1)) >> 1;

    void set_null(sa_index index) { isa[index] = THIS_IS_MAGIC; }
    bool is_null(sa_index index) { return (isa[index] == THIS_IS_MAGIC); }
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
            if(isa.is_null(chain_index)) {
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
    compare_uChain_elements(sa_index l, const string_span text)
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

        const character at_a = this->input_text[a + length];
        const character at_b = this->input_text[b + length];
        const bool is_a_smaller = at_a > at_b;
        if(at_a == at_b) {
            is_a_smaller = a < b;
        }
        return is_a_smaller;
    }

private:
    const string_span input_text;
}

// function that sorts newChainIDs and then re-links new uChains in isa and pushes new uChain tuples on stack
template <typename sa_index>
void refine_uChain(util::string_span text, special_bits<sa_index>& isa,
     std::stack<std::pair<sa_index, sa_index>>& cstack, span<sa_index> newChainIDs, sa_index length){
         compare_uChain_elements comparator(length, text);
         util::sort::introsort(newChainIDs, comparator);
         // TODO: Test if newChainIDs is sorted properly after this step!

         //TODO: Planning what exactly should be done here?!?!

         // set initial value to null as it marks the beginning of a u-Chain
         isa.set_null(newChainIDs[0]);
         for(sa_index i = 0; i < newChainIDs.size() - 1; i++) {
             auto current_element = text[newChainIDs[i] + length];
             auto next_element = text[newChainIDs[i+1] + length];
             // does this chain continue?
             if(current_element != next_element) {
                 // this element marks the end of a chain, push the new chain on stack!
                 std::pair<sa_index, sa_index> new_chain (newChainIDs[i], length + 1);
                 cstack.push(new_chain);
                 // same time, this means next element is beginning of a new chain:
                 // set it to null in isa
                 isa.set_null(newChainIDs[i+1]);
             }
         }
}

//TODO: Rubbish! This is NOT mSufSort, this is some buckety-thingy going on here!
template <typename sa_index>
void form_initial_chains(util::string_span text, size_t alphabet_size,
                         special_bits<sa_index>& isa, std::stack<std::pair<sa_index, sa_index>>& cstack) {
    // contains all last elements of uChains
    // alternatively: use a hashmap and dynamically add new u-Chains
    util::container<sa_index> uChain_links_ = util::make_container<sa_index>(alphabet_size + 1);

    special_bits<sa_index> uChain_links{util::span<sa_index>(uChain_links_)};
    for (sa_index i = 0; i < alphabet_size + 1; i++) {
        // Set initial values for uChains: end-character, null
        uChain_links.set_null(i);
    }
    // Link elements of same u-Chains within one left-to-right scan
    for (sa_index i = 0; i < isa.size(); i++) {
        //TODO: Check if type s suffix, ignore type l at this point!
        isa.set_link(i, uChain_links.get(text[i]));
        uChain_links.set_link(text[i], i);
    }
    // Fill uChains into cstack if they appear in the isa
    for(sa_index i = 1; i < uChain_links.size() + 1; i++) {
        // "largest" uChains first (so they are popped last)
        sa_index j = uChain_links.size() - i;
        if(!uChain_links.is_null(j)) {
            std::pair<size_t, size_t> u_chain (uChain_links.get(j), 1);
            cstack.push(u_chain);
        }
    }
}
} // namespace sacabench::m_suf_sort
