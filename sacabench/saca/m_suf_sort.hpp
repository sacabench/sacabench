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

        // initialize chain_stack
        std::stack<std::pair<sa_index, size_t>> chain_stack;
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
template <typename sa_index>
void form_initial_chains(util::string_span text, size_t alphabet_size,
                         special_bits<sa_index>& isa, std::stack<std::pair<sa_index, size_t>>& cstack) {
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
