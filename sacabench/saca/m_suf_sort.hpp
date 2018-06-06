
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
#include <util/ISAtoSA.hpp>

//#include<iostream>
#include <vector>
#include <stack>
#include <utility>

namespace sacabench::m_suf_sort {

template <typename sa_index> constexpr sa_index END = ((sa_index)(-1)) >> 1;
template <typename sa_index> constexpr sa_index NEG_BIT = ((sa_index)1) << (util::bits_of<sa_index> - 1);

template<typename content> using pairstack = std::stack<std::pair<content, content>>;


template <typename sa_index>
struct special_bits {
private:
    util::span<sa_index> isa;
    sa_index global_rank;

public:

    void set_END(sa_index index) { isa[index] = END<sa_index>; }
    bool is_END(sa_index index) { return (isa[index] == END<sa_index>); }
    bool is_rank(sa_index index) {
        return (isa[index] & NEG_BIT<sa_index>) > 0;
    }
    void set_rank(sa_index index) {
        isa[index] = global_rank | NEG_BIT<sa_index>;
        // if rank is set, increase global rank
        global_rank++;
    }
    sa_index get_rank(sa_index index) {
        return isa[index] & END<sa_index>;
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
    //return copy of global_rank if needed
    sa_index get_global_rank(){
        return global_rank;
    }

    special_bits(util::span<sa_index> isa_to_be) : isa(isa_to_be), global_rank(0) {}
};

struct m_suf_sort2 {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        // TODO: Check if sa_index type fits for text.size() and extra bits
        // Important: NO extra sentinel added. Catch all text[n] reads!!
        // TODO: Change sentinel handling to exactly 1 extra sentinel.

        // Just for special test case of string '':
        if(text.size() < 1) return;

        // make a special_bits struct out of out_sa for isa
        special_bits isa(out_sa);

        // initialize chain_stack containing pairs of
        // sa_index head of uChain and sa_index length of uChain elements (offset)
        pairstack<sa_index> chain_stack;
        // !! Warning: this HAS to be changed when only type S suffixes should be on stack !!
        util::container<sa_index> all_indices_ = util::make_container<sa_index>(text.size());
        util::span<sa_index> all_indices = util::span<sa_index>(all_indices_);
        for(sa_index i = 0; i < text.size(); i++) {
            all_indices[i] = i;
        }
        // initial length (offset) for u-chains is 0 (0 common prefix characters)
        sa_index length = 0;
        // fill chain_stack initially with length 1- u-Chains
        refine_uChain(text, isa, chain_stack, all_indices, length);

        // begin main loop:
        while(chain_stack.size() > 0){
            std::pair<sa_index, sa_index> current_chain = chain_stack.top();
            chain_stack.pop();
            sa_index chain_index = current_chain.first;
            sa_index length = current_chain.second;

            // if u-Chain is singleton rank it!
            if(isa.is_END(chain_index)) {
                assign_rank(chain_index, isa);
                continue;
            }


            // else follow the chain and refine it
            std::vector<sa_index> to_be_refined_;
            std::vector<sa_index> sorting_induced;

            // follow the chain
            while(true) {
                //TODO: Simple sorting by induction implementation

                // if is not sortable by simple induction
                // refine u-Chain with this element

                //if(!isa.is_rank(chain_index + length)) {
                    to_be_refined_.push_back(chain_index);
                //}
                //else {
                    //sorting_induced.push_back(chain_index);
                //}
                if(isa.is_END(chain_index)) {
                    break;
                }
                // update chain index by following the chain links
                chain_index = isa.get(chain_index);
            }

            //TODO: here sorting and ranking of induced suffixes
            util::span<sa_index> to_be_refined = util::span<sa_index>(to_be_refined_);
            refine_uChain(text, isa, chain_stack, to_be_refined, length);
        }

        // Here, hard coded isa2sa inplace conversion is used. Optimize later (try 2 other options)
        util::isa2sa_inplace2<sa_index>(isa.get_span());

    }
};

// private:
// TODO: Should later differentiate between different ranking scenarios
// in case of sorting repetition sequences no REAL rank is set.
// There is missing a third argument for that.
template <typename sa_index>
void assign_rank(sa_index index, special_bits<sa_index>& isa) {
        isa.set_rank(index);
}

// compare function for introsort that sorts first after text symbols at given indices
// and if both text symbols are the same compares (unique) indices.
template <typename sa_index>
struct compare_uChain_elements {
public:
    // Function for comparisons within introsort
    compare_uChain_elements(sa_index l, const util::string_span text)
        : length(l), input_text(text) {}

    const sa_index length;

    bool operator()(const sa_index& a, const sa_index& b) const {

        /*
        SENTINEL IS ADDED, NO NEED FOR THIS ANYMORE, RIGHT?

        const bool a_is_too_short = length + a >= input_text.size();
        const bool b_is_too_short = length + b >= input_text.size();

        if (a_is_too_short) {
            // b should be larger
            return false;
        }

        if (b_is_too_short) {
            // a should be larger
            return true;
        }
        */
        DCHECK_LT(length + a, input_text.size());
        DCHECK_LT(length + b, input_text.size());

        const util::character at_a = this->input_text[a + length];
        const util::character at_b = this->input_text[b + length];
        // Sort different characters descending
        // (to push "larger" chains first on stack)
        bool is_a_smaller = at_a > at_b;
        // in case of equality of the characters compare indices and sort ascending
        // (to pass chains from left to right and re-link)
        if(at_a == at_b) {
            is_a_smaller = a < b;
        }
        return is_a_smaller;
    }

private:
    const util::string_span input_text;
};

// function that sorts new_chain_IDs and then re-links new uChains in isa and pushes new uChain tuples on stack
template <typename sa_index>
void refine_uChain(util::string_span text, special_bits<sa_index>& isa,
     pairstack<sa_index>& cstack, util::span<sa_index> new_chain_IDs, sa_index length){



         compare_uChain_elements comparator(length, text);
         util::sort::introsort(new_chain_IDs, comparator);

         // last index that is to be linked
         sa_index last_ID = END<sa_index>;

         util::character next_element;
         util::character current_element = text[new_chain_IDs[0] + length];

         for(sa_index i = 1; i < new_chain_IDs.size(); i++) {
             const sa_index current_ID = new_chain_IDs[i-1];
             next_element = text[new_chain_IDs[i] + length];

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
             // set "current" to "next" element for next iteration
             current_element = next_element;
         }
         //last element is automatically beginning of the (lexikographically) smallest chain
         isa.set_link(new_chain_IDs[new_chain_IDs.size() - 1], last_ID);
         std::pair<sa_index, sa_index> new_chain (new_chain_IDs[new_chain_IDs.size() - 1], length + 1);
         cstack.push(new_chain);
}
} // namespace sacabench::m_suf_sort
