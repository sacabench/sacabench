/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
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
#include <util/kd_array.hpp>
#include <util/type_extraction.hpp>

//#include<iostream>
#include <vector>
#include <stack>
#include <utility>

namespace sacabench::m_suf_sort {

template <typename sa_index> constexpr sa_index END = ((sa_index)(-1)) >> 1;
template <typename sa_index> constexpr sa_index NEG_BIT = ((sa_index)1) << (util::bits_of<sa_index> - 1);

template<typename content> using pairstack = std::stack<std::pair<content, content>>;
template<typename content2> using pair_si = std::pair<content2, content2>;

template <typename sa_index>
struct type_l_lists {
private:
    util::array2d<pair_si<sa_index>> tupel_list;
    const pair_si<sa_index> null_pair = std::make_pair(END<sa_index>, END<sa_index>);

public:
    bool exists(util::character alpha, util::character beta) {
        return (tupel_list[{alpha,beta}] != null_pair);
    }
    void set_head(sa_index head_index, util::character alpha, util::character beta) {
        pair_si<sa_index> head_tail = std::make_pair(index, index);
        tupel_list[{alpha,beta}] = head_tail;
    }
    void set_tail(sa_index tail_index, util::character alpha, util::character beta) {
        tupel_list[{alpha,beta}] = tail_index;
    }
    sa_index get_tail(util::character alpha, util::character beta) {
        return tupel_list[{alpha,beta}].second;
    }
    sa_index get_head(util::character alpha, util::character beta) {
        return tupel_list[{alpha,beta}].first;
    }
    type_l_lists(size_t alphabet_size) : tupel_list({alphabet_size,alphabet_size}) {
        //initialize all relevant elements as null pairs of END symbols which link to nothing
        const sa_index END = ((sa_index)(-1)) >> 1;
        const pair_si<sa_index> null_pair = std::make_pair(END<sa_index>, END<sa_index>);

        for(size_t i = 0; i<alphabet_size; i++) {
            for(size_t j = 0; j<=i; j++) {
                tupel_list[{i,j}] = null_pair;
            }
        }
    }
};

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
        const auto rank = isa[index] & END<sa_index>;
        return rank;
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

    inline void print() const {
        for(const sa_index i : isa) {
            if(i == END<sa_index>) {
                std::cout << "[END]" << ", ";
            }
            else {
                if((i & NEG_BIT<sa_index>) > 0) {
                    std::cout << "R" << (i & END<sa_index>) << ", ";
                } else {
                    std::cout << "L" << (i & END<sa_index>) << ", ";
                }
            }
        }
    }

    special_bits(util::span<sa_index> isa_to_be) : isa(isa_to_be), global_rank(0) {
        for(size_t i=0; i<isa_to_be.size(); i++) {
            isa[i] = END<sa_index>;
        }
    }
};

template <typename sa_index>
struct m_suf_sort_attr {
    // make a special_bits struct out of out_sa for isa
    special_bits isa;
    //special_bits isa(out_sa);

    // make initial list for induced sort with type l lists
    type_l_lists<sa_index> m_list;
    //type_l_lists<sa_index> m_list{alphabet.max_character_value()};

    // initialize chain_stack containing pairs of
    // sa_index head of uChain and sa_index length of uChain elements (offset)
    pairstack<sa_index> chain_stack;

    util::string_span text;

    m_suf_sort_attr(util::span<sa_index> isa_to_be, size_t alphabet_size, util::string_span input_text) : isa(isa_to_be),
        m_list(alphabet_size), text(input_text) {}
};

struct m_suf_sort2 {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;

    template <typename sa_index>
    static void construct_sa(util::string_span text, util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {
        // TODO: Check if sa_index type fits for text.size() and extra bits
        // Important: NO extra sentinel added. Catch all text[n] reads!!

        // Just for special test case of string '':
        if(text.size() < 1) return;

        // initialize m_suf_sort_attr struct that holds slots text, isa, m_list, chain_stack
        m_suf_sort_attr<sa_index> attr{out_sa, alphabet.max_character_value(), text};

        std::vector<sa_index> type_s_indices_;
        bool is_last_type_l = false;

        for(util::ssize i = text.size()-1; i > -1; i--) {
            // collect all type s suffix indices
            bool is_current_type_l = util::get_type_rtl_dynamic(text, i, is_last_type_l);
            if(!is_current_type_l){
                type_s_indices_.push_back(i);
            }
            // update last type
            is_last_type_l = is_current_type_l;
        }
        util::span<sa_index> type_s_indices = util::span<sa_index>(type_s_indices_);
        // initial length (offset) for u-chains is 0 (0 common prefix characters)
        sa_index length = 0;
        // fill chain_stack initially with length 1- u-Chains
        refine_uChain(attr, all_indices, length);

        // begin main loop:
        while(attr.chain_stack.size() > 0) {
            std::pair<sa_index, sa_index> current_chain = attr.chain_stack.top();
            attr.chain_stack.pop();
            sa_index chain_index = current_chain.first;
            sa_index length = current_chain.second;
            util::character current_char = text[current_chain.first];

            // before anything happens with type s u-Chain elements, rank type l suffixes for same character (or smaller)
            //TODO: do not run through ALL first m_list elements (only those that are still possible)
            for(size_t i = 0; i<=current_char; i++){
                for(size_t j = 0; j <=i; j++) {
                    rank_type_l_list(i,j,attr);
                }
            }

            // if u-Chain is singleton rank it!
            if(attr.isa.is_END(chain_index)) {
                assign_rank(chain_index, false, attr);
                continue;
            }

            // else follow the chain and refine it
            std::vector<sa_index> to_be_refined_;
            std::vector<pair_si<sa_index>> sorting_induced_;

            // follow the chain
            while(true) {

                // if is not sortable by simple induction
                // refine u-Chain with this element

                if(!attr.isa.is_rank(chain_index + length)) {
                    to_be_refined_.push_back(chain_index);
                }
                else {
                    // make new pair from chain_index and sort key, rank of chain_index + length to be sorted by easy induced sort
                    pair_si<sa_index> new_sort_pair = std::make_pair(chain_index, attr.isa.get_rank(chain_index + length));
                    sorting_induced_.push_back(new_sort_pair);
                }
                if(attr.isa.is_END(chain_index)) {
                    break;
                }
                // update chain index by following the chain links
                chain_index = attr.isa.get(chain_index);
            }

            util::span<pair_si<sa_index>> sorting_induced = util::span<pair_si<sa_index>>(sorting_induced_);
            util::span<sa_index> to_be_refined = util::span<sa_index>(to_be_refined_);

            easy_induced_sort(text, attr.isa, sorting_induced);
            // only refine uChain if elements are left!
            if(to_be_refined.size() > 0) {
                refine_uChain(attr, to_be_refined, length);
            }
        }

        // Here, hard coded isa2sa inplace conversion is used. Optimize later (try 2 other options)
        util::isa2sa_inplace2<sa_index>(attr.isa.get_span());

    }
};



// TODO: Should later differentiate between different ranking scenarios
// in case of sorting repetition sequences no REAL rank is set.
// There is missing a third argument for that.
// Argument type_l: if called for a uChain element (either for singleton or for easy_induced_sort) false
// else (if called for element of a type_l_list) true
template <typename sa_index>
void assign_rank(sa_index index, bool type_l, m_suf_sort_attr<sa_index>& attr) {
    attr.isa.set_rank(index);
    // if suffix at index-1 (left neighbor) is type l
    if(index != 0 && util::get_type_rtl_dynamic(attr.text, (index-1), type_l)) {
        util::character alpha = attr.text[index-1];
        util::character beta = attr.text[index];
        sa_index new_tail = index-1;
        // if list for alpha, beta exists
        if(attr.m_list.exists(alpha,beta)) {
            // save last tail index
            sa_index last_tail = attr.m_list.get_tail(alpha,beta);
            // set new tail to currently found new type l element at index-1
            attr.m_list.set_tail(new_tail, alpha, beta);
            // link last and new tail (last tail leads to new tail)
            attr.isa.set_link(last_tail, new_tail);
        }
        else {
            attr.m_list.set_head(new_tail);
        }

        // set attr.isa to END symbol at new_tail
        attr.isa.set_END(new_tail);
    }
}

template <typename sa_index>
void rank_type_l_list(size_t i, size_t j, m_suf_sort_attr attr) {
    // check if list is empty
    if(!attr.m_list[{i,j}].exists()) {
        return;
    }
    else {
        pair_si<sa_index> head_tail = attr.m_list[{i,j}];
        sa_index current_element = attr.isa.get_link(head_tail.first);
        while(current_element != END<sa_index>) {
            //follow the u-Chain
        }
    }

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

// compare function for introsort that sorts a pair of sa_index types after second argument
// used for easy_induced_sort
template <typename sa_index>
struct compare_sortkey {
public:
    // Function for comparisons within introsort
    compare_sortkey(const util::string_span text)
        : input_text(text) {}

    bool operator()(const pair_si<sa_index> pair_a, const pair_si<sa_index> pair_b) const {

        const sa_index sortkey_a = pair_a.second;
        const sa_index sortkey_b = pair_b.second;

        return sortkey_a < sortkey_b;
    }

private:
    const util::string_span input_text;
};

// function that sorts new_chain_IDs and then re-links new uChains in isa and pushes new uChain tuples on stack
template <typename sa_index>
void refine_uChain(m_suf_sort_attr<sa_index>& attr, util::span<sa_index> new_chain_IDs, sa_index length){

         compare_uChain_elements comparator(length, attr.text);
         util::sort::introsort(new_chain_IDs, comparator);

         // last index that is to be linked
         sa_index last_ID = END<sa_index>;

         // declare next and current elements that are to be compared for linking
         util::character next_element;
         util::character current_element = attr.text[new_chain_IDs[0] + length];

         for(sa_index i = 1; i < new_chain_IDs.size(); i++) {
             const sa_index current_ID = new_chain_IDs[i-1];
             next_element = attr.text[new_chain_IDs[i] + length];

             // no matter what, we first link the last element:
             attr.isa.set_link(current_ID, last_ID);

             // does this chain continue?
             if(current_element != next_element) {

                 // this element marks the end of a chain, push the new chain on stack
                 std::pair<sa_index, sa_index> new_chain (current_ID, length + 1);
                 attr.chain_stack.push(new_chain);

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
         attr.isa.set_link(new_chain_IDs[new_chain_IDs.size() - 1], last_ID);
         std::pair<sa_index, sa_index> new_chain (new_chain_IDs[new_chain_IDs.size() - 1], length + 1);
         attr.chain_stack.push(new_chain);
}

//function to sort a set of suffix indices
template <typename sa_index>
void easy_induced_sort(m_suf_sort_attr<sa_index>& attr, util::span<pair_si<sa_index>> to_be_ranked) {
    //sort elements after their sortkey:
    compare_sortkey<sa_index> comparator(attr.text);
    util::sort::introsort(to_be_ranked, comparator);

    //rank all elements in sorted order:
    for(const auto current_index : to_be_ranked) {
        assign_rank(current_index.first, false, attr);
    }
}
} // namespace sacabench::m_suf_sort
