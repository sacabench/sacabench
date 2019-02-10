/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink <rosa.pink@tu-dortmund.de>
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/ISAtoSA.hpp>
#include <util/assertions.hpp>
#include <util/bits.hpp>
#include <util/container.hpp>
#include <util/kd_array.hpp>
#include <util/signed_size_type.hpp>
#include <util/sort/introsort.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/type_extraction.hpp>

#include <saca/deep_shallow/log.hpp>
using sacabench::deep_shallow::duration;
#include <chrono>

#include <tudocomp_stat/StatPhase.hpp>

#include<iostream>
#include <stack>
#include <utility>
#include <vector>

namespace sacabench::m_suf_sort_scan {

template <typename sa_index>
constexpr sa_index END = ((sa_index)(-1)) >> 1;
template <typename sa_index>
constexpr sa_index NEG_BIT = ((sa_index)1) << (util::bits_of<sa_index> - 1);

template <typename content>
using pairstack = std::stack<std::pair<content, content>>;
template <typename content2>
using pair_si = std::pair<content2, content2>;

template <typename sa_index>
struct type_l_lists {
private:
    util::array2d<pair_si<sa_index>> tupel_list;
    const pair_si<sa_index> null_pair =
        std::make_pair(END<sa_index>, END<sa_index>);

public:
    bool exists(util::character alpha, util::character beta) {
        // does no longer exist after tail is deleted, while deleted head only symbolizes
        // this chain is currently being ranked (and no new same chain has been constructed meanwhile)
        return (tupel_list[{alpha, beta}].second != END<sa_index>);
    }
    void set_new_list(sa_index head_index, util::character alpha,
                  util::character beta) {
        DCHECK_NE((head_index | NEG_BIT<sa_index>), head_index);
        pair_si<sa_index> head_tail = std::make_pair(head_index, head_index);
        tupel_list[{alpha, beta}] = head_tail;
    }
    void set_head(sa_index head_index, util::character alpha,
                util::character beta) {
        DCHECK_NE((head_index | NEG_BIT<sa_index>), head_index);
        tupel_list[{alpha, beta}].first = head_index;
    }
    void set_tail(sa_index tail_index, util::character alpha,
                  util::character beta) {
        DCHECK_NE((tail_index | NEG_BIT<sa_index>), tail_index);
        tupel_list[{alpha, beta}].second = tail_index;
    }
    sa_index get_tail(util::character alpha, util::character beta) {
        DCHECK(exists(alpha, beta));
        DCHECK_GE(alpha, beta);
        return tupel_list[{alpha, beta}].second;
    }
    sa_index get_head(util::character alpha, util::character beta) {
        DCHECK(exists(alpha, beta));
        DCHECK_GE(alpha, beta);
        return tupel_list[{alpha, beta}].first;
    }
    void set_empty(util::character alpha, util::character beta) {
        tupel_list[{alpha, beta}] = null_pair;
    }
    type_l_lists(size_t alphabet_size)
        : tupel_list({alphabet_size, alphabet_size}) {
        // initialize all relevant elements as null pairs of END symbols which
        // link to nothing

        for (size_t i = 0; i < alphabet_size; i++) {
            for (size_t j = 0; j <= i; j++) {
                tupel_list[{i, j}] = null_pair;
            }
        }
    }

    type_l_lists operator=(const type_l_lists& rhs) = delete;
    type_l_lists(const type_l_lists& rhs) = delete;
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
        ++global_rank;
    }
    sa_index get_rank(sa_index index) {
        const auto rank = isa[index] & END<sa_index>;
        return rank;
    }
    bool is_link(sa_index index) { return !is_rank(index); }
    void set_link(sa_index index, sa_index link) { isa[index] = link; }
    sa_index get(sa_index index) { return isa[index]; }
    util::span<sa_index> get_span() { return (isa); }
    sa_index size() { return isa.size(); }
    // return copy of global_rank if needed
    sa_index get_global_rank() { return global_rank; }

    inline void print() const {
        for (const sa_index i : isa) {
            if (i == END<sa_index>) {
                std::cout << "[END]"
                          << ", ";
            } else {
                if ((i & NEG_BIT<sa_index>) > 0) {
                    std::cout << "R" << (i & END<sa_index>) << ", ";
                } else {
                    std::cout << "L" << (i & END<sa_index>) << ", ";
                }
            }
        }
    }

    special_bits(util::span<sa_index> isa_to_be)
        : isa(isa_to_be), global_rank(0) {
        for (size_t i = 0; i < isa_to_be.size(); i++) {
            isa[i] = END<sa_index>;
        }
    }
};

template <typename sa_index>
struct m_suf_sort_attr {
    // make a special_bits struct out of out_sa for isa
    special_bits<sa_index> isa;
    // special_bits isa(out_sa);

    // make initial list for induced sort with type l lists
    type_l_lists<sa_index> m_list;
    // type_l_lists<sa_index> m_list{alphabet.max_character_value()};

    // initialize chain_stack containing pairs of
    // sa_index head of uChain and sa_index length of uChain elements (offset)
    pairstack<sa_index> chain_stack;

    // holds original text
    util::string_span text;

    void type_l_insert(size_t element, util::character alpha, util::character beta, bool type_l) {
        // Begin with easy case: if m_list of alpha, beta does not yet exist, make a new list:
        if(!m_list.exists(alpha, beta)) {
            m_list.set_new_list(element, alpha, beta);
        } // if list for alpha, beta exists
        else {
            // check for direct repetition bc else linking would overwrite rank
            // is index < index(sentinel)-1? (char gamma after (alpha, beta) !=
            // sentinel)
            if (element < text.size() - 3) {
                util::character gamma = text[element + 2];
                if (type_l && (alpha == beta) && (beta == gamma)) {
                    sa_index old_tail = m_list.get_tail(alpha, beta);
                    sa_index old_head = m_list.get_head(alpha, beta);
                    if(old_tail == old_head || isa.is_rank(old_tail)) {
                        // make new list bc new_tail is new head of list after this
                        // element is ranked OR tail of the list is already ranked
                        m_list.set_new_list(element, alpha, beta);
                    }
                    else {
                        // update tail so it contains newly found element
                        m_list.set_tail(element, alpha, beta);
                        // also link it in isa
                        isa.set_link(old_tail, element);
                    }

                } else {
                    // save last tail index
                    sa_index last_tail = m_list.get_tail(alpha, beta);
                    // set new tail to currently found new type l element at
                    // index-1
                    m_list.set_tail(element, alpha, beta);
                    // link last and new tail (last tail leads to new tail)
                    isa.set_link(last_tail, element);
                }
            } else {
                // save last tail index
                sa_index last_tail = m_list.get_tail(alpha, beta);
                // set new tail to currently found new type l element at index-1
                m_list.set_tail(element, alpha, beta);
                // link last and new tail (last tail leads to new tail)
                isa.set_link(last_tail, element);
            }
        }

        // set attr.isa to END symbol at new_tail (just to make sure!)
        isa.set_END(element);
    }

    m_suf_sort_attr(util::span<sa_index> isa_to_be, size_t alphabet_size,
                    util::string_span input_text)
        : isa(isa_to_be), m_list(alphabet_size), text(input_text) {}
};

struct m_suf_sort_scan2 {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "mSufSort_scan";
    static constexpr char const* DESCRIPTION = "mSufSort variant with new bucket approach";

    template <typename sa_index>
    static inline void construct_sa(util::string_span text,
                             util::alphabet const& alphabet,
                             util::span<sa_index> out_sa) {

        // Check if sa_index type fits for text.size(), END symbol and sign bit
        DCHECK(util::assert_text_length<sa_index>(text.size()+1, 1u));

        // Just for special test case of string '':
        if (text.size() < 2)
            return;

        // initialize m_suf_sort_attr struct that holds slots text, isa, m_list,
        // chain_stack
        m_suf_sort_attr<sa_index> attr{out_sa, alphabet.size_with_sentinel(),
                                       text};

        // make complete rtl scan for type s/type l suffixes

        bool is_last_type_l = false;

        // auto memory_ = util::make_container<sa_index>(2*alphabet.size_with_sentinel());
        // // array to contain uchain links (last elements)
        // auto init_uchain_links_ = memory_.slice(0, alphabet.size_with_sentinel());
        // // array to contain heads of uchains (rightmost elements)
        // auto head_uchains_ = memory_.slice(alphabet.size_with_sentinel());
        // for(size_t i=0; i<alphabet.size_with_sentinel(); i++) {
        //     init_uchain_links_[i] = END<sa_index>;
        //     head_uchains_[i] = END<sa_index>;
        // }
        auto links_heads = util::make_container<std::pair<sa_index, sa_index>>(alphabet.size_with_sentinel());
        // array to contain pairs of {uchain links (last elements) ; heads of uchains (rightmost elements)}
        auto links_heads_ = links_heads.slice();
        for(size_t i=0; i<alphabet.size_with_sentinel(); i++) {
            links_heads_[i] = std::make_pair(END<sa_index>, END<sa_index>);
        }

        // put head of sentinel chain in array (as it is not included in rtl scan)
        links_heads_[0].second = text.size()-1;

        double time_scan = 0;

        for (util::ssize i = text.size() - 2; i > -1; i--) {
            // collect all type s suffix indices
            bool is_current_type_l = util::get_type_rtl_dynamic(
                text.slice(0, text.size() - 1), i, is_last_type_l);
            if (!is_current_type_l) {
                // get last linked element of corresponding chain:
                const auto last_linked = links_heads_[text[i]].first;
                // if this is first occurence, put it into head array
                if(last_linked == END<sa_index>) {
                    links_heads_[text[i]].second = i;
                }
                // if its not a head, link elements
                else {
                    attr.isa.set_link(links_heads_[text[i]].first, i);
                }
                // and put new last element to link array
                links_heads_[text[i]].first = i;
            }
            // update last type
            is_last_type_l = is_current_type_l;
        }
        // fill stack with initial u-chains by simply iterating over head array
        // begin with greatest (last) element
        for(util::ssize i = alphabet.size_with_sentinel()-1; i>=0; i--) {
            sa_index current_head = links_heads_[i].second;
            if(current_head == END<sa_index>) {
                continue;
            }
            std::pair<sa_index, sa_index> new_chain(current_head, 1);
            attr.chain_stack.push(new_chain);
        }

        // initialize last chain character value
        util::character last_char = 0;

        // vectors for suffixes to be refined and
        // for suffixes to be sorted by induction
        // std::vector<sa_index> to_be_refined_;
        std::vector<pair_si<sa_index>> sorting_induced_;

        // begin main loop:
        while (attr.chain_stack.size() > 0) {
            // clear vectors (possibly filled from previous iteration)
            // to_be_refined_.clear();
            sorting_induced_.clear();

            // top & pop = get first element of stack and remove it from stack
            std::pair<sa_index, sa_index> current_chain =
                attr.chain_stack.top();
            attr.chain_stack.pop();
            // chain_index is the right-most index of a u-Chain
            sa_index chain_index = current_chain.first;
            // length is the length of the u-Chain (common prefix)
            sa_index length = current_chain.second;
            // get current character (u-Chain beginning)
            util::character current_char = text[chain_index];

            // before anything happens with type s u-Chain elements, rank type l
            // suffixes for same character (or smaller)
            for (size_t i = last_char; i <= current_char; i++) {
                for (size_t j = 0; j <= i; j++) {
                    while (attr.m_list.exists(i, j)) {
                        rank_type_l_list(i, j, attr);
                    }
                }
            }

            // if u-Chain is singleton rank it!
            if (attr.isa.is_END(chain_index)) {
                assign_rank(chain_index, false, attr);
                continue;
            }

            // re-fill scan memory for this new chain iteration
            // for(size_t i=0; i<alphabet.size_with_sentinel(); i++) {
            //     init_uchain_links_[i] = END<sa_index>;
            //     head_uchains_[i] = END<sa_index>;
            // }
            for(size_t i=0; i<alphabet.size_with_sentinel(); i++) {
                links_heads_[i] = std::make_pair(END<sa_index>, END<sa_index>);
            }

            // follow the chain
            while (true) {

                // if is not sortable by simple induction
                // refine u-Chain with this element

                if (!attr.isa.is_rank(chain_index + length)) {
                    //to_be_refined_.push_back(chain_index);

                    const auto time_temp = duration([&](){
                        // pointer to head and init memory
                        // auto head_pointer = &head_uchains_[attr.text[chain_index + length]];
                        // auto init_pointer = &init_uchain_links_[attr.text[chain_index + length]];
                        auto init_head_pointer = &links_heads_[attr.text[chain_index + length]];
                        // get last linked element of corresponding chain:
                        const auto last_linked = init_head_pointer->first;
                        // if this is first occurence, put it into head array
                        if(last_linked == END<sa_index>) {
                            init_head_pointer->second = chain_index;
                        } else {
                            // link this element to last element
                            attr.isa.set_link(init_head_pointer->first, chain_index);
                        }
                        // update init uchain links
                        init_head_pointer->first = chain_index;
                    });

                    time_scan += time_temp;
                } else {
                    // make new pair from chain_index and sort key, rank of
                    // chain_index + length to be sorted by easy induced sort
                    pair_si<sa_index> new_sort_pair = std::make_pair(
                        chain_index, attr.isa.get_rank(chain_index + length));
                    sorting_induced_.push_back(new_sort_pair);
                }
                if (attr.isa.is_END(chain_index)) {
                    break;
                }
                // update chain index by following the chain links
                chain_index = attr.isa.get(chain_index);
            }

            util::span<pair_si<sa_index>> sorting_induced =
                util::span<pair_si<sa_index>>(sorting_induced_);

            easy_induced_sort(attr, sorting_induced);

            // fill stack with refined u-chains by simply iterating over head array
            // begin with greatest (last) element
            for(util::ssize i = alphabet.size_with_sentinel()-1; i>=0; i--) {
                sa_index current_head = links_heads_[i].second;
                if(current_head == END<sa_index>) {
                    continue;
                }
                // set all (existing) new chain ends to end:
                attr.isa.set_END(links_heads_[i].first);
                std::pair<sa_index, sa_index> new_chain(current_head, length+sa_index(1));
                attr.chain_stack.push(new_chain);
            }

            // set last character to current
            last_char = current_char;
        }

        // After chain_stack is empty, rank all remaining type-l-lists:
        size_t max_char = alphabet.max_character_value();
        for (size_t i = 0; i <= max_char; i++) {
            for (size_t j = 0; j <= i; j++) {
                // repeat bc following the list could abrouptly end in case of
                // direct repetition
                //(start again = continue for same i,j)
                while (attr.m_list.exists(i, j)) {
                    rank_type_l_list(i, j, attr);
                }
            }
        }

        #ifdef DEBUG
            for (size_t i = 0; i < text.size(); i++) {
                if(attr.isa.is_END(i)) {
                    std::cout << "ERROR: ISA contains END symbol at position " << i << std::endl;
                    std::cout << "Surrounding Text: " << (size_t)text[i-2] << ";" << (size_t)text[i-1] <<
                    ";" << (size_t)text[i] << ";" << (size_t)text[i+1] << ";" << (size_t)text[i+2] << ";" << std::endl;
                }
                else if(attr.isa.is_link(i)) {
                    std::cout << "ERROR: ISA contains LINK!" << std::endl;
                }
            }
            // Debug information: global_rank at text.size()? (Next rank would be invalid)
            std::cout << "Is global rank == text.size()? - " << (attr.isa.get_global_rank() == text.size()) << std::endl;
        #endif

        // Here, hard coded isa2sa inplace conversion is used. Optimize later
        // (try 2 other options)
            util::isa2sa_inplace2<sa_index>(attr.isa.get_span());

    }
};

// TODO: Should later differentiate between different ranking scenarios
// in case of sorting repetition sequences no REAL rank is set.
// Argument type_l: if called for a uChain element (either for singleton or for
// easy_induced_sort) false else (if called for element of a type_l_list) true
template <typename sa_index>
inline void assign_rank(size_t index, bool type_l, m_suf_sort_attr<sa_index>& attr) {
    // set rank for index element (main functionality)
    attr.isa.set_rank(index);

    // check if suffix at index-1 (left neighbor) is type l (secondary
    // functionality)
    if (index != 0 &&
        util::get_type_rtl_dynamic(attr.text, (index - 1), type_l)) {
        util::character alpha = attr.text[index - 1];
        util::character beta = attr.text[index];
        sa_index new_tail = index - 1;

        attr.type_l_insert(new_tail, alpha, beta, type_l);
    }
}

template <typename sa_index>
inline void rank_type_l_list(size_t i, size_t j, m_suf_sort_attr<sa_index>& attr) {
    // check if list is empty
    if (!attr.m_list.exists(i, j)) {
        return;
    } else {
        //sa_index last_idx = attr.m_list.get_tail(i, j);
        sa_index current_idx = attr.m_list.get_head(i, j);
        sa_index next_idx = attr.isa.get(current_idx);

        // set head to END as sign for list is not changed in a strange way.
        // delete head (as it has been visited!)
        attr.m_list.set_head(END<sa_index>, i, j);

        while (current_idx != END<sa_index>) {
            next_idx = attr.isa.get(current_idx);
            // rank current element
            assign_rank(current_idx, true, attr);
            // follow the chain
            current_idx = next_idx;
        }
        // check if no new tail has been attached (if so, do not delete!)
        // if ((current_idx == last_idx) &&
        //     (last_idx == attr.m_list.get_tail(i, j))) {
        if(attr.m_list.get_head(i,j) == END<sa_index>) {
            // delete list after all elements of it have been ranked
            attr.m_list.set_empty(i, j);
        }
    }
}


// compare function for introsort that sorts a pair of sa_index types after
// second argument used for easy_induced_sort
template <typename sa_index>
struct compare_sortkey {
public:
    // Function for comparisons within introsort
    compare_sortkey(const util::string_span text) : input_text(text) {}

    bool operator()(const pair_si<sa_index> pair_a,
                    const pair_si<sa_index> pair_b) const {

        const sa_index sortkey_a = pair_a.second;
        const sa_index sortkey_b = pair_b.second;

        return sortkey_a < sortkey_b;
    }

private:
    const util::string_span input_text;
};


// function to sort a set of suffix indices
template <typename sa_index>
inline void easy_induced_sort(m_suf_sort_attr<sa_index>& attr,
                       util::span<pair_si<sa_index>> to_be_ranked) {
    // sort elements after their sortkey:
    compare_sortkey<sa_index> comparator(attr.text);
    util::sort::introsort(to_be_ranked, comparator);

    // rank all elements in sorted order:
    for (const auto current_index : to_be_ranked) {
        assign_rank(current_index.first, false, attr);
    }
}
} // namespace sacabench::m_suf_sort
