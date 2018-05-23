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

#include <tuple>
#include <vector>

namespace sacabench::m_suf_sort {
template <typename sa_index>
struct special_bits {
private:
    constexpr sa_index THIS_IS_MAGIC = ((sa_index)1) >> 1;
    span<sa_index> isa;

public:
    void set_null(sa_index index) { isa[index] = THIS_IS_MAGIC; }
    bool is_null(sa_index index) { return (isa[index] == THIS_IS_MAGIC); }
    bool is_rank(sa_index index) { return (isa[index] < 0); }
    bool set_rank(sa_index index, sa_index rank) {
        isa[index] = -rank;
        DCHECK_LT(-rank, 0);
    }
    span<sa_index> get_span(){return(isa);}

    special_bits(span<sa_index> isa_to_be) : isa(isa_to_be) {}
};

struct m_suf_sort {
public:
    template <typename sa_index>
    static void construct_sa(util::string_span text, size_t alphabet_size,
                             util::span<sa_index> out_sa) {
        // TODO: Check if sa_index type fits for text.size() and extra bits
        special_bits isa(out_sa);
        form_initial_chains(text, alphabet_size, isa);
    }
};

// private:
template <typename sa_index>
void form_initial_chains(util::string_span text, size_t alphabet_size,
                         special_bits<sa_index> isa) {
    // contains all last elements of uChains
    container<sa_index> uChain_links_ = make_container<sa_index>(alphabet_size);
    special_bits uChain_links(span(uChain_links_));
    for (sa_index i = 0; i < alphabet_size + 1; i++) {
        // Set initial values for uChains: end-character, null
        uChain_links.set_null(i);
    }
    for (sa_index i = 0; i < isa.size(); i++) {
        //TODO: Check if type s suffix, ignore type l at this point.
        isa[i] = uChain_links[text[i]];
        uChain_links[text[i]] = i;
    }
}
} // namespace sacabench::m_suf_sort
