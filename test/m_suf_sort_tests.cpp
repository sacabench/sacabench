/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/m_suf_sort.hpp>
#include <util/alphabet.hpp>

using namespace sacabench::m_suf_sort;

TEST(m_suf_sort, test) {
    test::saca_corner_cases<m_suf_sort<>>();
}

TEST(m_suf_sort, test_form_initial_chains) {
    util::string_span test_text = "caabaccaabacaa"_s;
    const alphabet alpha = alphabet(test_text);
    alphabet effective_alpha = apply_effective_alphabet(test_text, alpha);
    size_t alpha_size = effective_alpha.size;
    container<sa_index> dummy = make_container(alpha_size);
    span<sa_index> isa_span = span(dummy);
    special_bits isa(isa_span);
    //container<sa_index> expected_result = 

    form_initial_chains(test_text, alpha_size, isa);
    isa_span = isa.get_span();
    ASSERT_EQ(isa_span, )




}
