/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <util/container.hpp>
#include <util/span.hpp>
#include <saca/m_suf_sort.hpp>
#include <util/alphabet.hpp>
#include <stack>
#include <utility>

using namespace sacabench;
using namespace sacabench::util;
using namespace sacabench::m_suf_sort;

TEST(m_suf_sort, test) {
    //test::saca_corner_cases<m_suf_sort<>>();
}

TEST(m_suf_sort, test_form_initial_chains) {
    util::string test_text = util::make_string("caabaccaabacaa");
    const alphabet alpha = alphabet(test_text);
    apply_effective_alphabet(test_text, alpha);
    size_t alpha_size = alpha.size;
    util::container<size_t> sa = make_container<size_t>(test_text.size());
    span<size_t> sa_span = span(sa);
    special_bits<size_t> isa{sa_span};
    std::stack<std::pair<size_t, size_t>> chain_stack;
    auto m = special_bits<size_t>::THIS_IS_MAGIC;
    util::container<size_t> expected_result {m,m,1,m,2,0,5,4,7,3,8,6,10,12};

    form_initial_chains<size_t>(span(test_text), alpha_size, isa, chain_stack);
    ASSERT_EQ(chain_stack.size(), 3);
    ASSERT_EQ(isa.get_span(), span(expected_result));
}
