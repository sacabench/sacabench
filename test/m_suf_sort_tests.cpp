/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
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

TEST(m_suf_sort, test_refine_u_chains) {
    string test_text = util::make_string("aacbcb");
    container<size_t> new_chain_IDs_ {5,4,3,2,1,0};
    span<size_t> new_chains = span<size_t>(new_chain_IDs_);
    size_t length = 0;

    container<size_t> isa_expected_ {END<size_t>, 0, END<size_t>, END<size_t>, 2, 3};
    span<size_t> isa_expected = span<size_t>(isa_expected_);

    container<size_t> isa_ = make_container<size_t>(6);
    span<size_t> isa_x = span<size_t>(isa_);
    special_bits<size_t> isa(isa_x);

    std::stack<std::pair<size_t, size_t>> chain_stack;

    refine_uChain(test_text, isa, chain_stack, new_chains, length);

    ASSERT_EQ(isa.get_span(), isa_expected);
    ASSERT_EQ(chain_stack.size(), 3);
}
