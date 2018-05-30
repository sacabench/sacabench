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

TEST(m_suf_sort, test_introsort_u_chain_refinement) {
    string test_text = util::make_string("aacbcb");
    container<size_t> new_chain_IDs_ {5,4,3,2,1,0};
    span<size_t> new_chains = span<size_t>(new_chain_IDs_);
    size_t length = 0;
    container<size_t> expected_result {2,4,3,5,0,1};
    test_fun<size_t>(string_span(test_text), new_chains, length);
    ASSERT_EQ(new_chains, span<size_t>(expected_result));
}
