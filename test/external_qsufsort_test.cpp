/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/saca.hpp>
#include <saca/external/qsufsort/qsufsort_wrapper.hpp>
#include "test/saca.hpp"
#include <util/alphabet.hpp>

using namespace sacabench::util;

TEST(qsufsort_ext, construct_example)
{
    string test_span = "hello world"_s;
    auto alp = apply_effective_alphabet(test_span);
    auto out_sa = std::vector<int>(test_span.size());
    qsufsort_ext::construct_sa(test_span,alp,span(out_sa)); 
}

TEST(qsufsort_ext, saca_test) {

    test::saca_corner_cases<qsufsort_ext>();
}
