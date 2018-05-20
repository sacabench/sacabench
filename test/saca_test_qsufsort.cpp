/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/saca.hpp>
#include <saca/qsufsort.hpp>

using namespace sacabench::util;
using namespace sacabench::qsufsort;


TEST(qsufsort, init_sorting)
{
    const string_span test_span = "HONGKONG$"_s;
    auto test_set = std::vector<size_t>(test_span.size());//.fill(0);
    qsufsort::construct_sa(test_span,span(test_set));
}

/*
TEST(qsufsort, paper_example)
{
    const string_span test_span = "TOBEORNOTTOBE$"_s;
    auto test_set = std::vector<size_t>(test_span.size());//.fill(0);
    qsufsort::construct_sa(test_span,span(test_set));
}*/