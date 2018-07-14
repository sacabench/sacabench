/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/


#include <gtest/gtest.h>
#include <util/word_packing.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/alphabet.hpp>

using namespace sacabench::util;
using sacabench::util::word_packing;
TEST(word_packing, greece) {

    string test_span = "Εαμ ανσιλλαε περισυλα συαφιθαθε εξ, δυο ιδ ρεβυμ σομ"_s;
    auto result_set = make_container<size_t>(test_span.size());
    auto alp = apply_effective_alphabet(test_span);
    word_packing(test_span,result_set,alp,0,0);
}