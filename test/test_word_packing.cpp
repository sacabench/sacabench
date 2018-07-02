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
/*
TEST(word_packing, simple) {
    auto test_set = std::vector<size_t>{3,1,2,1,0};
    auto result_set = std::vector<size_t>(5);
    word_packing(test_set,result_set,4);
}*/
/*
TEST(word_packing, string_test) {

    string test_span = "hello world"_s;
    auto result_set = make_container<size_t>(test_span.size());
    auto alp = apply_effective_alphabet(test_span);
        word_packing(test_span,result_set,alp,0,0);
    for(auto elem : result_set)
    {
        std::cout<<elem<<std::endl;
    }
    std::cout<<"_____________"<<std::endl;

}
TEST(word_packing, test_with_sentinel) {

    string test_span = "hello world"_s;
    auto result_set = make_container<size_t>(test_span.size());
    auto alp = apply_effective_alphabet(test_span);
    word_packing(test_span,result_set,alp,0,1);
    for(auto elem : result_set)
    {
        std::cout<<elem<<std::endl;
    }
}*/