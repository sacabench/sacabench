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
#include "test/saca.hpp"
#include <util/alphabet.hpp>
#include <tudocomp_stat/StatPhase.hpp>

using namespace sacabench::util;
using namespace sacabench::qsufsort;

/*
TEST(qsufsort, slides_example)
{
    const string_span test_span = "HONGKONG$"_s;
    auto test_set = std::vector<size_t>(test_span.size());
    size_t alphabet_size = 255;
    qsufsort::construct_sa(test_span,alphabet_size,span(test_set));
    
}

TEST(qsufsort, paper_example)
{
    const string_span test_span = "TOBEORNOTTOBE$"_s;
    auto test_set = std::vector<size_t>(test_span.size());
    size_t alphabet_size = 255;
    qsufsort::construct_sa(test_span,alphabet_size,span(test_set));
}

TEST(qsufsort, example)
{
    const string_span test_span = "caabaccaabacaa$"_s;
    auto test_set = std::vector<size_t>(test_span.size());
    size_t alphabet_size = 255;
    qsufsort::construct_sa(test_span,alphabet_size,span(test_set));
}
*/
/*
TEST(qsufsort_naive, saca_test) {

    test::saca_corner_cases<qsufsort_naive>();
}*/

TEST(qsufsort, saca_test) {

    test::saca_corner_cases<qsufsort>();
}
/*
TEST(qsufsort, word_pack) {
    string test_span = "hello world$"_s;
    span<size_t> result_set = make_container<size_t>(test_span.size());
    const auto alp = apply_effective_alphabet(test_span);
    
    qsufsort::construct_sa(test_span,alp,result_set);
    for(auto elem : result_set)
    {
        std::cout<<elem<<", ";
    }
    std::cout<<std::endl;
}
*/
/*
TEST(qsufsort, word_pack) {
    tdc::StatPhase root("SACA");
    {
    string textspan = {'h','e','l','l','o',' ', 'w', 'o','r','l','d','\0'};
    span<size_t> result_set = make_container<size_t>(textspan.size());
    const auto alp = apply_effective_alphabet(textspan.slice(0,textspan.size()-2));
    std::cout<<"Start SA"<<std::endl;
    qsufsort::construct_sa(textspan,alp,result_set);
    std::cout<<"Constructed SA"<<std::endl;
    for(auto elem : result_set)
    {
        std::cout<<elem<<", ";
    }
    std::cout<<std::endl;
    }
}
*/