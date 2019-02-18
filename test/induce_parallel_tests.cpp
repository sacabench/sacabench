/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/induce_parallel.hpp>
#include <util/string.hpp>
#include <util/container.hpp>

TEST(induce_parallel, induce_l_types) {
    auto t = sacabench::util::container<size_t>{ 1,2,2,1,4,4,1,1,4,4,1,3,3,1,0 };

    const size_t UD = -1;
    //initialize suffix array with correct LMS values
    auto sa_cont = sacabench::util::container<size_t> { 14,UD,6,10,3,UD,UD,UD,UD,UD,UD,UD,UD,UD,UD };
    auto sa = sacabench::util::span<size_t>(sa_cont);

    //run method to test it
    sacabench::util::induce_type_l_parallel(t, sa, 4);

    //expected values
    auto expected = sacabench::util::container<size_t> { 14,13,6,10,3,UD,UD,2,1,12,11,5,9,4,8 };

    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(sa[i], expected[i]);
    }
}

TEST(induce_parallel, induce_s_types) {
    auto t = sacabench::util::container<size_t>{ 1,2,2,1,4,4,1,1,4,4,1,3,3,1,0 };

    const size_t UD = -1;
    //initialize suffix array after l-type-inducing
    auto sa_cont = sacabench::util::container<size_t> { 14,13,6,10,3,UD,UD,2,1,12,11,5,9,4,8 };
    auto sa = sacabench::util::span<size_t>(sa_cont);

    //run method to test it
    sacabench::util::induce_type_s_parallel(t, sa, 4);

    //expected values
    auto expected = sacabench::util::container<size_t> { 14,13,6,0,10,3,7,2,1,12,11,5,9,4,8 };

    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(sa[i], expected[i]);
    }
}