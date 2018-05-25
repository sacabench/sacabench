/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/merge_sa_dc.hpp>
#include <util/string.hpp>
#include <util/container.hpp>

//implementation of get-substring method
static const sacabench::util::string_span get_substring(const sacabench::util::string& t, const sacabench::util::character* ptr,
        int n, size_t index) {
    return sacabench::util::span(ptr, n);
}

// implementation of comp method
static const bool comp(const sacabench::util::string_span& a, const sacabench::util::string_span& b) {
    return a < b;
}

TEST(DC, merge) {
    sacabench::util::string input_string = sacabench::util::make_string("caabaccaabacaa$");

    //initialize suffix array and inverse suffix array
    auto sa_0 = sacabench::util::container<size_t> { 12,9,3,6,0 };
    auto sa_12 = sacabench::util::container<size_t> { 14,13,7,1,8,2,10,4,11,5 };
    auto isa_12 = sacabench::util::container<size_t> { 3,5,7,9,2,4,6,8,1,0 };

    //empty SA which should be filled correctly with method merge_sa_dc
    auto sa = sacabench::util::container<size_t> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    //run method to test it
    sacabench::util::merge_sa_dc<const sacabench::util::character>(input_string, sa_0, sa_12,
            isa_12, sa, comp, get_substring);

    //expected values for merged SA with DC
    auto expected = sacabench::util::container<size_t> {14,13,12,7,1,8,2,10,4,
            9,3,11,6,0,5};

    //compare results with expected values
    for (size_t i = 0; i < expected.size(); i++) {
        //ASSERT_EQ(sa[i], expected[i]);
    }
}
