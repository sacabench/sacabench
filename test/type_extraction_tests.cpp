/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Rosa Pink
 * Copyright (C) 2018 Jonas Bode
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <tuple>
#include "../sacabench/util/type_extraction.hpp"
#include "../sacabench/util/container.hpp"
#include "../sacabench/util/string.hpp"
#include "../sacabench/util/alphabet.hpp"
#include "../sacabench/util/insert_lms.hpp"
#include "../sacabench/saca/sacak.hpp"

using namespace sacabench;

TEST(type_extraction, test_type_l_easy) {
    util::string_span test_text = "caabaccaabacaa"_s;
    size_t test_ind = 0;
    bool is_type_l = std::get<0>(get_type_ltr_dynamic(test_text, test_ind));
    ASSERT_EQ(true, is_type_l);
}

TEST(insert_lms, test_1) {
    util::string_span test_text = "caabaccaabacaa"_s;
    util::container<size_t> sa = util::make_container<size_t>(15);
    util::alphabet alph = util::alphabet(test_text);

    for (size_t i = 0; i < sa.size(); i++)
    {
        sa[i] = -1;
    }

    util::string input = util::make_string(test_text);
    util::apply_effective_alphabet(input, alph);
    util::string input_w = util::make_container<util::character>(15);

    for (size_t i = 0; i < input_w.size(); i++) // Simulate appending one sentinel
    {
        if (i < input.size())
            input_w[i] = input[i];
        else
            input_w[i] = util::SENTINEL;
    }

    insert_lms_rtl(input_w, sa);

}

TEST(induced_sorting, test_1) {


    util::string_span test_text = "caabaccaabacaa"_s;
    util::container<size_t> sa = util::make_container<size_t>(15);
    util::alphabet alph = util::alphabet(test_text);

    for (size_t i = 0; i < sa.size(); i++)
    {
        sa[i] = -1;
    }

    util::string input = util::make_string(test_text);
    util::apply_effective_alphabet(input, alph);
    util::string input_w = util::make_container<util::character>(15);

    for (size_t i = 0; i < input_w.size(); i++)
    {
        if (i < input.size())
            input_w[i] = input[i];
        else
            input_w[i] = util::SENTINEL;
    }

    insert_lms_rtl(input_w, sa);
    saca::induced_sort(input_w, sa, alph.size_without_sentinel());

    ASSERT_EQ(sa[0], 14);
    ASSERT_EQ(sa[1], 13);
    ASSERT_EQ(sa[2], 12);
    ASSERT_EQ(sa[3], 7);
    ASSERT_EQ(sa[4], 1);
    ASSERT_EQ(sa[5], 8);
    ASSERT_EQ(sa[6], 2);
    ASSERT_EQ(sa[7], 10);
    ASSERT_EQ(sa[8], 4);
    ASSERT_EQ(sa[9], 9);
    ASSERT_EQ(sa[10], 3);
    ASSERT_EQ(sa[11], 11);
    ASSERT_EQ(sa[12], 6);
    ASSERT_EQ(sa[13], 0);
    ASSERT_EQ(sa[14], 5);
}

TEST(saca_k, test_1) {


    util::string_span test_text = "caabaccaabacaa"_s;
    util::container<size_t> sa = util::make_container<size_t>(15);
    util::alphabet alph = util::alphabet(test_text);

    for (size_t i = 0; i < sa.size(); i++)
    {
        sa[i] = -1;
    }

    util::string input = util::make_string(test_text);
    util::apply_effective_alphabet(input, alph);
    util::string input_w = util::make_container<util::character>(15);

    for (size_t i = 0; i < input_w.size(); i++)
    {
        if (i < input.size())
            input_w[i] = input[i];
        else
            input_w[i] = util::SENTINEL;
    }

    saca::calculate_sa(input_w, sa, alph.size_without_sentinel());

    ASSERT_EQ(sa[0], 14);
    ASSERT_EQ(sa[1], 13);
    ASSERT_EQ(sa[2], 12);
    ASSERT_EQ(sa[3], 7);
    ASSERT_EQ(sa[4], 1);
    ASSERT_EQ(sa[5], 8);
    ASSERT_EQ(sa[6], 2);
    ASSERT_EQ(sa[7], 10);
    ASSERT_EQ(sa[8], 4);
    ASSERT_EQ(sa[9], 9);
    ASSERT_EQ(sa[10], 3);
    ASSERT_EQ(sa[11], 11);
    ASSERT_EQ(sa[12], 6);
    ASSERT_EQ(sa[13], 0);
    ASSERT_EQ(sa[14], 5);
}
