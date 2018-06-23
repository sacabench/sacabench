/*******************************************************************************
 * test/example_tests.cpp
 *
 * Copyright (C) 2018 Jonas Bode
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <tuple>
#include "test/saca.hpp"
#include "../sacabench/util/type_extraction.hpp"
#include "../sacabench/util/container.hpp"
#include "../sacabench/util/string.hpp"
#include "../sacabench/util/alphabet.hpp"
#include "../sacabench/util/insert_lms.hpp"
#include "../sacabench/saca/sacak.hpp"

using namespace sacabench;


TEST(insert_lms_sacak, test_1) {
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

    util::insert_lms_rtl<util::string, size_t>(input_w, sa);

}

TEST(induced_sorting_sacak, test_1) {

    util::string_span test_text = "caabaccaabacaa"_s;
    util::container<size_t> sa = util::make_container<size_t>(15);
    util::alphabet alph = util::alphabet(test_text);

    for (size_t i = 0; i < sa.size(); i++)
    {
        sa[i] = -1;
    }

    util::string input = util::make_string(test_text);
    util::apply_effective_alphabet(input, alph);
    sacabench::util::string input_w = util::make_container<util::character>(15);

    for (size_t i = 0; i < input_w.size(); i++)
    {
        if (i < input.size())
            input_w[i] = input[i];
        else
            input_w[i] = util::SENTINEL;
    }

    util::insert_lms_rtl<util::string_span, size_t>((util::string_span)input_w, sa);
    sacak::sacak::induced_sort<util::string_span, size_t>((util::string_span)input_w, sa, alph.size_without_sentinel());

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

template<typename sa_index>
void make_sacak(util::string_span &test_text, util::container<sa_index> &sa, bool print)
{
    if(print)
        std::cout << "testing text: " << test_text << std::endl;

    util::alphabet alph = util::alphabet(test_text);

    for (size_t i = 0; i < sa.size(); i++)
    {
        sa[i] = -1;
    }

    util::string input = util::make_string(test_text);
    util::apply_effective_alphabet(input, alph);
    util::string input_w = util::make_container<util::character>(sa.size());

    if (print) {
        std::cout << "effective text is: ";

        for (size_t i = 0; i < input_w.size(); i++)
        {
            if (i < input.size())
                input_w[i] = input[i];
            else
                input_w[i] = util::SENTINEL;

            std::cout << int(input_w[i]) << " ";
        }

        std::cout << std::endl;
    }

    sacak::sacak::calculate_sa<size_t>(input_w, sa, alph.size_without_sentinel());

    if (print) {
        std::cout << "The resulting SA is: [ ";
        for (size_t i = 0; i < sa.size(); i++)
        {
            int symb = int(sa[i]) == -1 ? -1 : int(sa[i]);
            std::cout << symb << " ";
        }
        std::cout << "]" << std::endl;
    }
}

TEST(saca_k, nonrec_1) {

    util::string_span test_text = "kappa"_s;
    util::container<size_t> sa = util::make_container<size_t>(test_text.size() + 1);
    make_sacak(test_text, sa, true);

    ASSERT_EQ(sa[0], 5);
    ASSERT_EQ(sa[1], 4);
    ASSERT_EQ(sa[2], 1);
    ASSERT_EQ(sa[3], 0);
    ASSERT_EQ(sa[4], 3);
    ASSERT_EQ(sa[5], 2);
}



TEST(saca_k, nonrec_2) {

    util::string_span test_text = "abanana"_s;
    util::container<size_t> sa = util::make_container<size_t>(test_text.size() + 1);
    make_sacak(test_text, sa, true);

    ASSERT_EQ(sa[0], 7);
    ASSERT_EQ(sa[1], 6);
    ASSERT_EQ(sa[2], 0);
    ASSERT_EQ(sa[3], 4);
    ASSERT_EQ(sa[4], 2);
    ASSERT_EQ(sa[5], 1);
    ASSERT_EQ(sa[6], 5);
    ASSERT_EQ(sa[7], 3);
}

TEST(saca_k, recursion_1) { // For saca-k we differentiate between recursive (complex words with same LMS substrings) and non recursive calls (simple words)

    util::string_span test_text = "caabaccaabacaa"_s;
    util::container<size_t> sa = util::make_container<size_t>(test_text.size() + 1);
    make_sacak(test_text, sa, true);

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

TEST(saca_k, recursion_2) {

    util::string_span test_text = "minimumaluminium"_s;
    util::container<size_t> sa = util::make_container<size_t>(test_text.size() + 1);
    make_sacak(test_text, sa, true);

    ASSERT_EQ(sa[0], 16);
    ASSERT_EQ(sa[1], 7);
    ASSERT_EQ(sa[2], 3);
    ASSERT_EQ(sa[3], 1);
    ASSERT_EQ(sa[4], 11);
    ASSERT_EQ(sa[5], 13);
    ASSERT_EQ(sa[6], 8);
    ASSERT_EQ(sa[7], 15);
    ASSERT_EQ(sa[8], 6);
    ASSERT_EQ(sa[9], 0);
    ASSERT_EQ(sa[10], 10);
    ASSERT_EQ(sa[11], 4);
    ASSERT_EQ(sa[12], 2);
    ASSERT_EQ(sa[13], 12);
    ASSERT_EQ(sa[14], 14);
    ASSERT_EQ(sa[15], 5);
    ASSERT_EQ(sa[16], 9);

}

TEST(saca_k, error_test_1) {

    util::string_span test_text = "superkalifragilistisch expiallegorisch"_s;
    util::container<size_t> sa = util::make_container<size_t>(test_text.size() + 1);
    make_sacak(test_text, sa, false);

    return;

}

TEST(saca_k, error_test_2) {

    util::string_span test_text = "asdkjh3098u 54/n### 3+5asd333sadFFFFFFFFlorem ipsum und sowas,,/-_-"_s;
    util::container<size_t> sa = util::make_container<size_t>(test_text.size() + 1);
    make_sacak(test_text, sa, false);

    return;

}

TEST(saca_k, corner_cases) {
    test::saca_corner_cases<sacak::sacak>();
}





