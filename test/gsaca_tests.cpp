/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/gsaca.hpp>

using namespace sacabench::gsaca;

TEST(gsaca, test_corner_cases) {
    //test::saca_corner_cases<gsaca>();
}

TEST(gsaca, test_input_1) {
    sacabench::util::string_span text = "graindraining\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);
    int expected_result[14] = {13, 2, 7, 5, 12, 0, 3, 10, 8, 4, 11, 9, 1, 6};

    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < 14; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}

TEST(gsaca, test_input_2) {
    sacabench::util::string_span text = "hello world\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);
    int expected_result[12] = {11, 5, 10, 1, 0, 9, 2, 3, 4, 7, 8, 6};

    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < 12; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}

TEST(gsaca, test_input_3) {
    sacabench::util::string_span text = "caabaccaabacaa\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);
    int expected_result[15] = {14, 13, 12, 7, 1, 8, 2, 10, 4, 9, 3, 11, 6, 0, 5};

    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < 15; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}

TEST(gsaca, test_input_4) {
    sacabench::util::string_span text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);
    int expected_result[58] = {
        57, 56, 55, 54, 53, 52, 51, 50,
        49, 48, 47, 46, 45, 44, 43, 42, 41, 40,
        39, 38, 37, 36, 35, 34, 33, 32, 31, 30,
        29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };

    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < 58; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}
