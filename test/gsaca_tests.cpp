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

// succeeds with reference prev pointer calculation
// succeeds with paper prev pointer calculation
TEST(gsaca, test_input_0) {
    sacabench::util::string_span text = "banane\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);
    int expected_result[7] = {6, 1, 3, 0, 5, 2, 4};

    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < 7; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}

// succeeds with reference prev pointer calculation
// runtime_error with paper prev pointer calculation
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

// succeeds with reference prev pointer calculation
// succeeds with paper prev pointer calculation
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

// succeeds with reference prev pointer calculation
// fails with paper prev pointer calculation
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

// succeeds with reference prev pointer calculation
// succeeds with paper prev pointer calculation
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

// succeeds with reference prev pointer calculation
// runtime_error with paper prev pointer calculation
TEST(gsaca, test_input_5) {
    sacabench::util::string_span text = "asdfasctjkcbweasbebvtiwetwcnbwbbqnqxernqzezwuqwezuetqcrnzxbneqebwcbqwicbqcbtnqweqxcbwuexcbzqwezcqbwecqbwdassdasdfzdfgfsdfsdgfducezctzqwebctuiqwiiqcbnzcebzqc\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);
    int expected_result[157] = {
        156, 14, 4, 0, 109, 105, 30, 136, 16, 58,
        147, 71, 31, 66, 74, 18, 28, 63, 102, 11,
        97, 83, 152, 89, 155, 146, 70, 65, 73, 10,
        82, 88, 150, 127, 26, 100, 95, 53, 6, 137,
        130, 108, 104, 2, 114, 119, 111, 122, 125,
        13, 135, 17, 62, 151, 99, 60, 79, 36, 50,
        23, 86, 93, 128, 47, 41, 3, 124, 115, 117,
        120, 112, 123, 116, 69, 143, 144, 140, 21,
        8, 9, 27, 59, 76, 33, 38, 148, 55, 101, 96,
        154, 145, 72, 52, 61, 32, 133, 77, 91, 45,
        67, 141, 80, 34, 39, 37, 54, 15, 5, 107,
        1, 118, 110, 121, 106, 20, 7, 75, 51, 138,
        24, 131, 126, 49, 85, 139, 44, 19, 29, 64,
        25, 103, 12, 134, 98, 78, 22, 92, 46, 68,
        142, 84, 43, 57, 81, 87, 35, 149, 94, 129,
        113, 40, 153, 132, 90, 48, 42, 56
    };

    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < 157; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}