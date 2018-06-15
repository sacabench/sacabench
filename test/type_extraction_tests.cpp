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
#include "test/saca.hpp"
#include "../sacabench/util/type_extraction.hpp"
#include "../sacabench/util/container.hpp"
#include "../sacabench/util/string.hpp"
#include "../sacabench/util/alphabet.hpp"

using namespace sacabench;

TEST(type_extraction, test_type_l_easy) {
    util::string_span test_text = "caabaccaabacaa"_s;
    size_t test_ind = 0;
    bool is_type_l = std::get<0>(get_type_ltr_dynamic(test_text, test_ind));
    ASSERT_EQ(true, is_type_l);
}





