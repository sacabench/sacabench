/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "gtest/gtest.h"
#include <util/saca.hpp>
#include <util/span.hpp>
#include <util/sa_check.hpp>
#include <util/saca.hpp>

#include "bigtest_driver.hpp"

using namespace sacabench;

template<typename Algorithm, typename sa_index_type>
void run_test(const std::string& filename, const bool print_cases) {
    size_t slice_limit = 40;
    const util::string_span text = "hallo"_s;

    std::stringstream ss;

    ss << "Test SACA (big) on ";
    if (text.size() > slice_limit) {
        size_t i = slice_limit;
        while (i < text.size() && (text[i] >> 6 == 0b10)) {
            i++;
        }
        ss << "'" << text.slice(0, i) << "[...]'";
    } else {
        ss << "'" << text << "'";
    }
    ss << " (" << text.size() << " bytes)" << std::endl;

    if (print_cases) {
        std::cout << ss.str();
    }

    auto output = util::prepare_and_construct_sa<Algorithm, sa_index_type>(
        util::text_initializer_from_span(text));

    auto fast_result = sa_check(output.sa_without_sentinels(), text);
    if (fast_result != util::sa_check_result::ok) {
        if (!print_cases) {
            std::cout << ss.str();
        }
        auto slow_result =
            sa_check_naive(output.sa_without_sentinels(), text);
        ASSERT_EQ(bool(fast_result), bool(slow_result))
            << "BUG IN SA CHECKER DETECTED!";
        ASSERT_EQ(fast_result, util::sa_check_result::ok);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
