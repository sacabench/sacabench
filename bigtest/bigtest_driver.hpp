/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <gtest/gtest.h>
#include <util/sa_check.hpp>
#include <util/saca.hpp>
#include <util/span.hpp>

using namespace sacabench;

template <typename Algorithm, typename sa_index_type>
inline void run_bigtest(const std::string& filename, const size_t prefix_size, const bool print_cases) {
    const auto file_data = util::text_initializer_from_file("../../external/datasets/" + filename, prefix_size);

    ASSERT_LT(file_data.original_text_size(), size_t(-1));

    std::stringstream ss;
    ss << "Test SACA on big file " << "'" << filename << "'";
    ss << " (" << file_data.text_size() << " bytes)" << std::endl;

    if (print_cases) {
        std::cout << ss.str();
    }

    auto output = util::prepare_and_construct_sa<Algorithm, sa_index_type>(file_data);

    auto check_text = util::make_container<util::character>(file_data.text_size());
    file_data.initializer(check_text);

    auto fast_result = sa_check(output.sa_without_sentinels(), check_text);
    if (fast_result != util::sa_check_result::ok) {
        if (!print_cases) {
            std::cout << ss.str();
        }
        ASSERT_EQ(fast_result, util::sa_check_result::ok);
    }
}
