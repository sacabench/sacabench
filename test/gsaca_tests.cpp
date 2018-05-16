/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/gsaca.hpp>

using namespace sacabench::gsaca;

TEST(gsaca, test) {
    //test::saca_corner_cases<gsaca>();

    unsigned int n = 14;
    sacabench::util::string_span text = "graindraining\0"_s;
    sacabench::util::container<size_t> place = sacabench::util::make_container<size_t>(text.size());
    sacabench::util::span<size_t> out_sa = sacabench::util::span<size_t>(place);

    int expected_result[14] = {13, 2, 7, 5, 12, 0, 3, 10, 8, 4, 11, 9, 1, 6};
    gsaca::construct_sa(text, 0, out_sa);
    for (int index = 0; index < n; index++) {
        ASSERT_EQ(out_sa[index], expected_result[index]);
    }
}
