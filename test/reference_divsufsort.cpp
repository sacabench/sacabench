/*******************************************************************************
 * Copyright (C) 2018 Hermann Foot <hermann.foot@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/external/divsufsort.hpp>

TEST(reference_sacas, divsufsort) {
    test::saca_corner_cases<sacabench::reference_sacas::div_suf_sort>();
}
