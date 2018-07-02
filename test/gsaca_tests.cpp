/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <iostream>
#include "test/saca.hpp"

#include <saca/gsaca.hpp>
#include <saca/external/gsaca.hpp>
#include <saca/bucket_pointer_refinement.hpp>
#include <util/string.hpp>

using namespace sacabench::gsaca;

TEST(gsaca, test_corner_cases) {
    test::saca_corner_cases<sacabench::gsaca::gsaca>();
}

TEST(gsaca, test_wrapper) {
    test::saca_corner_cases<sacabench::reference_sacas::gsaca>();
}