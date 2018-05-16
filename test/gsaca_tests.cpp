/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/gsaca.hpp.hpp>

using namespace sacabench::gsaca;

TEST(gsaca, test) {
    test::saca_corner_cases<gsaca<>>();
}
