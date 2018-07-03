/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmunde.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/external/dc3/dc3.hpp>

TEST(reference_sacas, dc3) {
    test::saca_corner_cases<sacabench::reference_sacas::dc3>();
}