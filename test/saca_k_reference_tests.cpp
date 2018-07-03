/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/external/saca_k/saca_k.hpp>

TEST(saca_k_reference, construct_sa) {
	test::saca_corner_cases<sacabench::reference_sacas::saca_k>();
}
