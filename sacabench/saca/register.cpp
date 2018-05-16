/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "saca/example1.hpp"
#include "saca/example2.hpp"
#include "saca/gsaca.hpp"

#include "util/saca.hpp"

namespace sacabench::saca {

using saca_example1 = sacabench::example1::example1;
SACA_REGISTER("Example1", "Description of Example1", saca_example1)

using saca_example2 = sacabench::example2::example2;
SACA_REGISTER("Example2", "Description of Example2", saca_example2)

using saca_gsaca = sacabench::gsaca::gsaca<>;
SACA_REGISTER("GSACA", "Computes a suffix array with the algorithm gsaca.", saca_gsaca)

}

/******************************************************************************/
