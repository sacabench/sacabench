/*******************************************************************************
 * saca/register.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the TBD license in the LICENSE file.
 ******************************************************************************/

#include "saca/example1.hpp"
#include "saca/example2.hpp"

#include "util/saca.hpp"

using saca_example1 = sacabench::example1::example1;
SACA_REGISTER("Example1", "Description of Example1", saca_example1)

using saca_example2 = sacabench::example2::example2;
SACA_REGISTER("Example2", "Description of Example2", saca_example2)

/******************************************************************************/
