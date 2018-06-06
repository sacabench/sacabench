/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "saca/example1.hpp"
#include "saca/example2.hpp"
#include "saca/prefix_doubling.hpp"
#include "saca/dc7.hpp"

#include "util/saca.hpp"

namespace sacabench::saca {

using saca_example1 = sacabench::example1::example1;
SACA_REGISTER("Example1", "Description of Example1", saca_example1)

using saca_example2 = sacabench::example2::example2;
SACA_REGISTER("Example2", "Description of Example2", saca_example2)

using saca_prefix_doubling = sacabench::prefix_doubling::prefix_doubling;
SACA_REGISTER("Prefix Doubling", "TODO", saca_prefix_doubling)

using saca_prefix_doubling_discarding =
    sacabench::prefix_doubling::prefix_doubling_discarding;
SACA_REGISTER("Prefix Doubling+Discarding", "TODO",
              saca_prefix_doubling_discarding)

using saca_dc7 = sacabench::dc7::dc7;
SACA_REGISTER("DC7", "Description of DC7 TODO", saca_dc7)


} // namespace sacabench::saca

/******************************************************************************/
