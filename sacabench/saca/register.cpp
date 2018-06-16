/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "saca/bucket_pointer_refinement.hpp"

#include "saca/bucket_pointer_refinement.hpp"
#include "saca/dc3.hpp"
#include "saca/dc7.hpp"
#include "saca/deep_shallow/saca.hpp"
#include "saca/gsaca.hpp"
#include "saca/naive.hpp"
#include "saca/prefix_doubling.hpp"
#include "saca/qsufsort.hpp"
#include "saca/sacak.hpp"
#include "saca/sais.hpp"

#include "util/saca.hpp"

namespace sacabench::saca {

SACA_REGISTER(sacabench::deep_shallow::saca)
SACA_REGISTER(sacabench::bucket_pointer_refinement::bucket_pointer_refinement)
SACA_REGISTER(sacabench::prefix_doubling::prefix_doubling)
SACA_REGISTER(sacabench::prefix_doubling::prefix_doubling_discarding)
SACA_REGISTER(sacabench::sais::sais)
SACA_REGISTER(sacabench::gsaca::gsaca)
SACA_REGISTER(sacabench::dc7::dc7)
SACA_REGISTER(sacabench::qsufsort::qsufsort_naive)
SACA_REGISTER(sacabench::qsufsort::qsufsort)
SACA_REGISTER(sacabench::naive::naive)
SACA_REGISTER(sacabench::sacak::sacak)
SACA_REGISTER(sacabench::dc3::dc3)

} // namespace sacabench::saca

/******************************************************************************/
