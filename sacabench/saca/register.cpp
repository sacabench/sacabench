/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "util/saca.hpp"

#include "saca/deep_shallow/saca.hpp"
SACA_REGISTER(sacabench::deep_shallow::saca)

#include "saca/bucket_pointer_refinement.hpp"
SACA_REGISTER(sacabench::bucket_pointer_refinement::bucket_pointer_refinement)

#include "saca/prefix_doubling.hpp"
SACA_REGISTER(sacabench::prefix_doubling::prefix_doubling)
SACA_REGISTER(sacabench::prefix_doubling::prefix_doubling_discarding)

#include "saca/sais.hpp"
SACA_REGISTER(sacabench::sais::sais)

#include "saca/gsaca.hpp"
SACA_REGISTER(sacabench::gsaca::gsaca)

#include "saca/dc7.hpp"
SACA_REGISTER(sacabench::dc7::dc7)

#include "saca/qsufsort.hpp"
SACA_REGISTER(sacabench::qsufsort::qsufsort_naive)
SACA_REGISTER(sacabench::qsufsort::qsufsort)

#include "saca/naive.hpp"
SACA_REGISTER(sacabench::naive::naive)

#include "saca/sacak.hpp"
SACA_REGISTER(sacabench::sacak::sacak)

#include "saca/dc3.hpp"
SACA_REGISTER(sacabench::dc3::dc3)

/******************************************************************************/
