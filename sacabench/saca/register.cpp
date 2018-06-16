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

using deep_shallow = sacabench::deep_shallow::saca;
SACA_REGISTER(deep_shallow)

using saca_bucket_pointer_refinement =
    sacabench::bucket_pointer_refinement::bucket_pointer_refinement;
SACA_REGISTER(saca_bucket_pointer_refinement)

using saca_prefix_doubling = sacabench::prefix_doubling::prefix_doubling;
SACA_REGISTER(saca_prefix_doubling)

using saca_prefix_doubling_discarding =
    sacabench::prefix_doubling::prefix_doubling_discarding;
SACA_REGISTER(saca_prefix_doubling_discarding)

using saca_sais = sacabench::sais::sais;
SACA_REGISTER(saca_sais)

using saca_gsaca = sacabench::gsaca::gsaca;
SACA_REGISTER(saca_gsaca)

using saca_dc7 = sacabench::dc7::dc7;
SACA_REGISTER(saca_dc7)

using saca_qsufsort_naive = sacabench::qsufsort::qsufsort_naive;
SACA_REGISTER(saca_qsufsort_naive)

using saca_qsufsort = sacabench::qsufsort::qsufsort;
SACA_REGISTER(saca_qsufsort)

using saca_naive = sacabench::naive::naive;
SACA_REGISTER(saca_naive)

using sacak = sacabench::sacak::sacak;
SACA_REGISTER(sacak);

using saca_dc3 = sacabench::dc3::dc3;
SACA_REGISTER(saca_dc3)

} // namespace sacabench::saca

/******************************************************************************/
