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
SACA_REGISTER(deep_shallow::NAME, deep_shallow::DESCRIPTION, deep_shallow)

using saca_bucket_pointer_refinement =
    sacabench::bucket_pointer_refinement::bucket_pointer_refinement;
SACA_REGISTER(saca_bucket_pointer_refinement::NAME,
              saca_bucket_pointer_refinement::DESCRIPTION,
              saca_bucket_pointer_refinement)

using saca_prefix_doubling = sacabench::prefix_doubling::prefix_doubling;
SACA_REGISTER(saca_prefix_doubling::NAME, saca_prefix_doubling::DESCRIPTION,
              saca_prefix_doubling)

using saca_prefix_doubling_discarding =
    sacabench::prefix_doubling::prefix_doubling_discarding;
SACA_REGISTER(saca_prefix_doubling_discarding::NAME,
              saca_prefix_doubling_discarding::DESCRIPTION,
              saca_prefix_doubling_discarding)

using saca_sais = sacabench::sais::sais;
SACA_REGISTER(saca_sais::NAME, saca_sais::DESCRIPTION, saca_sais)

using saca_gsaca = sacabench::gsaca::gsaca;
SACA_REGISTER(saca_gsaca::NAME, saca_gsaca::DESCRIPTION, saca_gsaca)

using saca_dc7 = sacabench::dc7::dc7;
SACA_REGISTER(saca_dc7::NAME, saca_dc7::DESCRIPTION, saca_dc7)

using saca_qsufsort_naive = sacabench::qsufsort::qsufsort_naive;
SACA_REGISTER(saca_qsufsort_naive::NAME, saca_qsufsort_naive::DESCRIPTION,
              saca_qsufsort_naive)

using saca_qsufsort = sacabench::qsufsort::qsufsort;
SACA_REGISTER(saca_qsufsort::NAME, saca_qsufsort::DESCRIPTION, saca_qsufsort)

using saca_naive = sacabench::naive::naive;
SACA_REGISTER("Naiv", "Naiver Algorithmus. Sortiert Suffixe direkt.",
              saca_naive)

using sacak = sacabench::sacak::sacak;
SACA_REGISTER("SACA-K", "Constant-Space SA-Algorithm based on SAIS", sacak);

using saca_dc3 = sacabench::dc3::dc3;
SACA_REGISTER("DC3", "Difference Cover Modulo 3 SACA", saca_dc3)

} // namespace sacabench::saca

/******************************************************************************/
