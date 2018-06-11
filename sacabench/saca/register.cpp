/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "saca/bucket_pointer_refinement.hpp"

#include "saca/deep_shallow/saca.hpp"
#include "saca/example1.hpp"
#include "saca/example2.hpp"
#include "saca/gsaca.hpp"
#include "saca/deep_shallow/saca.hpp"
#include "saca/bucket_pointer_refinement.hpp"
#include "saca/naive.hpp"
#include "saca/prefix_doubling.hpp"
#include "saca/qsufsort.hpp"
#include "saca/sais.hpp"

#include "util/saca.hpp"

namespace sacabench::saca {

using saca_example1 = sacabench::example1::example1;
SACA_REGISTER("Example1", "Description of Example1", saca_example1)

using saca_example2 = sacabench::example2::example2;
SACA_REGISTER("Example2", "Description of Example2", saca_example2)

using deep_shallow = sacabench::deep_shallow::saca;
SACA_REGISTER("Deep-Shallow", "Deep Shallow SACA by Manzini and Ferragina",
              deep_shallow)

using saca_bucket_pointer_refinement =
    sacabench::bucket_pointer_refinement::bucket_pointer_refinement;
SACA_REGISTER("Bucket-Pointer Refinement",
              "Bucket-Pointer Refinement according to Klaus-Bernd Schürmann",
              saca_bucket_pointer_refinement)

using saca_prefix_doubling = sacabench::prefix_doubling::prefix_doubling;
SACA_REGISTER("Prefix Doubling", "TODO", saca_prefix_doubling)

using saca_prefix_doubling_discarding =
    sacabench::prefix_doubling::prefix_doubling_discarding;
SACA_REGISTER("Prefix Doubling+Discarding", "TODO",
              saca_prefix_doubling_discarding)
              
using saca_sais = sacabench::sais::sais;
SACA_REGISTER("SAIS", "Suffix Array Induced Sorting by Nong, Zhang and Chan", saca_sais)              

using saca_gsaca = sacabench::gsaca::gsaca;
SACA_REGISTER("GSACA", "Computes a suffix array with the algorithm gsaca by Uwe Baier.", saca_gsaca)

using saca_qsufsort_naive =
    sacabench::qsufsort::qsufsort_naive;
SACA_REGISTER("Naive qsufsort","Naive Version of N. Larssons and K. SADAKANES qsufsort",
              saca_qsufsort_naive)

using saca_qsufsort = sacabench::qsufsort::qsufsort;
SACA_REGISTER("qsufsort",
              "Improved Version of N. Larssons and K. SADAKANES qsufsort",
              saca_qsufsort)
using saca_naive = sacabench::naive::naive;
SACA_REGISTER("Naiv", "Naiver Algorithmus. Sortiert Suffixe direkt.",
              saca_naive)

} // namespace sacabench::saca

/******************************************************************************/
