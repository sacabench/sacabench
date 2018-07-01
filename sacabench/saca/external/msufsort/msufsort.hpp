/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <MSufSort.hpp>
#include <util/container.hpp>
#include <util/signed_size_type.hpp>
#include <util/span.hpp>
#include <util/string.hpp>

#include <tudocomp_stat/StatPhase.hpp>

namespace sacabench::reference_sacas {
using namespace sacabench::util;
class msufsort {
public:
    static constexpr size_t EXTRA_SENTINELS = 1;
    static constexpr char const* NAME = "Reference_MSufSort";
    static constexpr char const* DESCRIPTION =
        "Reference MSufSort by M. Maniscalco.";

    template <typename sa_index>
    static void construct_sa(util::string_span text,
                             sacabench::util::alphabet alphabet,
                             util::span<sa_index> out_sa) {
        //TODO: Call MSufSort
    }
};
} // namespace sacabench:reference_sacas
