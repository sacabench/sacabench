/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include "external_saca.hpp"
#include <cstdint>

extern int32_t _ds_Word_size;
extern int32_t Blind_sort_ratio;
extern "C" void ds_ssort(unsigned char* x, int32_t* p, int32_t n);
extern "C" void set_global_variables(void);
extern "C" int check_global_variables(void);

namespace sacabench::reference_sacas {
struct deep_shallow {
    static constexpr size_t EXTRA_SENTINELS = 575;
    static constexpr char const* NAME = "Reference-Deep-Shallow";
    static constexpr char const* DESCRIPTION =
        "Reference implementation of Deep-Shallow";

    template <typename sa_index>
    inline static void construct_sa(util::string_span text,
                                    const util::alphabet&,
                                    util::span<sa_index> out_sa) {
        set_global_variables();
        check_global_variables();
        external_saca_with_writable_text<sa_index, int32_t>(
            text, out_sa, text.size() - EXTRA_SENTINELS, ds_ssort);
    }
};
} // namespace sacabench::reference_sacas
